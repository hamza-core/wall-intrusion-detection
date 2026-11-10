"""
ML inference service — standalone Every-5-Frames Multi-Person Region LSTM.

Loads the YOLO pose model and the variable-length BiLSTM once at
application startup, then reuses them for every request.

Unlike the 30-frame V1/V2/V3 pipelines, this model:
    - samples the video every FRAME_STRIDE (5) frames (variable length),
    - consumes 104 features per person (55 pose + 48 wall + 1 flag),
    - requires a `lengths` argument in forward() (packed sequences).

The architecture class is imported verbatim from the copied training
script (`train_every_5_frames.py`) so it cannot drift from the trained
checkpoint. Feature extraction helpers are imported verbatim from the
copied extraction script (`extract_every_5_frames.py`), preserving the
exact numeric behaviour used at training time.
"""

import os
import base64
import sys
import logging
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ML_SOURCE_DIR = _PROJECT_DIR / "ml_source"

for _directory in (_PROJECT_DIR, _ML_SOURCE_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from extract_wall_region_features import (
    prepare_wall_boundaries,
    extract_frame_wall_features,
)

# NOTE: importing extract_every_5_frames loads the local YOLO pose
# model at module import time. Reusing that instance avoids loading
# the weights twice. The module raises at import if the weights file
# is missing, which is detected before the API can serve requests.
import extract_every_5_frames as _extractor

from extract_every_5_frames import (
    normalize_person,
    correct_box_from_keypoints,
    associate_tracks,
    IMAGE_SIZE,
    POSE_FEATURE_COUNT,
    REGION_FEATURE_COUNT,
)

from train_every_5_frames import (
    MultiPersonIntrusionLSTM_V2 as VariableLengthIntrusionLSTM,
    FEATURE_SIZE,
    MAX_PEOPLE,
)

from app.config import settings

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()


# ===========================================================================
# MODEL REGISTRY
# ===========================================================================

class ModelRegistry:
    def __init__(self):
        self.yolo: Optional[YOLO] = None
        self.lstm: Optional[nn.Module] = None
        self.device: torch.device = DEVICE
        self.model_version: str = "every_5_frames"
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading ML models on Device: %s", self.device)

        # YOLO — reuse the instance already loaded by the importer.
        self.yolo = _extractor.model
        if self.yolo is None or getattr(self.yolo, "task", None) != "pose":
            raise RuntimeError("YOLO pose model was not loaded correctly.")

        # LSTM checkpoint (raw state_dict — architecture is same as
        # VariableLengthIntrusionLSTM imported from train_every_5_frames)
        checkpoint_path = settings.lstm_checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {checkpoint_path}")

        self.lstm = VariableLengthIntrusionLSTM(
            input_size=settings.feature_size,
            hidden_size=settings.hidden_size,
            num_layers=settings.num_layers,
            dropout=settings.dropout,
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.lstm.load_state_dict(state_dict)
        self.lstm.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded


registry = ModelRegistry()


# ===========================================================================
# INFERENCE PIPELINE
# ===========================================================================

def _detect_people_on_frame(
    yolo_model: YOLO,
    frame: np.ndarray,
    device: torch.device,
) -> list[dict]:
    h, w = frame.shape[:2]
    result = yolo_model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        device=str(device),
        verbose=False,
        conf=settings.yolo_conf,
    )[0]

    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints_data = result.keypoints.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    people = []
    for idx in range(len(boxes)):
        kp = keypoints_data[idx]
        box = correct_box_from_keypoints(boxes[idx], kp, w, h)
        pose = normalize_person(box, kp, w, h)

        people.append({
            "box": box.astype(np.float32),
            "pose": pose,
            "keypoints": kp.astype(np.float32),
            "confidence": float(confidences[idx]),
            "area": float(max(0, box[2] - box[0]) * max(0, box[3] - box[1])),
        })

    people.sort(key=lambda p: p["confidence"], reverse=True)
    return people[:MAX_PEOPLE]


def _draw_annotated_overlay(
    frame: np.ndarray,
    detections: list[dict],
    assignments: dict,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
    caption: str,
) -> np.ndarray:
    """Draw wall boundaries, bounding boxes and YOLO COCO-17 pose keypoints
    (shared by both the JPEG previews and the annotated video)."""

    preview = frame.copy()

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    keypoint_threshold = 0.25

    # Wall boundary polylines (top blue, bottom orange).
    if len(top_boundary) >= 2:
        pts = top_boundary.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(preview, [pts], False, (255, 170, 90), 2, cv2.LINE_AA)
        cv2.putText(preview, "WALL TOP", (int(pts[0][0][0]) + 4, max(14, int(pts[0][0][1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 90), 1, cv2.LINE_AA)
    if len(bottom_boundary) >= 2:
        pts = bottom_boundary.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(preview, [pts], False, (90, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "WALL BOTTOM", (int(pts[0][0][0]) + 4, max(14, int(pts[0][0][1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 120, 255), 1, cv2.LINE_AA)

    for det_idx, person in enumerate(detections):
        x1, y1, x2, y2 = np.round(person["box"]).astype(int)
        keypoints = person["keypoints"]
        track_id = assignments.get(det_idx)

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 120), 2)

        label = f"Person {track_id} ({person['confidence']:.2f})"
        cv2.putText(
            preview, label, (max(0, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2, cv2.LINE_AA
        )

        for start_idx, end_idx in skeleton:
            start = keypoints[start_idx]
            end = keypoints[end_idx]

            if start[2] >= keypoint_threshold and end[2] >= keypoint_threshold:
                cv2.line(
                    preview,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    (255, 190, 0), 2, cv2.LINE_AA
                )

        for x, y, confidence in keypoints:
            if confidence >= keypoint_threshold:
                cv2.circle(
                    preview, (int(x), int(y)), 4,
                    (0, 80, 255), -1, cv2.LINE_AA
                )

    cv2.putText(preview, caption, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return preview


def _create_annotated_preview(
    frame: np.ndarray,
    detections: list[dict],
    assignments: dict,
    frame_number: int,
    source_frame_index: int,
    fps: float,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> dict:
    """One sampled frame with wall line + detections drawn, JPEG-encoded."""

    caption = f"Sample {frame_number + 1} | Video frame {source_frame_index}"

    overlay = _draw_annotated_overlay(
        frame, detections, assignments, top_boundary, bottom_boundary, caption
    )

    height, width = overlay.shape[:2]
    if width > 640:
        new_height = int(height * 640 / width)
        overlay = cv2.resize(overlay, (640, new_height), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(
        ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    if not success:
        raise RuntimeError("Could not encode annotated preview frame.")

    return {
        "frame_number": frame_number + 1,
        "source_frame_index": source_frame_index,
        "timestamp_seconds": round(source_frame_index / fps, 3) if fps > 0 else 0.0,
        "people_detected": len(detections),
        "image": "data:image/jpeg;base64," + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii"),
    }


def _encode_annotated_video(
    annotated_frames: list[np.ndarray],
    fps_out: float,
) -> Optional[str]:
    """Encode the sampled annotated frames into an H.264 MP4 (browser
    playable) and return it as a data URI, or None if encoding fails."""

    if not annotated_frames:
        return None

    import tempfile

    from app.config import settings

    writer = None
    video_path: Optional[Path] = None

    try:
        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f"annotated_{int(time.time() * 1000)}.mp4"

        height, width = annotated_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            max(1.0, float(fps_out)),
            (width, height),
        )

        if not writer.isOpened():
            logger.warning("H.264 VideoWriter unavailable — annotated video skipped.")
            return None

        for annotated in annotated_frames:
            writer.write(annotated)

        writer.release()
        writer = None

        if not video_path.exists() or video_path.stat().st_size == 0:
            return None

        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        return "data:video/mp4;base64," + encoded

    except Exception:
        logger.exception("Annotated video encoding failed.")
        return None

    finally:
        if writer is not None:
            writer.release()
        if video_path is not None and video_path.exists():
            try:
                video_path.unlink()
            except OSError:
                pass


def apply_velocity_gate(
    raw_score: float,
    pose_sequence: np.ndarray,
    track_sequence: np.ndarray,
    top_boundary: np.ndarray,
    actual_h: int,
) -> tuple[float, bool, Optional[str]]:
    """
    Velocity Gate / Stationary Posture Filter.

    Mitigates false alarms caused by people standing stationary on the ground
    in front of low obstacles or tables where upper body keypoints naturally
    appear above the top boundary line without any climbing action.

    NOTE: tuned for the fixed 30-frame pipeline. Disabled by default in
    this standalone package (settings.velocity_gate_enabled) so the API
    reproduces the every-5-frames model's own evaluation behaviour.
    """
    n_frames, n_people, _ = pose_sequence.shape
    top_y_norm = np.mean([pt[1] for pt in top_boundary]) / max(actual_h, 1)

    max_motion = 0.0
    any_legs_lifted = False
    has_valid_person = False

    for p in range(n_people):
        valid = (track_sequence[:, p] != -1)
        if not np.any(valid):
            continue
        has_valid_person = True
        hips_y = (pose_sequence[valid, p, 34] + pose_sequence[valid, p, 37]) / 2.0
        ankles_y = (pose_sequence[valid, p, 46] + pose_sequence[valid, p, 49]) / 2.0

        if len(hips_y) > 1:
            h_disp = float(hips_y.max() - hips_y.min())
            a_disp = float(ankles_y.max() - ankles_y.min())
            motion = max(h_disp, a_disp)
            if motion > max_motion:
                max_motion = motion

            # Ankles lifted up onto or over the wall top boundary (smaller Y is higher in image)
            if float(np.min(ankles_y)) <= (top_y_norm + 0.01):
                any_legs_lifted = True

    if not has_valid_person:
        return raw_score, False, None

    if max_motion >= 0.07 or any_legs_lifted:
        return raw_score, False, None

    gate_factor = max(0.04, min(1.0, (max_motion / 0.07) ** 2))
    gated_score = raw_score * gate_factor
    note = (
        f"Stationary posture detected (vertical motion={max_motion:.3f} < 0.070, ankles on ground). "
        f"Velocity gate attenuated score from {raw_score:.4f} to {gated_score:.4f}."
    )
    return gated_score, True, note


def run_inference(
    video_path: Path,
    annotation: dict,
    threshold: Optional[float] = None,
) -> dict:
    if not registry.is_loaded:
        raise RuntimeError("Models have not been loaded.")

    decision_threshold = threshold if threshold is not None else settings.threshold
    yolo_model = registry.yolo
    lstm_model = registry.lstm
    device = registry.device

    ann_width = int(annotation["frame_width"])
    ann_height = int(annotation["frame_height"])

    top_boundary, bottom_boundary = prepare_wall_boundaries(
        annotation, ann_width, ann_height
    )

    if len(top_boundary) < 2 or len(bottom_boundary) < 2:
        raise ValueError("Each wall boundary must have at least two points.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path.name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video appears to have no frames.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Variable-length sampling: one frame every FRAME_STRIDE (5) video
    # frames — identical to the training-time extractor.
    frame_indices = np.arange(0, total_frames, settings.frame_stride, dtype=int)
    sampled_frames = len(frame_indices)

    pose_sequence = np.zeros((sampled_frames, MAX_PEOPLE, POSE_FEATURE_COUNT), dtype=np.float32)
    region_sequence = np.zeros((sampled_frames, MAX_PEOPLE, REGION_FEATURE_COUNT), dtype=np.float32)
    track_sequence = np.full((sampled_frames, MAX_PEOPLE), -1, dtype=np.int32)
    detection_counts = np.zeros(sampled_frames, dtype=np.int32)

    previous_tracks: dict = {}
    next_track_id = 0
    preview_frames: list[dict] = []
    sampled_previews: list[int] = []
    annotated_video_frames: list[np.ndarray] = []

    # Evenly cap the returned annotated previews so long videos do not
    # produce hundreds of base64 images. The model still processes every
    # sampled frame regardless of this cap.
    # Annotated previews for the client. A positive value caps them
    # (evenly spaced across the sequence); 0 returns previews for
    # EVERY sampled frame. The model always sees every sampled frame
    # regardless of this setting.
    preview_cap = settings.max_preview_frames
    if preview_cap > 0:
        preview_count = min(preview_cap, sampled_frames)
        preview_indices = set(
            np.linspace(0, sampled_frames - 1, preview_count, dtype=int).tolist()
        )
    else:
        preview_indices = set(range(sampled_frames))

    for frame_number, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            previous_tracks = {}
            continue

        actual_h, actual_w = frame.shape[:2]
        detections = _detect_people_on_frame(yolo_model, frame, device)

        assignments, previous_tracks, next_track_id = associate_tracks(
            previous_tracks, detections, next_track_id, actual_w, actual_h
        )
        detection_counts[frame_number] = len(assignments)

        if frame_number in preview_indices:
            preview_frames.append(
                _create_annotated_preview(
                    frame, detections, assignments, frame_number,
                    int(frame_index), fps, top_boundary, bottom_boundary
                )
            )
            sampled_previews.append(int(frame_index))

        # Annotated video: caption + wall line + boxes + skeleton.
        caption_full = f"Frame {int(frame_index)} | Sample {frame_number + 1}"
        video_overlay = _draw_annotated_overlay(
            frame, detections, assignments,
            top_boundary, bottom_boundary, caption_full,
        )
        if video_overlay.shape[1] > 640:
            new_height = int(video_overlay.shape[0] * 640 / video_overlay.shape[1])
            video_overlay = cv2.resize(
                video_overlay, (640, new_height), interpolation=cv2.INTER_AREA
            )
        annotated_video_frames.append(video_overlay)

        for det_idx, person in enumerate(detections):
            if det_idx not in assignments:
                continue

            track_id = assignments[det_idx]
            slot = previous_tracks[track_id]["slot"]
            pose = person["pose"]

            pose_sequence[frame_number, slot] = pose
            track_sequence[frame_number, slot] = track_id

            wall = extract_frame_wall_features(
                pose, top_boundary, bottom_boundary, actual_w, actual_h
            )
            region_sequence[frame_number, slot] = np.asarray(wall, dtype=np.float32)

    cap.release()

    # Build the 104-feature array exactly like the training loader:
    #   [0:103]  55 pose + 48 wall-region
    #   [103]    wall annotation availability flag = 1.0
    combined = np.zeros(
        (sampled_frames, MAX_PEOPLE, FEATURE_SIZE),
        dtype=np.float32,
    )
    combined[:, :, :103] = np.concatenate([pose_sequence, region_sequence], axis=2)
    combined[:, :, 103] = 1.0

    if not np.isfinite(combined).all():
        raise RuntimeError("Non-finite feature values detected during extraction.")

    max_people_detected = int(detection_counts.max())

    if sampled_frames == 0 or max_people_detected == 0:
        raise RuntimeError(
            "No person was detected in any sampled frame — cannot run inference."
        )

    X = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)
    T = torch.tensor(track_sequence, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([sampled_frames], dtype=torch.long)

    with torch.no_grad():
        logit = lstm_model(X, T, lengths)
        raw_score = torch.sigmoid(logit).item()

    gate_active = False
    gate_note = None
    score = raw_score

    if settings.velocity_gate_enabled:
        score, gate_active, gate_note = apply_velocity_gate(
            raw_score, pose_sequence, track_sequence, top_boundary, actual_h
        )
        if gate_active:
            logger.info("Velocity Gate active: %s", gate_note)

    prediction_label = "Wall Crossing" if score >= decision_threshold else "Normal"

    # Playable annotated MP4 (boxes + skeleton + wall line) over the
    # whole sampled timeline. Plays at the same effective speed as the
    # real video: writer fps = original fps / stride.
    annotated_video = _encode_annotated_video(
        annotated_video_frames,
        fps / settings.frame_stride if fps > 0 else 6.0,
    )
    video_note = None
    if annotated_video is None:
        video_note = "Annotated video could not be encoded for this clip."

    return {
        "prediction": prediction_label,
        "model_score": round(score, 6),
        "raw_model_score": round(raw_score, 6),
        "velocity_gate_active": gate_active,
        "threshold_used": decision_threshold,
        "model_version": registry.model_version,
        "frame_stride": settings.frame_stride,
        "total_video_frames": total_frames,
        "people_detected": max_people_detected,
        "frames_processed": sampled_frames,
        "video_duration_seconds": (
            round((total_frames - 1) / fps, 2) if fps > 0 else None
        ),
        "preview_frames": preview_frames,
        "annotated_video": annotated_video,
        "warning": video_note or (gate_note if gate_active else None),
    }
