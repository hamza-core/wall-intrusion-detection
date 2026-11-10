"""
Standalone Inference — Every-5-Frames Multi-Person Region LSTM
==============================================================

Self-contained end-to-end prediction for the checkpoint:
    models/multi_person_region_lstm_every_5_frames.pth

Pipeline (identical to training data construction):
    1. Sample the video every FRAME_STRIDE (5) frames.
    2. Detect up to 5 people per sampled frame with the local
       YOLO11n-pose model (never downloaded, never cloud).
    3. Associate detections across frames into track slots.
    4. Build per-person features:
           55  normalized pose features
       +  48  wall-region features (distance to annotated boundaries)
       +   1  wall-annotation-available flag
       = 104 features per person per frame.
    5. Run the BiLSTM (peak + last temporal pooling, masked max
       pooling across people) and classify Wall Crossing / Normal.

Usage:
    python predict.py --video <video.mp4> --annotation <annotation.json>
    python predict.py --video <video.mp4> --annotation <annotation.json> \
        --threshold 0.2 --out result.json

Annotation JSON schema (same as the web frontend / training artifacts):
    {
        "frame_width": 320,
        "frame_height": 240,
        "wall_top_points":    [[x, y], ...],   # normalized [0, 1]
        "wall_bottom_points": [[x, y], ...]    # normalized [0, 1]
    }

Requirements: torch, ultralytics (YOLO), opencv-python, numpy.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Importing the extraction module loads the local YOLO pose model
# into memory (it is placed in this same folder).
import extract_every_5_frames as extractor

from extract_wall_region_features import (
    prepare_wall_boundaries,
    extract_frame_wall_features,
)

from train_every_5_frames import (
    MultiPersonIntrusionLSTM_V2,
    FEATURE_SIZE,
    MAX_PEOPLE,
    POSE_FEATURE_SIZE,
    REGION_FEATURE_SIZE,
)

# ============================================================
# PATHS (relative to this folder)
# ============================================================

STANDALONE_DIR = Path(__file__).resolve().parent

YOLO_PATH = STANDALONE_DIR / "yolo11n-pose.pt"

CHECKPOINT_PATH = (
    STANDALONE_DIR
    / "models"
    / "multi_person_region_lstm_every_5_frames.pth"
)

CLASS_NAMES = ["Normal", "Wall Crossing"]


# ============================================================
# FEATURE EXTRACTION FROM VIDEO
# ============================================================

def build_sequence(video_path, annotation):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video: no frames found")

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    annotation_width = int(annotation["frame_width"])
    annotation_height = int(annotation["frame_height"])

    if annotation_width <= 0 or annotation_height <= 0:
        raise RuntimeError(
            "Annotation frame_width/frame_height must be positive"
        )

    if (
        abs(actual_width - annotation_width) > 2
        or abs(actual_height - annotation_height) > 2
    ):
        print(
            f"WARNING: video is {int(actual_width)}x{int(actual_height)} "
            f"but annotation says {annotation_width}x{annotation_height}"
        )

    top_boundary, bottom_boundary = prepare_wall_boundaries(
        annotation,
        annotation_width,
        annotation_height,
    )

    if len(top_boundary) < 2 or len(bottom_boundary) < 2:
        raise RuntimeError(
            "Wall boundary requires at least 2 points per line"
        )

    frame_indices = np.arange(
        0,
        total_frames,
        extractor.FRAME_STRIDE,
        dtype=int,
    )

    sampled_frames = len(frame_indices)

    pose_sequence = np.zeros(
        (sampled_frames, MAX_PEOPLE, POSE_FEATURE_SIZE),
        dtype=np.float32,
    )

    region_sequence = np.zeros(
        (sampled_frames, MAX_PEOPLE, REGION_FEATURE_SIZE),
        dtype=np.float32,
    )

    track_sequence = np.full(
        (sampled_frames, MAX_PEOPLE),
        -1,
        dtype=np.int32,
    )

    people_detected_total = 0
    max_people_in_frame = 0

    previous_tracks = {}
    next_track_id = 0

    for frame_number, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            previous_tracks = {}
            continue

        actual_w, actual_h = frame.shape[1], frame.shape[0]

        detections = extractor.detect_people(frame)

        people_detected_total += len(detections)
        max_people_in_frame = max(
            max_people_in_frame,
            len(detections),
        )

        assignments, previous_tracks, next_track_id = (
            extractor.associate_tracks(
                previous_tracks,
                detections,
                next_track_id,
                actual_w,
                actual_h,
            )
        )

        for detection_index, person in enumerate(detections):

            if detection_index not in assignments:
                continue

            track_id = assignments[detection_index]
            slot = previous_tracks[track_id]["slot"]

            pose = person["pose"]

            pose_sequence[frame_number, slot] = pose
            track_sequence[frame_number, slot] = track_id

            region_sequence[frame_number, slot] = np.asarray(
                extract_frame_wall_features(
                    pose,
                    top_boundary,
                    bottom_boundary,
                    actual_w,
                    actual_h,
                ),
                dtype=np.float32,
            )

    cap.release()

    if not np.all(np.isfinite(pose_sequence)):
        raise RuntimeError("Non-finite pose features")

    if not np.all(np.isfinite(region_sequence)):
        raise RuntimeError("Non-finite wall-region features")

    # Build the 104-feature array used by the model:
    #   [0:103]  55 pose + 48 wall-region
    #   [103]    wall annotation availability flag (1.0)
    features = np.zeros(
        (sampled_frames, MAX_PEOPLE, FEATURE_SIZE),
        dtype=np.float32,
    )

    features[:, :, :103] = np.concatenate(
        [pose_sequence, region_sequence],
        axis=2,
    )
    features[:, :, 103] = 1.0

    return {
        "features": features,
        "track_ids": track_sequence.astype(np.int64),
        "sequence_length": sampled_frames,
        "total_video_frames": total_frames,
        "people_detected_total": people_detected_total,
        "max_people_in_frame": max_people_in_frame,
    }


# ============================================================
# MODEL LOADING
# ============================================================

def load_lstm(device):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    architecture = MultiPersonIntrusionLSTM_V2(
        input_size=FEATURE_SIZE,
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    architecture.load_state_dict(state_dict)
    architecture.to(device)
    architecture.eval()

    return architecture


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Every-5-Frames Multi-Person Region LSTM "
            "standalone inference"
        ),
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file",
    )

    parser.add_argument(
        "--annotation",
        required=True,
        help=(
            "Path to the wall annotation JSON "
            "(wall_top_points / wall_bottom_points, normalized)"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default: 0.5)",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for the JSON result",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    annotation_path = Path(args.annotation)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if not annotation_path.exists():
        raise SystemExit(f"Annotation not found: {annotation_path}")

    with open(annotation_path, "r") as file:
        annotation = json.load(file)

    device = extractor.DEVICE

    print("=" * 50)
    print("STANDALONE EVERY-5-FRAMES LSTM INFERENCE")
    print("Video:", video_path)
    print("Annotation:", annotation_path)
    print("Threshold:", args.threshold)
    print("Device:", device)
    print("=" * 50)

    started = time.time()

    lstm = load_lstm(device)

    batch = build_sequence(video_path, annotation)

    if batch["max_people_in_frame"] == 0:
        raise SystemExit(
            "No person was detected in any sampled frame."
        )

    features = torch.as_tensor(
        batch["features"],
        dtype=torch.float32,
    ).unsqueeze(0)

    track_ids = torch.as_tensor(
        batch["track_ids"],
        dtype=torch.long,
    ).unsqueeze(0)

    lengths = torch.tensor(
        [batch["sequence_length"]],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = lstm(
            features.to(device),
            track_ids.to(device),
            lengths,
        )

        probability = torch.sigmoid(logits).item()

    prediction = CLASS_NAMES[
        1 if probability >= args.threshold else 0
    ]

    elapsed = time.time() - started

    result = {
        "model": "multi_person_region_lstm_every_5_frames",
        "prediction": prediction,
        "label": CLASS_NAMES.index(prediction),
        "probability": probability,
        "threshold_used": args.threshold,
        "sequence_length": batch["sequence_length"],
        "frames_processed": batch["sequence_length"],
        "total_video_frames": batch["total_video_frames"],
        "max_people_in_frame": batch["max_people_in_frame"],
        "people_detected_total": batch["people_detected_total"],
        "processing_time_seconds": elapsed,
        "device": str(device),
    }

    print("\n========== RESULT ==========")
    print("Prediction:      ", result["prediction"])
    print(f"Probability:      {probability:.4f}")
    print("Frames sampled:  ", result["frames_processed"])
    print("Max people/frame:", result["max_people_in_frame"])
    print(f"Time:             {elapsed:.2f}s")
    print("============================")

    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=2)
        print("Saved:", args.out)

    return result


if __name__ == "__main__":
    main()
