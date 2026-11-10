"""
Feature Extraction — STEP 1 of training
=======================================

Run from the project root AFTER you put videos in `dataset/train/`:

    python extract_every_5_frames.py              # all videos
    python extract_every_5_frames.py --limit 5    # quick test on 5 videos
    python extract_every_5_frames.py --video 1.mp4

Beginner summary — what this script does to every training video:
    1. Samples EVERY 5TH FRAME (real 30 fps → 6 sampled fps).
       Still sees the whole action, but 5x less work.
    2. Runs YOLO11n-pose (yolo11n-pose.pt, fully offline) on each
       sampled frame → 17 body keypoints per detected person.
    3. Matches each detected person across frames into numbered
       "track slots" (person 0, 1, 2...), up to MAX_PEOPLE = 5,
       so the model can follow one person over time.
    4. Saves, per video:
         pose_sequences/*.npz          (frames, 5, 55)  skeleton numbers
         wall_region_features/*.npz    (frames, 5, 103) skeleton + wall numbers

The wall-region numbers come from the helper library
`extract_wall_region_features.py` — you never need to run it by hand.

What is an .npz file? A compressed folder of NumPy arrays. You never
open it manually — `train_every_5_frames.py` reads it for you.

Plain-English walkthrough: TRAINING_GUIDE.md
"""

from pathlib import Path
import argparse
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from extract_wall_region_features import (
    prepare_wall_boundaries,
    extract_frame_wall_features,
    WALL_FEATURE_NAMES,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "yolo11n-pose.pt"
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

ANNOTATION_ROOT = (
    PROJECT_DIR / "artifacts" / "wall_region_annotations" / "train"
)

OUTPUT_ROOT = (
    PROJECT_DIR / "artifacts" / "every_5_frames" / "train"
)

POSE_OUTPUT = OUTPUT_ROOT / "pose_sequences"
REGION_OUTPUT = OUTPUT_ROOT / "wall_region_features"

FRAME_STRIDE = 5
IMAGE_SIZE = 640
MAX_PEOPLE = 5

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.20

CLASS_LABELS = {
    "Normal Class": 0,
    "Wall crossing": 1,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

POSE_FEATURE_COUNT = 55
REGION_FEATURE_COUNT = len(WALL_FEATURE_NAMES)
TOTAL_FEATURE_COUNT = POSE_FEATURE_COUNT + REGION_FEATURE_COUNT


# ============================================================
# MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

print("Loading YOLO pose model:", MODEL_PATH)
print("Device:", DEVICE)

model = YOLO(str(MODEL_PATH))

if model.task != "pose":
    raise ValueError(f"Expected pose model, got {model.task}")


# ============================================================
# BOUNDING BOX UTILITIES
# ============================================================

def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def correct_box_from_keypoints(box, keypoints, width, height):
    """
    Reconstruct a bounding box from confident pose keypoints.

    Falls back to the original YOLO box if insufficient
    reliable keypoints are available.
    """

    valid = keypoints[:, 2] >= CONFIDENCE_THRESHOLD

    visible = keypoints[valid, :2]

    if len(visible) < 4:
        return box.astype(np.float32)

    x_min = float(np.min(visible[:, 0]))
    y_min = float(np.min(visible[:, 1]))
    x_max = float(np.max(visible[:, 0]))
    y_max = float(np.max(visible[:, 1]))

    body_width = max(1.0, x_max - x_min)
    body_height = max(1.0, y_max - y_min)

    # Add padding because keypoints do not cover the full body silhouette.
    pad_x = body_width * 0.10
    pad_y = body_height * 0.10

    corrected_box = np.array([
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(width, x_max + pad_x),
        min(height, y_max + pad_y),
    ], dtype=np.float32)

    return corrected_box


def normalize_person(box, keypoints, width, height):
    features = []

    for x, y, confidence in keypoints:
        features.extend([
            np.clip(x / width, 0, 1),
            np.clip(y / height, 0, 1),
            np.clip(confidence, 0, 1),
        ])

    x1, y1, x2, y2 = box

    features.extend([
        np.clip(((x1 + x2) / 2) / width, 0, 1),
        np.clip(((y1 + y2) / 2) / height, 0, 1),
        np.clip((x2 - x1) / width, 0, 1),
        np.clip((y2 - y1) / height, 0, 1),
    ])

    return np.asarray(features, dtype=np.float32)
# ============================================================
# DETECTION
# ============================================================

def detect_people(frame):
    height, width = frame.shape[:2]

    result = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
        conf=0.25,
    )[0]

    if (
        result.boxes is None
        or result.keypoints is None
        or len(result.boxes) == 0
    ):
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    people = []

    for index in range(len(boxes)):
        original_box = boxes[index]
        keypoint_data = keypoints[index]

        box = correct_box_from_keypoints(
            original_box,
            keypoint_data,
            width,
            height,
        )

        pose = normalize_person(
            box,
            keypoint_data,
            width,
            height,
        )

        people.append({
            "box": box.astype(np.float32),
            "pose": pose,
            "confidence": float(confidences[index]),
            "area": float(
                max(0, box[2] - box[0])
                * max(0, box[3] - box[1])
            ),
        })

    # Retain highest-confidence detections if there are too many.
    people.sort(
        key=lambda person: person["confidence"],
        reverse=True,
    )

    return people[:MAX_PEOPLE]


# ============================================================
# SIMPLE TRACK ASSOCIATION
# ============================================================
def associate_tracks(
    previous_tracks,
    detections,
    next_track_id,
    width,
    height,
):
    """
    Associate detections with existing tracks using IoU,
    normalized center distance, and bounding-box size similarity.

    Each track retains its assigned feature slot.
    Tracks can survive up to two missed sampled frames.
    """

    candidates = []
    frame_diagonal = np.hypot(width, height)

    for track_id, track in previous_tracks.items():
        previous_box = track["box"]

        ax = (previous_box[0] + previous_box[2]) / 2
        ay = (previous_box[1] + previous_box[3]) / 2

        previous_area = max(
            1.0,
            (previous_box[2] - previous_box[0])
            * (previous_box[3] - previous_box[1])
        )

        for detection_index, detection in enumerate(detections):
            box = detection["box"]

            bx = (box[0] + box[2]) / 2
            by = (box[1] + box[3]) / 2

            iou = box_iou(previous_box, box)

            center_distance = np.hypot(ax - bx, ay - by)
            normalized_distance = center_distance / frame_diagonal

            current_area = max(
                1.0,
                (box[2] - box[0]) * (box[3] - box[1])
            )

            size_similarity = (
                min(previous_area, current_area)
                / max(previous_area, current_area)
            )

            # Reject detections that are too far from the old track.
            if iou < 0.01 and normalized_distance > 0.12:
                continue

            distance_score = max(
                0.0,
                1.0 - normalized_distance / 0.12
            )

            score = (
                0.55 * iou
                + 0.30 * distance_score
                + 0.15 * size_similarity
            )

            candidates.append((
                score,
                track_id,
                detection_index
            ))

    candidates.sort(reverse=True)

    matched_tracks = set()
    matched_detections = set()
    assignments = {}

    for score, track_id, detection_index in candidates:
        if track_id in matched_tracks:
            continue

        if detection_index in matched_detections:
            continue

        assignments[detection_index] = track_id
        matched_tracks.add(track_id)
        matched_detections.add(detection_index)

    # Keep unmatched tracks for up to two sampled frames.
    current_tracks = {}

    for track_id, track in previous_tracks.items():
        missed = track["missed"] + 1

        if missed <= 2:
            current_tracks[track_id] = {
                "box": track["box"],
                "missed": missed,
                "slot": track["slot"],
            }

    # Matched detections retain their original slots.
    for detection_index, track_id in assignments.items():
        detection = detections[detection_index]

        current_tracks[track_id] = {
            "box": detection["box"],
            "missed": 0,
            "slot": previous_tracks[track_id]["slot"],
        }

    # Give unmatched detections a free slot.
    for detection_index, detection in enumerate(detections):
        if detection_index in assignments:
            continue

        occupied_slots = {
            track["slot"]
            for track in current_tracks.values()
        }

        free_slots = [
            slot for slot in range(MAX_PEOPLE)
            if slot not in occupied_slots
        ]

        if not free_slots:
            continue

        track_id = next_track_id
        next_track_id += 1

        slot = free_slots[0]

        assignments[detection_index] = track_id

        current_tracks[track_id] = {
            "box": detection["box"],
            "missed": 0,
            "slot": slot,
        }

    return assignments, current_tracks, next_track_id
# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(video_path, annotation_path):
    annotation = None
    wall_available = False

    if annotation_path.exists():

        with open(annotation_path, "r") as file:
            annotation = json.load(file)

        if annotation.get("annotation_version") != 2:
            return None, "Unsupported annotation version"

        if annotation.get("status") == "annotated":

            width = int(annotation["frame_width"])
            height = int(annotation["frame_height"])

            top_boundary, bottom_boundary = prepare_wall_boundaries(
                annotation,
                width,
                height,
            )

            if len(top_boundary) < 2 or len(bottom_boundary) < 2:
                return None, "Insufficient wall boundary points"

            wall_available = True

        elif annotation.get("status") == "skipped":

            wall_available = False

        else:
            return None, "Unknown annotation status"

    else:
        return None, "Missing wall annotation"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, "Could not open video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, "Invalid video frame count"

    frame_indices = np.arange(0, total_frames, FRAME_STRIDE, dtype=int)
    sampled_frames = len(frame_indices)

    # Each frame has MAX_PEOPLE fixed slots.
    pose_sequence = np.zeros(
        (sampled_frames, MAX_PEOPLE, POSE_FEATURE_COUNT),
        dtype=np.float32,
    )

    region_sequence = np.zeros(
        (sampled_frames, MAX_PEOPLE, REGION_FEATURE_COUNT),
        dtype=np.float32,
    )

    track_sequence = np.full(
        (sampled_frames, MAX_PEOPLE),
        -1,
        dtype=np.int32,
    )

    detection_counts = np.zeros(
        sampled_frames,
        dtype=np.int32,
    )

    previous_tracks = {}
    next_track_id = 0

    for frame_number, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            previous_tracks = {}
            continue

        actual_height, actual_width = frame.shape[:2]

        detections = detect_people(frame)

        assignments, previous_tracks, next_track_id = associate_tracks(
            previous_tracks,
            detections,
            next_track_id,
            actual_width,
            actual_height,
        )

        detection_counts[frame_number] = len(assignments)

        # Store each person in their persistent tracking slot.
        for detection_index, person in enumerate(detections):

            if detection_index not in assignments:
                continue

            track_id = assignments[detection_index]
            slot = previous_tracks[track_id]["slot"]

            pose = person["pose"]

            pose_sequence[frame_number, slot] = pose
            track_sequence[frame_number, slot] = track_id

            if wall_available:
                wall = extract_frame_wall_features(
                    pose, top_boundary, bottom_boundary,
                    actual_width, actual_height,
                )

                region_sequence[frame_number, slot] = np.asarray(
                    wall,
                    dtype=np.float32,
                )
    cap.release()

    if not np.all(np.isfinite(pose_sequence)):
     return None, "Non-finite pose features"

    if wall_available:
        if not np.all(np.isfinite(region_sequence)):
            return None, "Non-finite wall features"

        combined_sequence = np.concatenate(
            [pose_sequence, region_sequence],
            axis=2,
        )
    else:
        combined_sequence = None

    return {
        "features": combined_sequence,
        "pose_features": pose_sequence,
        "wall_features": region_sequence if wall_available else None,
        "track_ids": track_sequence,
        "detection_counts": detection_counts,
        "max_people": MAX_PEOPLE,
        "feature_count": (
            TOTAL_FEATURE_COUNT if wall_available
            else POSE_FEATURE_COUNT
        ),
        "wall_available": wall_available,
    }, "Success"

def wall_features_for_sequence(pose_seq, track_ids, top, bottom, w, h):
    T, P, _ = pose_seq.shape
    out = np.zeros((T, P, REGION_FEATURE_COUNT), np.float32)
    for t in range(T):
        for s in range(P):
            if track_ids[t, s] < 0:        # empty slot stays zero, as now
                continue
            out[t, s] = extract_frame_wall_features(
                pose_seq[t, s], top, bottom, w, h)
    return out

# ============================================================
# PROCESS DATASET
# ============================================================


def main(limit=None, target_video=None):
    videos = []

    for class_name in CLASS_LABELS:
        class_dir = TRAIN_DIR / class_name
        annotation_dir = ANNOTATION_ROOT / class_name

        for video_path in sorted(class_dir.glob("*.mp4")):

            annotation_path = (
                annotation_dir / f"{video_path.stem}.json"
            )

            if not annotation_path.exists():
                continue

            with open(annotation_path, "r") as file:
                annotation = json.load(file)

            if annotation.get("annotation_version") != 2:
                continue

            if annotation.get("status") not in ("annotated", "skipped"):
                continue

            videos.append((video_path, class_name))

    if target_video is not None:
        videos = [
            item for item in videos
            if item[0].stem == target_video
        ]

    if limit is not None:
        videos = videos[:limit]

    print("\n========== MULTI-PERSON EXTRACTION ==========")
    print("Videos:", len(videos))
    print("Frame sampling: every", FRAME_STRIDE, "frames")
    print("Maximum people:", MAX_PEOPLE)
    print("Pose features per person:", POSE_FEATURE_COUNT)
    print("Total features with wall:", TOTAL_FEATURE_COUNT)
    print("Output:", OUTPUT_ROOT)

    successful = 0
    skipped = 0
    failed = 0

    for index, (video_path, class_name) in enumerate(videos, 1):

        annotation_path = (
            ANNOTATION_ROOT
            / class_name
            / f"{video_path.stem}.json"
        )

        pose_path = (
            POSE_OUTPUT
            / class_name
            / f"{video_path.stem}.npz"
        )

        region_path = (
            REGION_OUTPUT
            / class_name
            / f"{video_path.stem}.npz"
        )

        # Read annotation status before checking cached features.
        annotation_status = None

        if annotation_path.exists():
            with open(annotation_path, "r") as file:
                annotation_status = json.load(file).get("status")

        # A skipped annotation means pose-only extraction.
        if annotation_status == "skipped":

            # Remove stale wall features from an earlier extraction.
            if region_path.exists():
                region_path.unlink()

            if pose_path.exists():
                print(
                    f"[{index}/{len(videos)}] "
                    f"Already processed (pose-only): {video_path.name}"
                )
                successful += 1
                continue

        # An annotated video needs both output files.
        elif (
            annotation_status == "annotated"
            and pose_path.exists()
            and region_path.exists()
        ):
            print(
                f"[{index}/{len(videos)}] "
                f"Already processed: {video_path.name}"
            )
            successful += 1
            continue

        # Extract features.
        data, message = extract_video(
            video_path,
            annotation_path,
        )

        if data is None:

            if message == "Annotation skipped":
                skipped += 1
            else:
                failed += 1

            print(
                f"[{index}/{len(videos)}] "
                f"{video_path.name}: {message}"
            )
            continue

        sequence_length = data["pose_features"].shape[0]
        expected_pose_shape = (sequence_length, MAX_PEOPLE, POSE_FEATURE_COUNT)
        expected_track_shape = (sequence_length, MAX_PEOPLE)

        if data["pose_features"].shape != expected_pose_shape:
            failed += 1
            print(f"Invalid pose shape: {data['pose_features'].shape}")
            continue

        if data["track_ids"].shape != expected_track_shape:
            failed += 1
            print(f"Invalid track shape: {data['track_ids'].shape}")
            continue

        if data["wall_available"]:
            expected_region_shape = (sequence_length, MAX_PEOPLE, TOTAL_FEATURE_COUNT)

            if data["features"].shape != expected_region_shape:
                failed += 1
                print(f"Invalid combined feature shape: {data['features'].shape}")
                continue

        label = CLASS_LABELS[class_name]

        # Always save pose features.
        pose_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            pose_path,
            features=data["pose_features"],
            track_ids=data["track_ids"],
            detection_counts=data["detection_counts"],
            label=np.array(label, dtype=np.int64),
        )

        # Save wall features only when a valid wall annotation exists.
        if data["wall_available"]:

            region_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                region_path,
                features=data["features"],
                pose_features=data["pose_features"],
                wall_features=data["wall_features"],
                track_ids=data["track_ids"],
                detection_counts=data["detection_counts"],
                wall_feature_names=np.asarray(WALL_FEATURE_NAMES),
                label=np.array(label, dtype=np.int64),
            )

        successful += 1

        feature_shape = (
            data["features"].shape
            if data["wall_available"]
            else data["pose_features"].shape
        )

        feature_type = (
            "pose + wall"
            if data["wall_available"]
            else "pose-only"
        )

        print(
            f"[{index}/{len(videos)}] "
            f"{class_name}/{video_path.name} "
            f"-> {feature_type}: {feature_shape}, "
            f"people/frame max={data['detection_counts'].max()}"
        )

    print("\n========== SUMMARY ==========")
    print("Successful:", successful)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    main(limit=args.limit, target_video=args.video)