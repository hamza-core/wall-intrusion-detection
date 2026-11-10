
import json
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

POSE_ROOT = (
    PROJECT_DIR / "artifacts" / "pose_sequences" / "train"
)

ANNOTATION_ROOT = (
    PROJECT_DIR / "artifacts" / "wall_region_annotations" / "train"
)

OUTPUT_ROOT = (
    PROJECT_DIR / "artifacts" / "wall_region_features" / "train"
)

CONFIDENCE_THRESHOLD = 0.25

SEQUENCE_LENGTH = 30
POSE_FEATURE_COUNT = 55

CLASSES = [
    "Normal Class",
    "Wall crossing"
]


# ============================================================
# BODY-PART GROUPS
# COCO KEYPOINT INDEXES
# ============================================================

BODY_GROUPS = {
    "nose": [0],
    "wrists": [9, 10],
    "shoulders": [5, 6],
    "hips": [11, 12],
    "knees": [13, 14],
    "ankles": [15, 16]
}


# ============================================================
# FEATURE NAMES
# ============================================================

WALL_FEATURE_NAMES = []

for group_name in [
    "nose",
    "wrists",
    "shoulders",
    "hips",
    "knees",
    "ankles",
    "bbox_center",
    "bbox_bottom"
]:

    WALL_FEATURE_NAMES.extend([
        f"{group_name}_top_distance",
        f"{group_name}_top_valid",

        f"{group_name}_bottom_distance",
        f"{group_name}_bottom_valid",

        f"{group_name}_region_position",
        f"{group_name}_region_valid"
    ])


# ============================================================
# DISTANCE BETWEEN POINT AND LINE SEGMENT
# ============================================================

def point_to_segment_distance(point, start, end):

    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segment = end - start

    length_squared = np.dot(segment, segment)

    if length_squared == 0:

        return float(np.linalg.norm(point - start))

    projection = np.dot(point - start, segment) / length_squared

    projection = np.clip(projection, 0.0, 1.0)

    nearest = start + projection * segment

    return float(np.linalg.norm(point - nearest))


# ============================================================
# DISTANCE TO POLYLINE
# ============================================================

def distance_to_polyline(point, polyline):

    distances = []

    for index in range(len(polyline) - 1):

        distance = point_to_segment_distance(
            point,
            polyline[index],
            polyline[index + 1]
        )

        distances.append(distance)

    if not distances:
        return None

    return min(distances)


# ============================================================
# INTERPOLATE BOUNDARY Y AT A GIVEN X
# ============================================================

def interpolate_y_at_x(x, polyline, tolerance=1e-8):

    candidates = []

    for index in range(len(polyline) - 1):

        x1, y1 = polyline[index]
        x2, y2 = polyline[index + 1]

        minimum_x = min(x1, x2)
        maximum_x = max(x1, x2)

        if x < minimum_x - tolerance:
            continue

        if x > maximum_x + tolerance:
            continue

        # Vertical segment.
        if abs(x2 - x1) < tolerance:

            if abs(x - x1) <= tolerance:

                candidates.append((y1 + y2) / 2)

            continue

        ratio = (x - x1) / (x2 - x1)

        y = y1 + ratio * (y2 - y1)

        candidates.append(y)

    if not candidates:
        return None

    # For ordinary left-to-right polylines,
    # one segment should normally provide the result.
    return float(np.mean(candidates))


# ============================================================
# PREPARE ANNOTATED WALL BOUNDARIES
# ============================================================

def prepare_wall_boundaries(annotation, width, height):

    top_normalized = np.asarray(
        annotation["wall_top_points"],
        dtype=float
    )

    bottom_normalized = np.asarray(
        annotation["wall_bottom_points"],
        dtype=float
    )

    top = top_normalized.copy()
    bottom = bottom_normalized.copy()

    top[:, 0] *= width - 1
    top[:, 1] *= height - 1

    bottom[:, 0] *= width - 1
    bottom[:, 1] *= height - 1

    return top, bottom


# ============================================================
# MEASURE ONE BODY-PART GROUP
# ============================================================

def measure_body_group(
    points,
    top_boundary,
    bottom_boundary,
    width,
    height
):

    diagonal = np.sqrt(width ** 2 + height ** 2)

    top_distances = []
    bottom_distances = []

    region_candidates = []

    for x, y in points:

        point = np.array([x, y], dtype=float)

        # Distance to top boundary.
        top_distance = distance_to_polyline(
            point,
            top_boundary
        )

        # Distance to bottom boundary.
        bottom_distance = distance_to_polyline(
            point,
            bottom_boundary
        )

        if top_distance is not None:

            top_distances.append(
                top_distance / diagonal
            )

        if bottom_distance is not None:

            bottom_distances.append(
                bottom_distance / diagonal
            )

        # Relative position requires both boundaries
        # to cover this point's horizontal coordinate.
        top_y = interpolate_y_at_x(
            x,
            top_boundary
        )

        bottom_y = interpolate_y_at_x(
            x,
            bottom_boundary
        )

        if top_y is None or bottom_y is None:
            continue

        wall_height = bottom_y - top_y

        if wall_height <= 1e-6:
            continue

        region_position = (y - top_y) / wall_height

        # Select the body point closest to either boundary.
        boundary_distance = min(
            abs(y - top_y),
            abs(y - bottom_y)
        )

        region_candidates.append((
            boundary_distance,
            region_position
        ))

    # Top distance and validity.
    if top_distances:

        top_distance = min(top_distances)
        top_valid = 1.0

    else:

        top_distance = 1.0
        top_valid = 0.0

    # Bottom distance and validity.
    if bottom_distances:

        bottom_distance = min(bottom_distances)
        bottom_valid = 1.0

    else:

        bottom_distance = 1.0
        bottom_valid = 0.0

    # Relative position and validity.
    if region_candidates:

        region_candidates.sort(
            key=lambda item: item[0]
        )

        region_position = region_candidates[0][1]
        region_valid = 1.0

    else:

        region_position = 0.0
        region_valid = 0.0

    return [
        top_distance,
        top_valid,
        bottom_distance,
        bottom_valid,
        region_position,
        region_valid
    ]


# ============================================================
# EXTRACT FEATURES FROM ONE FRAME
# ============================================================

def extract_frame_wall_features(
    pose_frame,
    top_boundary,
    bottom_boundary,
    width,
    height
):

    frame_features = []

    # --------------------------------------------------------
    # KEYPOINT GROUPS
    # --------------------------------------------------------

    for group_name, keypoint_indexes in BODY_GROUPS.items():

        points = []

        for keypoint_index in keypoint_indexes:

            base = keypoint_index * 3

            x = float(pose_frame[base])
            y = float(pose_frame[base + 1])
            confidence = float(pose_frame[base + 2])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            points.append((
                x * (width - 1),
                y * (height - 1)
            ))

        measurements = measure_body_group(
            points,
            top_boundary,
            bottom_boundary,
            width,
            height
        )

        frame_features.extend(measurements)

    # --------------------------------------------------------
    # BOUNDING BOX CENTER
    # --------------------------------------------------------

    center_x = float(pose_frame[51])
    center_y = float(pose_frame[52])

    box_width = float(pose_frame[53])
    box_height = float(pose_frame[54])

    bbox_valid = (
        box_width > 0
        and box_height > 0
        and all(np.isfinite([
            center_x,
            center_y,
            box_width,
            box_height
        ]))
    )

    if bbox_valid:

        center_point = [(
            center_x * (width - 1),
            center_y * (height - 1)
        )]

        bottom_point = [(
            center_x * (width - 1),
            (center_y + box_height / 2) * (height - 1)
        )]

    else:

        center_point = []
        bottom_point = []

    center_measurements = measure_body_group(
        center_point,
        top_boundary,
        bottom_boundary,
        width,
        height
    )

    bottom_measurements = measure_body_group(
        bottom_point,
        top_boundary,
        bottom_boundary,
        width,
        height
    )

    frame_features.extend(center_measurements)
    frame_features.extend(bottom_measurements)

    return frame_features


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(class_name, annotation_path):

    video_id = annotation_path.stem

    pose_path = (
        POSE_ROOT / class_name / f"{video_id}.npz"
    )

    output_path = (
        OUTPUT_ROOT / class_name / f"{video_id}.npz"
    )

    if not pose_path.exists():

        return False, "Missing pose file"

    with open(annotation_path, "r") as file:

        annotation = json.load(file)

    if annotation.get("status") != "annotated":

        return False, "Annotation skipped"

    if annotation.get("annotation_version") != 2:

        return False, "Unsupported annotation version"

    width = int(annotation["frame_width"])
    height = int(annotation["frame_height"])

    if width <= 0 or height <= 0:

        return False, "Invalid frame dimensions"

    top_boundary, bottom_boundary = prepare_wall_boundaries(
        annotation,
        width,
        height
    )

    if len(top_boundary) < 2 or len(bottom_boundary) < 2:

        return False, "Insufficient boundary points"

    with np.load(pose_path) as data:

        pose_features = data["features"].astype(np.float32)
        label = int(data["label"])

    if pose_features.shape != (
        SEQUENCE_LENGTH,
        POSE_FEATURE_COUNT
    ):

        return False, (
            f"Unexpected pose shape: {pose_features.shape}"
        )

    if not np.all(np.isfinite(pose_features)):

        return False, "Non-finite pose features"

    wall_features = []

    for frame_index in range(SEQUENCE_LENGTH):

        frame_features = extract_frame_wall_features(
            pose_features[frame_index],
            top_boundary,
            bottom_boundary,
            width,
            height
        )

        wall_features.append(frame_features)

    wall_features = np.asarray(
        wall_features,
        dtype=np.float32
    )

    combined_features = np.concatenate(
        [pose_features, wall_features],
        axis=1
    )

    if combined_features.shape != (
        SEQUENCE_LENGTH,
        POSE_FEATURE_COUNT + len(WALL_FEATURE_NAMES)
    ):

        return False, (
            f"Unexpected combined shape: {combined_features.shape}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        features=combined_features,
        pose_features=pose_features,
        wall_features=wall_features,
        label=label,
        wall_feature_names=np.asarray(WALL_FEATURE_NAMES)
    )

    return True, "Success"


# ============================================================
# MAIN
# ============================================================

def main():

    total_success = 0
    total_skipped = 0
    total_failed = 0

    print("\n========== WALL REGION FEATURE EXTRACTION ==========\n")

    for class_name in CLASSES:

        annotation_dir = ANNOTATION_ROOT / class_name

        annotation_files = sorted(
            annotation_dir.glob("*.json")
        )

        success_count = 0
        skipped_count = 0
        failed_count = 0

        print(f"\n{class_name}")
        print("-" * 55)

        for index, annotation_path in enumerate(
            annotation_files,
            start=1
        ):

            print(
                f"[{index}/{len(annotation_files)}] "
                f"{annotation_path.name}",
                end=" "
            )

            success, message = process_video(
                class_name,
                annotation_path
            )

            if success:

                success_count += 1
                print("OK")

            elif message == "Annotation skipped":

                skipped_count += 1
                print("SKIPPED")

            else:

                failed_count += 1
                print("FAILED:", message)

        print("\nClass summary:")
        print("Successful:", success_count)
        print("Skipped:", skipped_count)
        print("Failed:", failed_count)

        total_success += success_count
        total_skipped += skipped_count
        total_failed += failed_count

    print("\n========== FINAL RESULTS ==========")
    print("Successful:", total_success)
    print("Skipped:", total_skipped)
    print("Failed:", total_failed)
    print("Total:", total_success + total_skipped + total_failed)
    print("===================================")


if __name__ == "__main__":
    main()
