
"""
Multi-Person Wall Intrusion Detection
Every-5-Frames Variable-Length BiLSTM Trainer

Input:
    (batch, variable_frames, 5, 104)

Features:
    0:55    Pose features
    55:103  Wall-region features
    103     Annotation availability (1 = annotated, 0 = pose-only)

Architecture:
    Shared per-person BiLSTM
    Peak temporal pooling
    Last-valid-frame pooling
    Person-level masked max pooling

Important:
    - Does not modify the existing 30/45-frame experiments.
    - Does not use horizontal-flip augmentation.
    - Does not truncate long videos.
    - Batch padding is excluded from BiLSTM processing.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "experiment_split_every_5_frames.json"
)

MODEL_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "models_every_5_frames"
)

EXPERIMENT_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "experiments_every_5_frames"
)

MODEL_NAME = "multi_person_region_lstm_every_5_frames"

MAX_PEOPLE = 5
FEATURE_SIZE = 104

POSE_FEATURE_SIZE = 55
REGION_FEATURE_SIZE = 48

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 4
LEARNING_RATE = 0.001
MAX_EPOCHS = 60
PATIENCE = 12

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ============================================================
# PATH HELPER
# ============================================================

def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(samples, dataset_name):

    features_list = []
    track_ids_list = []
    labels_list = []
    lengths_list = []

    annotated_count = 0

    for index, sample in enumerate(samples, start=1):

        class_name = sample["class_name"]
        video_id = str(sample["video_id"])
        expected_label = int(sample["label"])

        pose_path = resolve_path(sample["pose_path"])
        region_path = sample["region_path"]

        if not pose_path.exists():
            raise FileNotFoundError(
                f"Missing pose file: {pose_path}"
            )

        # ----------------------------------------------------
        # LOAD POSE FEATURES
        # ----------------------------------------------------

        with np.load(pose_path) as data:

            pose_features = data["features"].copy()
            track_ids = data["track_ids"].copy()
            label = int(data["label"])

        if pose_features.ndim != 3:
            raise ValueError(
                f"Invalid pose dimensions: {class_name}/{video_id}"
            )

        if pose_features.shape[1:] != (
            MAX_PEOPLE,
            POSE_FEATURE_SIZE,
        ):
            raise ValueError(
                f"Invalid pose shape for {class_name}/{video_id}: "
                f"{pose_features.shape}"
            )

        sequence_length = pose_features.shape[0]

        if sequence_length == 0:
            raise ValueError(
                f"Empty sequence: {class_name}/{video_id}"
            )

        if track_ids.shape != (
            sequence_length,
            MAX_PEOPLE,
        ):
            raise ValueError(
                f"Invalid track shape for {class_name}/{video_id}: "
                f"{track_ids.shape}"
            )

        if label != expected_label:
            raise ValueError(
                f"Pose label mismatch: {class_name}/{video_id}"
            )

        # ----------------------------------------------------
        # CREATE 104-FEATURE ARRAY
        # ----------------------------------------------------

        features = np.zeros(
            (
                sequence_length,
                MAX_PEOPLE,
                FEATURE_SIZE,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # ANNOTATED VIDEO
        # ----------------------------------------------------

        if region_path is not None:

            region_path = resolve_path(region_path)

            if not region_path.exists():
                raise FileNotFoundError(
                    f"Missing region file: {region_path}"
                )

            with np.load(region_path) as data:

                combined_features = data["features"].copy()
                region_tracks = data["track_ids"].copy()
                region_label = int(data["label"])

                # Optional consistency check if available.
                region_pose = (
                    data["pose_features"].copy()
                    if "pose_features" in data
                    else None
                )

            expected_shape = (
                sequence_length,
                MAX_PEOPLE,
                POSE_FEATURE_SIZE + REGION_FEATURE_SIZE,
            )

            if combined_features.shape != expected_shape:
                raise ValueError(
                    f"Invalid region feature shape for "
                    f"{class_name}/{video_id}: "
                    f"{combined_features.shape}"
                )

            if region_tracks.shape != track_ids.shape:
                raise ValueError(
                    f"Region track shape mismatch: "
                    f"{class_name}/{video_id}"
                )

            if not np.array_equal(region_tracks, track_ids):
                raise ValueError(
                    f"Pose/region track IDs differ: "
                    f"{class_name}/{video_id}"
                )

            if region_label != expected_label:
                raise ValueError(
                    f"Region label mismatch: "
                    f"{class_name}/{video_id}"
                )

            if region_pose is not None:
                if region_pose.shape != pose_features.shape:
                    raise ValueError(
                        f"Stored pose shape mismatch: "
                        f"{class_name}/{video_id}"
                    )

                if not np.allclose(
                    region_pose,
                    pose_features,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"Pose features differ between files: "
                        f"{class_name}/{video_id}"
                    )

            # First 103 features:
            # 55 pose + 48 wall-region.
            features[:, :, :103] = combined_features

            # Final feature:
            # 1 means annotation is available.
            features[:, :, 103] = 1.0

            annotated_count += 1

        # ----------------------------------------------------
        # POSE-ONLY VIDEO
        # ----------------------------------------------------

        else:

            # First 55 features contain pose data.
            features[:, :, :55] = pose_features

            # Wall features remain zero.
            # Annotation availability remains zero.
            features[:, :, 103] = 0.0

        # ----------------------------------------------------
        # STORE SAMPLE
        # ----------------------------------------------------

        features_list.append(features.astype(np.float32))
        track_ids_list.append(track_ids.astype(np.int64))

        labels_list.append(expected_label)
        lengths_list.append(sequence_length)

        if index % 100 == 0:
            print(
                f"Loaded {index}/{len(samples)} "
                f"{dataset_name} videos"
            )

    labels_array = np.asarray(
        labels_list,
        dtype=np.float32,
    )

    print("\n" + "=" * 55)
    print(f"{dataset_name} DATASET")
    print("=" * 55)

    print("Samples:", len(samples))
    print("Total sampled frames:", sum(lengths_list))
    print("Minimum sequence length:", min(lengths_list))
    print("Maximum sequence length:", max(lengths_list))

    print("Annotated:", annotated_count)
    print("Pose-only:", len(samples) - annotated_count)

    print(
        "Normal:",
        int(np.sum(labels_array == 0)),
    )

    print(
        "Wall crossing:",
        int(np.sum(labels_array == 1)),
    )

    print("=" * 55)

    return (
        features_list,
        track_ids_list,
        labels_array,
        lengths_list,
    )


# ============================================================
# VARIABLE-LENGTH DATASET
# ============================================================

class VariableSequenceDataset(Dataset):

    def __init__(
        self,
        features_list,
        track_ids_list,
        labels,
        lengths,
    ):

        self.features = features_list
        self.track_ids = track_ids_list
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.track_ids[index],
            self.labels[index],
            self.lengths[index],
        )


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def collate_variable_sequences(batch):

    """
    Pad videos to the longest sequence in the current batch.

    Features:
        Padding value = 0

    Track IDs:
        Padding value = -1

    Lengths:
        Original sequence lengths, before padding.
    """

    features_list, tracks_list, labels_list, lengths_list = zip(
        *batch
    )

    batch_size = len(batch)
    max_length = max(lengths_list)

    batch_features = torch.zeros(
        batch_size,
        max_length,
        MAX_PEOPLE,
        FEATURE_SIZE,
        dtype=torch.float32,
    )

    batch_tracks = torch.full(
        (
            batch_size,
            max_length,
            MAX_PEOPLE,
        ),
        -1,
        dtype=torch.long,
    )

    for i, (features, tracks, length) in enumerate(
        zip(features_list, tracks_list, lengths_list)
    ):

        batch_features[i, :length] = torch.as_tensor(
            features,
            dtype=torch.float32,
        )

        batch_tracks[i, :length] = torch.as_tensor(
            tracks,
            dtype=torch.long,
        )

    batch_labels = torch.tensor(
        labels_list,
        dtype=torch.float32,
    ).unsqueeze(1)

    batch_lengths = torch.tensor(
        lengths_list,
        dtype=torch.long,
    )

    return (
        batch_features,
        batch_tracks,
        batch_labels,
        batch_lengths,
    )


# ============================================================
# MULTI-PERSON LSTM V2 ARCHITECTURE
# ============================================================

class MultiPersonIntrusionLSTM_V2(nn.Module):

    """
    Shared per-person BiLSTM.

    Processing:
        1. Encode each person's temporal sequence.
        2. Ignore batch padding through packed sequences.
        3. Peak temporal pooling.
        4. Last-valid-frame pooling.
        5. Combine peak and last embeddings.
        6. Masked max pooling across people.
        7. Binary classification.
    """

    def __init__(
        self,
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ):

        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        lstm_output_dim = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(32, 1),
        )

    def forward(self, x, track_ids, lengths):

        # x:
        # (batch, frames, people, features)

        # track_ids:
        # (batch, frames, people)

        # lengths:
        # (batch,)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        n_people = x.shape[2]
        feat_dim = x.shape[3]

        # ----------------------------------------------------
        # 1. CREATE INDEPENDENT PERSON SEQUENCES
        # ----------------------------------------------------

        person_sequences = x.permute(
            0, 2, 1, 3
        ).reshape(
            batch_size * n_people,
            seq_len,
            feat_dim,
        )

        # ----------------------------------------------------
        # 2. PACK VARIABLE-LENGTH SEQUENCES
        # ----------------------------------------------------

        person_lengths = (
            lengths
            .repeat_interleave(n_people)
            .cpu()
        )

        packed_sequences = pack_padded_sequence(
            person_sequences,
            person_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed_sequences)

        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_len,
        )

        lstm_output_dim = lstm_out.shape[-1]

        # ----------------------------------------------------
        # 3. RESTORE PERSON DIMENSION
        # ----------------------------------------------------

        lstm_out = lstm_out.reshape(
            batch_size,
            n_people,
            seq_len,
            lstm_output_dim,
        )

        # ----------------------------------------------------
        # 4. VALID TRACK FRAMES
        # ----------------------------------------------------

        valid_frames = (
            track_ids != -1
        ).permute(0, 2, 1)

        valid_people = valid_frames.any(dim=2)

        # ----------------------------------------------------
        # 5. PEAK TEMPORAL POOLING
        # ----------------------------------------------------

        masked_for_peak = lstm_out.masked_fill(
            ~valid_frames.unsqueeze(-1),
            float("-inf"),
        )

        peak_embedding = masked_for_peak.max(
            dim=2
        ).values

        peak_embedding = torch.where(
            valid_people.unsqueeze(-1),
            peak_embedding,
            torch.zeros_like(peak_embedding),
        )

        # ----------------------------------------------------
        # 6. LAST VALID FRAME POOLING
        # ----------------------------------------------------

        frame_indices = torch.arange(
            seq_len,
            device=x.device,
        ).view(1, 1, seq_len)

        last_valid_indices = torch.where(
            valid_frames,
            frame_indices,
            torch.full_like(frame_indices, -1),
        ).max(dim=2).values

        safe_indices = last_valid_indices.clamp(min=0)

        gather_idx = safe_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            1,
            lstm_output_dim,
        )

        last_embedding = torch.gather(
            lstm_out,
            dim=2,
            index=gather_idx,
        ).squeeze(2)

        last_embedding = torch.where(
            valid_people.unsqueeze(-1),
            last_embedding,
            torch.zeros_like(last_embedding),
        )

        # ----------------------------------------------------
        # 7. COMBINE PEAK + LAST REPRESENTATIONS
        # ----------------------------------------------------

        person_embedding = torch.cat(
            [
                peak_embedding,
                last_embedding,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # 8. MASK EMPTY PERSON SLOTS
        # ----------------------------------------------------

        person_embedding = person_embedding.masked_fill(
            ~valid_people.unsqueeze(-1),
            float("-inf"),
        )

        # ----------------------------------------------------
        # 9. MAX POOL ACROSS PEOPLE
        # ----------------------------------------------------

        pooled = person_embedding.max(dim=1).values

        has_any_person = valid_people.any(
            dim=1,
            keepdim=True,
        )

        pooled = torch.where(
            has_any_person,
            pooled,
            torch.zeros_like(pooled),
        )

        # ----------------------------------------------------
        # 10. CLASSIFICATION
        # ----------------------------------------------------

        return self.classifier(pooled)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    acc = accuracy_score(
        y_true,
        predictions,
    )

    prec = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),

        "confusion_matrix": {
            "true_normal": int(tn),
            "false_intrusion": int(fp),
            "missed_intrusion": int(fn),
            "detected_intrusion": int(tp),
        },
    }


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def find_optimal_threshold(
    y_true,
    probabilities,
):

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics = None

    thresholds = np.linspace(
        0.20,
        0.80,
        25,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold=threshold,
        )

        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    train_loader,
    validation_loader,
    validation_y,
):

    print("\n" + "=" * 65)
    print("EVERY-5-FRAMES MULTI-PERSON REGION LSTM")
    print("=" * 65)

    set_seed(RANDOM_SEED)

    model = MultiPersonIntrusionLSTM_V2(
        input_size=FEATURE_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=1e-5,
    )

    best_val_loss = float("inf")
    best_epoch = 0

    patience_counter = 0
    best_state = None

    history = []

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        total_train_loss = 0.0

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for (
            batch_X,
            batch_tracks,
            batch_y,
            batch_lengths,
        ) in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_tracks = batch_tracks.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Lengths intentionally remain on CPU.
            optimizer.zero_grad()

            logits = model(
                batch_X,
                batch_tracks,
                batch_lengths,
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0,
            )

            optimizer.step()

            total_train_loss += (
                loss.item() * len(batch_X)
            )

        scheduler.step()

        train_loss = (
            total_train_loss
            / len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for (
                batch_X,
                batch_tracks,
                batch_y,
                batch_lengths,
            ) in validation_loader:

                batch_X = batch_X.to(DEVICE)
                batch_tracks = batch_tracks.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits = model(
                    batch_X,
                    batch_tracks,
                    batch_lengths,
                )

                loss = criterion(
                    logits,
                    batch_y,
                )

                total_val_loss += (
                    loss.item() * len(batch_X)
                )

        val_loss = (
            total_val_loss
            / len(validation_loader.dataset)
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "lr": float(current_lr),
        })

        print(
            f"Epoch [{epoch:02d}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping triggered at epoch {epoch}."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint produced."
        )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(best_state)

    checkpoint_path = (
        MODEL_DIR / f"{MODEL_NAME}.pth"
    )

    torch.save(
        best_state,
        checkpoint_path,
    )

    print("\nBest Epoch:", best_epoch)
    print("Best Validation Loss:", best_val_loss)
    print("Saved checkpoint:", checkpoint_path)

    # ========================================================
    # FINAL VALIDATION EVALUATION
    # ========================================================

    model.eval()

    all_probabilities = []

    with torch.no_grad():

        for (
            batch_X,
            batch_tracks,
            _,
            batch_lengths,
        ) in validation_loader:

            batch_X = batch_X.to(DEVICE)
            batch_tracks = batch_tracks.to(DEVICE)

            logits = model(
                batch_X,
                batch_tracks,
                batch_lengths,
            )

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.cpu().numpy().flatten()
            )

    y_true = np.asarray(
        validation_y,
        dtype=int,
    ).flatten()

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # DEFAULT THRESHOLD
    # --------------------------------------------------------

    default_metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    # --------------------------------------------------------
    # OPTIMAL THRESHOLD
    # --------------------------------------------------------

    optimal_threshold, optimal_metrics = (
        find_optimal_threshold(
            y_true,
            probabilities,
        )
    )

    auc_score = float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS — THRESHOLD 0.50")
    print("=" * 65)

    for key, value in default_metrics.items():
        print(f"{key}: {value}")

    print(f"ROC-AUC: {auc_score:.4f}")

    print("\n" + "=" * 65)
    print(
        f"OPTIMAL THRESHOLD — {optimal_threshold:.2f}"
    )
    print("=" * 65)

    for key, value in optimal_metrics.items():
        print(f"{key}: {value}")

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (
        EXPERIMENT_DIR
        / f"{MODEL_NAME}_history.json"
    )

    with open(history_path, "w") as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_data = {

        "experiment": MODEL_NAME,

        "architecture": (
            "Shared per-person Bidirectional LSTM "
            "+ Peak and Last Temporal Pooling "
            "+ Person-level Masked Max Pooling"
        ),

        "sampling": {
            "frame_stride": 5,
            "truncation": False,
            "sequence_length": "variable",
        },

        "features": {
            "pose": POSE_FEATURE_SIZE,
            "wall_region": REGION_FEATURE_SIZE,
            "annotation_availability": 1,
            "total": FEATURE_SIZE,
        },

        "architecture_config": {
            "max_people": MAX_PEOPLE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
        },

        "training_config": {
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "random_seed": RANDOM_SEED,
            "augmentation": "disabled",
        },

        "input_shape": [
            "batch",
            "variable_frames",
            MAX_PEOPLE,
            FEATURE_SIZE,
        ],

        "best_epoch": best_epoch,

        "best_validation_loss": float(
            best_val_loss
        ),

        "roc_auc": auc_score,

        "default_metrics_threshold_0_5": (
            default_metrics
        ),

        "optimal_threshold": float(
            optimal_threshold
        ),

        "optimal_threshold_metrics": (
            optimal_metrics
        ),

        "checkpoint": str(checkpoint_path),

        "history": str(history_path),
    }

    results_path = (
        EXPERIMENT_DIR
        / f"{MODEL_NAME}_results.json"
    )

    with open(results_path, "w") as file:
        json.dump(
            results_data,
            file,
            indent=4,
        )

    print("\nSaved history:", history_path)
    print("Saved results:", results_path)

    return results_data


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD SPLIT
    # --------------------------------------------------------

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    with open(SPLIT_PATH, "r") as file:
        split = json.load(file)

    train_samples = split["train_samples"]
    validation_samples = split["validation_samples"]

    print("\nLoaded split:", SPLIT_PATH)
    print("Training videos:", len(train_samples))
    print("Validation videos:", len(validation_samples))

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    (
        train_features,
        train_tracks,
        train_y,
        train_lengths,
    ) = load_dataset(
        train_samples,
        "TRAINING",
    )

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    (
        validation_features,
        validation_tracks,
        validation_y,
        validation_lengths,
    ) = load_dataset(
        validation_samples,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # CREATE DATASETS
    # --------------------------------------------------------

    train_dataset = VariableSequenceDataset(
        train_features,
        train_tracks,
        train_y,
        train_lengths,
    )

    validation_dataset = VariableSequenceDataset(
        validation_features,
        validation_tracks,
        validation_y,
        validation_lengths,
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate_variable_sequences,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_sequences,
        num_workers=0,
    )

    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    train_model(
        train_loader,
        validation_loader,
        validation_y,
    )


if __name__ == "__main__":
    main()
