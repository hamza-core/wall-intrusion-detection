"""
Live CCTV simulation service.

The frontend plays a local surveillance file in real time ("as if it
were streaming from a CCTV camera"), captures the current frame a few
times per second, and POSTs it as JPEG to POST /api/live/frame.

Each arriving frame:
    1. goes through the local YOLO pose model,
    2. is associated with existing tracks (same logic as training),
    3. is appended to the session's growing feature sequence
       (55 pose + 48 wall-region + 1 annotation flag per person),
    4. and every ingestion re-evaluates a trailing window with the
       Every-5-Frames variable-length BiLSTM. When the window
       probability crosses the decision threshold, an ALARM event is
       emitted with the frame number and timestamp of that moment.

This gives alarm behaviour during the stream — the alarm fires when a
person actually crosses, not only at the end of the clip. Everything
runs offline on the local models.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from app.config import settings
from app.services.ml_inference import (
    registry,
    _detect_people_on_frame,
)
from extract_wall_region_features import (
    prepare_wall_boundaries,
    extract_frame_wall_features,
)
from extract_every_5_frames import (
    associate_tracks,
    POSE_FEATURE_COUNT,
    REGION_FEATURE_COUNT,
)
from train_every_5_frames import (
    FEATURE_SIZE,
    MAX_PEOPLE,
)

logger = logging.getLogger(__name__)

# Number of trailing sampled frames the LSTM sees per evaluation.
# Long enough for the BiLSTM to capture an approach + crossing, short
# enough to stay near-real-time on a laptop.
WINDOW_MAX_FRAMES = 60

# Alarm is released when the window probability falls this far below
# the decision threshold (hysteresis avoids alert flapping).
ALARM_RELEASE_MARGIN = 0.10


@dataclass
class AlarmEvent:
    frame_number: int      # index within the live stream (sampled frames)
    timestamp_seconds: float
    probability: float
    threshold: float


@dataclass
class LiveSession:
    session_id: str
    top_boundary: np.ndarray
    bottom_boundary: np.ndarray
    threshold: float
    created_at: float = field(default_factory=time.time)

    # Growing stream buffers (list of per-frame arrays; sample = every
    # frame the frontend posts, treated as one sampled frame).
    pose_list: list[np.ndarray] = field(default_factory=list)
    region_list: list[np.ndarray] = field(default_factory=list)
    track_list: list[np.ndarray] = field(default_factory=list)
    people_in_frame: list[int] = field(default_factory=list)
    max_frame_count: int = 0

    previous_tracks: dict = field(default_factory=dict)
    next_track_id: int = 0

    alarms: list[AlarmEvent] = field(default_factory=list)
    alarm_active: bool = False
    last_probability: Optional[float] = None
    last_timestamp_seconds: float = 0.0
    max_probability: float = 0.0

    # Buffers as tensors for the model evaluation.
    def _window_arrays(self) -> tuple[np.ndarray, np.ndarray, int]:
        total = len(self.pose_list)
        if total == 0:
            return (
                np.zeros((0, MAX_PEOPLE, FEATURE_SIZE), np.float32),
                np.zeros((0, MAX_PEOPLE), np.int64),
                0,
            )

        window = min(total, WINDOW_MAX_FRAMES)
        start = total - window

        pose = np.asarray(self.pose_list[start:], dtype=np.float32)
        region = np.asarray(self.region_list[start:], dtype=np.float32)
        tracks = np.asarray(self.track_list[start:], dtype=np.int64)

        features = np.zeros((window, MAX_PEOPLE, FEATURE_SIZE), dtype=np.float32)
        features[:, :, :103] = np.concatenate([pose, region], axis=2)
        features[:, :, 103] = 1.0
        return features, tracks, window

    def ingest(self, frame: np.ndarray, timestamp_seconds: float) -> dict:
        """Process one arriving JPEG frame; returns overlay data."""
        device = registry.device
        detections = _detect_people_on_frame(registry.yolo, frame, device)

        assignments, self.previous_tracks, self.next_track_id = (
            associate_tracks(
                self.previous_tracks,
                detections,
                self.next_track_id,
                frame.shape[1],
                frame.shape[0],
            )
        )

        pose_frame = np.zeros((MAX_PEOPLE, POSE_FEATURE_COUNT), np.float32)
        region_frame = np.zeros((MAX_PEOPLE, REGION_FEATURE_COUNT), np.float32)
        track_frame = np.full((MAX_PEOPLE,), -1, np.int32)

        for det_idx, person in enumerate(detections):
            if det_idx not in assignments:
                continue
            track_id = assignments[det_idx]
            slot = self.previous_tracks[track_id]["slot"]

            pose = person["pose"]
            pose_frame[slot] = pose
            track_frame[slot] = track_id
            region_frame[slot] = np.asarray(
                extract_frame_wall_features(
                    pose,
                    self.top_boundary,
                    self.bottom_boundary,
                    frame.shape[1],
                    frame.shape[0],
                ),
                dtype=np.float32,
            )

        self.pose_list.append(pose_frame)
        self.region_list.append(region_frame)
        self.track_list.append(track_frame)
        self.people_in_frame.append(len(detections))
        self.last_timestamp_seconds = float(timestamp_seconds)
        stream_index = len(self.pose_list)  # 1-based stream counter
        self.max_frame_count = max(self.max_frame_count, stream_index)

        h, w = frame.shape[:2]
        return {
            "frame_number": stream_index,
            "timestamp_seconds": timestamp_seconds,
            "people": [
                {
                    "track_id": assignments.get(det_idx, -1),
                    "box": [
                        round(float(v) / w, 4) if i % 2 == 0 else round(float(v) / h, 4)
                        for i, v in enumerate(person["box"])
                    ],
                    "keypoints": [
                        [
                            round(float(x) / w, 4),
                            round(float(y) / h, 4),
                            round(float(c), 4),
                        ]
                        for x, y, c in person["keypoints"]
                    ],
                    "confidence": round(float(person["confidence"]), 3),
                }
                for det_idx, person in enumerate(detections)
                if det_idx in assignments
            ],
        }

    def evaluate(self) -> tuple[float, bool, bool]:
        """Run the trailing-window LSTM. Returns
        (probability, alarm_active, new_alarm_raised)."""
        features, tracks, window = self._window_arrays()
        if window < 3 or registry.lstm is None:
            return 0.0, self.alarm_active, False

        if not np.isfinite(features).all():
            return self.last_probability or 0.0, self.alarm_active, False

        device = registry.device
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        T = torch.tensor(tracks, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([window], dtype=torch.long)

        with torch.no_grad():
            logit = registry.lstm(X, T, lengths)
            probability = torch.sigmoid(logit).item()

        self.last_probability = probability
        self.max_probability = max(self.max_probability, probability)

        new_alarm = False
        if probability >= self.threshold and not self.alarm_active:
            self.alarm_active = True
            new_alarm = True
            self.alarms.append(AlarmEvent(
                frame_number=len(self.pose_list),
                timestamp_seconds=self.last_timestamp_seconds,
                probability=probability,
                threshold=self.threshold,
            ))
        elif self.alarm_active and probability < self.threshold - ALARM_RELEASE_MARGIN:
            self.alarm_active = False

        return probability, self.alarm_active, new_alarm

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "frames_streamed": len(self.pose_list),
            "alarms": [
                {
                    "frame_number": alarm.frame_number,
                    "timestamp_seconds": alarm.timestamp_seconds,
                    "probability": round(alarm.probability, 4),
                    "threshold": alarm.threshold,
                }
                for alarm in self.alarms
            ],
            "alarm_count": len(self.alarms),
            "max_probability": round(self.max_probability, 4),
            "last_probability": (
                round(self.last_probability, 4)
                if self.last_probability is not None else None
            ),
            "max_people_in_frame": max(self.people_in_frame, default=0),
        }


class LiveSessionManager:
    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def create(self, boundaries: dict, threshold: float) -> LiveSession:
        ann_width = int(boundaries["frame_width"])
        ann_height = int(boundaries["frame_height"])

        top, bottom = prepare_wall_boundaries(boundaries, ann_width, ann_height)

        session = LiveSession(
            session_id=uuid.uuid4().hex[:12],
            top_boundary=top,
            bottom_boundary=bottom,
            threshold=threshold,
        )

        with self._lock:
            self._sessions[session.session_id] = session

        logger.info(
            "Live session started: %s (threshold=%.2f, %d +%d wall points)",
            session.session_id, threshold, len(top), len(bottom),
        )
        return session

    def get(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def decay_survivors(self, frame_index: int):
        """No-op hook kept for parity with the extractor semantics."""
        return frame_index

    def stop(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


manager = LiveSessionManager()
