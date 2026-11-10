"""
Pydantic schemas for request validation and API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Wall boundary input
# ---------------------------------------------------------------------------

class WallPoint(BaseModel):
    """A single normalized coordinate on a wall boundary."""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate [0, 1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate [0, 1]")


class WallBoundaries(BaseModel):
    """
    Wall annotation provided by the frontend.

    Mirrors the annotation schema (version 2) used during training,
    but arrives as structured JSON rather than a file on disk.
    """
    wall_top_points: list[WallPoint] = Field(
        ...,
        min_length=2,
        description="At least two points defining the top wall boundary (normalized)",
    )
    wall_bottom_points: list[WallPoint] = Field(
        ...,
        min_length=2,
        description="At least two points defining the bottom wall boundary (normalized)",
    )
    frame_width: int = Field(
        ..., gt=0, description="Actual video frame width in pixels"
    )
    frame_height: int = Field(
        ..., gt=0, description="Actual video frame height in pixels"
    )

    @field_validator("wall_top_points", "wall_bottom_points", mode="before")
    @classmethod
    def at_least_two_points(cls, value):
        if len(value) < 2:
            raise ValueError("Each wall boundary must have at least two points.")
        return value

    def to_annotation_dict(self) -> dict:
        """
        Convert to the annotation dict format expected by
        prepare_wall_boundaries() in extract_wall_region_features.py.
        """
        return {
            "annotation_version": 2,
            "status": "annotated",
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "wall_top_points": [[p.x, p.y] for p in self.wall_top_points],
            "wall_bottom_points": [[p.x, p.y] for p in self.wall_bottom_points],
        }


# ---------------------------------------------------------------------------
# Prediction response
# ---------------------------------------------------------------------------



class PreviewFrame(BaseModel):
    """One sampled frame with YOLO detections drawn on it."""

    frame_number: int
    source_frame_index: int
    timestamp_seconds: float
    people_detected: int
    image: str = Field(..., description="Annotated JPEG data URI")


class PredictionResponse(BaseModel):
    """
    Response returned by POST /api/predict.

    model_score is the raw sigmoid output from the LSTM [0, 1].
    It measures how strongly the model leans toward 'Wall Crossing'.
    It is NOT a statistically calibrated confidence value — the model
    has not been evaluated on a held-out test set yet.
    """
    success: bool
    prediction: str = Field(
        ..., description="'Normal' or 'Wall Crossing'"
    )
    model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Raw sigmoid output [0, 1]. Values above 0.5 predict Wall Crossing. "
            "This is a model probability, not a calibrated confidence score."
        ),
    )
    processing_time_seconds: float
    model_name: str = Field(default="multi_person_region_lstm_every_5_frames")
    model_version: Optional[str] = Field(
        default="every_5_frames",
        description="Model variant served by this package (every_5_frames)",
    )
    frame_stride: Optional[int] = Field(
        default=None,
        description="Video stride used for sampling (one frame every N video frames)",
    )
    total_video_frames: Optional[int] = Field(
        default=None,
        description="Total frames in the original video",
    )
    threshold_used: Optional[float] = Field(default=0.5, description="Decision threshold applied")
    velocity_gate_active: bool = Field(
        default=False,
        description="Whether the velocity gate / stationary filter attenuated a false alarm",
    )
    raw_model_score: Optional[float] = Field(
        default=None,
        description="Raw LSTM score before velocity gate attenuation",
    )
    people_detected: int = Field(
        ..., description="Maximum number of people detected in any single frame"
    )
    frames_processed: int = Field(
        ..., description="Number of frames sampled from the video (every frame_stride-th frame)"
    )
    preview_frames: list[PreviewFrame] = Field(default_factory=list)
    annotated_video: Optional[str] = Field(
        default=None,
        description=(
            "Base64 data URI of an H.264 MP4 replaying the sampled frames "
            "with wall boundaries, person boxes and pose skeletons drawn"
        ),
    )
    video_duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the original video in seconds",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Non-fatal warning (e.g. no person detected in some frames)",
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    lstm_loaded: bool
    device: str
