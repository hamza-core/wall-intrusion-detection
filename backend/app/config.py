"""
Application configuration.

All values can be overridden through environment variables or a .env file.
The backend never hard-codes filesystem paths — every path is derived
from the project root, which is located automatically.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# The backend/ folder sits one level below the ML project root.
# backend/app/config.py  →  parent = app/  →  parent = backend/  →  parent = project root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_DIR = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    Runtime settings.

    Defaults work out-of-the-box for local development.
    Override any value by setting the matching environment variable
    or adding it to backend/.env
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------
    # Paths — relative to the ML project root
    # -----------------------------------------------------------------

    # YOLO pose weights
    yolo_weights: Path = _PROJECT_DIR / "models" / "yolo11n-pose.pt"

    # Selected LSTM checkpoint — every-5-frames variable-length variant
    lstm_checkpoint: Path = (
        _PROJECT_DIR / "models" / "multi_person_region_lstm_every_5_frames.pth"
    )

    # Temporary upload directory (inside backend/)
    upload_dir: Path = _BACKEND_DIR / "uploads"

    # -----------------------------------------------------------------
    # Inference hyper-parameters
    # These MUST match the values used during training.
    # Do not change them unless you retrain the model.
    # -----------------------------------------------------------------

    frame_stride: int = 5           # one sampled frame every N video frames (variable-length sequences)
    max_people: int = 5             # fixed person slots per frame
    feature_size: int = 104         # 55 pose + 48 wall features + 1 annotation-available flag
    hidden_size: int = 64           # LSTM hidden dimension
    num_layers: int = 2             # LSTM layers
    dropout: float = 0.3            # dropout (also active at inference for arch match)
    threshold: float = 0.5          # sigmoid output → class boundary
    image_size: int = 640           # YOLO imgsz
    yolo_conf: float = 0.25         # YOLO detection confidence threshold

    # Annotated preview frames returned to the frontend.
    # 0 = return previews for every sampled frame (no cap).
    max_preview_frames: int = 0

    # Velocity gate / stationary-posture filter. Tuned for the fixed
    # 30-frame pipeline — disabled by default so the API reproduces the
    # every-5-frames model's own test-evaluation behaviour exactly.
    velocity_gate_enabled: bool = False

    # -----------------------------------------------------------------
    # API settings
    # -----------------------------------------------------------------

    # Comma-separated list of allowed CORS origins
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:3000"

    # Maximum upload size in bytes (default: 500 MB)
    max_upload_bytes: int = 500 * 1024 * 1024

    # Allowed video extensions (checked on the server side)
    allowed_extensions: str = ".mp4,.avi,.mov,.mkv,.m4v"

    # -----------------------------------------------------------------
    # Computed helpers (not env-configurable)
    # -----------------------------------------------------------------

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}


# Single shared instance used throughout the application.
settings = Settings()
