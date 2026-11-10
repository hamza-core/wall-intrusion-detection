"""
POST /api/predict

Accepts a video file and wall boundary annotation, runs the full
YOLO + Multi-person Region LSTM inference pipeline, and returns
the prediction result.

Wall boundaries are submitted as a JSON string in the form field
'boundaries' (multipart/form-data). This design keeps the API
compatible with both React and future Flutter clients.

Request (multipart/form-data):
    video       : video file  (mp4, avi, mov, mkv, m4v)
    boundaries  : JSON string — WallBoundaries schema

Response (application/json):
    PredictionResponse schema
"""

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import (
    ErrorResponse,
    PredictionResponse,
    WallBoundaries,
)
from app.services.ml_inference import registry, run_inference
from app.services.video_service import cleanup, save_upload, validate_video_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict wall intrusion",
    description=(
        "Upload a video and wall boundary annotation. "
        "Returns the Multi-person Region LSTM prediction."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Models not loaded"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
)
async def predict(
    video: UploadFile = File(..., description="Video file to analyse"),
    boundaries: str = Form(
        ...,
        description=(
            "JSON string containing wall_top_points, wall_bottom_points, "
            "frame_width, frame_height"
        ),
    ),
    threshold: Optional[float] = Form(
        None,
        description="Optional decision threshold (default: 0.50). Values >= threshold predict Wall Crossing.",
    ),
) -> PredictionResponse:

    # ------------------------------------------------------------------ #
    # 1. Guard — models must be loaded
    # ------------------------------------------------------------------ #
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded. Check application startup logs.",
        )

    # ------------------------------------------------------------------ #
    # 2. Validate video file (extension, non-empty filename)
    # ------------------------------------------------------------------ #
    validate_video_file(video)

    # ------------------------------------------------------------------ #
    # 3. Parse and validate wall boundaries
    # ------------------------------------------------------------------ #
    try:
        boundaries_data = json.loads(boundaries)
        wall = WallBoundaries.model_validate(boundaries_data)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'boundaries' field is not valid JSON: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid wall boundary data: {exc}",
        )

    # ------------------------------------------------------------------ #
    # 4. Save video to a temporary file
    # ------------------------------------------------------------------ #
    temp_path = await save_upload(video)

    # ------------------------------------------------------------------ #
    # 5. Run inference
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()
    try:
        result = run_inference(
            video_path=temp_path,
            annotation=wall.to_annotation_dict(),
            threshold=threshold,
        )
    except RuntimeError as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except ValueError as exc:
        logger.warning("Inference input error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during inference.",
        )
    finally:
        # Always delete the temporary file, even if inference raised.
        cleanup(temp_path)

    elapsed = round(time.perf_counter() - t_start, 2)

    logger.info(
        "Prediction: %s  score=%.4f  people=%d  time=%.2fs",
        result["prediction"],
        result["model_score"],
        result["people_detected"],
        elapsed,
    )

    return PredictionResponse(
        success=True,
        prediction=result["prediction"],
        model_score=result["model_score"],
        raw_model_score=result.get("raw_model_score"),
        velocity_gate_active=result.get("velocity_gate_active", False),
        processing_time_seconds=elapsed,
        model_name="multi_person_region_lstm_every_5_frames",
        model_version=result.get("model_version", "every_5_frames"),
        frame_stride=result.get("frame_stride"),
        total_video_frames=result.get("total_video_frames"),
        threshold_used=result.get("threshold_used", 0.5),
        people_detected=result["people_detected"],
        frames_processed=result["frames_processed"],
        warning=result.get("warning"),
        preview_frames=result.get("preview_frames", []),
        annotated_video=result.get("annotated_video"),
        video_duration_seconds=result.get("video_duration_seconds"),
    )
