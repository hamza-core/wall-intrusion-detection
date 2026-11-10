"""
Live CCTV simulation API.

Endpoints:
    POST /api/live/start  — open a live session (wall boundaries)
    POST /api/live/frame  — stream one JPEG frame from the "camera",
                            returns detections for client-side overlay
                            plus any NEW alarm events
    POST /api/live/stop   — close the session, returns a summary

The frontend plays the surveillance file in real time and continuously
sends the current frame. Inference runs on every arriving frame against
a trailing window of the variable-length BiLSTM; the moment the window
probability crosses the threshold, an alarm event is raised with the
frame number and timestamp when the crossing actually happened.
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.prediction import (
    ErrorResponse,
)
from app.services.live_service import manager
from app.services.ml_inference import registry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/live/start",
    summary="Start live CCTV monitoring session",
    description=(
        "Provide the same wall boundary annotation used for batch "
        "analysis. Returns a session id to stream frames against."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Models not loaded"},
        422: {"model": ErrorResponse, "description": "Invalid boundaries"},
    },
)
async def live_start(
    boundaries: str = Form(..., description="JSON string with wall_top_points, wall_bottom_points, frame_width, frame_height"),
    threshold: float = Form(0.5, description="Alarm decision threshold"),
):
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded. Check application startup logs.",
        )

    import json
    from app.schemas.prediction import WallBoundaries

    try:
        boundaries_data = json.loads(boundaries)
        wall = WallBoundaries.model_validate(boundaries_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid wall boundary data: {exc}",
        )

    threshold = min(max(float(threshold), 0.05), 0.95)
    session = manager.create(wall.to_annotation_dict(), threshold)

    return {
        "success": True,
        "session_id": session.session_id,
        "threshold": session.threshold,
    }


@router.post(
    "/live/frame",
    summary="Stream one frame into the live session",
    description=(
        "Single JPEG frame from the playing feed. Runs YOLO detection, "
        "track association and trailing-window LSTM inference. Returns "
        "normalized detections for the client overlay and whether the "
        "alarm is currently active."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Unknown session"},
        503: {"model": ErrorResponse, "description": "Models not loaded"},
    },
)
async def live_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(..., description="Current JPEG frame"),
    frame_index: int = Form(0, description="Video frame index at capture time"),
    timestamp_seconds: float = Form(0.0, description="Video playback timestamp"),
):
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded.",
        )

    session = manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or closed live session: {session_id}",
        )

    data = await frame.read()
    encoded = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Frame payload is not a valid JPEG image.",
        )

    overlay = session.ingest(decoded, float(timestamp_seconds))
    probability, alarm_active, new_alarm = session.evaluate()

    return {
        "success": True,
        "session_id": session_id,
        "frame_index": int(frame_index),
        "frames_streamed": len(session.pose_list),
        "latest_probability": round(probability, 4),
        "alarm_active": alarm_active,
        "alarm_raised": new_alarm,
        "alarms": [
            {
                "frame_number": a.frame_number,
                "timestamp_seconds": a.timestamp_seconds,
                "probability": round(a.probability, 4),
                "threshold": a.threshold,
            }
            for a in session.alarms
        ],
        "people": overlay["people"],
    }


@router.post(
    "/live/stop",
    summary="Stop a live session",
    description="Closes the session and returns a full summary "
                "including every alarm that occurred during the stream.",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown session"},
    },
)
async def live_stop(
    session_id: str = Form(...),
):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or closed live session: {session_id}",
        )

    summary = session.summary()
    manager.stop(session_id)
    logger.info(
        "Live session stopped: %s frames=%d alarms=%d",
        session_id, summary["frames_streamed"], summary["alarm_count"],
    )
    return {"success": True, **summary}
