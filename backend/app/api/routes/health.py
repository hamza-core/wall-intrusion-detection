"""
GET /api/health

Returns whether the API is running and whether both models are loaded.
"""

from fastapi import APIRouter
from app.schemas.prediction import HealthResponse
from app.services.ml_inference import registry

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API status and model load state.",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        yolo_loaded=registry.yolo is not None,
        lstm_loaded=registry.lstm is not None,
        device=str(registry.device),
    )
