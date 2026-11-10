from fastapi import APIRouter
from app.api.routes import health, prediction, live

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(prediction.router, tags=["Prediction"])
api_router.include_router(live.router, tags=["Live CCTV"])
