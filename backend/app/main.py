"""
Wall Intrusion Detection System — FastAPI backend (standalone every-5-frames variant).

Startup sequence:
  1. FastAPI lifespan begins.
  2. YOLO pose model is loaded from yolo11n-pose.pt (local, no download).
  3. The variable-length MultiPersonIntrusionLSTM_V2 is loaded from
     models/multi_person_region_lstm_every_5_frames.pth.
  4. Both models are kept in memory for the lifetime of the server.
  5. POST /api/predict serves inference requests using those loaded models,
     sampling one frame every 5 video frames (variable-length sequences).

The server never downloads models at runtime and never calls any cloud AI API.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.config import settings
from app.services.ml_inference import registry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — model loading at startup, cleanup at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Starting Wall Intrusion Detection backend")
    logger.info("Python: %s", sys.version)
    logger.info("Project root: %s", Path(__file__).resolve().parent.parent.parent)

    try:
        registry.load()
        logger.info("Models loaded successfully. API is ready.")
    except FileNotFoundError as exc:
        logger.critical("Model file missing — cannot start: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.critical("Model loading failed: %s", exc)
        raise SystemExit(1) from exc

    yield  # Server is running here

    # --- SHUTDOWN ---
    logger.info("Shutting down Wall Intrusion Detection backend")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Wall Intrusion Detection API",
    description=(
        "Offline REST API for detecting wall intrusion events in video footage. "
        "Uses YOLO pose detection and a multi-person region LSTM classifier. "
        "All inference runs locally — no cloud AI services are used."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server and any other configured origins
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix="/api")


# ---------------------------------------------------------------------------
# Root — redirect browsers to the interactive docs
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Wall Intrusion Detection API",
        "docs": "/docs",
        "health": "/api/health",
    }
