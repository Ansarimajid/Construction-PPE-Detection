from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import configure_root_logger, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_root_logger(settings.LOG_LEVEL)
    logger.info("Starting PPE Detection API (env=%s)", settings.APP_ENV)

    # Ensure violation frames directory exists
    os.makedirs(settings.FRAMES_DIR, exist_ok=True)

    # Initialize database
    from app.db.session import init_db
    await init_db()
    logger.info("Database initialised")

    # Load YOLO model once and store on app state
    from app.core.detector import PPEDetector
    app.state.detector = PPEDetector(settings.MODEL_PATH, settings.DETECTION_CONFIDENCE)
    logger.info("YOLO model loaded from %s", settings.MODEL_PATH)

    # Optional pose model; fall back to box-only PPE assignment if it can't be loaded
    app.state.pose_estimator = None
    if settings.ENABLE_POSE:
        from app.core.pose import PoseEstimator
        try:
            app.state.pose_estimator = PoseEstimator(
                settings.POSE_MODEL_PATH, settings.POSE_CONFIDENCE
            )
        except Exception as exc:
            logger.warning("Pose model %s unavailable, continuing without pose: %s",
                           settings.POSE_MODEL_PATH, exc)

    # Start camera manager
    from app.camera.manager import CameraManager
    app.state.camera_manager = CameraManager(app.state.detector, app.state.pose_estimator)
    await app.state.camera_manager.start()
    logger.info("Camera manager started")

    yield

    # Shutdown
    logger.info("Shutting down camera manager...")
    await app.state.camera_manager.stop()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Construction PPE Detection API",
        description="Real-time PPE compliance monitoring for construction sites",
        version="2.0.0",
        lifespan=lifespan,
    )

    from app.api.routes.health import router as health_router
    from app.api.routes.cameras import router as cameras_router
    from app.api.routes.violations import router as violations_router
    from app.api.routes.stream import router as stream_router

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(cameras_router, prefix="/api/v1")
    app.include_router(violations_router, prefix="/api/v1")
    app.include_router(stream_router, prefix="/api/v1")

    # Serve violation frame images
    os.makedirs(settings.FRAMES_DIR, exist_ok=True)
    app.mount("/frames", StaticFiles(directory=settings.FRAMES_DIR), name="frames")

    # Serve web dashboard (must be last)
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

    return app


app = create_app()
