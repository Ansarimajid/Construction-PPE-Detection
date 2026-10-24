from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    APP_ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./ppe_detection.db"

    # Model
    MODEL_PATH: str = "Model/ppe.pt"
    DETECTION_CONFIDENCE: float = 0.5

    # Person tracking (per-worker IDs and per-worker violations)
    ENABLE_TRACKING: bool = True
    TRACK_IOU_THRESHOLD: float = 0.3
    TRACK_MAX_AGE: int = 30  # frames a lost track is kept alive
    TRACK_MIN_HITS: int = 3  # frames before a track gets an ID
    PPE_SMOOTHING_WINDOW: int = 15  # frames of majority vote per worker (1 = off)

    # Pose estimation (keypoints used to locate head/torso for PPE assignment)
    ENABLE_POSE: bool = True
    POSE_MODEL_PATH: str = "yolo11n-pose.pt"
    POSE_CONFIDENCE: float = 0.5

    # Violation frames storage
    FRAMES_DIR: str = "violation_frames"

    # Alert timing
    ALERT_COOLDOWN_SECONDS: int = 10
    VIOLATION_PERSIST_SECONDS: int = 10

    # Email
    SENDER_EMAIL: str = ""
    RECEIVER_EMAIL: str = ""
    EMAIL_PASSWORD: str = ""
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587

    # Optional webhook alerts
    SLACK_WEBHOOK_URL: str = ""
    WEBHOOK_URL: str = ""


settings = Settings()
