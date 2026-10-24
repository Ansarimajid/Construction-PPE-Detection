from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # webcam | rtsp | file
    source_uri: Mapped[str] = mapped_column(String(500), nullable=False)  # index, URL, or path
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    violations: Mapped[List["Violation"]] = relationship("Violation", back_populates="camera")


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"), nullable=False)
    violation_type: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    frame_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # tracked worker ID
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="violations")
    alert_logs: Mapped[List["AlertLog"]] = relationship("AlertLog", back_populates="violation")


class AlertLog(Base):
    __tablename__ = "alert_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    violation_id: Mapped[int] = mapped_column(Integer, ForeignKey("violations.id"), nullable=False)
    handler_type: Mapped[str] = mapped_column(String(50), nullable=False)  # email | webhook | slack | db
    sent_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_msg: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    violation: Mapped["Violation"] = relationship("Violation", back_populates="alert_logs")
