from __future__ import annotations

from app.alerts.base import AlertHandler
from app.core.logging import get_logger
from app.core.violation_checker import ViolationEvent

logger = get_logger(__name__)


class DatabaseHandler(AlertHandler):
    handler_type = "db"

    async def send(self, violation: ViolationEvent) -> bool:
        try:
            from app.db.models import AlertLog, Violation
            from app.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                db_violation = Violation(
                    camera_id=violation.camera_id,
                    violation_type=violation.violation_type,
                    confidence=violation.confidence,
                    frame_path=violation.frame_path,
                    track_id=violation.track_id,
                )
                session.add(db_violation)
                await session.flush()  # get db_violation.id

                log = AlertLog(
                    violation_id=db_violation.id,
                    handler_type=self.handler_type,
                    success=True,
                )
                session.add(log)
                await session.commit()

            logger.info(
                "Violation saved to DB: camera=%d type=%s",
                violation.camera_id,
                violation.violation_type,
            )
            return True
        except Exception as exc:
            logger.error("Failed to save violation to DB: %s", exc)
            return False
