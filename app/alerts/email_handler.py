from __future__ import annotations

import os

import aiosmtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

from app.alerts.base import AlertHandler
from app.core.config import settings
from app.core.logging import get_logger
from app.core.violation_checker import ViolationEvent

logger = get_logger(__name__)


class EmailHandler(AlertHandler):
    handler_type = "email"

    async def send(self, violation: ViolationEvent) -> bool:
        if not settings.SENDER_EMAIL or not settings.EMAIL_PASSWORD:
            logger.warning("Email not configured, skipping alert")
            return False

        message = MIMEMultipart()
        message["From"] = settings.SENDER_EMAIL
        message["To"] = settings.RECEIVER_EMAIL
        message["Subject"] = f"Alert: {violation.violation_type} on Camera {violation.camera_id}"

        worker_line = (
            f"Worker ID: #{violation.track_id}\n" if violation.track_id is not None else ""
        )
        body = (
            f"A safety violation was detected on Camera {violation.camera_id}.\n"
            f"Violation type: {violation.violation_type}\n"
            f"{worker_line}"
            f"Confidence: {violation.confidence:.0%}\n"
            "Please review the attached frame."
        )
        message.attach(MIMEText(body, "plain"))

        if violation.frame_path and os.path.exists(violation.frame_path):
            with open(violation.frame_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(violation.frame_path)}",
                )
                message.attach(part)

        try:
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SENDER_EMAIL,
                password=settings.EMAIL_PASSWORD,
                start_tls=True,
            )
            logger.info(
                "Email alert sent for camera %d (%s)",
                violation.camera_id,
                violation.violation_type,
            )
            return True
        except Exception as exc:
            logger.error("Failed to send email alert: %s", exc)
            return False
