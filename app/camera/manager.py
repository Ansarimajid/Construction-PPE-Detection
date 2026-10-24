from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.camera.source import CameraSource
from app.core.config import settings
from app.core.detector import Detection, PPEDetector
from app.core.frame_annotator import annotate_frame
from app.core.logging import get_logger
from app.core.pose import Pose, PoseEstimator
from app.core.ppe_assigner import PersonStatus, PPEStatusSmoother, analyze_persons
from app.core.tracker import IoUTracker
from app.core.violation_checker import ViolationChecker

logger = get_logger(__name__)


@dataclass
class _CameraEntry:
    camera_id: int
    source: CameraSource
    task: asyncio.Task | None = None
    latest_frame: bytes | None = None  # JPEG bytes for MJPEG stream
    latest_counts: dict = field(default_factory=dict)
    alert_sent_until: float = 0.0  # show overlay until this timestamp
    tracker: IoUTracker | None = None
    smoother: PPEStatusSmoother | None = None


class CameraManager:
    def __init__(self, detector: PPEDetector, pose_estimator: PoseEstimator | None = None) -> None:
        self.detector = detector
        self.pose_estimator = pose_estimator
        self._entries: dict[int, _CameraEntry] = {}
        self._checker = ViolationChecker(
            cooldown_seconds=settings.ALERT_COOLDOWN_SECONDS,
            persist_seconds=settings.VIOLATION_PERSIST_SECONDS,
        )
        self._ws_subscribers: dict[int, list[asyncio.Queue]] = {}

    async def start(self) -> None:
        # Restore active cameras from DB on startup
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import Camera
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                result = await session.execute(select(Camera).where(Camera.is_active == True))  # noqa: E712
                cameras = result.scalars().all()
                for cam in cameras:
                    await self._launch_camera(cam.id, cam.source_type, cam.source_uri)
        except Exception as exc:
            logger.warning("Could not restore active cameras from DB: %s", exc)

    async def stop(self) -> None:
        for entry in list(self._entries.values()):
            if entry.task:
                entry.task.cancel()
                try:
                    await entry.task
                except asyncio.CancelledError:
                    pass
            await entry.source.release()
        self._entries.clear()

    def _build_source(self, source_type: str, source_uri: str) -> CameraSource:
        if source_type == "webcam":
            from app.camera.webcam_source import WebcamSource
            return WebcamSource(int(source_uri))
        elif source_type == "rtsp":
            from app.camera.rtsp_source import RTSPSource
            return RTSPSource(source_uri)
        elif source_type == "file":
            from app.camera.file_source import FileSource
            return FileSource(source_uri)
        else:
            raise ValueError(f"Unknown source_type: {source_type!r}")

    async def _launch_camera(
        self, camera_id: int, source_type: str, source_uri: str
    ) -> bool:
        if camera_id in self._entries and self._entries[camera_id].task and not self._entries[camera_id].task.done():
            logger.warning("Camera %d already running", camera_id)
            return False

        source = self._build_source(source_type, source_uri)
        connected = await source.connect()
        if not connected:
            return False

        entry = _CameraEntry(camera_id=camera_id, source=source)
        if settings.ENABLE_TRACKING:
            entry.tracker = IoUTracker(
                iou_threshold=settings.TRACK_IOU_THRESHOLD,
                max_age=settings.TRACK_MAX_AGE,
                min_hits=settings.TRACK_MIN_HITS,
            )
            entry.smoother = PPEStatusSmoother(window=settings.PPE_SMOOTHING_WINDOW)
        self._entries[camera_id] = entry
        entry.task = asyncio.create_task(
            self._process_loop(entry), name=f"camera-{camera_id}"
        )
        logger.info("Camera %d processing started", camera_id)
        return True

    async def start_camera(self, camera_id: int, source_type: str, source_uri: str) -> bool:
        return await self._launch_camera(camera_id, source_type, source_uri)

    async def stop_camera(self, camera_id: int) -> bool:
        entry = self._entries.get(camera_id)
        if entry is None:
            return False
        if entry.task:
            entry.task.cancel()
            try:
                await entry.task
            except asyncio.CancelledError:
                pass
        await entry.source.release()
        self._entries.pop(camera_id, None)
        self._checker.reset(camera_id)
        logger.info("Camera %d stopped", camera_id)
        return True

    def _infer(self, frame: np.ndarray) -> tuple[list[Detection], list[Pose]]:
        detections = self.detector.detect(frame)
        poses = self.pose_estimator.estimate(frame) if self.pose_estimator else []
        return detections, poses

    async def _process_loop(self, entry: _CameraEntry) -> None:
        from app.alerts.dispatcher import AlertDispatcher
        from app.alerts.email_handler import EmailHandler
        from app.alerts.webhook_handler import WebhookHandler
        from app.alerts.db_handler import DatabaseHandler

        handlers = [DatabaseHandler()]
        if settings.SENDER_EMAIL:
            handlers.append(EmailHandler())
        if settings.WEBHOOK_URL:
            handlers.append(WebhookHandler(settings.WEBHOOK_URL))
        dispatcher = AlertDispatcher(handlers)

        camera_id = entry.camera_id
        loop = asyncio.get_event_loop()

        try:
            while True:
                frame = await entry.source.read_frame()
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue

                # Run YOLO in thread pool so we don't block the event loop
                detections, poses = await loop.run_in_executor(None, self._infer, frame)

                persons: list[PersonStatus] | None = None
                if entry.tracker is not None:
                    tracked = entry.tracker.update(
                        [d for d in detections if d.class_name == "Person"]
                    )
                    persons = entry.smoother.update(analyze_persons(tracked, detections, poses))

                hardhat_count = sum(1 for d in detections if d.class_name == "Hardhat")
                vest_count = sum(1 for d in detections if d.class_name == "Safety Vest")
                person_count = sum(1 for d in detections if d.class_name == "Person")

                entry.latest_counts = {
                    "hardhat_count": hardhat_count,
                    "vest_count": vest_count,
                    "person_count": person_count,
                    "total_detections": len(detections),
                }
                if persons is not None:
                    entry.latest_counts.update({
                        "unique_person_count": entry.tracker.total_tracks,
                        "non_compliant_count": sum(1 for p in persons if not p.has_hardhat),
                        "persons": [
                            {
                                "track_id": p.track_id,
                                "has_hardhat": p.has_hardhat,
                                "has_vest": p.has_vest,
                                "has_pose": p.pose is not None,
                            }
                            for p in persons
                        ],
                    })

                # Check for violations (quick pre-check before saving frame)
                if persons is not None:
                    violations = self._checker.check_tracks(camera_id, persons)
                else:
                    violation = self._checker.check(camera_id, detections)
                    violations = [violation] if violation is not None else []

                if violations:
                    frame_dir = os.path.join(settings.FRAMES_DIR, f"camera_{camera_id}")
                    os.makedirs(frame_dir, exist_ok=True)
                    fname = f"violation_{int(time.time() * 1000)}.jpg"
                    frame_path = os.path.join(frame_dir, fname)
                    await loop.run_in_executor(None, cv2.imwrite, frame_path, frame)
                    entry.alert_sent_until = time.time() + 3.0
                    for violation in violations:
                        violation.frame_path = frame_path
                        await dispatcher.dispatch(violation)

                # Annotate frame for MJPEG stream
                show_alert = time.time() < entry.alert_sent_until
                annotated = annotate_frame(
                    frame, detections, hardhat_count, vest_count, person_count, show_alert,
                    persons=persons,
                )
                resized = cv2.resize(annotated, (640, 480))
                _, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                entry.latest_frame = jpeg.tobytes()

                # Push detection event to WebSocket subscribers
                await self._broadcast(camera_id, entry.latest_counts)

                await asyncio.sleep(0)  # yield to event loop

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in camera %d processing loop: %s", camera_id, exc)

    async def _broadcast(self, camera_id: int, data: dict) -> None:
        queues = self._ws_subscribers.get(camera_id, [])
        for q in list(queues):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def subscribe(self, camera_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._ws_subscribers.setdefault(camera_id, []).append(q)
        return q

    def unsubscribe(self, camera_id: int, q: asyncio.Queue) -> None:
        subs = self._ws_subscribers.get(camera_id, [])
        if q in subs:
            subs.remove(q)

    def get_latest_frame(self, camera_id: int) -> bytes | None:
        entry = self._entries.get(camera_id)
        return entry.latest_frame if entry else None

    def is_running(self, camera_id: int) -> bool:
        entry = self._entries.get(camera_id)
        return entry is not None and entry.task is not None and not entry.task.done()
