from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.detector import Detection
from app.core.logging import get_logger
from app.core.ppe_assigner import PersonStatus

logger = get_logger(__name__)


@dataclass
class ViolationEvent:
    camera_id: int
    violation_type: str
    confidence: float
    frame_path: Optional[str] = None
    track_id: Optional[int] = None


@dataclass
class _CameraState:
    last_hardhat_time: float = field(default_factory=time.time)
    last_alert_time: float = field(default_factory=lambda: 0.0)


@dataclass
class _TrackState:
    last_hardhat_time: float
    last_seen: float
    last_alert_time: float = 0.0


class ViolationChecker:
    """
    Tracks per-camera state and emits ViolationEvent when a person is detected
    without a hardhat for longer than ALERT_COOLDOWN_SECONDS.

    ``check`` works on whole frames; ``check_tracks`` works per tracked worker so
    each person gets their own persistence timer and alert cooldown.
    """

    # Forget a track's state once it hasn't been seen for this long
    TRACK_STATE_TTL_SECONDS = 60.0

    def __init__(self, cooldown_seconds: int = 10, persist_seconds: int = 10) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.persist_seconds = persist_seconds
        self._states: dict[int, _CameraState] = {}
        self._track_states: dict[tuple[int, int], _TrackState] = {}

    def _get_state(self, camera_id: int) -> _CameraState:
        if camera_id not in self._states:
            self._states[camera_id] = _CameraState()
        return self._states[camera_id]

    def check(
        self,
        camera_id: int,
        detections: list[Detection],
        frame_path: Optional[str] = None,
    ) -> Optional[ViolationEvent]:
        state = self._get_state(camera_id)
        now = time.time()

        hardhat_detected = any(d.class_name == "Hardhat" for d in detections)
        person_detected = any(d.class_name == "Person" for d in detections)

        if hardhat_detected:
            state.last_hardhat_time = now

        # Emit violation if: person present, no hardhat, cooldown elapsed
        if person_detected and not hardhat_detected:
            time_without_hardhat = now - state.last_hardhat_time
            time_since_last_alert = now - state.last_alert_time

            if (
                time_without_hardhat >= self.persist_seconds
                and time_since_last_alert >= self.cooldown_seconds
            ):
                state.last_alert_time = now
                person_detections = [d for d in detections if d.class_name == "Person"]
                confidence = max((d.confidence for d in person_detections), default=0.0)
                logger.info(
                    "Violation detected on camera %d: no hardhat for %.1fs",
                    camera_id,
                    time_without_hardhat,
                )
                return ViolationEvent(
                    camera_id=camera_id,
                    violation_type="NO-Hardhat",
                    confidence=confidence,
                    frame_path=frame_path,
                )

        return None

    def check_tracks(
        self,
        camera_id: int,
        persons: list[PersonStatus],
        frame_path: Optional[str] = None,
    ) -> list[ViolationEvent]:
        now = time.time()
        events: list[ViolationEvent] = []

        for status in persons:
            if status.track_id is None:
                continue
            key = (camera_id, status.track_id)
            state = self._track_states.get(key)
            if state is None:
                state = self._track_states[key] = _TrackState(last_hardhat_time=now, last_seen=now)
            state.last_seen = now

            if status.has_hardhat:
                state.last_hardhat_time = now
                continue

            time_without_hardhat = now - state.last_hardhat_time
            if (
                time_without_hardhat >= self.persist_seconds
                and now - state.last_alert_time >= self.cooldown_seconds
            ):
                state.last_alert_time = now
                logger.info(
                    "Violation detected on camera %d: worker #%d no hardhat for %.1fs",
                    camera_id,
                    status.track_id,
                    time_without_hardhat,
                )
                events.append(
                    ViolationEvent(
                        camera_id=camera_id,
                        violation_type="NO-Hardhat",
                        confidence=status.person.confidence,
                        frame_path=frame_path,
                        track_id=status.track_id,
                    )
                )

        self._prune_tracks(now)
        return events

    def _prune_tracks(self, now: float) -> None:
        stale = [
            key
            for key, state in self._track_states.items()
            if now - state.last_seen > self.TRACK_STATE_TTL_SECONDS
        ]
        for key in stale:
            del self._track_states[key]

    def reset(self, camera_id: int) -> None:
        self._states.pop(camera_id, None)
        for key in [k for k in self._track_states if k[0] == camera_id]:
            del self._track_states[key]
