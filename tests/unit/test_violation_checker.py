from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.core.detector import Detection
from app.core.ppe_assigner import PersonStatus
from app.core.violation_checker import ViolationChecker


def make_detection(class_name: str, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        x1=0, y1=0, x2=100, y2=100,
        color=(255, 0, 0),
    )


def make_status(track_id: int | None, has_hardhat: bool) -> PersonStatus:
    person = make_detection("Person")
    person.track_id = track_id
    return PersonStatus(person=person, pose=None, has_hardhat=has_hardhat, has_vest=True)


@pytest.fixture
def checker():
    return ViolationChecker(cooldown_seconds=10, persist_seconds=5)


def test_no_violation_when_hardhat_present(checker):
    detections = [make_detection("Hardhat"), make_detection("Person")]
    result = checker.check(camera_id=1, detections=detections)
    assert result is None


def test_no_violation_when_no_person(checker):
    detections = [make_detection("Hardhat")]
    result = checker.check(camera_id=1, detections=detections)
    assert result is None


def test_no_violation_before_persist_threshold(checker):
    detections = [make_detection("Person")]
    # Should not trigger immediately (persist_seconds=5)
    result = checker.check(camera_id=1, detections=detections)
    assert result is None


def test_violation_after_persist_threshold(checker):
    camera_id = 1
    detections = [make_detection("Person")]

    # Push last_hardhat_time far into the past
    with patch("time.time", return_value=1000.0):
        checker._get_state(camera_id).last_hardhat_time = 990.0  # 10s ago

    with patch("time.time", return_value=1000.0):
        result = checker.check(camera_id=camera_id, detections=detections)

    assert result is not None
    assert result.violation_type == "NO-Hardhat"
    assert result.camera_id == camera_id


def test_cooldown_prevents_duplicate_alert(checker):
    camera_id = 2
    detections = [make_detection("Person")]

    # First alert at t=1000
    with patch("time.time", return_value=1000.0):
        checker._get_state(camera_id).last_hardhat_time = 980.0
        first = checker.check(camera_id=camera_id, detections=detections)

    # Second check at t=1005 (only 5s later, cooldown=10)
    with patch("time.time", return_value=1005.0):
        second = checker.check(camera_id=camera_id, detections=detections)

    assert first is not None
    assert second is None


def test_alert_fires_again_after_cooldown(checker):
    camera_id = 3
    detections = [make_detection("Person")]

    with patch("time.time", return_value=1000.0):
        checker._get_state(camera_id).last_hardhat_time = 980.0
        first = checker.check(camera_id=camera_id, detections=detections)

    # After cooldown (10s) + a bit more
    with patch("time.time", return_value=1015.0):
        second = checker.check(camera_id=camera_id, detections=detections)

    assert first is not None
    assert second is not None


def test_per_camera_isolation(checker):
    detections_no_hardhat = [make_detection("Person")]
    detections_with_hardhat = [make_detection("Hardhat"), make_detection("Person")]

    with patch("time.time", return_value=1000.0):
        checker._get_state(1).last_hardhat_time = 980.0
        checker._get_state(2).last_hardhat_time = 980.0

    with patch("time.time", return_value=1000.0):
        result_cam1 = checker.check(1, detections_no_hardhat)
        result_cam2 = checker.check(2, detections_with_hardhat)

    assert result_cam1 is not None
    assert result_cam2 is None


def test_reset_clears_state(checker):
    camera_id = 4
    checker._get_state(camera_id)
    assert camera_id in checker._states
    checker.reset(camera_id)
    assert camera_id not in checker._states


def test_track_violation_after_persist_threshold(checker):
    with patch("time.time", return_value=1000.0):
        assert checker.check_tracks(1, [make_status(7, has_hardhat=False)]) == []

    with patch("time.time", return_value=1006.0):
        events = checker.check_tracks(1, [make_status(7, has_hardhat=False)])

    assert len(events) == 1
    assert events[0].track_id == 7
    assert events[0].violation_type == "NO-Hardhat"


def test_track_violation_only_for_offending_worker(checker):
    with patch("time.time", return_value=1000.0):
        checker.check_tracks(1, [make_status(1, True), make_status(2, False)])

    with patch("time.time", return_value=1006.0):
        events = checker.check_tracks(1, [make_status(1, True), make_status(2, False)])

    assert [e.track_id for e in events] == [2]


def test_track_cooldown_is_per_worker(checker):
    with patch("time.time", return_value=1000.0):
        checker.check_tracks(1, [make_status(1, False)])
    with patch("time.time", return_value=1006.0):
        first = checker.check_tracks(1, [make_status(1, False)])
    with patch("time.time", return_value=1007.0):
        checker.check_tracks(1, [make_status(1, False), make_status(2, False)])
    with patch("time.time", return_value=1013.0):
        second = checker.check_tracks(1, [make_status(1, False), make_status(2, False)])

    assert [e.track_id for e in first] == [1]
    # Worker 1 still in cooldown, worker 2 has now persisted long enough
    assert [e.track_id for e in second] == [2]


def test_putting_on_hardhat_resets_track_timer(checker):
    with patch("time.time", return_value=1000.0):
        checker.check_tracks(1, [make_status(1, False)])
    with patch("time.time", return_value=1004.0):
        checker.check_tracks(1, [make_status(1, True)])
    with patch("time.time", return_value=1008.0):
        events = checker.check_tracks(1, [make_status(1, False)])

    assert events == []


def test_untracked_people_ignored(checker):
    with patch("time.time", return_value=1000.0):
        assert checker.check_tracks(1, [make_status(None, False)]) == []
    assert checker._track_states == {}


def test_stale_track_state_pruned(checker):
    with patch("time.time", return_value=1000.0):
        checker.check_tracks(1, [make_status(1, False)])
    with patch("time.time", return_value=1000.0 + checker.TRACK_STATE_TTL_SECONDS + 1):
        checker.check_tracks(1, [])
    assert checker._track_states == {}


def test_reset_clears_track_state(checker):
    with patch("time.time", return_value=1000.0):
        checker.check_tracks(1, [make_status(1, False)])
        checker.check_tracks(2, [make_status(1, False)])
    checker.reset(1)
    assert list(checker._track_states) == [(2, 1)]
