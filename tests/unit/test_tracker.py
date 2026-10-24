from __future__ import annotations

import numpy as np

from app.core.detector import Detection
from app.core.tracker import IoUTracker, box_iou, greedy_match


def make_person(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=5,
        class_name="Person",
        confidence=confidence,
        x1=x1, y1=y1, x2=x2, y2=y2,
        color=(0, 255, 255),
    )


def test_box_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = box_iou(a, b)
    assert iou[0, 0] == 1.0
    assert iou[0, 1] == 0.0


def test_greedy_match_prefers_highest_iou():
    iou = np.array([[0.9, 0.5], [0.8, 0.1]])
    assert sorted(greedy_match(iou, 0.3)) == [(0, 0)]


def test_track_confirmed_after_min_hits():
    tracker = IoUTracker(min_hits=3)
    assert tracker.update([make_person(0, 0, 50, 100)]) == []
    assert tracker.update([make_person(2, 0, 52, 100)]) == []
    tracked = tracker.update([make_person(4, 0, 54, 100)])
    assert [d.track_id for d in tracked] == [1]


def test_ids_stable_while_people_move():
    tracker = IoUTracker(min_hits=1)
    for step in range(10):
        dx = step * 5
        tracked = tracker.update([
            make_person(0 + dx, 0, 50 + dx, 100),
            make_person(300 - dx, 0, 350 - dx, 100),
        ])
        ids = {d.x1 < 150: d.track_id for d in tracked}
        assert ids == {True: 1, False: 2}
    assert tracker.total_tracks == 2


def test_track_survives_short_occlusion():
    tracker = IoUTracker(min_hits=1, max_age=5)
    tracker.update([make_person(0, 0, 50, 100)])
    for _ in range(3):
        tracker.update([])
    tracked = tracker.update([make_person(0, 0, 50, 100)])
    assert tracked[0].track_id == 1


def test_track_dropped_after_max_age():
    tracker = IoUTracker(min_hits=1, max_age=2)
    tracker.update([make_person(0, 0, 50, 100)])
    for _ in range(3):
        tracker.update([])
    tracked = tracker.update([make_person(0, 0, 50, 100)])
    assert tracked[0].track_id == 2


def test_reset_restarts_ids():
    tracker = IoUTracker(min_hits=1)
    tracker.update([make_person(0, 0, 50, 100)])
    tracker.reset()
    assert tracker.update([make_person(0, 0, 50, 100)])[0].track_id == 1
