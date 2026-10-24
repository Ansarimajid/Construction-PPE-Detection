from __future__ import annotations

import numpy as np

from app.core.detector import Detection
from app.core.pose import (
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Pose,
)
from app.core.ppe_assigner import (
    PersonStatus,
    PPEStatusSmoother,
    analyze_persons,
    head_region,
    match_poses,
)


def make_detection(
    class_name: str, x1: int, y1: int, x2: int, y2: int, track_id: int | None = None
) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2,
        color=(255, 0, 0),
        track_id=track_id,
    )


def make_pose(x1: int, y1: int, x2: int, y2: int, points: dict[int, tuple[float, float]]) -> Pose:
    kpts = np.zeros((17, 3))
    for idx, (x, y) in points.items():
        kpts[idx] = (x, y, 0.9)
    return Pose(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, keypoints=kpts)


def test_hardhat_assigned_to_correct_person():
    alice = make_detection("Person", 0, 0, 100, 300, track_id=1)
    bob = make_detection("Person", 200, 0, 300, 300, track_id=2)
    hardhat = make_detection("Hardhat", 220, -10, 280, 40)

    statuses = analyze_persons([alice, bob], [alice, bob, hardhat], poses=[])

    assert [(s.track_id, s.has_hardhat) for s in statuses] == [(1, False), (2, True)]


def test_hardhat_held_at_waist_does_not_count():
    worker = make_detection("Person", 0, 0, 100, 300, track_id=1)
    hardhat_in_hand = make_detection("Hardhat", 60, 150, 110, 190)

    [status] = analyze_persons([worker], [worker, hardhat_in_hand], poses=[])

    assert status.has_hardhat is False


def test_vest_assigned_via_torso():
    worker = make_detection("Person", 0, 0, 100, 300, track_id=1)
    vest = make_detection("Safety Vest", 10, 80, 90, 180)

    [status] = analyze_persons([worker], [worker, vest], poses=[])

    assert status.has_vest is True
    assert status.has_hardhat is False


def test_pose_matched_and_used_for_head_region():
    # Worker crouching: head is in the middle of the box, not at the top
    worker = make_detection("Person", 0, 0, 100, 300, track_id=1)
    pose = make_pose(0, 0, 100, 300, {
        NOSE: (50, 160), LEFT_EYE: (45, 155), RIGHT_EYE: (55, 155),
        LEFT_SHOULDER: (30, 190), RIGHT_SHOULDER: (70, 190),
        LEFT_HIP: (35, 260), RIGHT_HIP: (65, 260),
    })
    hardhat = make_detection("Hardhat", 35, 115, 65, 145)

    [status] = analyze_persons([worker], [worker, hardhat], poses=[pose])

    assert status.pose is pose
    assert status.has_hardhat is True
    region = head_region(worker, pose)
    assert region[1] > 0  # head region moved down from the top of the box


def test_match_poses_ignores_non_overlapping():
    worker = make_detection("Person", 0, 0, 100, 300, track_id=1)
    far_pose = make_pose(500, 0, 600, 300, {})
    assert match_poses([worker], [far_pose]) == [None]


def make_status(track_id: int | None, has_hardhat: bool, has_vest: bool = True) -> PersonStatus:
    person = make_detection("Person", 0, 0, 100, 300, track_id=track_id)
    return PersonStatus(person=person, pose=None, has_hardhat=has_hardhat, has_vest=has_vest)


def test_smoother_ignores_single_missed_frame():
    smoother = PPEStatusSmoother(window=5)
    for _ in range(4):
        smoother.update([make_status(1, True)])
    [status] = smoother.update([make_status(1, False)])
    assert status.has_hardhat is True


def test_smoother_follows_sustained_change():
    smoother = PPEStatusSmoother(window=5)
    for _ in range(5):
        smoother.update([make_status(1, True)])
    for _ in range(3):
        [status] = smoother.update([make_status(1, False)])
    assert status.has_hardhat is False


def test_smoother_keeps_workers_separate():
    smoother = PPEStatusSmoother(window=5)
    for _ in range(4):
        smoother.update([make_status(1, True), make_status(2, False)])
    a, b = smoother.update([make_status(1, False), make_status(2, True)])
    assert (a.has_hardhat, b.has_hardhat) == (True, False)


def test_smoother_passes_through_untracked_and_disabled():
    assert PPEStatusSmoother(window=5).update([make_status(None, False)])[0].has_hardhat is False
    disabled = PPEStatusSmoother(window=1)
    disabled.update([make_status(1, True)])
    assert disabled.update([make_status(1, False)])[0].has_hardhat is False


def test_smoother_forgets_idle_workers():
    smoother = PPEStatusSmoother(window=5, max_idle_frames=2)
    smoother.update([make_status(1, True)])
    for _ in range(3):
        smoother.update([])
    assert smoother._history == {}
