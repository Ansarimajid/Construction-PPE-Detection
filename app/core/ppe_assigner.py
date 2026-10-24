from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.detector import Detection
from app.core.pose import HEAD_KEYPOINTS, TORSO_KEYPOINTS, Pose
from app.core.tracker import box_iou, greedy_match

POSE_MATCH_IOU = 0.3
MIN_ITEM_OVERLAP = 0.3  # fraction of a PPE box that must fall inside the body region


@dataclass
class PersonStatus:
    person: Detection
    pose: Optional[Pose]
    has_hardhat: bool
    has_vest: bool

    @property
    def track_id(self) -> Optional[int]:
        return self.person.track_id


def _box(obj: Detection | Pose) -> np.ndarray:
    return np.array([obj.x1, obj.y1, obj.x2, obj.y2], dtype=float)


def head_region(person: Detection, pose: Optional[Pose]) -> np.ndarray:
    """Region where a worn hardhat should appear. Uses face keypoints when available."""
    w = person.x2 - person.x1
    h = person.y2 - person.y1
    pts = pose.visible(HEAD_KEYPOINTS) if pose is not None else np.empty((0, 2))
    if len(pts):
        cx, cy = pts.mean(axis=0)
        size = max(float(np.ptp(pts[:, 0])), 0.25 * w)
        # Hardhat sits above the eyes, so extend further upwards than downwards
        return np.array([cx - size, cy - 1.5 * size, cx + size, cy + 0.75 * size])
    return np.array([person.x1, person.y1 - 0.1 * h, person.x2, person.y1 + 0.3 * h])


def torso_region(person: Detection, pose: Optional[Pose]) -> np.ndarray:
    """Region where a worn safety vest should appear. Uses shoulder/hip keypoints when available."""
    w = person.x2 - person.x1
    h = person.y2 - person.y1
    pts = pose.visible(TORSO_KEYPOINTS) if pose is not None else np.empty((0, 2))
    if len(pts) >= 2:
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        if y2 - y1 < 0.2 * h:  # hips not visible
            y2 = y1 + 0.4 * h
        pad = 0.2 * w
        return np.array([x1 - pad, y1 - 0.05 * h, x2 + pad, y2])
    return np.array([person.x1, person.y1 + 0.2 * h, person.x2, person.y1 + 0.7 * h])


def _overlap_fraction(items: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """(N, M) fraction of each item box's area lying inside each region box."""
    if len(items) == 0 or len(regions) == 0:
        return np.zeros((len(items), len(regions)))
    x1 = np.maximum(items[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(items[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(items[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(items[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (items[:, 2] - items[:, 0]) * (items[:, 3] - items[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def _assign(items: list[Detection], regions: np.ndarray) -> set[int]:
    """Assign each PPE item to at most one person region; return indices of people that got one."""
    if not items or len(regions) == 0:
        return set()
    overlap = _overlap_fraction(np.array([_box(d) for d in items]), regions)
    owners: set[int] = set()
    for row in overlap:
        best = int(np.argmax(row))
        if row[best] >= MIN_ITEM_OVERLAP:
            owners.add(best)
    return owners


def match_poses(persons: list[Detection], poses: list[Pose]) -> list[Optional[Pose]]:
    """Pair each person detection with the pose skeleton covering the same body."""
    matched: list[Optional[Pose]] = [None] * len(persons)
    if not persons or not poses:
        return matched
    iou = box_iou(
        np.array([_box(p) for p in persons]),
        np.array([_box(p) for p in poses]),
    )
    for p_idx, k_idx in greedy_match(iou, POSE_MATCH_IOU):
        matched[p_idx] = poses[k_idx]
    return matched


def analyze_persons(
    persons: list[Detection],
    detections: list[Detection],
    poses: list[Pose],
) -> list[PersonStatus]:
    """Work out which tracked person is wearing which PPE item."""
    matched_poses = match_poses(persons, poses)

    heads = np.array([head_region(p, k) for p, k in zip(persons, matched_poses)]).reshape(-1, 4)
    torsos = np.array([torso_region(p, k) for p, k in zip(persons, matched_poses)]).reshape(-1, 4)

    with_hardhat = _assign([d for d in detections if d.class_name == "Hardhat"], heads)
    with_vest = _assign([d for d in detections if d.class_name == "Safety Vest"], torsos)

    return [
        PersonStatus(
            person=person,
            pose=pose,
            has_hardhat=i in with_hardhat,
            has_vest=i in with_vest,
        )
        for i, (person, pose) in enumerate(zip(persons, matched_poses))
    ]
