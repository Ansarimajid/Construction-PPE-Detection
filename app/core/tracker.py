from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.detector import Detection


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedily pair rows and columns by descending IoU, ignoring pairs below threshold."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    rows, cols = np.unravel_index(np.argsort(-iou, axis=None), iou.shape)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if iou[r, c] < threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        pairs.append((r, c))
    return pairs


def _det_box(det: Detection) -> np.ndarray:
    return np.array([det.x1, det.y1, det.x2, det.y2], dtype=float)


@dataclass
class _Track:
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    time_since_update: int = 0
    track_id: Optional[int] = None  # assigned once the track is confirmed

    def predict(self) -> np.ndarray:
        return self.box + self.velocity * (self.time_since_update + 1)

    def update(self, box: np.ndarray) -> None:
        step = (box - self.box) / (self.time_since_update + 1)
        self.velocity = 0.5 * self.velocity + 0.5 * step
        self.box = box
        self.hits += 1
        self.time_since_update = 0


class IoUTracker:
    """
    Lightweight multi-object tracker (SORT-style, constant-velocity prediction + IoU
    matching). One instance per camera so IDs never leak between streams.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def total_tracks(self) -> int:
        """Number of distinct confirmed tracks seen so far."""
        return self._next_id - 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Associate this frame's detections with existing tracks.
        Returns the detections belonging to confirmed tracks, with ``track_id`` set.
        """
        det_boxes = np.array([_det_box(d) for d in detections]).reshape(-1, 4)
        pred_boxes = np.array([t.predict() for t in self._tracks]).reshape(-1, 4)

        matches = greedy_match(box_iou(pred_boxes, det_boxes), self.iou_threshold)
        matched_tracks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}

        det_track: dict[int, _Track] = {}
        for t_idx, d_idx in matches:
            track = self._tracks[t_idx]
            track.update(det_boxes[d_idx])
            det_track[d_idx] = track

        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                track.time_since_update += 1

        for d_idx in range(len(detections)):
            if d_idx not in matched_dets:
                track = _Track(box=det_boxes[d_idx])
                self._tracks.append(track)
                det_track[d_idx] = track

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        tracked: list[Detection] = []
        for d_idx, det in enumerate(detections):
            track = det_track[d_idx]
            if track.track_id is None and track.hits >= self.min_hits:
                track.track_id = self._next_id
                self._next_id += 1
            if track.track_id is not None:
                tracked.append(dataclasses.replace(det, track_id=track.track_id))
        return tracked

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
