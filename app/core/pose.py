from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# COCO-17 keypoint indices
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
TORSO_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# Limb connections for drawing the skeleton
SKELETON: list[tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # legs + hips
    (5, 11), (6, 12), (5, 6),                          # torso
    (5, 7), (7, 9), (6, 8), (8, 10),                   # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head
]

KEYPOINT_CONFIDENCE = 0.5


@dataclass
class Pose:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    keypoints: np.ndarray  # (17, 3): x, y, confidence

    def visible(self, indices: tuple[int, ...]) -> np.ndarray:
        """Return (K, 2) xy of the given keypoints whose confidence passes the threshold."""
        kpts = self.keypoints[list(indices)]
        return kpts[kpts[:, 2] >= KEYPOINT_CONFIDENCE, :2]


class PoseEstimator:
    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        logger.info("PoseEstimator loaded: %s (conf=%.2f)", model_path, confidence)

    def estimate(self, frame: np.ndarray) -> list[Pose]:
        results = self.model(frame, conf=self.confidence, verbose=False)
        poses: list[Pose] = []

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()
            for box, conf, kpts in zip(boxes, confs, keypoints):
                poses.append(
                    Pose(
                        x1=int(box[0]),
                        y1=int(box[1]),
                        x2=int(box[2]),
                        y2=int(box[3]),
                        confidence=float(conf),
                        keypoints=kpts,
                    )
                )

        return poses
