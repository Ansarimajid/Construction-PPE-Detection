from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# BGR colours matching original webcam.py
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),    # Hardhat
    1: (0, 255, 0),    # Mask
    2: (0, 0, 255),    # NO-Hardhat
    3: (255, 255, 0),  # NO-Mask
    4: (255, 0, 255),  # NO-Safety Vest
    5: (0, 255, 255),  # Person
    6: (128, 0, 128),  # Safety Cone
    7: (128, 128, 0),  # Safety Vest
    8: (0, 128, 128),  # Machinery
    9: (128, 128, 128),  # Vehicle
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    color: tuple[int, int, int]
    track_id: Optional[int] = None  # set by IoUTracker for tracked objects


class PPEDetector:
    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        logger.info("PPEDetector loaded: %s (conf=%.2f)", model_path, confidence)

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                detections.append(
                    Detection(
                        class_id=cls,
                        class_name=self.model.names[cls],
                        confidence=float(box.conf[0]),
                        x1=int(box.xyxy[0][0]),
                        y1=int(box.xyxy[0][1]),
                        x2=int(box.xyxy[0][2]),
                        y2=int(box.xyxy[0][3]),
                        color=CLASS_COLORS.get(cls, (200, 200, 200)),
                    )
                )

        return detections
