from __future__ import annotations

import cv2
import numpy as np

from app.core.detector import Detection
from app.core.pose import KEYPOINT_CONFIDENCE, SKELETON, Pose
from app.core.ppe_assigner import PersonStatus

COMPLIANT_COLOR = (0, 200, 0)
NO_VEST_COLOR = (0, 165, 255)
NO_HARDHAT_COLOR = (0, 0, 255)
SKELETON_COLOR = (255, 255, 255)


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.4,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
    padding: int = 5,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    x, y = position

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        bg_color,
        -1,
    )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_skeleton(frame: np.ndarray, pose: Pose) -> None:
    kpts = pose.keypoints
    for a, b in SKELETON:
        if kpts[a, 2] >= KEYPOINT_CONFIDENCE and kpts[b, 2] >= KEYPOINT_CONFIDENCE:
            pt_a = (int(kpts[a, 0]), int(kpts[a, 1]))
            pt_b = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(frame, pt_a, pt_b, SKELETON_COLOR, 1, cv2.LINE_AA)
    for x, y, conf in kpts:
        if conf >= KEYPOINT_CONFIDENCE:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_person(frame: np.ndarray, status: PersonStatus) -> None:
    p = status.person
    if not status.has_hardhat:
        color = NO_HARDHAT_COLOR
    elif not status.has_vest:
        color = NO_VEST_COLOR
    else:
        color = COMPLIANT_COLOR

    if status.pose is not None:
        draw_skeleton(frame, status.pose)

    cv2.rectangle(frame, (p.x1, p.y1), (p.x2, p.y2), color, 2)
    label = f"ID {status.track_id}"
    if not status.has_hardhat:
        label += " | No Hardhat"
    if not status.has_vest:
        label += " | No Vest"
    draw_text_with_background(
        frame,
        label,
        (p.x1, p.y1 - 10),
        font_scale=0.45,
        color=(255, 255, 255),
        bg_color=color,
        alpha=0.8,
        padding=4,
    )


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    hardhat_count: int,
    vest_count: int,
    person_count: int,
    show_email_sent: bool = False,
    persons: list[PersonStatus] | None = None,
) -> np.ndarray:
    annotated = frame.copy()

    for det in detections:
        # Tracked people are drawn separately with their ID and PPE status
        if persons is not None and det.class_name == "Person":
            continue
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), det.color, 2)
        label = f"{det.class_name} ({det.confidence:.2f})"
        draw_text_with_background(
            annotated,
            label,
            (det.x1, det.y1 - 10),
            font_scale=0.4,
            color=(255, 255, 255),
            bg_color=det.color,
            alpha=0.8,
            padding=4,
        )

    for status in persons or []:
        draw_person(annotated, status)

    # Sideboard counts
    for i, text in enumerate([
        f"Hardhats: {hardhat_count}",
        f"Safety Vests: {vest_count}",
        f"People: {person_count}",
    ]):
        draw_text_with_background(
            annotated,
            text,
            (10, 30 + i * 30),
            font_scale=0.5,
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
            alpha=0.7,
            padding=5,
        )

    if show_email_sent:
        draw_text_with_background(
            annotated,
            "Alert Sent",
            (annotated.shape[1] - 110, 30),
            font_scale=0.5,
            color=(0, 255, 0),
            bg_color=(0, 0, 0),
            alpha=0.8,
            padding=5,
        )

    return annotated
