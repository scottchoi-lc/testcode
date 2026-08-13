"""Frame extraction and simple geometric helpers used by the analysis pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds from clip start
    image: np.ndarray  # BGR, as read by OpenCV


def extract_frames(video_path: str, target_fps: float) -> tuple[list[Frame], float]:
    """Sample a video at ``target_fps`` and return the frames plus the clip duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / native_fps if native_fps > 0 else 0.0

    step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1

    frames: list[Frame] = []
    index = 0
    sampled = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if index % step == 0:
            timestamp = index / native_fps
            frames.append(Frame(index=sampled, timestamp=timestamp, image=image))
            sampled += 1
        index += 1
    cap.release()

    if duration == 0.0 and frames:
        duration = frames[-1].timestamp

    return frames, duration


def probe_video(video_path: str) -> tuple[float, int, int]:
    """Return (duration_seconds, width, height) without decoding every frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return duration, width, height


def extract_frame_at(video_path: str, timestamp: float) -> np.ndarray:
    """Return the single BGR frame nearest ``timestamp`` seconds into the clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_index = max(0, round(timestamp * fps))
    if frame_count > 0:
        target_index = min(target_index, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame near timestamp {timestamp}s")
    return image


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_diag(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
