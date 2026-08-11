"""Person + basketball detection using a Hugging Face object detection model.

Uses ``hustvl/yolos-tiny`` (YOLOS, trained on COCO) via the `transformers`
`object-detection` pipeline. COCO includes both a "person" class and a
"sports ball" class, which is what we key off of to find players and the
ball in each sampled frame. The heavy `torch`/`transformers` imports are
deferred to first use so importing this module (and starting the API
server) never requires them to be installed or a model to be downloaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings

BALL_LABELS = {"sports ball"}
PERSON_LABELS = {"person"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class DetectionModel:
    """Thin wrapper around a HF object-detection pipeline, loaded lazily."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.DETECTION_MODEL
        self.device = device or settings.DEVICE
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        import torch
        from transformers import pipeline as hf_pipeline

        device = self.device
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            device = -1
        self._pipeline = hf_pipeline(
            "object-detection", model=self.model_name, device=device
        )

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR (OpenCV-style) frame."""
        self._ensure_loaded()
        from PIL import Image

        rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        raw = self._pipeline(pil_image)
        results: list[Detection] = []
        for item in raw:
            box = item["box"]
            results.append(
                Detection(
                    label=item["label"],
                    score=float(item["score"]),
                    box=(box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
                )
            )
        return results

    def detect_ball_and_players(
        self, image_bgr: np.ndarray
    ) -> tuple[list[Detection], list[Detection]]:
        """Convenience helper: return (ball_detections, person_detections) filtered by score."""
        detections = self.detect(image_bgr)
        balls = [
            d
            for d in detections
            if d.label in BALL_LABELS and d.score >= settings.BALL_SCORE_THRESHOLD
        ]
        people = [
            d
            for d in detections
            if d.label in PERSON_LABELS and d.score >= settings.PERSON_SCORE_THRESHOLD
        ]
        return balls, people


@lru_cache(maxsize=1)
def get_detection_model() -> DetectionModel:
    return DetectionModel()
