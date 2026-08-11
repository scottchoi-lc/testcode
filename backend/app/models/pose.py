"""Player pose estimation using a Hugging Face keypoint-detection model.

Uses ``usyd-community/vitpose-base-simple`` (ViTPose) via `transformers`.
ViTPose is a top-down pose estimator: it needs a person bounding box (from
`DetectionModel`) per image and returns COCO-17 keypoints for that person.
Wrist/shoulder/elbow keypoints are what the fusion logic uses to tell a
shooting motion (wrist rises above the shoulder) apart from dribbling or
passing (wrist stays low).

Pose estimation is treated as an optional signal: if the model/classes
aren't available in the installed `transformers` version, or loading fails
for any reason, we log once and fall back to detection-only heuristics
instead of crashing the whole pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# COCO-17 keypoint order returned by ViTPose.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class PoseResult:
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, score)

    def get(self, name: str) -> tuple[float, float, float] | None:
        return self.keypoints.get(name)


class PoseModel:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.POSE_MODEL
        self.device = device or settings.DEVICE
        self._processor = None
        self._model = None
        self._unavailable = False

    def _ensure_loaded(self) -> bool:
        """Returns True if the model is ready to use, False if unavailable."""
        if self._unavailable:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoProcessor, VitPoseForPoseEstimation

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = VitPoseForPoseEstimation.from_pretrained(self.model_name)
            self._model.to(device)
            self._model.eval()
            self._device = device
            return True
        except Exception:  # noqa: BLE001 - genuinely optional dependency
            logger.warning(
                "Pose model %s unavailable; continuing without pose keypoints.",
                self.model_name,
                exc_info=True,
            )
            self._unavailable = True
            return False

    def estimate(
        self, image_bgr: np.ndarray, person_box: tuple[float, float, float, float]
    ) -> PoseResult | None:
        """Estimate keypoints for a single person box in a single frame."""
        if not self._ensure_loaded():
            return None
        import torch
        from PIL import Image

        rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        x1, y1, x2, y2 = person_box
        coco_box = [[x1, y1, x2 - x1, y2 - y1]]  # ViTPose expects [x, y, w, h]

        inputs = self._processor(pil_image, boxes=[coco_box], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        parsed = self._processor.post_process_pose_estimation(outputs, boxes=[coco_box])
        if not parsed or not parsed[0]:
            return None
        person_result = parsed[0][0]
        keypoints = person_result["keypoints"]
        scores = person_result["scores"]

        kp_dict: dict[str, tuple[float, float, float]] = {}
        for name, (x, y), score in zip(COCO_KEYPOINT_NAMES, keypoints, scores):
            kp_dict[name] = (float(x), float(y), float(score))
        return PoseResult(keypoints=kp_dict)


@lru_cache(maxsize=1)
def get_pose_model() -> PoseModel:
    return PoseModel()
