"""Coarse action classification using a Hugging Face video-classification model.

Uses ``MCG-NJU/videomae-base-finetuned-kinetics`` (VideoMAE fine-tuned on
Kinetics-400). Kinetics-400 happens to include basketball-relevant classes
such as "dribbling basketball", "shooting basketball", "dunking basketball"
and "playing basketball" (general/ambiguous). It does *not* have a distinct
"passing basketball" class, and its "moving without the ball" concept
doesn't exist as a label at all - which is exactly why this signal is fused
with detection/pose heuristics in `app/pipeline/fusion.py` rather than used
on its own. See that module for how the four target labels are derived.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings


@dataclass
class ClipPrediction:
    top_labels: list[tuple[str, float]]  # (label, probability), sorted desc


class ActionClassifier:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.ACTION_MODEL
        self.device = device or settings.DEVICE
        self._processor = None
        self._model = None
        self._num_frames = settings.ACTION_WINDOW_FRAMES

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        self._model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._num_frames = getattr(self._model.config, "num_frames", self._num_frames)

    def classify_window(self, frames_bgr: list[np.ndarray], top_k: int = 5) -> ClipPrediction:
        """Classify a short clip. `frames_bgr` should have ~`ACTION_WINDOW_FRAMES` frames;
        it is padded/subsampled to the exact count VideoMAE was trained with."""
        self._ensure_loaded()
        import torch

        frames_rgb = [f[:, :, ::-1] for f in frames_bgr]
        frames_rgb = _resample_to_length(frames_rgb, self._num_frames)

        inputs = self._processor(frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        id2label = self._model.config.id2label
        labels = [(id2label[int(idx)], float(score)) for score, idx in zip(top.values, top.indices)]
        return ClipPrediction(top_labels=labels)


def _resample_to_length(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Cannot classify an empty window of frames")
    if len(frames) == length:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=length).round().astype(int)
    return [frames[i] for i in indices]


@lru_cache(maxsize=1)
def get_action_classifier() -> ActionClassifier:
    return ActionClassifier()
