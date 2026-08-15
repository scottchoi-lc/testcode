"""Coarse action classification using a Hugging Face video-text model.

Uses ``microsoft/xclip-base-patch32`` (X-CLIP), a CLIP-style model trained
contrastively on (video, text) pairs. Two reasons this replaced
``MCG-NJU/videomae-base-finetuned-kinetics`` (VideoMAE fine-tuned on
Kinetics-400):

1. License. VideoMAE-Kinetics is CC-BY-NC-4.0, which blocks commercial use
   of the model *and* of anything fine-tuned from it. X-CLIP is MIT.
2. Fit. VideoMAE's classification head is fixed to Kinetics-400's 400
   classes, which has no "passing basketball" concept at all - a real,
   longstanding gap (see fusion.py's history). X-CLIP does zero-shot
   video-text similarity instead of reading from a fixed head, so we
   supply the exact candidate phrases we want scored
   (``settings.ACTION_CANDIDATE_LABELS``), including a real passing
   candidate.

Note for anyone tuning fusion.py's KINETICS_OVERRIDE_MIN and similar
thresholds: those were calibrated against Kinetics-400's ~400-way softmax,
where an unrelated class's "noise floor" is close to 1/400 (~0.0025), so
even a small nonzero score was already meaningfully above chance. X-CLIP's
softmax here runs over only ``len(ACTION_CANDIDATE_LABELS)`` candidates
(6 by default), so the chance-level baseline is much higher (~1/6 ~= 0.17) -
thresholds tuned against the old model's output distribution are not
automatically valid against the new one and should be re-checked against
real logged scores before being trusted.
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
        self.candidate_labels = list(settings.ACTION_CANDIDATE_LABELS)
        self._processor = None
        self._model = None
        self._num_frames = settings.ACTION_WINDOW_FRAMES

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = XCLIPProcessor.from_pretrained(self.model_name)
        self._model = XCLIPModel.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._num_frames = getattr(self._model.config.vision_config, "num_frames", self._num_frames)

    def classify_window(
        self,
        frames_bgr: list[np.ndarray],
        top_k: int = 5,
        candidate_labels: list[str] | None = None,
    ) -> ClipPrediction:
        """Classify a short clip via X-CLIP's zero-shot video-text
        similarity, against `self.candidate_labels` by default. `frames_bgr`
        should have ~`ACTION_WINDOW_FRAMES` frames; it is padded/subsampled
        to the exact count X-CLIP was trained with.

        `candidate_labels`, if given, scores against a different phrase set
        for this one call instead (e.g. shot-outcome phrases rather than
        action-type phrases - see `settings.SHOT_OUTCOME_CANDIDATE_LABELS`)
        without needing a second `ActionClassifier`/model load - the
        candidate labels are just the text side of a CLIP-style similarity
        score, not something baked into the loaded weights.
        """
        self._ensure_loaded()
        import torch

        labels = candidate_labels if candidate_labels is not None else self.candidate_labels
        frames_rgb = [f[:, :, ::-1] for f in frames_bgr]
        frames_rgb = _resample_to_length(frames_rgb, self._num_frames)

        # `images=`, not `videos=`: newer `transformers` added `videos=` as a
        # convenience alias that maps internally to `images=`, but older
        # releases (e.g. the last ones still installable on Python 3.9) never
        # had a `videos` parameter at all - passed as a keyword, it silently
        # falls into **kwargs and gets ignored, leaving `images=None` and no
        # `pixel_values` in the output at all (confirmed from a real
        # `AttributeError: 'NoneType' object has no attribute 'shape'` on
        # `pixel_values.shape` inside the model - not a shape mismatch, a
        # missing key). `images=` is understood by both: even the newest
        # processor's `videos=` support is just `if videos is not None and
        # images is None: images = videos` before delegating onward, so
        # passing `images=` directly skips that alias and works identically.
        inputs = self._processor(
            text=labels, images=frames_rgb, return_tensors="pt", padding=True
        )
        if inputs.get("pixel_values") is None:
            raise RuntimeError(
                "XCLIPProcessor did not produce pixel_values for the video frames. "
                "This usually means the installed `transformers` version's XCLIPProcessor "
                "expects a different argument for video frames than `images=`/`videos=` - "
                "check `XCLIPProcessor.__call__`'s signature for the installed version."
            )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits_per_video = self._model(**inputs).logits_per_video[0]
        probs = torch.softmax(logits_per_video, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        top_labels = [
            (labels[int(idx)], float(score)) for score, idx in zip(top.values, top.indices)
        ]
        return ClipPrediction(top_labels=top_labels)


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
