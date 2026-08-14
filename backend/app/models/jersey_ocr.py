"""Best-effort jersey number reading using a Hugging Face OCR model.

Uses ``microsoft/trocr-base-printed`` (TrOCR, an image-to-text transformer
for a single line of printed text) on a crop of the tracked player's torso.

Reading a jersey number off casual phone video is a genuinely hard OCR
target: motion blur, an oblique camera angle, a curved/wrinkled surface,
and non-standard fonts all work against it. So this is treated as an
optional enhancement layered on top of the rest of the pipeline, the same
way pose estimation is: readings are collected across several frames and
`JerseyNumberAggregator` only returns a number if it was read consistently
enough to be a plausible majority vote. Otherwise the narration falls back
to a generic "Player 1" subject rather than reporting a guess.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_DIGIT_RE = re.compile(r"\d{1,2}")


class JerseyNumberReader:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.JERSEY_OCR_MODEL
        self.device = device or settings.DEVICE
        self._processor = None
        self._model = None
        self._unavailable = False

    def _ensure_loaded(self) -> bool:
        if self._unavailable:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self._model.to(device)
            self._model.eval()
            self._device = device
            return True
        except Exception:  # noqa: BLE001 - genuinely optional signal
            logger.warning(
                "Jersey OCR model %s unavailable; continuing without jersey numbers.",
                self.model_name,
                exc_info=True,
            )
            self._unavailable = True
            return False

    def read_crop(
        self, image_bgr: np.ndarray, player_box: tuple[float, float, float, float]
    ) -> str | None:
        """Best-effort read of a 1-2 digit jersey number from the player's torso."""
        if not self._ensure_loaded():
            return None
        try:
            return self._read_crop(image_bgr, player_box)
        except Exception:  # noqa: BLE001 - one bad frame shouldn't sink the job
            logger.warning("Jersey OCR failed on a frame; skipping.", exc_info=True)
            return None

    def _read_crop(
        self, image_bgr: np.ndarray, player_box: tuple[float, float, float, float]
    ) -> str | None:
        import torch
        from PIL import Image

        x1, y1, x2, y2 = player_box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return None

        # Heuristic torso region: numbers usually sit on the chest/back, so
        # take the upper-middle band of the box and trim the side margins
        # (arms, background) rather than trying to localize the digits.
        top = y1 + 0.15 * height
        bottom = y1 + 0.55 * height
        left = x1 + 0.15 * width
        right = x2 - 0.15 * width

        img_h, img_w = image_bgr.shape[:2]
        top, bottom = int(max(0, top)), int(min(img_h, bottom))
        left, right = int(max(0, left)), int(min(img_w, right))
        if bottom - top < 10 or right - left < 10:
            return None

        crop_rgb = image_bgr[top:bottom, left:right, ::-1]
        pil_image = Image.fromarray(crop_rgb)

        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, max_new_tokens=8)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        match = _DIGIT_RE.search(text)
        reading = match.group(0) if match else None
        logger.debug("Jersey OCR raw read: %r -> %r", text, reading)
        return reading


class JerseyNumberAggregator:
    """Collects per-frame OCR readings and produces a majority-vote guess.

    Two separate things determine confidence, and they must not be blended
    into one fraction: how often a frame yields *any* digit reading at all
    (often low on hard footage - blur, angle, occlusion - and that's fine),
    versus, among the readings that did happen, how much they agree with
    each other (the real signal for "did we read the right number"). Only
    the latter is checked against MIN_FRACTION; a clip where 5 of 24 frames
    read anything but 3 of those 5 agreed is a good result, not a bad one.
    """

    MIN_VOTES = 3
    MIN_FRACTION = 0.5

    def __init__(self):
        self._counter: Counter[str] = Counter()
        self._total = 0

    def add(self, reading: str | None) -> None:
        self._total += 1
        if reading:
            self._counter[reading] += 1

    def best_guess(self) -> str | None:
        if not self._counter:
            return None
        number, votes = self._counter.most_common(1)[0]
        if votes < self.MIN_VOTES:
            return None
        total_readings = sum(self._counter.values())
        if votes / total_readings < self.MIN_FRACTION:
            return None
        return number

    def debug_summary(self) -> str:
        """Human-readable vote breakdown, for logging why a guess was/wasn't made."""
        votes = dict(self._counter.most_common())
        total_readings = sum(self._counter.values())
        return f"{self._total} samples, {total_readings} with a digit, readings={votes or '{}'}"


@lru_cache(maxsize=1)
def get_jersey_number_reader() -> JerseyNumberReader:
    return JerseyNumberReader()
