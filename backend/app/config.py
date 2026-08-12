"""Central configuration. All knobs can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    # Hugging Face model checkpoints. Swap these to try alternative models.
    # yolos-small over yolos-tiny: a real clip logged the ball detected in
    # only 19/48 sampled frames (40%) with yolos-tiny, which starved the
    # dribbling/passing/shooting heuristics (and hand-dribbling detection)
    # of ball-position evidence. yolos-small is the same architecture/COCO
    # classes (no code changes needed) with materially better accuracy at
    # ~5x the parameters - the trade-off is slower CPU inference per frame.
    # If that's a problem, yolos-tiny is still a drop-in via this env var.
    DETECTION_MODEL: str = os.getenv("DETECTION_MODEL", "hustvl/yolos-small")
    POSE_MODEL: str = os.getenv("POSE_MODEL", "usyd-community/vitpose-base-simple")
    ACTION_MODEL: str = os.getenv("ACTION_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics")
    JERSEY_OCR_MODEL: str = os.getenv("JERSEY_OCR_MODEL", "microsoft/trocr-base-printed")

    # Torch device: "cuda", "mps", or "cpu". Auto-detected at runtime if left as "auto".
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Frame sampling rate used for detection/pose (frames per second).
    ANALYSIS_FPS: float = float(os.getenv("ANALYSIS_FPS", "6"))

    # Sliding window (in analyzed frames) fed to the VideoMAE action classifier.
    ACTION_WINDOW_FRAMES: int = int(os.getenv("ACTION_WINDOW_FRAMES", "16"))
    ACTION_WINDOW_STRIDE: int = int(os.getenv("ACTION_WINDOW_STRIDE", "8"))

    # Detection confidence thresholds. Ball is lower than person because a
    # basketball is small, fast-moving, and often motion-blurred - the
    # detector's confidence on real hits tends to run lower than it does for
    # a whole person, so 0.3 was filtering out plausible ball detections
    # along with genuine false positives. Lowering it trades some extra
    # false positives (e.g. a head or light mistaken for a ball) for
    # meaningfully better recall on frames where the ball actually appears.
    PERSON_SCORE_THRESHOLD: float = float(os.getenv("PERSON_SCORE_THRESHOLD", "0.5"))
    BALL_SCORE_THRESHOLD: float = float(os.getenv("BALL_SCORE_THRESHOLD", "0.15"))

    # Jersey number OCR is run on at most this many frames per clip (evenly
    # spaced), regardless of clip length - it's a per-frame model call, and
    # results are majority-voted, so more than ~20 samples adds runtime
    # without meaningfully improving the vote.
    JERSEY_OCR_MAX_SAMPLES: int = int(os.getenv("JERSEY_OCR_MAX_SAMPLES", "20"))

    # Storage.
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/basketball_analyzer/uploads"))
    RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", "/tmp/basketball_analyzer/results"))

    # Max upload size in bytes (default 300 MB).
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(300 * 1024 * 1024)))

    CORS_ALLOW_ORIGINS: list[str] = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
