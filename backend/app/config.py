"""Central configuration. All knobs can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    # Hugging Face model checkpoints. Swap these to try alternative models.
    DETECTION_MODEL: str = os.getenv("DETECTION_MODEL", "hustvl/yolos-tiny")
    POSE_MODEL: str = os.getenv("POSE_MODEL", "usyd-community/vitpose-base-simple")
    ACTION_MODEL: str = os.getenv("ACTION_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics")

    # Torch device: "cuda", "mps", or "cpu". Auto-detected at runtime if left as "auto".
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Frame sampling rate used for detection/pose (frames per second).
    ANALYSIS_FPS: float = float(os.getenv("ANALYSIS_FPS", "6"))

    # Sliding window (in analyzed frames) fed to the VideoMAE action classifier.
    ACTION_WINDOW_FRAMES: int = int(os.getenv("ACTION_WINDOW_FRAMES", "16"))
    ACTION_WINDOW_STRIDE: int = int(os.getenv("ACTION_WINDOW_STRIDE", "8"))

    # Detection confidence thresholds.
    PERSON_SCORE_THRESHOLD: float = float(os.getenv("PERSON_SCORE_THRESHOLD", "0.5"))
    BALL_SCORE_THRESHOLD: float = float(os.getenv("BALL_SCORE_THRESHOLD", "0.3"))

    # Storage.
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/basketball_analyzer/uploads"))
    RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", "/tmp/basketball_analyzer/results"))

    # Max upload size in bytes (default 300 MB).
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(300 * 1024 * 1024)))

    CORS_ALLOW_ORIGINS: list[str] = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
