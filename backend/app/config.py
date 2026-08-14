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
    # microsoft/xclip-base-patch32, not MCG-NJU/videomae-base-finetuned-kinetics
    # (swapped out): the VideoMAE-Kinetics checkpoint is CC-BY-NC-4.0, which
    # blocks commercial use of the model *and* anything fine-tuned from it -
    # a real problem if this app is ever meant to be a commercial product.
    # X-CLIP is MIT licensed. It's also a better fit for this task on the
    # merits, not just the license: VideoMAE's fixed Kinetics-400 head has no
    # "passing basketball" class at all (a real, longstanding gap - see
    # fusion.py's history), whereas X-CLIP does zero-shot video-text
    # similarity against whatever candidate phrases we supply
    # (ACTION_CANDIDATE_LABELS below), so we can name the exact concept we
    # want scored, including a real passing candidate.
    ACTION_MODEL: str = os.getenv("ACTION_MODEL", "microsoft/xclip-base-patch32")
    JERSEY_OCR_MODEL: str = os.getenv("JERSEY_OCR_MODEL", "microsoft/trocr-base-printed")

    # Candidate phrases scored per fusion window via X-CLIP's zero-shot
    # video-text similarity (softmax over just these candidates, not a fixed
    # 400-way head) - app/pipeline/fusion.py's KINETICS_*_LABELS constants
    # must reference these exact strings to match on them. Includes a
    # generic/ambiguous catch-all ("playing basketball", mirroring the old
    # Kinetics-400 signal of the same name) and an explicit negative anchor
    # (unrelated activity) so the softmax has somewhere for probability mass
    # to go on a clip that doesn't clearly match any specific action, rather
    # than being forced to spread only across basketball-specific phrases.
    ACTION_CANDIDATE_LABELS: list[str] = [
        "dribbling a basketball",
        "shooting a basketball",
        "passing a basketball to a teammate",
        "catching or receiving a basketball pass",
        "a basketball player moving without the ball",
        "playing basketball",
        "a person doing an activity unrelated to basketball",
    ]

    # Torch device: "cuda", "mps", or "cpu". Auto-detected at runtime if left as "auto".
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Frame sampling rate used for detection/pose (frames per second).
    ANALYSIS_FPS: float = float(os.getenv("ANALYSIS_FPS", "6"))

    # Sliding window (in analyzed frames) fed to the action classifier
    # (X-CLIP - see ACTION_MODEL above). This stays relatively coarse/
    # expensive per inference call on CPU - narrative granularity comes from
    # FUSION_WINDOW_SECONDS below, not from running this more often. X-CLIP's
    # own num_frames (8 by default) can differ from this; ActionClassifier
    # resamples whatever's collected here down/up to match, so this setting
    # mainly controls collection/coarse-window granularity, not the exact
    # frame count the model sees.
    ACTION_WINDOW_FRAMES: int = int(os.getenv("ACTION_WINDOW_FRAMES", "16"))
    ACTION_WINDOW_STRIDE: int = int(os.getenv("ACTION_WINDOW_STRIDE", "8"))

    # Window size (in seconds) for the rule-based fusion scoring in
    # fusion.py - deliberately decoupled from ACTION_WINDOW_FRAMES/STRIDE
    # above. The rule-based scoring only needs already-computed per-frame
    # ball/pose signals (cheap), unlike the action classifier (expensive CPU
    # inference), so it can run at a much finer grain without added model
    # cost - each fusion window just borrows the label scores from whichever
    # coarse classifier window is temporally closest. Finer windows mean
    # merge_adjacent_segments produces more, shorter segments instead of
    # smoothing quick individual actions into one long block.
    FUSION_WINDOW_SECONDS: float = float(os.getenv("FUSION_WINDOW_SECONDS", "0.8"))

    # Stride (in seconds) between fusion windows - deliberately less than
    # FUSION_WINDOW_SECONDS so consecutive windows overlap, the same way
    # ACTION_WINDOW_STRIDE < ACTION_WINDOW_FRAMES makes Kinetics windows
    # overlap above. Without overlap, a brief ball release (pass or shot)
    # that happens to land right at a window boundary is split across two
    # windows and neither one ever sees the "starts with the ball, ends
    # without it" pattern score_window's shooting/passing branches look
    # for - a real clip's logs showed exactly this (one window still had
    # the ball, the very next had lost it entirely, and the release in
    # between was never classified as anything but a gap). 0.4 (50% of the
    # default 0.8s window) guarantees any such transition is fully
    # contained in at least one window.
    FUSION_WINDOW_STRIDE_SECONDS: float = float(os.getenv("FUSION_WINDOW_STRIDE_SECONDS", "0.4"))

    # Longest run of consecutive ball-detection misses (in sampled frames)
    # that gets linearly interpolated from the nearest valid reading on
    # each side, when the player was still tracked throughout. The ball is
    # the least reliable detection in the pipeline - small, fast, and prone
    # to motion blur exactly when it matters most (a release mid-flight) -
    # and a real clip logged a whole fusion window with zero ball
    # detections right where a pass actually happened, leaving the
    # passing/shooting heuristics with no position data at all even after
    # widening the fusion windows to overlap.
    #
    # Started at 2 frames (~0.3s) but a real clip's release still went
    # completely undetected at that bound - two adjacent fusion windows
    # showed ball_frames_detected of 1 and 0 respectively (out of ~5 raw
    # frames each), meaning the actual gap ran longer than 2 consecutive
    # frames, confirmed directly by pipeline.py's "Ball gap interpolation:
    # ... longest gap seen=N frames" log line rather than inferred. Raised
    # to 4 (~0.7s), then to 8 (~1.3s) - both evidence-motivated, not
    # re-guesses: with tracking correctly locked onto the labeled player
    # (ruling out "wrong player" as the cause), the same clip's actual
    # release still logged longest gap seen=8 frames, straight ball
    # detection dropout during the fastest part of the motion (matches the
    # mechanical expectation - the release is exactly when the ball moves
    # fastest, so it's exactly when motion blur is most likely to defeat
    # the detector).
    #
    # 8 frames (~1.3s) is a real stretch to trust a straight-line
    # interpolation for - if `longest_ball_gap_frames` still exceeds this
    # bound on a future clip, that's the signal to stop raising it and
    # address detection recall directly instead (e.g. BALL_SCORE_THRESHOLD,
    # or reconsidering the detection model), rather than keep widening a
    # window that's already covering more of the ball's real trajectory
    # than a straight line can respect.
    BALL_GAP_INTERPOLATION_MAX_FRAMES: int = int(os.getenv("BALL_GAP_INTERPOLATION_MAX_FRAMES", "8"))

    # How far past a fusion window's end to look, when deciding whether a
    # candidate PASSING window is actually a crossover/hesitation dribble:
    # both have the same "released, lateral, no wrist raise" signature, but
    # a crossover's ball comes straight back into the same player's hands a
    # fraction of a second later while a real pass to a teammate doesn't. A
    # real clip confirmed this exact case (a crossover scored as passing,
    # with the ball back in close possession in the very next fusion
    # window). 0.5s is long enough to catch that quick a return without
    # being so long it could also swallow a genuine, fast give-and-go.
    PASSING_RETURN_CHECK_SECONDS: float = float(os.getenv("PASSING_RETURN_CHECK_SECONDS", "0.5"))

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

    # JPEG quality (0-100) for preview frames returned by
    # POST /videos/{id}/preview-frame - the select-a-player screen only
    # needs to look decent at phone-screen size, so this favors a smaller/
    # faster response over maximum fidelity.
    PREVIEW_FRAME_JPEG_QUALITY: int = int(os.getenv("PREVIEW_FRAME_JPEG_QUALITY", "80"))

    # Storage.
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/basketball_analyzer/uploads"))
    RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", "/tmp/basketball_analyzer/results"))

    # Max upload size in bytes (default 300 MB).
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(300 * 1024 * 1024)))

    CORS_ALLOW_ORIGINS: list[str] = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
