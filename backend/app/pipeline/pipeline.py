"""End-to-end analysis pipeline: video file in, `AnalysisResult` out.

Steps:
  1. Sample frames from the video at `settings.ANALYSIS_FPS`.
  2. Run the HF object-detection model on every sampled frame to find the
     ball and players, and track a single "primary player" across frames.
  3. Run the HF pose model on the primary player's box each frame (best
     effort - skipped automatically if unavailable). When dribbling, also
     use wrist-to-ball proximity to call which hand is dribbling.
  4. Run the HF OCR model on a sparse sample of the primary player's torso
     to read a jersey number (also best effort; majority-voted across
     frames so a few bad reads don't win).
  5. Slide a window over the sampled frames; run the HF video-classification
     model (VideoMAE/Kinetics) on the raw frames in each window.
  6. Fuse all of the above per window via `app.pipeline.fusion.score_window`.
  7. Merge adjacent same-label windows into the final segment timeline, and
     generate a plain-English narrative from them.
"""
from __future__ import annotations

import logging
from typing import Callable

from app.config import settings
from app.models.action_classifier import get_action_classifier
from app.models.detection import Detection, get_detection_model
from app.models.jersey_ocr import JerseyNumberAggregator, get_jersey_number_reader
from app.models.pose import get_pose_model
from app.pipeline.fusion import FrameSignals, WindowSignals, merge_adjacent_segments, score_window
from app.pipeline.narration import narrate
from app.pipeline.video_utils import Frame, bbox_center, bbox_diag, euclidean, extract_frames
from app.schemas import ActionLabel, AnalysisResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def _pick_primary_player(
    people: list[Detection], previous_center: tuple[float, float] | None
) -> Detection | None:
    if not people:
        return None
    if previous_center is None:
        return max(people, key=lambda d: bbox_diag(d.box))
    return min(people, key=lambda d: euclidean(bbox_center(d.box), previous_center))


def _pick_ball(balls: list[Detection], player_center: tuple[float, float] | None) -> Detection | None:
    if not balls:
        return None
    if player_center is None:
        return max(balls, key=lambda d: d.score)
    return min(balls, key=lambda d: euclidean(bbox_center(d.box), player_center))


def _build_frame_signals(
    frame: Frame,
    player: Detection | None,
    ball: Detection | None,
    wrist_above_shoulder: bool | None,
    dribbling_hand: str | None = None,
) -> FrameSignals:
    if player is None:
        return FrameSignals(
            timestamp=frame.timestamp,
            player_center=None,
            ball_center=None,
            ball_player_distance=None,
            wrist_above_shoulder=None,
            dribbling_hand=None,
        )

    scale = bbox_diag(player.box) or 1.0
    player_center = bbox_center(player.box)
    player_center_norm = (player_center[0] / scale, player_center[1] / scale)

    ball_center_norm = None
    distance_norm = None
    if ball is not None:
        ball_center = bbox_center(ball.box)
        ball_center_norm = (ball_center[0] / scale, ball_center[1] / scale)
        distance_norm = euclidean(player_center, ball_center) / scale

    return FrameSignals(
        timestamp=frame.timestamp,
        player_center=player_center_norm,
        ball_center=ball_center_norm,
        ball_player_distance=distance_norm,
        wrist_above_shoulder=wrist_above_shoulder,
        dribbling_hand=dribbling_hand,
    )


def _wrist_above_shoulder(pose_result) -> bool | None:
    if pose_result is None:
        return None
    left_wrist = pose_result.get("left_wrist")
    right_wrist = pose_result.get("right_wrist")
    left_shoulder = pose_result.get("left_shoulder")
    right_shoulder = pose_result.get("right_shoulder")
    if not (left_wrist and right_wrist and left_shoulder and right_shoulder):
        return None
    from app.pipeline.fusion import WRIST_ABOVE_SHOULDER_MARGIN

    left_high = left_wrist[1] < left_shoulder[1] - WRIST_ABOVE_SHOULDER_MARGIN
    right_high = right_wrist[1] < right_shoulder[1] - WRIST_ABOVE_SHOULDER_MARGIN
    return bool(left_high or right_high)


def _dribbling_hand(
    pose_result, ball_box: tuple[float, float, float, float] | None, scale: float
) -> str | None:
    """Which wrist is nearest the ball this frame, if close enough to plausibly
    be controlling it. None if pose/ball data is missing or neither wrist is close.

    ViTPose's "left"/"right" keypoint names follow the COCO convention: they
    identify the *subject's* own left/right hand (as an annotator looking at
    the subject would label it), not image-left/right - so this is correct
    regardless of which way the player is facing the camera.
    """
    if pose_result is None or ball_box is None:
        return None
    left_wrist = pose_result.get("left_wrist")
    right_wrist = pose_result.get("right_wrist")
    if not left_wrist or not right_wrist:
        return None

    from app.pipeline.fusion import DRIBBLE_HAND_MAX_WRIST_BALL_DIST

    ball_center = bbox_center(ball_box)
    left_dist = euclidean((left_wrist[0], left_wrist[1]), ball_center) / scale
    right_dist = euclidean((right_wrist[0], right_wrist[1]), ball_center) / scale
    if min(left_dist, right_dist) > DRIBBLE_HAND_MAX_WRIST_BALL_DIST:
        return None
    return "left" if left_dist < right_dist else "right"


def run_pipeline(video_path: str, progress_cb: ProgressCallback | None = None) -> AnalysisResult:
    def report(fraction: float) -> None:
        if progress_cb:
            progress_cb(min(1.0, max(0.0, fraction)))

    frames, duration = extract_frames(video_path, settings.ANALYSIS_FPS)
    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video")

    report(0.05)

    detection_model = get_detection_model()
    pose_model = get_pose_model()
    jersey_reader = get_jersey_number_reader()
    jersey_votes = JerseyNumberAggregator()
    jersey_ocr_stride = max(1, len(frames) // settings.JERSEY_OCR_MAX_SAMPLES)

    frame_signals: list[FrameSignals] = []
    previous_center: tuple[float, float] | None = None

    for i, frame in enumerate(frames):
        balls, people = detection_model.detect_ball_and_players(frame.image)
        player = _pick_primary_player(people, previous_center)
        ball = _pick_ball(balls, bbox_center(player.box) if player else None)

        wrist_above_shoulder = None
        dribbling_hand = None
        if player is not None:
            previous_center = bbox_center(player.box)
            pose_result = pose_model.estimate(frame.image, player.box)
            wrist_above_shoulder = _wrist_above_shoulder(pose_result)
            dribbling_hand = _dribbling_hand(
                pose_result, ball.box if ball else None, bbox_diag(player.box) or 1.0
            )

            if i % jersey_ocr_stride == 0:
                jersey_votes.add(jersey_reader.read_crop(frame.image, player.box))

        frame_signals.append(
            _build_frame_signals(frame, player, ball, wrist_above_shoulder, dribbling_hand)
        )
        report(0.05 + 0.55 * (i + 1) / len(frames))

    player_number = jersey_votes.best_guess()
    logger.info("Jersey OCR result: guess=%r (%s)", player_number, jersey_votes.debug_summary())

    action_classifier = get_action_classifier()
    window_size = settings.ACTION_WINDOW_FRAMES
    stride = settings.ACTION_WINDOW_STRIDE

    scored_windows = []
    window_starts = list(range(0, max(1, len(frames) - 1), stride)) or [0]
    for wi, start_idx in enumerate(window_starts):
        end_idx = min(len(frames), start_idx + window_size)
        if end_idx - start_idx < 2:
            continue
        window_frames = frames[start_idx:end_idx]
        window_frame_signals = frame_signals[start_idx:end_idx]

        prediction = action_classifier.classify_window([f.image for f in window_frames])

        window = WindowSignals(
            start_time=window_frames[0].timestamp,
            end_time=window_frames[-1].timestamp,
            frames=window_frame_signals,
            kinetics_top_labels=prediction.top_labels,
        )
        scored = score_window(window)
        scored_windows.append((window, scored))
        report(0.6 + 0.4 * (wi + 1) / len(window_starts))

    segments = merge_adjacent_segments(scored_windows)

    summary: dict[str, float] = {label.value: 0.0 for label in ActionLabel}
    for segment in segments:
        summary[segment.label.value] += segment.end_time - segment.start_time

    report(1.0)

    return AnalysisResult(
        duration_seconds=duration,
        fps_analyzed=settings.ANALYSIS_FPS,
        segments=segments,
        summary=summary,
        narrative=narrate(segments, summary, player_number=player_number),
        detected_player_number=player_number,
    )
