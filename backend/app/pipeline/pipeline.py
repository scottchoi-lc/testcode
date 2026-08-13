"""End-to-end analysis pipeline: video file in, `AnalysisResult` out.

Steps:
  1. Sample frames from the video at `settings.ANALYSIS_FPS`.
  1a. If the caller tapped a specific player in a preview frame
      (`selected_player_box`/`selected_timestamp`), tracking is seeded from
      that exact detection at the frame nearest that timestamp, and
      processed *bidirectionally* - forward to the end of the clip, then
      backward to the start - since the selection can be anywhere in the
      clip, not just the beginning. Otherwise frames are processed in the
      normal 0..N order, seeded by the default heuristic (largest person in
      frame 0).
  2. Run the HF object-detection model on every sampled frame to find the
     ball and players, and track a single "primary player" across frames.
  3. Run the HF pose model on the primary player's box each frame (best
     effort - skipped automatically if unavailable). When dribbling, also
     use wrist-to-ball proximity to call which hand is dribbling.
  4. Run the HF OCR model on a sparse sample of the primary player's torso
     to read a jersey number (also best effort; majority-voted across
     frames so a few bad reads don't win) - purely for display in the
     narrative, unrelated to player selection.
  5. Slide a *coarse* window (ACTION_WINDOW_FRAMES/STRIDE) over the sampled
     frames; run the HF video-classification model (VideoMAE/Kinetics) on
     the raw frames in each - this is the expensive step, so it stays
     relatively infrequent.
  6. Slide a separate, much *finer* window (FUSION_WINDOW_SECONDS) over the
     same per-frame ball/pose signals - cheap, since no model inference is
     needed here - and fuse each one via `app.pipeline.fusion.score_window`,
     borrowing whichever coarse window's kinetics label is temporally
     closest. This is what gives the narrative event-level granularity
     instead of one label per ~2.5s coarse window.
  7. Merge adjacent same-label fine windows into the final segment
     timeline, and generate a plain-English narrative from them.
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
    ball_wrist_distance: float | None = None,
) -> FrameSignals:
    if player is None:
        return FrameSignals(
            timestamp=frame.timestamp,
            player_center=None,
            ball_center=None,
            ball_player_distance=None,
            wrist_above_shoulder=None,
            dribbling_hand=None,
            ball_wrist_distance=None,
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
        ball_wrist_distance=ball_wrist_distance,
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


def _wrist_ball_signal(
    pose_result,
    ball_box: tuple[float, float, float, float] | None,
    scale: float,
    stats: dict | None = None,
) -> tuple[str | None, float | None]:
    """Returns (closest_hand, min_wrist_ball_distance).

    `closest_hand` names which wrist is nearest the ball this frame, but
    only when close enough to plausibly be controlling it - it's the signal
    behind ActionSegment.dominant_hand. `min_wrist_ball_distance` is
    returned whenever pose+ball data exist regardless of how far apart they
    are; it feeds `fusion.py`'s possession distance as a tighter alternative
    to ball-to-player-bbox-center distance (which under-detects possession
    during an extended-arm dribble/pass, where the ball sits away from the
    torso center even while a wrist has it in hand).

    ViTPose's "left"/"right" keypoint names follow the COCO convention: they
    identify the *subject's* own left/right hand (as an annotator looking at
    the subject would label it), not image-left/right - so `closest_hand` is
    correct regardless of which way the player is facing the camera.

    `stats`, if given, is mutated with counters for diagnosing *why* a clip
    isn't producing hand calls (no pose data vs. wrist-to-ball too far, etc).
    """
    if stats is not None:
        stats["frames_with_pose"] += pose_result is not None
        stats["frames_with_ball"] += ball_box is not None

    if pose_result is None or ball_box is None:
        return None, None
    left_wrist = pose_result.get("left_wrist")
    right_wrist = pose_result.get("right_wrist")
    if not left_wrist or not right_wrist:
        if stats is not None:
            stats["frames_missing_wrist_keypoints"] += 1
        return None, None

    from app.pipeline.fusion import BALL_POSSESSION_MAX_DIST

    ball_center = bbox_center(ball_box)
    left_dist = euclidean((left_wrist[0], left_wrist[1]), ball_center) / scale
    right_dist = euclidean((right_wrist[0], right_wrist[1]), ball_center) / scale
    closest = min(left_dist, right_dist)

    if stats is not None:
        stats["closest_dist_min"] = min(stats["closest_dist_min"], closest)

    hand = "left" if left_dist < right_dist else "right"
    if closest > BALL_POSSESSION_MAX_DIST:
        return None, closest
    return hand, closest


def _bidirectional_frame_order(num_frames: int, seed_index: int) -> tuple[list[int], list[int]]:
    """Frame processing order for tracking anchored at `seed_index`: forward
    from the seed to the end of the clip, then backward from just before the
    seed to the start. Returned as two separate lists (rather than one
    combined list) because tracking continuity resets to the seed position
    at the start of *each* direction - it's the one frame with a confirmed
    player position, so both directions walk outward from it independently
    rather than the backward pass continuing from wherever the forward pass
    ended up."""
    forward = list(range(seed_index, num_frames))
    backward = list(range(seed_index - 1, -1, -1))
    return forward, backward


def _nearest_kinetics_labels(
    kinetics_windows: list[tuple[float, float, list[tuple[str, float]]]], midpoint: float
) -> list[tuple[str, float]]:
    """Pick the coarse VideoMAE window whose time span is temporally closest
    to a fine fusion window's midpoint, and return its top labels. Lets many
    small fusion windows share one (expensive) kinetics inference call from
    whichever coarse window covers roughly the same moment."""
    if not kinetics_windows:
        return []
    best = min(kinetics_windows, key=lambda kw: abs((kw[0] + kw[1]) / 2 - midpoint))
    return best[2]


def run_pipeline(
    video_path: str,
    progress_cb: ProgressCallback | None = None,
    selected_player_box: tuple[float, float, float, float] | None = None,
    selected_timestamp: float | None = None,
) -> AnalysisResult:
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

    frame_signals: list[FrameSignals | None] = [None] * len(frames)
    hand_debug_stats = {
        "frames_with_player": 0,
        "frames_with_pose": 0,
        "frames_with_ball": 0,
        "frames_missing_wrist_keypoints": 0,
        "closest_dist_min": float("inf"),
    }
    hand_votes: dict[str, int] = {"left": 0, "right": 0}
    processed_count = 0

    def process_index(i: int, previous_center: tuple[float, float] | None) -> tuple[float, float] | None:
        frame = frames[i]
        balls, people = detection_model.detect_ball_and_players(frame.image)
        player = _pick_primary_player(people, previous_center)
        ball = _pick_ball(balls, bbox_center(player.box) if player else None)

        wrist_above_shoulder = None
        dribbling_hand = None
        ball_wrist_distance = None
        new_previous_center = previous_center
        if player is not None:
            hand_debug_stats["frames_with_player"] += 1
            new_previous_center = bbox_center(player.box)
            pose_result = pose_model.estimate(frame.image, player.box)
            wrist_above_shoulder = _wrist_above_shoulder(pose_result)
            dribbling_hand, ball_wrist_distance = _wrist_ball_signal(
                pose_result, ball.box if ball else None, bbox_diag(player.box) or 1.0, hand_debug_stats
            )
            if dribbling_hand:
                hand_votes[dribbling_hand] += 1

            if i % jersey_ocr_stride == 0:
                jersey_votes.add(jersey_reader.read_crop(frame.image, player.box))

        frame_signals[i] = _build_frame_signals(
            frame, player, ball, wrist_above_shoulder, dribbling_hand, ball_wrist_distance
        )
        return new_previous_center

    def report_frame_progress() -> None:
        nonlocal processed_count
        processed_count += 1
        report(0.05 + 0.55 * processed_count / len(frames))

    player_selected = selected_player_box is not None and selected_timestamp is not None
    if player_selected:
        seed_index = max(0, min(len(frames) - 1, round(selected_timestamp * settings.ANALYSIS_FPS)))
        forward_order, backward_order = _bidirectional_frame_order(len(frames), seed_index)
        seed_center = bbox_center(selected_player_box)

        previous_center = seed_center
        for i in forward_order:
            previous_center = process_index(i, previous_center)
            report_frame_progress()

        previous_center = seed_center
        for i in backward_order:
            previous_center = process_index(i, previous_center)
            report_frame_progress()
    else:
        previous_center = None
        for i in range(len(frames)):
            previous_center = process_index(i, previous_center)
            report_frame_progress()

    logger.info(
        "Player selection: %s. Dribbling-hand debug: %d frames total, %s, votes=%s",
        "tapped player" if player_selected else "default heuristic",
        len(frames),
        hand_debug_stats,
        hand_votes,
    )

    player_number = jersey_votes.best_guess()
    logger.info("Jersey OCR result: guess=%r (%s)", player_number, jersey_votes.debug_summary())

    action_classifier = get_action_classifier()
    kinetics_window_size = settings.ACTION_WINDOW_FRAMES
    kinetics_stride = settings.ACTION_WINDOW_STRIDE

    kinetics_windows: list[tuple[float, float, list[tuple[str, float]]]] = []
    kinetics_starts = list(range(0, max(1, len(frames) - 1), kinetics_stride)) or [0]
    for wi, start_idx in enumerate(kinetics_starts):
        end_idx = min(len(frames), start_idx + kinetics_window_size)
        if end_idx - start_idx < 2:
            continue
        window_frames = frames[start_idx:end_idx]
        prediction = action_classifier.classify_window([f.image for f in window_frames])
        kinetics_windows.append(
            (window_frames[0].timestamp, window_frames[-1].timestamp, prediction.top_labels)
        )
        report(0.6 + 0.25 * (wi + 1) / len(kinetics_starts))

    fusion_window_frames = max(3, round(settings.FUSION_WINDOW_SECONDS * settings.ANALYSIS_FPS))
    scored_windows = []
    fusion_starts = list(range(0, len(frames), fusion_window_frames)) or [0]
    for fi, start_idx in enumerate(fusion_starts):
        end_idx = min(len(frames), start_idx + fusion_window_frames)
        if end_idx - start_idx < 2:
            continue
        window_frames = frames[start_idx:end_idx]
        window_frame_signals = frame_signals[start_idx:end_idx]
        midpoint = (window_frames[0].timestamp + window_frames[-1].timestamp) / 2

        window = WindowSignals(
            start_time=window_frames[0].timestamp,
            end_time=window_frames[-1].timestamp,
            frames=window_frame_signals,
            kinetics_top_labels=_nearest_kinetics_labels(kinetics_windows, midpoint),
        )
        scored = score_window(window)
        scored_windows.append((window, scored))
        logger.info(
            "Window %.1f-%.1f: label=%s conf=%.2f evidence=%s",
            window.start_time,
            window.end_time,
            scored.label.value,
            scored.confidence,
            scored.evidence,
        )
        report(0.85 + 0.15 * (fi + 1) / len(fusion_starts))

    segments = merge_adjacent_segments(scored_windows)
    logger.info(
        "Segments: %s",
        [
            (s.label.value, round(s.start_time, 1), round(s.end_time, 1), s.dominant_hand)
            for s in segments
        ],
    )

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
        player_selected=player_selected,
    )
