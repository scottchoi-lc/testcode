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
     frames; run the HF video-text model (X-CLIP) on the raw frames in each,
     scoring them against `settings.ACTION_CANDIDATE_LABELS` - this is the
     expensive step, so it stays relatively infrequent. (Code/variable names
     below still say "kinetics" - a holdover from when this was a
     Kinetics-400 classifier; see `app/models/action_classifier.py` for why
     that was swapped out and what "kinetics_top_labels" actually means now.)
  6. Slide a separate, much *finer* window (FUSION_WINDOW_SECONDS) over the
     same per-frame ball/pose signals - cheap, since no model inference is
     needed here - and fuse each one via `app.pipeline.fusion.score_window`,
     borrowing whichever coarse window's label scores are temporally
     closest. This is what gives the narrative event-level granularity
     instead of one label per ~2.5s coarse window.
  7. Merge adjacent same-label fine windows into the final segment
     timeline, and generate a plain-English narrative from them.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from app.config import settings
from app.models.action_classifier import get_action_classifier
from app.models.detection import Detection, get_detection_model
from app.models.jersey_ocr import JerseyNumberAggregator, get_jersey_number_reader
from app.models.pose import get_pose_model
from app.pipeline.fusion import (
    BALL_POSSESSION_MAX_DIST,
    FrameSignals,
    ScoredLabel,
    WindowSignals,
    _effective_ball_distance,
    merge_adjacent_segments,
    score_window,
)
from app.pipeline.narration import narrate
from app.pipeline.video_utils import Frame, bbox_center, bbox_diag, euclidean, extract_frames
from app.schemas import ActionLabel, AnalysisResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


# Maximum plausible displacement (in the *candidate* detection's own
# bbox-diagonal units) between one analyzed frame and the next for this to
# plausibly still be the same physical player. Real motion between samples
# a fraction of a second apart rarely covers more than one or two body
# lengths; a real clip logged a 6.9-diagonal "jump" in a single window,
# which is a different person, not continued tracking - accepting it
# meant everything downstream (which action, which hand, the narrative
# order) reflected the wrong player from that point on. Treating an
# implausibly-far "nearest" match as "no detection this frame" instead
# (previous_center stays frozen at the last good position rather than
# snapping to someone else) is a deliberately generous cap, not a tight
# one - tune down if wrong-person jumps still slip through.
MAX_PLAUSIBLE_TRACKING_JUMP = 3.0


# Minimum appearance similarity (cv2.HISTCMP_CORREL, roughly -1..1) a
# candidate must reach to be trusted as the same person once a reference
# appearance exists. This is checked on *every* frame, not just when
# multiple detections are ambiguously close by position - a single wrong-
# person detection near the last known position would otherwise sail
# through untested, since there was nothing to disambiguate it *against*.
# That mattered in practice: once tracking drifted, later frames usually
# only had one nearby candidate (the wrong person), so a tie-breaker that
# only activates when there's a tie never got a chance to catch it.
MIN_APPEARANCE_SIMILARITY = 0.3


def _color_histogram(image_bgr: np.ndarray, box: tuple[float, float, float, float]):
    """A cheap appearance signature for a person crop (HSV color histogram).

    Compared against a reference signature (captured once, from whichever
    frame first successfully tracks the player - the exact tapped frame
    when a player was selected) to verify continuity every frame, not just
    to break ties among multiple candidates - see MIN_APPEARANCE_SIMILARITY.
    Not a general re-identification model and not robust to major lighting/
    angle changes across a whole clip, or to two players in matching
    uniforms.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    height, width = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _appearance_similarity(reference, candidate) -> float:
    """Higher is more similar. 0.0 (treated as "no signal") if either
    histogram is unavailable, rather than favoring or penalizing a
    candidate we simply couldn't compute a signature for."""
    if reference is None or candidate is None:
        return 0.0
    return float(cv2.compareHist(reference, candidate, cv2.HISTCMP_CORREL))


def _pick_primary_player(
    people: list[Detection],
    previous_center: tuple[float, float] | None,
    image_bgr: np.ndarray | None = None,
    reference_appearance=None,
    stats: dict | None = None,
) -> Detection | None:
    if not people:
        return None
    if previous_center is None:
        return max(people, key=lambda d: bbox_diag(d.box))

    def jump_for(d: Detection) -> float:
        scale = bbox_diag(d.box) or 1.0
        return euclidean(bbox_center(d.box), previous_center) / scale

    plausible = [d for d in people if jump_for(d) <= MAX_PLAUSIBLE_TRACKING_JUMP]
    if not plausible:
        if stats is not None:
            nearest = min(people, key=jump_for)
            stats["implausible_jumps_rejected"] += 1
            stats["max_jump_seen"] = max(stats["max_jump_seen"], jump_for(nearest))
        return None

    if image_bgr is None or reference_appearance is None:
        return min(plausible, key=jump_for)

    # A reference appearance exists (either from a tapped selection or the
    # default heuristic's first pick) - verify every plausible candidate
    # against it, not just when there's more than one to choose between.
    scored = [
        (d, _appearance_similarity(reference_appearance, _color_histogram(image_bgr, d.box)))
        for d in plausible
    ]
    best, best_score = max(scored, key=lambda item: item[1])
    if best_score < MIN_APPEARANCE_SIMILARITY:
        if stats is not None:
            stats["appearance_mismatches_rejected"] += 1
        return None
    if len(plausible) > 1 and stats is not None:
        stats["ambiguous_frames_disambiguated_by_appearance"] += 1
    return best


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
    """Pick the coarse X-CLIP window whose time span is temporally closest
    to a fine fusion window's midpoint, and return its top labels. Lets many
    small fusion windows share one (expensive) classifier inference call from
    whichever coarse window covers roughly the same moment."""
    if not kinetics_windows:
        return []
    best = min(kinetics_windows, key=lambda kw: abs((kw[0] + kw[1]) / 2 - midpoint))
    return best[2]


def _fusion_window_bounds(num_frames: int, window_frames: int, stride_frames: int) -> list[tuple[int, int]]:
    """Start/end frame indices for the fine fusion-scoring windows.

    `stride_frames < window_frames` makes windows overlap, which matters for
    catching a brief release (pass or shot) that happens to land right at
    what would otherwise be a hard tile boundary. A real clip's logs showed
    exactly this: one non-overlapping window still had the ball close
    (fraction_possessed=0.6), the very next had it gone entirely - the
    release straddled the boundary, so neither window ever saw a "starts
    with the ball, ends without it" pattern, and score_window's
    shooting/passing branches (which both require that pattern within a
    single window) never got a chance to fire. Overlap guarantees any such
    transition is fully contained in at least one window."""
    starts = list(range(0, max(1, num_frames - 1), stride_frames)) or [0]
    bounds = []
    for start in starts:
        end = min(num_frames, start + window_frames)
        if end - start >= 2:
            bounds.append((start, end))
    return bounds


def _interpolate_ball_gaps(
    frame_signals: list[FrameSignals | None], max_gap_frames: int
) -> tuple[list[FrameSignals | None], int, int]:
    """Linearly interpolate ball position/distance across short runs of
    consecutive frames where the player was tracked but the ball wasn't
    detected, bounded by `max_gap_frames` on each side. Returns the filled
    list, how many frames were interpolated, and the longest such gap seen
    in the clip regardless of whether it was short enough to fill (for
    logging - `longest_gap_frames` is what tells you whether
    `max_gap_frames` needs raising, versus `filled` alone which stays 0
    whether the real gap is 1 frame too long or 20).

    Only fills a gap when there's a real ball reading immediately before
    *and* after it (so leading/trailing gaps, or gaps longer than the
    bound, are left as-is rather than fabricated) and only when the player
    was tracked the whole way through the gap (so a genuine tracking loss
    isn't papered over with an invented ball position for a player we
    don't actually know the location of)."""
    n = len(frame_signals)
    result = list(frame_signals)
    filled = 0
    longest_gap_frames = 0
    i = 0
    while i < n:
        f = frame_signals[i]
        if f is not None and f.player_center is not None and f.ball_center is None:
            start = i
            while (
                i < n
                and frame_signals[i] is not None
                and frame_signals[i].player_center is not None
                and frame_signals[i].ball_center is None
            ):
                i += 1
            end = i
            gap_len = end - start
            longest_gap_frames = max(longest_gap_frames, gap_len)
            before = frame_signals[start - 1] if start > 0 else None
            after = frame_signals[end] if end < n else None
            if (
                gap_len <= max_gap_frames
                and before is not None
                and after is not None
                and before.ball_center is not None
                and after.ball_center is not None
            ):
                for k in range(gap_len):
                    t = (k + 1) / (gap_len + 1)
                    idx = start + k
                    orig = frame_signals[idx]
                    ball_center = (
                        before.ball_center[0] + (after.ball_center[0] - before.ball_center[0]) * t,
                        before.ball_center[1] + (after.ball_center[1] - before.ball_center[1]) * t,
                    )
                    ball_wrist_distance = None
                    if before.ball_wrist_distance is not None and after.ball_wrist_distance is not None:
                        ball_wrist_distance = (
                            before.ball_wrist_distance
                            + (after.ball_wrist_distance - before.ball_wrist_distance) * t
                        )
                    result[idx] = replace(
                        orig,
                        ball_center=ball_center,
                        ball_player_distance=euclidean(orig.player_center, ball_center),
                        ball_wrist_distance=ball_wrist_distance,
                    )
                    filled += 1
        else:
            i += 1
    return result, filled, longest_gap_frames


def _ball_returns_to_possession_soon(
    frame_signals: list[FrameSignals | None], end_idx: int, lookahead_frames: int
) -> bool:
    """Whether the ball is back within BALL_POSSESSION_MAX_DIST of the
    tracked player in any of the `lookahead_frames` frames right after
    `end_idx`. Distinguishes a genuine release (pass or shot - the ball
    stays away, now with someone else or in the air) from a retained-
    possession move like a crossover dribble, where the ball swings wide of
    the player's bbox center and then straight back into close range a
    fraction of a second later, having never actually left that player's
    hand."""
    for f in frame_signals[end_idx : end_idx + lookahead_frames]:
        if f is None:
            continue
        distance = _effective_ball_distance(f)
        if distance is not None and distance <= BALL_POSSESSION_MAX_DIST:
            return True
    return False


def run_pipeline(
    video_path: str,
    progress_cb: ProgressCallback | None = None,
    selected_player_box: tuple[float, float, float, float] | None = None,
    selected_timestamp: float | None = None,
    debug_scored_windows: list[tuple[WindowSignals, ScoredLabel]] | None = None,
) -> AnalysisResult:
    """See module docstring for the pipeline steps. `debug_scored_windows`,
    if passed a list, gets extended in place with every (WindowSignals,
    ScoredLabel) pair scored during the run - the raw per-window evidence
    and prediction that `segments` (the merged, narrated output) is built
    from. Exists so tooling (e.g. comparing predictions against hand-
    labeled ground truth) can reuse the exact production pipeline - same
    tracking, same windowing, same scoring - without duplicating any of it,
    rather than reading it out of log lines."""

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
    tracking_debug_stats = {
        "implausible_jumps_rejected": 0,
        "max_jump_seen": 0.0,
        "ambiguous_frames_disambiguated_by_appearance": 0,
        "appearance_mismatches_rejected": 0,
    }
    # Appearance signature of the tracked player, established from whichever
    # frame first successfully picks one (the seed frame itself when a
    # player was selected, since forward_order starts there; otherwise
    # frame 0's largest-bbox pick) - see _pick_primary_player.
    tracking_state = {"reference_appearance": None}
    processed_count = 0

    def process_index(i: int, previous_center: tuple[float, float] | None) -> tuple[float, float] | None:
        frame = frames[i]
        balls, people = detection_model.detect_ball_and_players(frame.image)
        player = _pick_primary_player(
            people,
            previous_center,
            frame.image,
            tracking_state["reference_appearance"],
            tracking_debug_stats,
        )
        ball = _pick_ball(balls, bbox_center(player.box) if player else None)

        wrist_above_shoulder = None
        dribbling_hand = None
        ball_wrist_distance = None
        new_previous_center = previous_center
        if player is not None:
            hand_debug_stats["frames_with_player"] += 1
            new_previous_center = bbox_center(player.box)
            if tracking_state["reference_appearance"] is None:
                tracking_state["reference_appearance"] = _color_histogram(frame.image, player.box)
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
        "Player selection: %s. Tracking debug: %s. Dribbling-hand debug: %d frames total, %s, votes=%s",
        "tapped player" if player_selected else "default heuristic",
        tracking_debug_stats,
        len(frames),
        hand_debug_stats,
        hand_votes,
    )

    frame_signals, ball_gap_frames_filled, longest_ball_gap_frames = _interpolate_ball_gaps(
        frame_signals, settings.BALL_GAP_INTERPOLATION_MAX_FRAMES
    )
    logger.info(
        "Ball gap interpolation: %d frame(s) filled (max_gap=%d), longest gap seen=%d frames",
        ball_gap_frames_filled,
        settings.BALL_GAP_INTERPOLATION_MAX_FRAMES,
        longest_ball_gap_frames,
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
    fusion_stride_frames = max(1, round(settings.FUSION_WINDOW_STRIDE_SECONDS * settings.ANALYSIS_FPS))
    passing_lookahead_frames = max(1, round(settings.PASSING_RETURN_CHECK_SECONDS * settings.ANALYSIS_FPS))
    scored_windows = []
    fusion_bounds = _fusion_window_bounds(len(frames), fusion_window_frames, fusion_stride_frames)
    for fi, (start_idx, end_idx) in enumerate(fusion_bounds):
        window_frames = frames[start_idx:end_idx]
        window_frame_signals = frame_signals[start_idx:end_idx]
        midpoint = (window_frames[0].timestamp + window_frames[-1].timestamp) / 2

        window = WindowSignals(
            start_time=window_frames[0].timestamp,
            end_time=window_frames[-1].timestamp,
            frames=window_frame_signals,
            kinetics_top_labels=_nearest_kinetics_labels(kinetics_windows, midpoint),
            ball_returns_to_possession_soon=_ball_returns_to_possession_soon(
                frame_signals, end_idx, passing_lookahead_frames
            ),
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
        report(0.85 + 0.15 * (fi + 1) / len(fusion_bounds))

    if debug_scored_windows is not None:
        debug_scored_windows.extend(scored_windows)

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
