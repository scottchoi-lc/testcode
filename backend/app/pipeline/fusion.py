"""Rule-based fusion of per-frame signals into basketball action segments.

No single Hugging Face model classifies exactly "dribbling / shooting /
passing / moving without the ball" for basketball, so this module is the
domain-specific glue: it combines

  * ball <-> player proximity and ball trajectory (from `DetectionModel`)
  * wrist/shoulder motion (from `PoseModel`, optional)
  * a coarse activity label from `ActionClassifier` (VideoMAE / Kinetics)

into one label per analysis window. Everything here is plain, deterministic
Python over small numeric structures - no ML inference happens in this
module - which keeps it fast to unit test without downloading any models.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from app.schemas import ActionLabel

# Kinetics-400 label -> which of our buckets it supports, and how much it
# should nudge the heuristic score when present in the classifier's top-k.
KINETICS_DRIBBLE_LABELS = {"dribbling basketball"}
KINETICS_SHOOT_LABELS = {"shooting basketball", "dunking basketball", "shooting goal (soccer)"}
KINETICS_GENERIC_BASKETBALL_LABELS = {
    "dribbling basketball",
    "shooting basketball",
    "dunking basketball",
    "playing basketball",
    "playing kickball",
}

# Tunable thresholds for the heuristics below. Distances are normalized by
# the player's bounding-box diagonal so they're roughly scale-invariant.
BALL_POSSESSION_MAX_DIST = 0.9
BALL_RELEASE_MIN_DIST = 1.6
DRIBBLE_VERTICAL_STD_MAX = 0.35
WRIST_ABOVE_SHOULDER_MARGIN = 0.02

# Player displacement needed to call a window "moving", expressed as a rate
# (player-bbox-diagonals per second) rather than a flat per-window amount,
# since analysis window duration is a tunable (FUSION_WINDOW_SECONDS) and a
# flat threshold would silently need re-tuning any time that changes. 0.2/s
# reproduces the same idle-vs-moving calls a flat 0.5-per-~2.5s threshold
# made on real logged windows (e.g. 0.34/2.5s=0.14/s stayed idle, 0.53/2.5s
# =0.21/s was moving), so it's a like-for-like conversion, not a new guess.
PLAYER_MOVE_MIN_DISPLACEMENT_RATE = 0.2

# Minimum fraction of ball-detected frames within a window that must read as
# "possessed" (<= BALL_POSSESSION_MAX_DIST) to call it dribbling. Lower than
# it might look: real dribbling has the ball in flight, away from the hand,
# for most of each bounce cycle - only briefly close at the top of the
# bounce - so requiring a majority of frames to show "close" systematically
# under-detects genuine dribbling. Tuned against one real clip (the only
# window with any real ball-proximity signal measured fraction_possessed=0.3,
# well under a naive 0.6 majority bar) rather than a validated dataset, so
# revisit if it starts producing false-positive dribbling calls elsewhere.
DRIBBLE_POSSESSION_MIN_FRACTION = 0.3

# Minimum Kinetics softmax score for "dribbling basketball"/"shooting
# basketball" to count as corroborating evidence that can substitute for a
# missing direct-signal check (vertical std / wrist height / release). A
# bare `> 0` here is a bug, not a threshold: VideoMAE's softmax spreads a
# little probability mass over most of its 400 classes, so an unrelated
# window can show e.g. kinetics_shoot_score=0.088 with zero real shooting
# evidence (fraction_wrist_high=0.0, ball_released=False) and still get
# labeled SHOOTING purely from that noise floor - this was caught from a
# real clip's logs where exactly that happened. 0.3 matches the existing
# fraction_wrist_high >= 0.3 bar so a Kinetics-only override needs to be as
# convincing as the direct-evidence threshold it's standing in for.
KINETICS_OVERRIDE_MIN = 0.3


@dataclass
class FrameSignals:
    """Per-frame signals for the single player being tracked, already
    normalized: distances/displacements are in units of that frame's player
    bbox diagonal; positions are (x, y) in the same normalized frame."""

    timestamp: float
    player_center: tuple[float, float] | None
    ball_center: tuple[float, float] | None
    ball_player_distance: float | None  # ball <-> player-bbox-center distance; None if either is missing
    wrist_above_shoulder: bool | None  # None if pose unavailable
    dribbling_hand: str | None = None  # "left" | "right" | None; which wrist is nearest the ball
    ball_wrist_distance: float | None = None  # ball <-> nearest-wrist distance; None if pose/ball unavailable


@dataclass
class WindowSignals:
    start_time: float
    end_time: float
    frames: list[FrameSignals]
    kinetics_top_labels: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ScoredLabel:
    label: ActionLabel
    confidence: float
    evidence: dict


def _kinetics_boost(labels: set[str], top_labels: list[tuple[str, float]]) -> float:
    for name, score in top_labels:
        if name in labels:
            return score
    return 0.0

def _has_generic_basketball_signal(top_labels: list[tuple[str, float]]) -> float:
    return _kinetics_boost(KINETICS_GENERIC_BASKETBALL_LABELS, top_labels)


def _effective_ball_distance(f: FrameSignals) -> float | None:
    """Best available possession-distance signal for a frame: the closer of
    ball-to-player-bbox-center and ball-to-nearest-wrist, when both exist.

    Bbox-center distance alone under-detects possession for an extended-arm
    dribble/pass, where the ball sits well away from the torso center even
    while a wrist has it in hand - wrist distance is a tighter, more direct
    signal when pose data is available, so we take whichever is smaller
    rather than relying on bbox-center distance alone.
    """
    candidates = [d for d in (f.ball_player_distance, f.ball_wrist_distance) if d is not None]
    return min(candidates) if candidates else None


def score_window(window: WindowSignals) -> ScoredLabel:
    """Classify a single analysis window into one of the four action labels
    (or IDLE when there isn't enough evidence for any of them)."""
    frames = window.frames
    if not frames:
        return ScoredLabel(ActionLabel.IDLE, 0.0, {"reason": "no_frames"})

    distances = [d for d in (_effective_ball_distance(f) for f in frames) if d is not None]
    fraction_possessed = (
        sum(1 for d in distances if d <= BALL_POSSESSION_MAX_DIST) / len(distances)
        if distances
        else 0.0
    )

    ball_ys = [f.ball_center[1] for f in frames if f.ball_center is not None]
    ball_xs = [f.ball_center[0] for f in frames if f.ball_center is not None]
    player_centers = [f.player_center for f in frames if f.player_center is not None]

    wrist_flags = [f.wrist_above_shoulder for f in frames if f.wrist_above_shoulder is not None]
    fraction_wrist_high = sum(1 for w in wrist_flags if w) / len(wrist_flags) if wrist_flags else 0.0

    player_displacement = 0.0
    if len(player_centers) >= 2:
        (x0, y0), (x1, y1) = player_centers[0], player_centers[-1]
        player_displacement = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    window_duration = frames[-1].timestamp - frames[0].timestamp
    player_displacement_rate = player_displacement / window_duration if window_duration > 0 else 0.0

    kinetics_labels = window.kinetics_top_labels
    dribble_boost = _kinetics_boost(KINETICS_DRIBBLE_LABELS, kinetics_labels)
    shoot_boost = _kinetics_boost(KINETICS_SHOOT_LABELS, kinetics_labels)
    generic_basketball = _has_generic_basketball_signal(kinetics_labels)

    candidates: list[ScoredLabel] = []

    # --- Dribbling: ball stays close to the player most of the window and
    # bounces rhythmically (low-to-moderate vertical variance, not a single
    # big upward launch). ---
    if fraction_possessed >= DRIBBLE_POSSESSION_MIN_FRACTION and len(ball_ys) >= 3:
        vertical_std = statistics.pstdev(ball_ys)
        if vertical_std <= DRIBBLE_VERTICAL_STD_MAX or dribble_boost >= KINETICS_OVERRIDE_MIN:
            confidence = min(1.0, 0.45 + 0.3 * fraction_possessed + 0.25 * dribble_boost)
            candidates.append(
                ScoredLabel(
                    ActionLabel.DRIBBLING,
                    confidence,
                    {
                        "fraction_possessed": fraction_possessed,
                        "ball_vertical_std": vertical_std,
                        "kinetics_dribble_score": dribble_boost,
                    },
                )
            )

    # --- Shooting: player starts with the ball, wrist rises above the
    # shoulder, and the ball ends up released well away from the player
    # (moving up/away rather than staying put). ---
    if distances and distances[0] <= BALL_POSSESSION_MAX_DIST:
        released = distances[-1] >= BALL_RELEASE_MIN_DIST if len(distances) > 1 else False
        if (fraction_wrist_high >= 0.3 or shoot_boost >= KINETICS_OVERRIDE_MIN) and (
            released or shoot_boost >= KINETICS_OVERRIDE_MIN
        ):
            confidence = min(
                1.0, 0.4 + 0.3 * fraction_wrist_high + 0.3 * shoot_boost + (0.15 if released else 0)
            )
            candidates.append(
                ScoredLabel(
                    ActionLabel.SHOOTING,
                    confidence,
                    {
                        "fraction_wrist_high": fraction_wrist_high,
                        "ball_released": released,
                        "kinetics_shoot_score": shoot_boost,
                        # Raw trail behind `released`, which currently reads
                        # only the single last detected-ball distance - kept
                        # here so a "released" call driven by one noisy/
                        # misdetected frame is visible in logs rather than
                        # indistinguishable from a real sustained release.
                        "ball_distances": [round(d, 2) for d in distances],
                    },
                )
            )

    # --- Passing: player starts with the ball, it leaves possession quickly
    # (released) but *without* the wrist-above-shoulder shooting motion and
    # without a strong shooting signal from the action classifier. Ball
    # travels laterally more than vertically. ---
    if distances and distances[0] <= BALL_POSSESSION_MAX_DIST and len(distances) > 1:
        released = distances[-1] >= BALL_RELEASE_MIN_DIST
        lateral_move = abs(ball_xs[-1] - ball_xs[0]) if len(ball_xs) > 1 else 0.0
        vertical_move = abs(ball_ys[-1] - ball_ys[0]) if len(ball_ys) > 1 else 0.0
        looks_like_pass = released and fraction_wrist_high < 0.3 and shoot_boost < 0.15
        if looks_like_pass and lateral_move >= vertical_move:
            confidence = min(1.0, 0.5 + 0.3 * min(1.0, lateral_move) - 0.2 * shoot_boost)
            candidates.append(
                ScoredLabel(
                    ActionLabel.PASSING,
                    max(0.0, confidence),
                    {
                        "lateral_move": lateral_move,
                        "vertical_move": vertical_move,
                        "fraction_wrist_high": fraction_wrist_high,
                        "ball_distances": [round(d, 2) for d in distances],
                    },
                )
            )

    # --- Moving without the ball: the player is not in possession for
    # (almost) the whole window, yet is clearly displacing on the court. ---
    if fraction_possessed <= 0.15 and player_displacement_rate >= PLAYER_MOVE_MIN_DISPLACEMENT_RATE:
        confidence = min(1.0, 0.4 + 0.4 * min(1.0, player_displacement_rate))
        candidates.append(
            ScoredLabel(
                ActionLabel.MOVING_WITHOUT_BALL,
                confidence,
                {
                    "fraction_possessed": fraction_possessed,
                    "player_displacement": player_displacement,
                    "player_displacement_rate": player_displacement_rate,
                    # 0 here is ambiguous between "ball detected but always
                    # far" and "ball never detected this window at all" -
                    # ball_frames_detected disambiguates for log-reading:
                    # a real 0 means total detection dropout (the harder
                    # case a wider fusion window can't fix by itself, since
                    # there's no position data to reason about at all).
                    "ball_frames_detected": len(distances),
                },
            )
        )

    if not candidates:
        return ScoredLabel(
            ActionLabel.IDLE,
            0.3,
            {
                "fraction_possessed": fraction_possessed,
                "player_displacement": player_displacement,
                "generic_basketball_signal": generic_basketball,
                "ball_frames_detected": len(distances),
            },
        )

    return max(candidates, key=lambda c: c.confidence)


def merge_adjacent_segments(scored_windows: list[tuple[WindowSignals, ScoredLabel]]):
    """Merge consecutive windows sharing the same label into single segments.

    Both the coarse Kinetics windows (ACTION_WINDOW_STRIDE < ACTION_WINDOW_
    FRAMES) and the fine fusion windows this function actually receives
    (FUSION_WINDOW_STRIDE_SECONDS < FUSION_WINDOW_SECONDS) overlap by
    design, so a new window's start_time can fall before the previous
    segment's end_time. A new segment's start is clamped to the previous
    segment's end so the output timeline is contiguous and non-overlapping,
    which is what the mobile timeline UI and the narrative's sequential
    wording both assume.

    Also majority-votes a dominant dribbling hand ("left"/"right") across
    every frame in a merged DRIBBLING segment, from each frame's
    `dribbling_hand` (whichever wrist was nearest the ball that frame).
    """
    from app.schemas import ActionSegment

    segments: list[ActionSegment] = []
    hand_counters: list[Counter] = []
    for window, scored in scored_windows:
        window_hand_votes = Counter(f.dribbling_hand for f in window.frames if f.dribbling_hand)
        if segments and segments[-1].label == scored.label:
            segments[-1].end_time = window.end_time
            segments[-1].confidence = max(segments[-1].confidence, scored.confidence)
            hand_counters[-1].update(window_hand_votes)
        else:
            start_time = window.start_time
            if segments:
                start_time = max(start_time, segments[-1].end_time)
            segments.append(
                ActionSegment(
                    start_time=start_time,
                    end_time=window.end_time,
                    label=scored.label,
                    confidence=scored.confidence,
                    evidence=scored.evidence,
                )
            )
            hand_counters.append(window_hand_votes)

    for segment, counter in zip(segments, hand_counters):
        if segment.label == ActionLabel.DRIBBLING and counter:
            segment.dominant_hand = counter.most_common(1)[0][0]

    return segments
