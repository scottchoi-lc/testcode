"""Unit tests for pure-logic helpers in app.pipeline.pipeline. No models,
no video I/O - these only need numpy/opencv importable (see README)."""
import numpy as np
import pytest

from app.config import settings
from app.models.action_classifier import ClipPrediction
from app.models.detection import Detection
from app.pipeline.fusion import FrameSignals
from app.pipeline.pipeline import (
    _ball_returns_to_possession_soon,
    _ball_was_already_possessed_before,
    _bidirectional_frame_order,
    _build_frame_signals,
    _classify_shot_outcome,
    _color_histogram,
    _fusion_window_bounds,
    _interpolate_ball_gaps,
    _nearest_kinetics_labels,
    _pick_primary_player,
)
from app.pipeline.video_utils import Frame


def _fs(t, player=None, ball=None, dist=None):
    return FrameSignals(
        timestamp=t,
        player_center=player,
        ball_center=ball,
        ball_player_distance=dist,
        wrist_above_shoulder=None,
    )


def test_nearest_kinetics_labels_picks_closest_midpoint():
    kinetics_windows = [
        (0.0, 2.5, [("dribbling basketball", 0.8)]),
        (1.3, 3.8, [("shooting basketball", 0.7)]),
        (5.0, 7.5, [("playing basketball", 0.5)]),
    ]
    # Fine window centered at 0.7s is closest to the first kinetics window's
    # midpoint (1.25), not the second (2.55) or third (6.25).
    assert _nearest_kinetics_labels(kinetics_windows, midpoint=0.7) == [("dribbling basketball", 0.8)]
    assert _nearest_kinetics_labels(kinetics_windows, midpoint=2.6) == [("shooting basketball", 0.7)]
    assert _nearest_kinetics_labels(kinetics_windows, midpoint=6.0) == [("playing basketball", 0.5)]


def test_nearest_kinetics_labels_empty_input_returns_empty():
    assert _nearest_kinetics_labels([], midpoint=1.0) == []


def test_bidirectional_frame_order_seed_in_middle():
    forward, backward = _bidirectional_frame_order(num_frames=10, seed_index=4)
    assert forward == [4, 5, 6, 7, 8, 9]
    assert backward == [3, 2, 1, 0]
    # Every index appears exactly once across both lists.
    assert sorted(forward + backward) == list(range(10))


def test_bidirectional_frame_order_seed_at_start():
    forward, backward = _bidirectional_frame_order(num_frames=5, seed_index=0)
    assert forward == [0, 1, 2, 3, 4]
    assert backward == []


def test_bidirectional_frame_order_seed_at_end():
    forward, backward = _bidirectional_frame_order(num_frames=5, seed_index=4)
    assert forward == [4]
    assert backward == [3, 2, 1, 0]


def test_fusion_window_bounds_no_overlap_tiles_exactly():
    # stride == window size reproduces the old non-overlapping tiling, so
    # this also pins the pre-overlap behavior as a regression check.
    assert _fusion_window_bounds(num_frames=10, window_frames=5, stride_frames=5) == [(0, 5), (5, 10)]


def test_fusion_window_bounds_overlap_covers_every_boundary():
    # stride < window size: every consecutive pair of frames must fall
    # inside at least one common window, so a brief transition (e.g. a
    # ball release) landing anywhere can't be split across a hard boundary
    # the way it was with non-overlapping tiling.
    bounds = _fusion_window_bounds(num_frames=10, window_frames=5, stride_frames=2)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10
    for i in range(9):
        assert any(start <= i and i + 1 < end for start, end in bounds)


def test_fusion_window_bounds_drops_windows_shorter_than_two_frames():
    assert _fusion_window_bounds(num_frames=1, window_frames=5, stride_frames=2) == []


def test_interpolate_ball_gaps_fills_short_gap_between_known_positions():
    # Player tracked throughout; ball missing for exactly 1 frame between
    # two real detections - short enough (<= max_gap_frames=2) to fill.
    frames = [
        _fs(0.0, player=(0.0, 0.0), ball=(0.0, 0.0), dist=0.0),
        _fs(0.2, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.4, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
    ]
    result, filled, longest_gap = _interpolate_ball_gaps(frames, max_gap_frames=2)
    assert filled == 1
    assert longest_gap == 1
    assert result[1].ball_center == (1.0, 0.0)
    assert result[1].ball_player_distance == 1.0
    # Real readings are untouched.
    assert result[0] is frames[0]
    assert result[2] is frames[2]


def test_interpolate_ball_gaps_leaves_gap_longer_than_bound():
    frames = [
        _fs(0.0, player=(0.0, 0.0), ball=(0.0, 0.0), dist=0.0),
        _fs(0.2, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.4, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.6, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.8, player=(0.0, 0.0), ball=(6.0, 0.0), dist=6.0),
    ]
    result, filled, longest_gap = _interpolate_ball_gaps(frames, max_gap_frames=2)
    assert filled == 0
    # Reported even though it wasn't filled - this is what tells a caller
    # how far over the bound a real clip's gap actually ran, rather than
    # just "0 filled" with no way to tell a 1-frame miss from a 20-frame one.
    assert longest_gap == 3
    assert result[1].ball_center is None
    assert result[2].ball_center is None
    assert result[3].ball_center is None


def test_interpolate_ball_gaps_leaves_gap_without_valid_bounds_on_both_sides():
    # Leading gap: no real detection before it to interpolate from.
    frames = [
        _fs(0.0, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.2, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
    ]
    result, filled, longest_gap = _interpolate_ball_gaps(frames, max_gap_frames=2)
    assert filled == 0
    assert longest_gap == 1
    assert result[0].ball_center is None


def test_interpolate_ball_gaps_does_not_bridge_across_untracked_player_frame():
    # The gap includes a frame where the player itself wasn't tracked
    # (player_center=None) - that frame can't anchor an interpolation
    # (no known ball_center either), so the preceding ball-only miss is
    # left alone rather than assuming continuity through a tracking loss.
    frames = [
        _fs(0.0, player=(0.0, 0.0), ball=(0.0, 0.0), dist=0.0),
        _fs(0.2, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.4, player=None, ball=None, dist=None),
        _fs(0.6, player=(0.0, 0.0), ball=(4.0, 0.0), dist=4.0),
    ]
    result, filled, longest_gap = _interpolate_ball_gaps(frames, max_gap_frames=2)
    assert filled == 0
    assert longest_gap == 1
    assert result[1].ball_center is None


def test_ball_returns_to_possession_soon_true_when_close_reading_in_lookahead():
    # The frame right after the window ends shows the ball back close -
    # the crossover-dribble case: distance looked "released" by the end of
    # the previous window, but it's back in the same player's hands almost
    # immediately.
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
        _fs(0.2, player=(0.0, 0.0), ball=(0.1, 0.0), dist=0.1),
    ]
    assert _ball_returns_to_possession_soon(frame_signals, end_idx=1, lookahead_frames=1) is True


def test_ball_returns_to_possession_soon_false_when_ball_stays_away():
    # A real pass/shot: the ball stays far for the whole lookahead window.
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
        _fs(0.2, player=(0.0, 0.0), ball=(2.5, 0.0), dist=2.5),
    ]
    assert _ball_returns_to_possession_soon(frame_signals, end_idx=1, lookahead_frames=1) is False


def test_ball_returns_to_possession_soon_false_when_lookahead_runs_past_clip_end():
    frame_signals = [_fs(0.0, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0)]
    assert _ball_returns_to_possession_soon(frame_signals, end_idx=5, lookahead_frames=3) is False


def test_ball_returns_to_possession_soon_skips_frames_with_no_ball_detected():
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.2, player=(0.0, 0.0), ball=(0.1, 0.0), dist=0.1),
    ]
    assert _ball_returns_to_possession_soon(frame_signals, end_idx=0, lookahead_frames=2) is True


def test_ball_was_already_possessed_before_true_when_close_reading_in_lookback():
    # Mirror of the return-soon case: the frame right before the window
    # starts shows the ball was already close - the hand-switch case,
    # where a same-player crossover briefly looks like a fresh arrival.
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=(0.1, 0.0), dist=0.1),
        _fs(0.2, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
    ]
    assert _ball_was_already_possessed_before(frame_signals, start_idx=1, lookback_frames=1) is True


def test_ball_was_already_possessed_before_false_when_ball_was_always_away():
    # A real reception: the ball wasn't with this player at all beforehand.
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=(2.5, 0.0), dist=2.5),
        _fs(0.2, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
    ]
    assert _ball_was_already_possessed_before(frame_signals, start_idx=1, lookback_frames=1) is False


def test_ball_was_already_possessed_before_false_when_lookback_runs_before_clip_start():
    frame_signals = [_fs(0.0, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0)]
    assert _ball_was_already_possessed_before(frame_signals, start_idx=0, lookback_frames=3) is False


def test_ball_was_already_possessed_before_skips_frames_with_no_ball_detected():
    frame_signals = [
        _fs(0.0, player=(0.0, 0.0), ball=(0.1, 0.0), dist=0.1),
        _fs(0.2, player=(0.0, 0.0), ball=None, dist=None),
        _fs(0.4, player=(0.0, 0.0), ball=(2.0, 0.0), dist=2.0),
    ]
    assert _ball_was_already_possessed_before(frame_signals, start_idx=2, lookback_frames=2) is True


def _person(box, score=0.9):
    return Detection(label="person", score=score, box=box)


def _frame(t=0.0):
    return Frame(index=0, timestamp=t, image=np.zeros((1, 1, 3), dtype=np.uint8))


def test_build_frame_signals_normalizes_other_people_centers():
    player = _person((0.0, 0.0, 10.0, 10.0))  # center (5, 5), diag = 10*sqrt(2)
    other = _person((20.0, 0.0, 30.0, 10.0))  # center (25, 5)
    scale = (10.0**2 + 10.0**2) ** 0.5
    signals = _build_frame_signals(
        _frame(), player, ball=None, wrist_above_shoulder=None, other_people=[other]
    )
    assert signals.other_people_centers == [pytest.approx((25.0 / scale, 5.0 / scale))]


def test_build_frame_signals_empty_other_people_by_default():
    player = _person((0.0, 0.0, 10.0, 10.0))
    signals = _build_frame_signals(_frame(), player, ball=None, wrist_above_shoulder=None)
    assert signals.other_people_centers == []


def test_build_frame_signals_no_other_people_when_player_untracked():
    other = _person((20.0, 0.0, 30.0, 10.0))
    signals = _build_frame_signals(
        _frame(), player=None, ball=None, wrist_above_shoulder=None, other_people=[other]
    )
    assert signals.player_center is None
    assert signals.other_people_centers == []


def test_pick_primary_player_picks_largest_when_no_previous_center():
    small = _person((0, 0, 10, 10))
    large = _person((0, 0, 100, 100))
    assert _pick_primary_player([small, large], previous_center=None) is large


def test_pick_primary_player_continues_nearest_within_plausible_range():
    # previous_center is (5, 5); nearby candidate should win over a far one.
    nearby = _person((3, 3, 13, 13))  # center (8, 8), diagonal ~14.1
    far = _person((500, 500, 510, 510))
    result = _pick_primary_player([nearby, far], previous_center=(5.0, 5.0))
    assert result is nearby


def test_pick_primary_player_rejects_implausibly_far_jump_instead_of_snapping():
    # The only candidate is ~35 diagonals away from the last known position -
    # a real clip logged exactly this kind of jump (6.9 diagonals) snapping
    # onto a different person. Should return None (no detection this frame)
    # rather than silently tracking whoever happens to be nearest.
    far_only = _person((1000, 1000, 1010, 1010))  # diagonal ~14.1, center (1005, 1005)
    result = _pick_primary_player([far_only], previous_center=(5.0, 5.0))
    assert result is None


def test_pick_primary_player_records_rejection_stats():
    far_only = _person((1000, 1000, 1010, 1010))
    stats = {"implausible_jumps_rejected": 0, "max_jump_seen": 0.0}
    result = _pick_primary_player([far_only], previous_center=(5.0, 5.0), stats=stats)
    assert result is None
    assert stats["implausible_jumps_rejected"] == 1
    assert stats["max_jump_seen"] > 3.0


def _solid_color_image(colors_and_boxes, size=200):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for color, box in colors_and_boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        image[y1:y2, x1:x2] = color
    return image


def test_pick_primary_player_disambiguates_ambiguous_candidates_by_appearance():
    # Two players near each other: both within MAX_PLAUSIBLE_TRACKING_JUMP of
    # previous_center, so position alone can't tell them apart - this is the
    # exact scenario that let tracking silently drift from a passer onto a
    # nearby receiver in a real clip. blue_box is the closer of the two by
    # raw distance, but the reference appearance matches red_box; appearance
    # should win over "merely closer".
    red_box = (10, 10, 40, 40)
    blue_box = (20, 20, 50, 50)
    image = _solid_color_image([((0, 0, 255), red_box), ((255, 0, 0), blue_box)])  # BGR: red, blue

    red_person = _person(red_box)
    blue_person = _person(blue_box)
    previous_center = (28.0, 28.0)  # closer to blue_box's center (35, 35) than red's (25, 25)

    reference = _color_histogram(image, red_box)
    result = _pick_primary_player(
        [red_person, blue_person], previous_center, image_bgr=image, reference_appearance=reference
    )
    assert result is red_person


def test_pick_primary_player_records_disambiguation_stat():
    red_box = (10, 10, 40, 40)
    blue_box = (20, 20, 50, 50)
    image = _solid_color_image([((0, 0, 255), red_box), ((255, 0, 0), blue_box)])
    reference = _color_histogram(image, red_box)
    stats = {
        "implausible_jumps_rejected": 0,
        "max_jump_seen": 0.0,
        "ambiguous_frames_disambiguated_by_appearance": 0,
    }
    _pick_primary_player(
        [_person(red_box), _person(blue_box)],
        previous_center=(28.0, 28.0),
        image_bgr=image,
        reference_appearance=reference,
        stats=stats,
    )
    assert stats["ambiguous_frames_disambiguated_by_appearance"] == 1


def test_pick_primary_player_rejects_lone_candidate_with_mismatched_appearance():
    # Only one candidate is within jump range - no ambiguous tie for a
    # tie-breaker to resolve - but its appearance doesn't match the
    # reference at all. Once tracking has drifted onto a wrong person,
    # later frames typically only have *one* nearby candidate (the wrong
    # person), so this must be checked even without a tie, or a lost
    # identity is never recovered.
    box = (10, 10, 40, 40)
    blue_frame = _solid_color_image([((255, 0, 0), box)])  # candidate is blue
    red_reference_frame = _solid_color_image([((0, 0, 255), box)])  # reference is red
    reference = _color_histogram(red_reference_frame, box)

    result = _pick_primary_player(
        [_person(box)], previous_center=(25.0, 25.0), image_bgr=blue_frame, reference_appearance=reference
    )
    assert result is None


def test_pick_primary_player_records_appearance_mismatch_stat():
    box = (10, 10, 40, 40)
    blue_frame = _solid_color_image([((255, 0, 0), box)])
    red_reference_frame = _solid_color_image([((0, 0, 255), box)])
    reference = _color_histogram(red_reference_frame, box)
    stats = {
        "implausible_jumps_rejected": 0,
        "max_jump_seen": 0.0,
        "ambiguous_frames_disambiguated_by_appearance": 0,
        "appearance_mismatches_rejected": 0,
    }
    result = _pick_primary_player(
        [_person(box)],
        previous_center=(25.0, 25.0),
        image_bgr=blue_frame,
        reference_appearance=reference,
        stats=stats,
    )
    assert result is None
    assert stats["appearance_mismatches_rejected"] == 1


def test_pick_primary_player_accepts_lone_candidate_matching_appearance():
    box = (10, 10, 40, 40)
    image = _solid_color_image([((0, 0, 255), box)])
    reference = _color_histogram(image, box)
    result = _pick_primary_player(
        [_person(box)], previous_center=(25.0, 25.0), image_bgr=image, reference_appearance=reference
    )
    assert result is not None


def test_pick_primary_player_falls_back_to_nearest_without_appearance_signal():
    # Same ambiguous setup, but no reference appearance provided (e.g. it
    # couldn't be computed) - falls back to plain nearest-by-position rather
    # than crashing or picking arbitrarily.
    red_box = (10, 10, 40, 40)
    blue_box = (20, 20, 50, 50)
    result = _pick_primary_player(
        [_person(red_box), _person(blue_box)], previous_center=(28.0, 28.0)
    )
    assert result is not None


class _FakeActionClassifier:
    """Stand-in for `ActionClassifier` - returns a fixed prediction and
    records what it was called with, so these tests don't need to load a
    real X-CLIP model."""

    def __init__(self, top_labels):
        self._top_labels = top_labels
        self.calls: list[tuple[int, list[str] | None]] = []

    def classify_window(self, frames_bgr, top_k=5, candidate_labels=None):
        self.calls.append((len(frames_bgr), candidate_labels))
        return ClipPrediction(top_labels=self._top_labels)


def test_classify_shot_outcome_returns_none_with_too_few_frames_after_shot():
    made_label, _ = settings.SHOT_OUTCOME_CANDIDATE_LABELS
    # Shot ends at 6.3s; only one frame (6.4s) falls inside the outcome
    # window that follows - not enough to bother calling the classifier.
    frames = [_frame(0.0), _frame(6.4)]
    classifier = _FakeActionClassifier([(made_label, 0.9)])
    result = _classify_shot_outcome(classifier, frames, shot_end_time=6.3)
    assert result is None
    assert classifier.calls == []


def test_classify_shot_outcome_true_when_made_label_wins_confidently():
    made_label, missed_label = settings.SHOT_OUTCOME_CANDIDATE_LABELS
    frames = [_frame(6.3), _frame(6.5), _frame(6.7)]
    classifier = _FakeActionClassifier([(made_label, 0.9), (missed_label, 0.1)])
    assert _classify_shot_outcome(classifier, frames, shot_end_time=6.3) is True
    assert classifier.calls == [(3, settings.SHOT_OUTCOME_CANDIDATE_LABELS)]


def test_classify_shot_outcome_false_when_missed_label_wins_confidently():
    made_label, missed_label = settings.SHOT_OUTCOME_CANDIDATE_LABELS
    frames = [_frame(6.3), _frame(6.5), _frame(6.7)]
    classifier = _FakeActionClassifier([(missed_label, 0.85), (made_label, 0.15)])
    assert _classify_shot_outcome(classifier, frames, shot_end_time=6.3) is False


def test_classify_shot_outcome_none_when_top_score_below_confidence_threshold():
    made_label, missed_label = settings.SHOT_OUTCOME_CANDIDATE_LABELS
    # Nearly a coin flip - shouldn't be reported as a confident make.
    frames = [_frame(6.3), _frame(6.5), _frame(6.7)]
    classifier = _FakeActionClassifier([(made_label, 0.55), (missed_label, 0.45)])
    assert _classify_shot_outcome(classifier, frames, shot_end_time=6.3) is None


def test_classify_shot_outcome_only_passes_frames_within_the_outcome_window():
    made_label, _ = settings.SHOT_OUTCOME_CANDIDATE_LABELS
    # Shot ends at 6.3s; SHOT_OUTCOME_WINDOW_SECONDS defaults to 2.0s, so a
    # frame at 9.0s is well past it and must not be included.
    frames = [_frame(6.3), _frame(6.5), _frame(6.7), _frame(9.0)]
    classifier = _FakeActionClassifier([(made_label, 0.9)])
    _classify_shot_outcome(classifier, frames, shot_end_time=6.3)
    assert classifier.calls[0][0] == 3
