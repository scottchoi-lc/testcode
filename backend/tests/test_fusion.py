"""Unit tests for the rule-based fusion logic. Pure Python/numeric inputs -
no ML models, no network, no GPU required."""
from app.pipeline.fusion import FrameSignals, WindowSignals, merge_adjacent_segments, score_window
from app.schemas import ActionLabel


def _frame(t, player=None, ball=None, dist=None, wrist_high=None, hand=None, wrist_dist=None):
    return FrameSignals(
        timestamp=t,
        player_center=player,
        ball_center=ball,
        ball_player_distance=dist,
        wrist_above_shoulder=wrist_high,
        dribbling_hand=hand,
        ball_wrist_distance=wrist_dist,
    )


def test_dribbling_detected_from_ball_bounce_near_player():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.7), dist=0.3, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.9), dist=0.4, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.7), dist=0.3, wrist_high=False),
        _frame(0.6, player=(0.5, 0.5), ball=(0.5, 0.9), dist=0.4, wrist_high=False),
    ]
    window = WindowSignals(0.0, 0.6, frames, kinetics_top_labels=[("dribbling basketball", 0.8)])
    result = score_window(window)
    assert result.label == ActionLabel.DRIBBLING
    assert result.confidence > 0.5


def test_shooting_detected_from_wrist_raise_and_release():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=0.5, wrist_high=True),
        _frame(0.4, player=(0.5, 0.5), ball=(0.55, 0.1), dist=1.8, wrist_high=True),
        _frame(0.6, player=(0.5, 0.5), ball=(0.6, 0.0), dist=2.0, wrist_high=True),
    ]
    window = WindowSignals(0.0, 0.6, frames, kinetics_top_labels=[("shooting basketball", 0.7)])
    result = score_window(window)
    assert result.label == ActionLabel.SHOOTING
    assert result.confidence > 0.5


def test_passing_detected_from_lateral_release_without_shooting_motion():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.9, 0.5), dist=1.7, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(1.3, 0.5), dist=2.2, wrist_high=False),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.PASSING
    assert result.confidence > 0.4


def test_dribbling_detected_with_low_possession_fraction_from_real_bounce_pattern():
    # Real dribbling spends most of each bounce cycle with the ball in
    # flight, away from the hand - only 1 of 3 detected-ball frames here
    # reads as "possessed" (fraction_possessed = 1/3 ~= 0.33), which is
    # exactly the kind of window DRIBBLE_POSSESSION_MIN_FRACTION=0.3 exists
    # to still catch, versus a naive majority-of-frames requirement.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.6), dist=0.3),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.9), dist=1.1),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.6), dist=1.2),
    ]
    window = WindowSignals(0.0, 0.4, frames)
    result = score_window(window)
    assert result.label == ActionLabel.DRIBBLING


def test_dribbling_detected_via_wrist_distance_despite_far_bbox_center():
    # Extended-arm dribble: the ball is far from the player's bbox *center*
    # (dist=1.2, above BALL_POSSESSION_MAX_DIST=0.9) every frame, which alone
    # would read as no-possession, but the wrist is right on the ball
    # (wrist_dist=0.1) - the tighter signal should win and still call this
    # dribbling rather than moving_without_ball/idle.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.9, 0.7), dist=1.2, wrist_dist=0.1),
        _frame(0.2, player=(0.5, 0.5), ball=(0.9, 0.9), dist=1.2, wrist_dist=0.1),
        _frame(0.4, player=(0.5, 0.5), ball=(0.9, 0.7), dist=1.2, wrist_dist=0.1),
        _frame(0.6, player=(0.5, 0.5), ball=(0.9, 0.9), dist=1.2, wrist_dist=0.1),
    ]
    window = WindowSignals(0.0, 0.6, frames)
    result = score_window(window)
    assert result.label == ActionLabel.DRIBBLING


def test_moving_without_ball_when_no_possession_but_player_displaces():
    frames = [
        _frame(0.0, player=(0.2, 0.5), ball=None, dist=None, wrist_high=None),
        _frame(0.2, player=(0.4, 0.5), ball=None, dist=None, wrist_high=None),
        _frame(0.4, player=(0.7, 0.5), ball=None, dist=None, wrist_high=None),
        _frame(0.6, player=(1.0, 0.5), ball=None, dist=None, wrist_high=None),
    ]
    window = WindowSignals(0.0, 0.6, frames)
    result = score_window(window)
    assert result.label == ActionLabel.MOVING_WITHOUT_BALL


def test_idle_when_no_ball_and_no_movement():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=None, dist=None, wrist_high=None),
        _frame(0.2, player=(0.5, 0.5), ball=None, dist=None, wrist_high=None),
    ]
    window = WindowSignals(0.0, 0.2, frames)
    result = score_window(window)
    assert result.label == ActionLabel.IDLE


def test_empty_window_is_idle_with_zero_confidence():
    window = WindowSignals(0.0, 0.0, [])
    result = score_window(window)
    assert result.label == ActionLabel.IDLE
    assert result.confidence == 0.0


def test_merge_adjacent_segments_combines_same_label_windows():
    from app.pipeline.fusion import ScoredLabel

    w1 = WindowSignals(0.0, 0.5, [])
    w2 = WindowSignals(0.5, 1.0, [])
    w3 = WindowSignals(1.0, 1.5, [])
    scored = [
        (w1, ScoredLabel(ActionLabel.DRIBBLING, 0.6, {})),
        (w2, ScoredLabel(ActionLabel.DRIBBLING, 0.7, {})),
        (w3, ScoredLabel(ActionLabel.SHOOTING, 0.8, {})),
    ]
    segments = merge_adjacent_segments(scored)
    assert len(segments) == 2
    assert segments[0].label == ActionLabel.DRIBBLING
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 1.0
    assert segments[0].confidence == 0.7
    assert segments[1].label == ActionLabel.SHOOTING


def test_merge_adjacent_segments_produces_non_overlapping_boundaries():
    from app.pipeline.fusion import ScoredLabel

    # Overlapping windows (50% overlap, like ACTION_WINDOW_STRIDE <
    # ACTION_WINDOW_FRAMES in real usage): window 2 starts before window 1
    # ends. A naive label change should not let segment 2 start earlier
    # than segment 1's end.
    w1 = WindowSignals(0.0, 2.5, [])
    w2 = WindowSignals(1.3, 3.8, [])
    w3 = WindowSignals(2.7, 5.2, [])
    scored = [
        (w1, ScoredLabel(ActionLabel.IDLE, 0.3, {})),
        (w2, ScoredLabel(ActionLabel.PASSING, 0.6, {})),
        (w3, ScoredLabel(ActionLabel.MOVING_WITHOUT_BALL, 0.5, {})),
    ]
    segments = merge_adjacent_segments(scored)
    assert len(segments) == 3
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 2.5
    # Clamped to the previous segment's end instead of the raw (overlapping) window start.
    assert segments[1].start_time == 2.5
    assert segments[1].end_time == 3.8
    assert segments[2].start_time == 3.8
    assert segments[2].end_time == 5.2
    for a, b in zip(segments, segments[1:]):
        assert a.end_time <= b.start_time


def test_merge_adjacent_segments_votes_dominant_dribbling_hand():
    from app.pipeline.fusion import ScoredLabel

    w1 = WindowSignals(
        0.0,
        0.5,
        [
            _frame(0.0, hand="left"),
            _frame(0.2, hand="left"),
        ],
    )
    w2 = WindowSignals(
        0.5,
        1.0,
        [
            _frame(0.5, hand="left"),
            _frame(0.7, hand="right"),
        ],
    )
    scored = [
        (w1, ScoredLabel(ActionLabel.DRIBBLING, 0.6, {})),
        (w2, ScoredLabel(ActionLabel.DRIBBLING, 0.7, {})),
    ]
    segments = merge_adjacent_segments(scored)
    assert len(segments) == 1
    assert segments[0].dominant_hand == "left"


def test_merge_adjacent_segments_leaves_dominant_hand_none_without_signal():
    from app.pipeline.fusion import ScoredLabel

    w1 = WindowSignals(0.0, 0.5, [_frame(0.0)])
    scored = [(w1, ScoredLabel(ActionLabel.DRIBBLING, 0.6, {}))]
    segments = merge_adjacent_segments(scored)
    assert segments[0].dominant_hand is None


def test_merge_adjacent_segments_only_sets_dominant_hand_for_dribbling():
    from app.pipeline.fusion import ScoredLabel

    w1 = WindowSignals(0.0, 0.5, [_frame(0.0, hand="right")])
    scored = [(w1, ScoredLabel(ActionLabel.SHOOTING, 0.6, {}))]
    segments = merge_adjacent_segments(scored)
    assert segments[0].dominant_hand is None
