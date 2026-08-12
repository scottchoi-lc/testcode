"""Unit tests for the rule-based fusion logic. Pure Python/numeric inputs -
no ML models, no network, no GPU required."""
from app.pipeline.fusion import FrameSignals, WindowSignals, merge_adjacent_segments, score_window
from app.schemas import ActionLabel


def _frame(t, player=None, ball=None, dist=None, wrist_high=None, hand=None):
    return FrameSignals(
        timestamp=t,
        player_center=player,
        ball_center=ball,
        ball_player_distance=dist,
        wrist_above_shoulder=wrist_high,
        dribbling_hand=hand,
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
