"""Unit tests for the rule-based fusion logic. Pure Python/numeric inputs -
no ML models, no network, no GPU required."""
from app.pipeline.fusion import (
    FrameSignals,
    WindowSignals,
    _first_frame_with_ball,
    _has_other_player_near_ball,
    _last_frame_with_ball,
    merge_adjacent_segments,
    score_window,
)
from app.schemas import ActionLabel


def _frame(
    t, player=None, ball=None, dist=None, wrist_high=None, hand=None, wrist_dist=None, other_people=None
):
    return FrameSignals(
        timestamp=t,
        player_center=player,
        ball_center=ball,
        ball_player_distance=dist,
        wrist_above_shoulder=wrist_high,
        dribbling_hand=hand,
        ball_wrist_distance=wrist_dist,
        other_people_centers=other_people or [],
    )


def test_dribbling_detected_from_ball_bounce_near_player():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.7), dist=0.3, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.9), dist=0.4, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.7), dist=0.3, wrist_high=False),
        _frame(0.6, player=(0.5, 0.5), ball=(0.5, 0.9), dist=0.4, wrist_high=False),
    ]
    window = WindowSignals(0.0, 0.6, frames, kinetics_top_labels=[("dribbling a basketball", 0.8)])
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
    window = WindowSignals(0.0, 0.6, frames, kinetics_top_labels=[("shooting a basketball", 0.7)])
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


def test_passing_evidence_flags_nearby_other_player_at_final_ball_position():
    # Someone else detected within BALL_POSSESSION_MAX_DIST of the ball at
    # the last frame it was seen - best-effort, moment-only evidence the
    # pass landed near another player, surfaced in narration.py's wording.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.9, 0.5), dist=1.7, wrist_high=False),
        _frame(
            0.4,
            player=(0.5, 0.5),
            ball=(1.3, 0.5),
            dist=2.2,
            wrist_high=False,
            other_people=[(1.35, 0.55)],  # within 0.9 of the ball at (1.3, 0.5)
        ),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.PASSING
    assert result.evidence["nearby_other_player"] is True


def test_passing_evidence_no_nearby_other_player_when_none_detected():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.9, 0.5), dist=1.7, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(1.3, 0.5), dist=2.2, wrist_high=False),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.PASSING
    assert result.evidence["nearby_other_player"] is False


def test_receiving_detected_from_ball_arriving():
    # Mirror image of passing: the ball starts far from the player and
    # ends close - a caught pass (or a loose-ball pickup; the distance
    # signal can't tell those apart, which is why the narrative wording
    # stays neutral rather than presuming a teammate threw it).
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.0), dist=2.0),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=1.0),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.RECEIVING
    assert result.confidence > 0.5


def test_receiving_evidence_flags_nearby_other_player_at_first_ball_position():
    # Someone else detected within BALL_POSSESSION_MAX_DIST of the ball at
    # the first frame it was seen (before it arrived) - best-effort,
    # moment-only evidence it came from near another player.
    frames = [
        _frame(
            0.0,
            player=(0.5, 0.5),
            ball=(0.5, 0.0),
            dist=2.0,
            other_people=[(0.55, 0.05)],  # within 0.9 of the ball at (0.5, 0.0)
        ),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=1.0),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.RECEIVING
    assert result.evidence["nearby_other_player"] is True


def test_receiving_evidence_no_nearby_other_player_when_none_detected():
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.0), dist=2.0),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=1.0),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[])
    result = score_window(window)
    assert result.label == ActionLabel.RECEIVING
    assert result.evidence["nearby_other_player"] is False


def test_has_other_player_near_ball_true_within_threshold():
    frame = _frame(0.0, ball=(0.5, 0.5), other_people=[(0.9, 0.5)])  # distance 0.4 <= 0.9
    assert _has_other_player_near_ball(frame) is True


def test_has_other_player_near_ball_false_beyond_threshold():
    frame = _frame(0.0, ball=(0.5, 0.5), other_people=[(2.0, 0.5)])  # distance 1.5 > 0.9
    assert _has_other_player_near_ball(frame) is False


def test_has_other_player_near_ball_false_without_ball_or_frame():
    assert _has_other_player_near_ball(None) is False
    assert _has_other_player_near_ball(_frame(0.0, ball=None, other_people=[(0.5, 0.5)])) is False
    assert _has_other_player_near_ball(_frame(0.0, ball=(0.5, 0.5), other_people=[])) is False


def test_last_and_first_frame_with_ball():
    f0 = _frame(0.0, ball=None)
    f1 = _frame(0.2, ball=(0.1, 0.1))
    f2 = _frame(0.4, ball=None)
    f3 = _frame(0.6, ball=(0.9, 0.9))
    f4 = _frame(0.8, ball=None)
    frames = [f0, f1, f2, f3, f4]
    assert _first_frame_with_ball(frames) is f1
    assert _last_frame_with_ball(frames) is f3
    assert _first_frame_with_ball([f0, f2, f4]) is None
    assert _last_frame_with_ball([f0, f2, f4]) is None


def test_receiving_not_triggered_by_weak_kinetics_noise_alone():
    # Ball starts far and stays far (never actually arrives) - a weak
    # "catching or receiving" classifier score shouldn't manufacture a
    # reception on its own, same KINETICS_OVERRIDE_MIN discipline as every
    # other branch.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.0), dist=2.0),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.1), dist=1.8),
    ]
    window = WindowSignals(
        0.0, 0.2, frames, kinetics_top_labels=[("catching or receiving a basketball pass", 0.1)]
    )
    result = score_window(window)
    assert result.label != ActionLabel.RECEIVING


def test_receiving_corroborated_by_strong_kinetics_score_without_clear_catch():
    # The ball only closes to dist=1.0 (still above BALL_POSSESSION_MAX_
    # DIST=0.9), so the direct "caught" check alone wouldn't call this a
    # reception - but a strong classifier score should be able to
    # corroborate it, the same way dribble_boost/shoot_boost/pass_boost
    # can corroborate their branches.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.0), dist=2.0),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=1.0),
    ]
    window = WindowSignals(
        0.0, 0.2, frames, kinetics_top_labels=[("catching or receiving a basketball pass", 0.35)]
    )
    result = score_window(window)
    assert result.label == ActionLabel.RECEIVING


def test_crossover_dribble_not_mistaken_for_passing_when_ball_returns_soon():
    # Real clip: a crossover (ball swings laterally across the body, no
    # wrist raise) has the exact same signature the PASSING branch looks
    # for - released, lateral > vertical, no shot motion - which got it
    # mislabeled as a pass. The tell is what happens right after: a real
    # pass doesn't come back to the passer's hands a fraction of a second
    # later, but a crossover's ball does (this is what
    # ball_returns_to_possession_soon, computed in pipeline.py from frames
    # beyond this window, is for).
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.9, 0.5), dist=1.7, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(1.3, 0.5), dist=2.2, wrist_high=False),
    ]
    window = WindowSignals(
        0.0, 0.4, frames, kinetics_top_labels=[], ball_returns_to_possession_soon=True
    )
    result = score_window(window)
    assert result.label != ActionLabel.PASSING


def test_passing_corroborated_by_strong_kinetics_pass_score_without_clear_release():
    # The ball only makes it to dist=1.0 (below BALL_RELEASE_MIN_DIST=1.6),
    # so the direct "released" check alone wouldn't call this a pass - but a
    # strong classifier score for "passing a basketball to a teammate"
    # should be able to corroborate it, the same way dribble_boost/
    # shoot_boost can corroborate their branches.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.8, 0.5), dist=1.0, wrist_high=False),
    ]
    window = WindowSignals(
        0.0, 0.2, frames, kinetics_top_labels=[("passing a basketball to a teammate", 0.35)]
    )
    result = score_window(window)
    assert result.label == ActionLabel.PASSING


def test_passing_not_corroborated_by_weak_kinetics_pass_score():
    # Same as above, but the score is below KINETICS_OVERRIDE_MIN - too weak
    # to stand in for the missing direct release evidence.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.8, 0.5), dist=1.0, wrist_high=False),
    ]
    window = WindowSignals(
        0.0, 0.2, frames, kinetics_top_labels=[("passing a basketball to a teammate", 0.1)]
    )
    result = score_window(window)
    assert result.label != ActionLabel.PASSING


def test_passing_kinetics_corroboration_still_vetoed_by_ball_returns_to_possession_soon():
    # A strong "passing" classifier score doesn't get to override the
    # physical fact that the ball came right back to the same player - that
    # veto applies regardless of which path (direct release or kinetics
    # corroboration) would otherwise have triggered PASSING.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.8, 0.5), dist=1.0, wrist_high=False),
    ]
    window = WindowSignals(
        0.0,
        0.2,
        frames,
        kinetics_top_labels=[("passing a basketball to a teammate", 0.35)],
        ball_returns_to_possession_soon=True,
    )
    result = score_window(window)
    assert result.label != ActionLabel.PASSING


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


def test_shooting_not_triggered_by_weak_kinetics_noise_alone():
    # Real clip: fraction_wrist_high=0.0 and the ball never released, yet
    # kinetics_shoot_score=0.088 - softmax noise, not a real "shooting
    # basketball" prediction - was enough to trigger SHOOTING under the old
    # `shoot_boost > 0` bypass, which treated *any* nonzero Kinetics score as
    # a hard override for both the wrist-height and release checks. Neither
    # real signal is present here (ball stays put, wrist never rises), so a
    # weak score below KINETICS_OVERRIDE_MIN must not manufacture a shot.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.3, wrist_high=False),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=False),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[("shooting a basketball", 0.088)])
    result = score_window(window)
    assert result.label != ActionLabel.SHOOTING
    # The ball sitting still and close the whole window is a textbook (if
    # boring) dribble/possession read, not a shot - pin the actual label so
    # this test also catches any future regression in the dribbling branch.
    assert result.label == ActionLabel.DRIBBLING


def test_dribbling_not_triggered_by_weak_kinetics_noise_alone():
    # Same bug, other branch: a bouncy/erratic ball trajectory
    # (vertical_std well above DRIBBLE_VERTICAL_STD_MAX) shouldn't be
    # relabeled DRIBBLING just because the Kinetics classifier assigned it a
    # tiny, likely-noise probability for "dribbling basketball".
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.1), dist=1.0),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.9), dist=0.3),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.1), dist=1.0),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[("dribbling a basketball", 0.1)])
    result = score_window(window)
    assert result.label != ActionLabel.DRIBBLING


def test_shooting_still_detected_when_kinetics_score_clears_override_threshold():
    # Sanity check for the other direction: a *strong* Kinetics signal (at
    # or above KINETICS_OVERRIDE_MIN) should still be able to corroborate a
    # shot even when the direct release check alone wouldn't quite clear the
    # bar - the fix narrows the override, it doesn't remove it.
    frames = [
        _frame(0.0, player=(0.5, 0.5), ball=(0.5, 0.5), dist=0.2, wrist_high=True),
        _frame(0.2, player=(0.5, 0.5), ball=(0.5, 0.3), dist=0.5, wrist_high=True),
        _frame(0.4, player=(0.5, 0.5), ball=(0.5, 0.2), dist=0.6, wrist_high=True),
    ]
    window = WindowSignals(0.0, 0.4, frames, kinetics_top_labels=[("shooting a basketball", 0.6)])
    result = score_window(window)
    assert result.label == ActionLabel.SHOOTING


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
