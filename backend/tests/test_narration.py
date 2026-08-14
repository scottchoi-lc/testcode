from app.pipeline.narration import NO_ACTIONS_MESSAGE, narrate
from app.schemas import ActionLabel, ActionSegment


def _segment(label, start, end, dominant_hand=None, evidence=None):
    return ActionSegment(
        start_time=start,
        end_time=end,
        label=label,
        confidence=0.8,
        dominant_hand=dominant_hand,
        evidence=evidence or {},
    )


def test_narrate_empty_segments_returns_no_actions_message():
    assert narrate([], {}) == NO_ACTIONS_MESSAGE


def test_narrate_all_idle_returns_no_actions_message():
    segments = [_segment(ActionLabel.IDLE, 0.0, 0.5)]
    summary = {"idle": 0.5}
    assert narrate(segments, summary) == NO_ACTIONS_MESSAGE


def test_narrate_builds_play_by_play_in_order():
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 3.2),
        _segment(ActionLabel.MOVING_WITHOUT_BALL, 3.2, 4.7),
        _segment(ActionLabel.SHOOTING, 4.7, 5.5),
    ]
    summary = {
        "dribbling": 3.2,
        "shooting": 0.8,
        "passing": 0.0,
        "moving_without_ball": 1.5,
        "idle": 0.0,
    }
    result = narrate(segments, summary)
    assert result.startswith(
        "Player 1 dribbled the ball, then moved without the ball, then took a shot."
    )
    assert "3.2s" not in result.split("Overall:")[0]  # no per-event durations before "Overall:"
    assert "Overall: 3.2s dribbling, 1.5s moving without the ball, 0.8s shooting." in result


def test_narrate_play_by_play_omits_per_event_durations():
    # Per-event durations ("dribbled the ball for 0.7s") were noise, not
    # useful narration detail - removed from the play-by-play clauses.
    # The "Overall: Xs ..." totals sentence is a different construct (an
    # aggregate breakdown, matching the mobile app's summary bar chart)
    # and keeps its durations.
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 0.7),
        _segment(ActionLabel.PASSING, 0.7, 1.4),
    ]
    summary = {"dribbling": 0.7, "passing": 0.7, "shooting": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary)
    play_by_play, _, totals = result.partition("Overall:")
    assert "0.7s" not in play_by_play
    assert "for" not in play_by_play
    assert play_by_play.strip() == "Player 1 dribbled the ball, then passed the ball."
    assert "0.7s" in totals


def test_narrate_skips_short_idle_gaps_but_keeps_long_ones():
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 2.0),
        _segment(ActionLabel.IDLE, 2.0, 2.3),  # too short to mention
        _segment(ActionLabel.PASSING, 2.3, 3.0),
        _segment(ActionLabel.IDLE, 3.0, 5.0),  # long enough to mention
    ]
    summary = {"dribbling": 2.0, "passing": 0.7, "shooting": 0.0, "moving_without_ball": 0.0, "idle": 2.3}
    result = narrate(segments, summary)
    assert "then passed the ball," in result
    assert "then paused." in result
    assert result.count("paused") == 1


def test_narrate_uses_player_number_when_given():
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 2.0),
        _segment(ActionLabel.SHOOTING, 2.0, 2.8),
    ]
    summary = {"dribbling": 2.0, "shooting": 0.8, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary, player_number="23")
    assert result.startswith("Number 23 dribbled the ball,")
    assert "Player 1" not in result


def test_narrate_falls_back_to_generic_wording_without_a_number():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0)]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary, player_number=None)
    assert result.startswith("Player 1 dribbled")


def test_narrate_mentions_dominant_hand_when_known():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0, dominant_hand="left")]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 dribbled the ball with the left hand.")


def test_narrate_omits_hand_when_unknown():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0, dominant_hand=None)]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 dribbled the ball.")
    assert "hand" not in result


def test_narrate_uses_neutral_wording_for_receiving():
    # Deliberately doesn't say "caught a pass" - the ball-distance signal
    # behind RECEIVING can't distinguish a caught pass from picking up a
    # loose ball or grabbing a rebound, so the wording shouldn't presume
    # a teammate threw it.
    segments = [_segment(ActionLabel.RECEIVING, 0.0, 1.0)]
    summary = {"receiving": 1.0}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 received the ball.")
    assert "pass" not in result.lower()


def test_narrate_mentions_nearby_player_for_passing_when_flagged():
    segments = [_segment(ActionLabel.PASSING, 0.0, 0.5, evidence={"nearby_other_player": True})]
    summary = {"passing": 0.5}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 passed the ball to a nearby player.")


def test_narrate_mentions_nearby_player_for_receiving_when_flagged():
    segments = [_segment(ActionLabel.RECEIVING, 0.0, 0.5, evidence={"nearby_other_player": True})]
    summary = {"receiving": 0.5}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 received the ball from a nearby player.")


def test_narrate_uses_plain_wording_when_no_nearby_player_flagged():
    # Default evidence (no nearby_other_player key at all, or explicitly
    # False) falls back to the plain phrasing rather than erroring.
    segments = [
        _segment(ActionLabel.PASSING, 0.0, 0.5),
        _segment(ActionLabel.RECEIVING, 0.5, 1.0, evidence={"nearby_other_player": False}),
    ]
    summary = {"passing": 0.5, "receiving": 0.5}
    result = narrate(segments, summary)
    assert result.startswith("Player 1 passed the ball, then received the ball.")
