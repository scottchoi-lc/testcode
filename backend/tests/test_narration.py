from app.pipeline.narration import NO_ACTIONS_MESSAGE, narrate, player_identification_note
from app.schemas import ActionLabel, ActionSegment


def _segment(label, start, end, dominant_hand=None):
    return ActionSegment(
        start_time=start, end_time=end, label=label, confidence=0.8, dominant_hand=dominant_hand
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
    assert result.startswith("The player dribbled the ball for 3.2s")
    assert "then moved without the ball for 1.5s" in result
    assert "then took a shot for 0.8s" in result
    assert "Overall: 3.2s dribbling, 1.5s moving without the ball, 0.8s shooting." in result


def test_narrate_skips_short_idle_gaps_but_keeps_long_ones():
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 2.0),
        _segment(ActionLabel.IDLE, 2.0, 2.3),  # too short to mention
        _segment(ActionLabel.PASSING, 2.3, 3.0),
        _segment(ActionLabel.IDLE, 3.0, 5.0),  # long enough to mention
    ]
    summary = {"dribbling": 2.0, "passing": 0.7, "shooting": 0.0, "moving_without_ball": 0.0, "idle": 2.3}
    result = narrate(segments, summary)
    assert "then passed the ball for 0.7s" in result
    assert "then paused for 2.0s" in result
    assert result.count("paused") == 1


def test_narrate_uses_player_number_when_given():
    segments = [
        _segment(ActionLabel.DRIBBLING, 0.0, 2.0),
        _segment(ActionLabel.SHOOTING, 2.0, 2.8),
    ]
    summary = {"dribbling": 2.0, "shooting": 0.8, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary, player_number="23")
    assert result.startswith("Player #23 dribbled the ball for 2.0s")
    assert "The player" not in result


def test_narrate_falls_back_to_generic_wording_without_a_number():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0)]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary, player_number=None)
    assert result.startswith("The player dribbled")


def test_narrate_mentions_dominant_hand_when_known():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0, dominant_hand="left")]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary)
    assert result.startswith("The player dribbled the ball with the left hand for 2.0s")


def test_narrate_omits_hand_when_unknown():
    segments = [_segment(ActionLabel.DRIBBLING, 0.0, 2.0, dominant_hand=None)]
    summary = {"dribbling": 2.0, "shooting": 0.0, "passing": 0.0, "moving_without_ball": 0.0, "idle": 0.0}
    result = narrate(segments, summary)
    assert result.startswith("The player dribbled the ball for 2.0s")
    assert "hand" not in result


def test_player_identification_note_none_when_no_number_requested():
    assert player_identification_note(None, None) is None


def test_player_identification_note_none_when_match_found():
    assert player_identification_note("23", True) is None


def test_player_identification_note_present_when_no_match():
    note = player_identification_note("23", False)
    assert note is not None
    assert "23" in note
