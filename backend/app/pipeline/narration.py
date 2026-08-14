"""Turns a list of merged action segments into a plain-English play-by-play.

Pure string formatting over already-computed segments/summary - no ML, no
I/O - so it's cheap to unit test with synthetic segments the same way
`fusion.py` is.
"""
from __future__ import annotations

from app.schemas import ActionLabel, ActionSegment

# Idle stretches shorter than this are treated as noise between real events
# and left out of the narrative rather than called out on their own.
MIN_IDLE_SECONDS_TO_MENTION = 1.0

_VERB_PHRASES = {
    ActionLabel.DRIBBLING: "dribbled the ball",
    ActionLabel.SHOOTING: "took a shot",
    ActionLabel.PASSING: "passed the ball",
    ActionLabel.RECEIVING: "received the ball",
    ActionLabel.MOVING_WITHOUT_BALL: "moved without the ball",
    ActionLabel.IDLE: "paused",
}

_NOUN_PHRASES = {
    ActionLabel.DRIBBLING: "dribbling",
    ActionLabel.SHOOTING: "shooting",
    ActionLabel.PASSING: "passing",
    ActionLabel.RECEIVING: "receiving",
    ActionLabel.MOVING_WITHOUT_BALL: "moving without the ball",
    ActionLabel.IDLE: "paused",
}

NO_ACTIONS_MESSAGE = "No clear basketball actions were detected in this clip."


def _should_mention(segment: ActionSegment) -> bool:
    if segment.label != ActionLabel.IDLE:
        return True
    return (segment.end_time - segment.start_time) >= MIN_IDLE_SECONDS_TO_MENTION


def _verb_phrase(segment: ActionSegment) -> str:
    if segment.label == ActionLabel.DRIBBLING and segment.dominant_hand:
        return f"dribbled the ball with the {segment.dominant_hand} hand"
    return _VERB_PHRASES[segment.label]


def _play_by_play(events: list[ActionSegment], subject: str) -> str:
    clauses = []
    for i, segment in enumerate(events):
        duration = segment.end_time - segment.start_time
        verb = _verb_phrase(segment)
        prefix = subject if i == 0 else "then"
        clauses.append(f"{prefix} {verb} for {duration:.1f}s")
    return ", ".join(clauses) + "."


def _totals_summary(summary: dict[str, float]) -> str:
    ordered = sorted(
        (
            (label, seconds)
            for label, seconds in summary.items()
            if label != ActionLabel.IDLE.value and seconds > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ordered:
        return ""
    parts = [f"{seconds:.1f}s {_NOUN_PHRASES[ActionLabel(label)]}" for label, seconds in ordered]
    return "Overall: " + ", ".join(parts) + "."


def narrate(
    segments: list[ActionSegment],
    summary: dict[str, float],
    player_number: str | None = None,
) -> str:
    events = [s for s in segments if _should_mention(s)]
    if not events:
        return NO_ACTIONS_MESSAGE

    subject = f"Player #{player_number}" if player_number else "The player"
    narrative = _play_by_play(events, subject)
    totals = _totals_summary(summary)
    if totals:
        narrative = f"{narrative} {totals}"
    return narrative
