"""Identify which detected person matches a user-requested jersey number,
scanning the first few sampled frames before the main per-frame tracking
loop begins.

This is deliberately a lightweight single-pass greedy clustering, not a
full multi-object tracker: across a short window of early frames, every
detected person is OCR'd and matched to the nearest already-seen
"candidate" (by bbox-center proximity, scaled by that detection's own bbox
diagonal) or starts a new one. Whichever candidate accumulates enough OCR
reads matching the requested number is picked, and its most recent box
seeds the main tracker's continuity search - the same nearest-neighbor
matching `_pick_primary_player` already does frame-to-frame, just given a
better starting point than "largest person in frame 0".

The OCR reader itself is injected as a plain callable rather than imported
directly, so this stays testable with synthetic reads and no real model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from app.pipeline.video_utils import bbox_center, bbox_diag, euclidean

# How close (in a detection's own bbox-diagonal units) a new person's center
# must be to a candidate's last-seen center to be treated as the same
# physical person across frames, within this short identification window.
CANDIDATE_MATCH_MAX_DIST = 1.0

# A candidate needs at least this many OCR reads matching the requested
# number before it's trusted enough to seed the tracker.
MIN_MATCHING_READS = 2


class _BoxLike(Protocol):
    box: tuple[float, float, float, float]


ReaderFn = Callable[[tuple[float, float, float, float]], "str | None"]


@dataclass
class _Candidate:
    matching_reads: int = 0
    total_reads: int = 0
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_center: tuple[float, float] = field(default=(0.0, 0.0))


class PlayerIdentifier:
    def __init__(self, target_number: str):
        self.target_number = target_number
        self._candidates: list[_Candidate] = []

    def observe(self, people: list[_BoxLike], read_number: ReaderFn) -> None:
        """Process one frame's worth of detected people. `read_number(box)`
        should return an OCR reading for that person's box, or None."""
        for person in people:
            center = bbox_center(person.box)
            scale = bbox_diag(person.box) or 1.0
            candidate = self._match_candidate(center, scale)
            if candidate is None:
                candidate = _Candidate()
                self._candidates.append(candidate)
            candidate.total_reads += 1
            candidate.last_box = person.box
            candidate.last_center = center
            if read_number(person.box) == self.target_number:
                candidate.matching_reads += 1

    def _match_candidate(
        self, center: tuple[float, float], scale: float
    ) -> _Candidate | None:
        for candidate in self._candidates:
            if euclidean(candidate.last_center, center) / scale <= CANDIDATE_MATCH_MAX_DIST:
                return candidate
        return None

    def best_match_box(self) -> tuple[float, float, float, float] | None:
        matching = [c for c in self._candidates if c.matching_reads >= MIN_MATCHING_READS]
        if not matching:
            return None
        return max(matching, key=lambda c: c.matching_reads).last_box

    def debug_summary(self) -> str:
        return (
            f"{len(self._candidates)} candidates, "
            f"reads={[(c.matching_reads, c.total_reads) for c in self._candidates]}"
        )
