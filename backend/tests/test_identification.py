"""Unit tests for PlayerIdentifier. Pure Python, synthetic detections and a
fake OCR reader - no models, no video, no network."""
from dataclasses import dataclass

from app.pipeline.identification import PlayerIdentifier


@dataclass
class FakeDetection:
    box: tuple[float, float, float, float]


def _reader(readings: dict[tuple[float, float, float, float], str]):
    return lambda box: readings.get(box)


def test_no_match_when_number_never_read():
    identifier = PlayerIdentifier("23")
    frames = [
        [FakeDetection((0, 0, 10, 10)), FakeDetection((50, 0, 60, 10))],
        [FakeDetection((0, 1, 10, 11)), FakeDetection((50, 1, 60, 11))],
    ]
    reader = _reader({})  # nobody ever reads as "23"
    for people in frames:
        identifier.observe(people, reader)
    assert identifier.best_match_box() is None


def test_matches_candidate_with_enough_consistent_reads():
    identifier = PlayerIdentifier("23")
    box_a = (0, 0, 10, 10)
    box_b = (50, 0, 60, 10)
    # Candidate A (near box_a each frame) reads "23" twice; candidate B never does.
    frames = [
        ([FakeDetection(box_a), FakeDetection(box_b)], {box_a: "23", box_b: "7"}),
        ([FakeDetection((1, 1, 11, 11)), FakeDetection((51, 1, 61, 11))], {(1, 1, 11, 11): "23", (51, 1, 61, 11): None}),
    ]
    for people, readings in frames:
        identifier.observe(people, _reader(readings))
    match = identifier.best_match_box()
    assert match is not None
    # The matched box should be candidate A's most recent position.
    assert match == (1, 1, 11, 11)


def test_requires_minimum_matching_reads():
    identifier = PlayerIdentifier("23")
    box = (0, 0, 10, 10)
    # Only one frame, one matching read - below MIN_MATCHING_READS=2.
    identifier.observe([FakeDetection(box)], _reader({box: "23"}))
    assert identifier.best_match_box() is None


def test_picks_candidate_with_most_matching_reads_when_multiple_qualify():
    identifier = PlayerIdentifier("23")
    box_a = (0, 0, 10, 10)
    box_b = (100, 0, 110, 10)
    readings_by_frame = [
        {box_a: "23", box_b: "23"},
        {box_a: "23", box_b: "23"},
        {box_a: "23", box_b: "7"},
    ]
    for readings in readings_by_frame:
        identifier.observe([FakeDetection(box_a), FakeDetection(box_b)], _reader(readings))
    match = identifier.best_match_box()
    assert match == box_a  # 3 matching reads vs candidate B's 2


def test_far_apart_detections_are_treated_as_different_candidates():
    identifier = PlayerIdentifier("23")
    near_a = (0, 0, 10, 10)
    far_away = (1000, 1000, 1010, 1010)
    identifier.observe([FakeDetection(near_a)], _reader({near_a: "23"}))
    identifier.observe([FakeDetection(far_away)], _reader({far_away: "23"}))
    # Two separate single-read candidates, neither reaching MIN_MATCHING_READS.
    assert identifier.best_match_box() is None
    assert len(identifier._candidates) == 2
