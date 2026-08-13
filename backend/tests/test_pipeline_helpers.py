"""Unit tests for pure-logic helpers in app.pipeline.pipeline. No models,
no video I/O - these only need numpy/opencv importable (see README)."""
import numpy as np

from app.models.detection import Detection
from app.pipeline.pipeline import (
    _bidirectional_frame_order,
    _color_histogram,
    _nearest_kinetics_labels,
    _pick_primary_player,
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


def _person(box, score=0.9):
    return Detection(label="person", score=score, box=box)


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
