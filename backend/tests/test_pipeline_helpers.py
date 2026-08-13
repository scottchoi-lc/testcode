"""Unit tests for pure-logic helpers in app.pipeline.pipeline. No models,
no video I/O - these only need numpy/opencv importable (see README)."""
from app.pipeline.pipeline import _bidirectional_frame_order, _nearest_kinetics_labels


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
