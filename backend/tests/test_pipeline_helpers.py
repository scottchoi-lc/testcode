"""Unit tests for pure-logic helpers in app.pipeline.pipeline. No models,
no video I/O - these only need numpy/opencv importable (see README)."""
from app.pipeline.pipeline import _nearest_kinetics_labels


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
