"""Unit tests for the pure-logic parts of scripts/compare_labels.py (CSV
parsing, ground-truth lookup). No models, no video I/O."""
import csv

import pytest

from scripts.compare_labels import _ground_truth_at, _load_labels


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_filename", "segment_start_sec", "segment_end_sec", "label", "dominant_hand", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_labels_groups_segments_by_clip(tmp_path):
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            {"clip_filename": "a.mov", "segment_start_sec": "0.0", "segment_end_sec": "1.5", "label": "dribbling", "dominant_hand": "left", "notes": ""},
            {"clip_filename": "a.mov", "segment_start_sec": "1.5", "segment_end_sec": "2.0", "label": "passing", "dominant_hand": "", "notes": "crossover?"},
            {"clip_filename": "b.mov", "segment_start_sec": "0.0", "segment_end_sec": "3.0", "label": "idle", "dominant_hand": "", "notes": ""},
        ],
    )
    result = _load_labels(csv_path)
    assert set(result.keys()) == {"a.mov", "b.mov"}
    assert result["a.mov"] == [
        {"start": 0.0, "end": 1.5, "label": "dribbling"},
        {"start": 1.5, "end": 2.0, "label": "passing"},
    ]
    assert result["b.mov"] == [{"start": 0.0, "end": 3.0, "label": "idle"}]


def test_load_labels_rejects_unrecognized_label(tmp_path):
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            {"clip_filename": "a.mov", "segment_start_sec": "0.0", "segment_end_sec": "1.0", "label": "dunking", "dominant_hand": "", "notes": ""},
        ],
    )
    with pytest.raises(ValueError, match="Unrecognized label"):
        _load_labels(csv_path)


def test_ground_truth_at_picks_covering_segment():
    segments = [
        {"start": 0.0, "end": 1.5, "label": "dribbling"},
        {"start": 1.5, "end": 2.0, "label": "passing"},
    ]
    assert _ground_truth_at(segments, midpoint=0.7) == "dribbling"
    assert _ground_truth_at(segments, midpoint=1.6) == "passing"


def test_ground_truth_at_returns_unlabeled_outside_any_segment():
    segments = [{"start": 0.0, "end": 1.5, "label": "dribbling"}]
    assert _ground_truth_at(segments, midpoint=5.0) == "unlabeled"


def test_ground_truth_at_boundary_is_inclusive_of_start_exclusive_of_end():
    segments = [{"start": 1.0, "end": 2.0, "label": "shooting"}]
    assert _ground_truth_at(segments, midpoint=1.0) == "shooting"
    assert _ground_truth_at(segments, midpoint=2.0) == "unlabeled"
