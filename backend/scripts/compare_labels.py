"""Compare the pipeline's per-window predictions against hand-labeled
ground truth for a clip, using the exact production pipeline (same
tracking/windowing/scoring as a real /analyze call) via run_pipeline's
`debug_scored_windows` hook - no separate/duplicated logic to keep in sync
with fusion.py as it evolves.

Usage:
    python scripts/compare_labels.py --csv labels.csv --clips-dir ./clips
    python scripts/compare_labels.py --csv labels.csv --clips-dir ./clips \
        --clip my_clip.mov --selected-box "[120,80,340,420]" --selected-timestamp 2.1

The CSV format matches the labeling spreadsheet template: one row per
labeled segment, columns clip_filename,segment_start_sec,segment_end_sec,
label,dominant_hand,notes. `label` must be one of the ActionLabel values
(dribbling/shooting/passing/receiving/moving_without_ball/idle) -
`dominant_hand` and `notes` are read but not used by this script yet.

--selected-box/--selected-timestamp seed player tracking the same way
tapping a player in the app does (same [x1,y1,x2,y2]/seconds format the
API takes) - only meaningful with --clip, since they apply to one clip at
a time. Without them, the clip is analyzed with the default "largest
person in frame 0" heuristic, which only produces a meaningful comparison
if that heuristic happens to track the same player you labeled.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.pipeline import run_pipeline  # noqa: E402
from app.schemas import ActionLabel  # noqa: E402

VALID_LABELS = {label.value for label in ActionLabel}


def _load_labels(csv_path: Path) -> dict[str, list[dict]]:
    by_clip: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unrecognized label {label!r} for clip {row['clip_filename']!r} "
                    f"(must be one of {sorted(VALID_LABELS)})"
                )
            by_clip[row["clip_filename"].strip()].append(
                {
                    "start": float(row["segment_start_sec"]),
                    "end": float(row["segment_end_sec"]),
                    "label": label,
                }
            )
    return dict(by_clip)


def _ground_truth_at(segments: list[dict], midpoint: float) -> str:
    """Which labeled segment (if any) covers a window's midpoint. Assumes
    non-overlapping segments, as produced by the labeling spreadsheet
    workflow - the first match wins if that's ever violated."""
    for seg in segments:
        if seg["start"] <= midpoint < seg["end"]:
            return seg["label"]
    return "unlabeled"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", required=True, type=Path, help="Labeling spreadsheet (CSV)")
    parser.add_argument(
        "--clips-dir", required=True, type=Path, help="Directory containing the video files"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("label_comparison.csv"),
        help="Where to write the per-window comparison report (default: label_comparison.csv)",
    )
    parser.add_argument("--clip", help="Only process this one clip_filename from the CSV")
    parser.add_argument("--selected-box", help="JSON [x1,y1,x2,y2] - requires --clip")
    parser.add_argument(
        "--selected-timestamp", type=float, help="Seconds the box was taken at - requires --clip"
    )
    args = parser.parse_args()

    if bool(args.selected_box) != (args.selected_timestamp is not None):
        parser.error("--selected-box and --selected-timestamp must be provided together")
    if (args.selected_box or args.selected_timestamp is not None) and not args.clip:
        parser.error("--selected-box/--selected-timestamp require --clip (they apply to one clip)")

    labels_by_clip = _load_labels(args.csv)
    if args.clip:
        if args.clip not in labels_by_clip:
            parser.error(f"{args.clip!r} not found in {args.csv} (clips present: {sorted(labels_by_clip)})")
        labels_by_clip = {args.clip: labels_by_clip[args.clip]}

    selected_box = tuple(json.loads(args.selected_box)) if args.selected_box else None

    rows = []
    total = 0
    agree = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for clip_filename, segments in labels_by_clip.items():
        video_path = args.clips_dir / clip_filename
        if not video_path.exists():
            print(f"WARNING: {video_path} not found, skipping", file=sys.stderr)
            continue

        print(f"Processing {clip_filename}...")
        scored_windows: list = []
        run_pipeline(
            str(video_path),
            selected_player_box=selected_box,
            selected_timestamp=args.selected_timestamp,
            debug_scored_windows=scored_windows,
        )

        for window, scored in scored_windows:
            midpoint = (window.start_time + window.end_time) / 2
            truth = _ground_truth_at(segments, midpoint)
            predicted = scored.label.value
            if truth != "unlabeled":
                total += 1
                agree += int(truth == predicted)
                confusion[(truth, predicted)] += 1
            rows.append(
                {
                    "clip": clip_filename,
                    "window_start": round(window.start_time, 2),
                    "window_end": round(window.end_time, 2),
                    "ground_truth": truth,
                    "predicted": predicted,
                    "confidence": round(scored.confidence, 2),
                    "agree": (truth == predicted) if truth != "unlabeled" else "",
                    "evidence": json.dumps(scored.evidence),
                }
            )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "window_start",
                "window_end",
                "ground_truth",
                "predicted",
                "confidence",
                "agree",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} window(s) to {args.out}")
    if total:
        print(f"\nAgreement on labeled windows: {agree}/{total} ({100 * agree / total:.0f}%)")
        print("\nConfusion (ground_truth -> predicted): count")
        for (truth, predicted), count in sorted(confusion.items()):
            marker = "" if truth == predicted else "  <-- mismatch"
            print(f"  {truth:>20} -> {predicted:<20} {count}{marker}")
    else:
        print(
            "\nNo windows fell inside a labeled segment - check your CSV's timestamps "
            "against the clip (they might not overlap the analyzed portion)."
        )


if __name__ == "__main__":
    main()
