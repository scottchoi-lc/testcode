"""Save a frame from a clip with every detected person's box drawn and
numbered, so you can visually pick out which one is the player you want to
track - then pass that box's coordinates to compare_labels.py's
--selected-box, the same way tapping a player in the app seeds tracking.

Exists because compare_labels.py's --selected-box needs plain pixel
coordinates, and getting those normally means going through the mobile
app's tap-to-select screen - this gets you the same information from the
command line, using the exact same detection model/thresholds the real
pipeline uses, so the box you pick is exactly as valid as one the app
would have produced.

Usage:
    python scripts/preview_frame.py --video path/to/clip.mov --timestamp 2.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.detection import get_detection_model  # noqa: E402
from app.pipeline.video_utils import extract_frame_at  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to the video clip")
    parser.add_argument(
        "--timestamp", required=True, type=float, help="Seconds into the clip to preview"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("preview.jpg"), help="Where to save the annotated frame"
    )
    args = parser.parse_args()

    frame = extract_frame_at(str(args.video), args.timestamp)
    _, people = get_detection_model().detect_ball_and_players(frame)

    if not people:
        print(
            "No people detected at this timestamp - try a nearby moment where "
            "the player is clearly visible."
        )
        return

    annotated = frame.copy()
    print(f"{len(people)} people detected at t={args.timestamp}s:\n")
    for i, person in enumerate(people):
        x1, y1, x2, y2 = (round(v) for v in person.box)
        print(f"  [{i}] box=[{x1}, {y1}, {x2}, {y2}]  score={person.score:.2f}")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(
            annotated,
            str(i),
            (x1 + 4, max(20, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
        )

    cv2.imwrite(str(args.out), annotated)
    print(f"\nSaved {args.out} - open it, find the number on the player you want, then pass")
    print("that number's [x1,y1,x2,y2] from the list above to compare_labels.py, e.g.:")
    print(
        f'  --selected-box "[<that box>]" --selected-timestamp {args.timestamp}'
    )


if __name__ == "__main__":
    main()
