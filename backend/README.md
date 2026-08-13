# Basketball Movement Analyzer — Backend

A FastAPI service that takes a basketball video clip and returns a timeline
of what the tracked player was doing: **dribbling**, **shooting**,
**passing**, or **moving without the ball**.

## How it works

There is no single Hugging Face model that classifies exactly these four
basketball-specific actions, so the pipeline combines several off-the-shelf
models with a small rule-based fusion layer:

| Stage | Model | Purpose |
|---|---|---|
| Detection | [`hustvl/yolos-small`](https://huggingface.co/hustvl/yolos-small) | Finds the ball ("sports ball") and players ("person") in each sampled frame (COCO classes). Upgraded from `yolos-tiny`, which under-detected the ball on real footage (~40% of frames in one test clip) — `yolos-small` trades slower CPU inference for meaningfully better recall on a small, fast-moving object; `yolos-tiny` is a drop-in fallback via `DETECTION_MODEL` if that trade-off doesn't work for you. |
| Pose | [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) | Estimates the tracked player's keypoints (wrists, shoulders, ...) per frame. Optional — the pipeline degrades gracefully if it's unavailable. |
| Action context | [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) | Classifies short clip windows against Kinetics-400, which includes classes like "dribbling basketball" and "shooting basketball". |
| Jersey number | [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) | Best-effort OCR on the tracked player's torso, majority-voted across a sample of frames, to personalize the narrative ("Player #23 dribbled...") when a number is legible. Falls back to generic "the player" wording otherwise — see `app/models/jersey_ocr.py`. |

`app/pipeline/fusion.py` combines ball-to-player distance/trajectory, wrist
height relative to the shoulder, and the Kinetics label into one of the four
target labels per analysis window (`app/pipeline/pipeline.py` orchestrates
the whole thing). Adjacent windows with the same label are merged into
segments with a `start_time`/`end_time`/`confidence`, and
`app/pipeline/narration.py` turns the merged segments into a plain-English
play-by-play (`AnalysisResult.narrative`).

For DRIBBLING segments, the pipeline also calls which hand is doing the
dribbling: each frame it checks which of the pose model's `left_wrist` /
`right_wrist` keypoints is nearest the ball, and the merged segment's
`dominant_hand` is a majority vote of those per-frame calls (`None` if pose
data wasn't available/confident enough). ViTPose's left/right keypoint
labels are anatomical (the *player's* left/right), not image-left/right, so
this is correct regardless of camera angle.

The main possession signal (`fraction_possessed` in `fusion.py`) also uses
this wrist-to-ball distance, not just ball-to-player-bbox-center distance:
an extended-arm dribble or pass keeps the ball away from the torso center
even while a wrist has it in hand, so bbox-center distance alone
under-detects possession in exactly that case. `_effective_ball_distance()`
takes whichever of the two signals is smaller per frame.

`DRIBBLE_POSSESSION_MIN_FRACTION` (0.3, down from an earlier 0.6) is
deliberately less than "the ball must be close most of the time": real
dribbling has the ball in flight, away from the hand, for most of each
bounce cycle, so requiring a frame-majority of "close" readings
systematically under-detects it. This was tuned against one real clip
rather than a validated dataset — if it starts over-calling dribbling on
other footage (e.g. brief incidental ball-near-player contact getting
mislabeled), that threshold is the first place to revisit.

This is a heuristic, best-effort system, not a validated basketball-specific
classifier — see "Improving accuracy" below for the natural next step
(fine-tuning a dedicated model on labeled basketball footage). Jersey-number
OCR in particular is a hard target on casual phone video (motion blur,
camera angle, curved fabric) — expect `detected_player_number` to often be
`null` rather than a wrong guess, by design (see `JerseyNumberAggregator`'s
voting thresholds).

### Narrative granularity

The rule-based scoring in `fusion.py` runs on much finer windows
(`FUSION_WINDOW_SECONDS`, default 0.8s) than the VideoMAE classifier
(`ACTION_WINDOW_FRAMES`/`ACTION_WINDOW_STRIDE`, ~2.5s per inference call).
These are deliberately decoupled: the fine windows only need already-cheap
per-frame ball/pose signals, so `pipeline.py` runs the expensive VideoMAE
inference at its normal (coarser) cadence and has each fine window borrow
the kinetics label from whichever coarse window is temporally closest
(`_nearest_kinetics_labels`). This is what lets quick individual events show
up as their own segments instead of getting smoothed into one long block —
without adding more model inference calls.

### Focusing on a specific player

Rather than guessing (an earlier version of this let you type a jersey
number and tried to OCR-match it — replaced because OCR matching is
unreliable and gives no way to be sure it picked the right person), the app
lets you *tap* the player you want directly on a preview frame. This is a
three-endpoint flow:

1. `POST /videos` — upload the clip once; returns a `video_id` plus probed
   `duration_seconds`/`frame_width`/`frame_height`. The file is kept on
   disk (`app/videos/store.py`'s `VideoStore`) so it doesn't need
   re-uploading for the next two steps.
2. `POST /videos/{video_id}/preview-frame` (form field `timestamp`) —
   extracts the nearest frame at that timestamp (`extract_frame_at` in
   `video_utils.py`), runs the same HF detection model used during
   analysis to find every visible person, and returns the frame as a JPEG
   plus each person's box. The client can call this as many times as it
   wants (e.g. while scrubbing) before committing to a selection.
3. `POST /analyze` — takes `video_id` plus optional `selected_box`
   (JSON-encoded `[x1, y1, x2, y2]`, in the *same pixel space* as the
   preview frame it came from) and `selected_timestamp`. If provided,
   `AnalysisResult.player_selected` is `true` and tracking is seeded from
   that exact detection - no ambiguity, since it's a confirmed tap rather
   than an inferred match.

Because the tap can happen at *any* point in the clip, tracking can't just
seed frame 0 and walk forward - the player may have moved a lot between
t=0 and the selected timestamp. Instead `run_pipeline` finds the sampled
frame nearest `selected_timestamp` and processes *bidirectionally* from
there: forward to the end of the clip, then backward to the start,
resetting to the exact selected position at the start of each direction
(`_bidirectional_frame_order`, `_process_index` in `pipeline.py`). With no
selection, frames are processed in the normal 0..N order seeded by the
default heuristic (largest person in frame 0), same as before this
feature existed.

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first request that hits `/analyze` will download the four model
checkpoints from Hugging Face (several GB total) and cache them under the
default `~/.cache/huggingface`. Set `DEVICE=cuda` (see `app/config.py`) if
you have a GPU available — inference on CPU works but is slow for anything
beyond short clips.

## API

- `POST /videos` — multipart upload with a `video` field (any format
  OpenCV/ffmpeg can decode). Returns `UploadVideoResponse`
  (`video_id`/`duration_seconds`/`frame_width`/`frame_height`).
- `POST /videos/{video_id}/preview-frame` — form field `timestamp` (seconds).
  Returns `PreviewFrameResponse`: the nearest frame as a base64 JPEG plus
  every detected person's box, for the tap-to-select-a-player UI.
- `POST /analyze` — form fields `video_id` (required), and optionally
  `selected_box` (JSON `[x1, y1, x2, y2]` from a preview-frame response)
  and `selected_timestamp` — see "Focusing on a specific player" above.
  Returns `{ job_id, status }` immediately; analysis runs in the background.
- `GET /jobs/{job_id}` — poll for `{ status, progress, result, error }`.
  `status` is one of `queued | processing | done | failed`. `result` is
  populated once `status == "done"` and matches `AnalysisResult` in
  `app/schemas.py`.
- `GET /health` — liveness check.

## Configuration

All tunables live in `app/config.py` and can be overridden via environment
variables, e.g.:

```bash
ANALYSIS_FPS=8 ACTION_WINDOW_FRAMES=16 DEVICE=cuda uvicorn app.main:app
```

## Tests

```bash
pip install -r requirements.txt
pytest
```

`tests/test_fusion.py` covers the rule-based fusion logic directly with
synthetic signals — no models, no network. `tests/test_api_smoke.py`
exercises the FastAPI app with `run_pipeline` monkeypatched out, so the full
test suite runs in well under a second with no GPU or model downloads
required. (`numpy`/`opencv-python-headless`/`Pillow` are imported at module
load time for frame handling and are needed to run the suite; `torch`/
`transformers` are imported lazily inside the model wrapper classes and are
only required for real inference.)

## Improving accuracy

The rule-based fusion layer is a reasonable starting point but is not a
substitute for a model trained specifically on this task. The natural next
step is to collect labeled basketball clips (per-frame or per-segment
dribble/shoot/pass/move labels) and fine-tune the VideoMAE (or a similar
video transformer) classification head directly on those four classes,
replacing or augmenting `fusion.py`'s heuristics.
