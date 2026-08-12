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
| Detection | [`hustvl/yolos-tiny`](https://huggingface.co/hustvl/yolos-tiny) | Finds the ball ("sports ball") and players ("person") in each sampled frame (COCO classes). |
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

This is a heuristic, best-effort system, not a validated basketball-specific
classifier — see "Improving accuracy" below for the natural next step
(fine-tuning a dedicated model on labeled basketball footage). Jersey-number
OCR in particular is a hard target on casual phone video (motion blur,
camera angle, curved fabric) — expect `detected_player_number` to often be
`null` rather than a wrong guess, by design (see `JerseyNumberAggregator`'s
voting thresholds).

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

- `POST /analyze` — multipart upload with a `video` field (any format
  OpenCV/ffmpeg can decode). Returns `{ job_id, status }` immediately;
  analysis runs in the background.
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
