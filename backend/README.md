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

Frame-to-frame continuity (`_pick_primary_player`) is otherwise nearest-
neighbor by position, with two layers of protection against tracking the
wrong physical person - both found from a real clip's logs, in order:

1. **`MAX_PLAUSIBLE_TRACKING_JUMP`** rejects a "nearest" match that's
   implausibly far from the last known position (occlusion, a missed
   detection, a fast cut), treating that frame as "no detection" (position
   stays frozen) instead of snapping onto someone else. This alone turned
   out to be insufficient: a real clip's `Segments:` log showed a narrative
   claiming one player passed the ball and then immediately took a shot -
   impossible in basketball, and a tell that tracking had drifted onto a
   different person - yet `tracking_debug_stats` showed
   `implausible_jumps_rejected: 0`. The drift was *gradual*: several
   individually-small per-frame steps (each under the cap) that added up
   to a large net displacement across a window, most likely because two
   players were near each other and position alone couldn't tell them
   apart.
2. **Appearance verification**: once a reference appearance exists (an HSV
   color histogram, `_color_histogram`/`_appearance_similarity`, captured
   once from whichever frame first successfully tracks the player - the
   exact tapped frame when a player was selected), every plausible-by-
   position candidate is checked against it on *every* frame, not just
   when there's an ambiguous tie between multiple detections. That
   distinction mattered: a tie-breaker that only activates when there's a
   tie never gets a chance to catch a *lone* wrong-person detection, which
   is exactly what later frames tend to have once tracking has already
   drifted - by then there's usually only one nearby candidate (the wrong
   person), not an ambiguous pair. `MIN_APPEARANCE_SIMILARITY` is the
   minimum match required to accept a candidate as a continuation; below
   it, the frame is treated as "no detection" (same as an implausible
   jump) rather than accepted.

Both layers are cheap, local checks (a few dozen pixels' worth of jersey/
skin color, not a trained re-identification model) tuned against one real
clip's evidence rather than a validated dataset - `tracking_debug_stats`
(logged per run: `implausible_jumps_rejected`, `max_jump_seen`,
`ambiguous_frames_disambiguated_by_appearance`,
`appearance_mismatches_rejected`) is there to check whether either needs
retuning on other footage. Known limitations: two players in matching
uniforms won't be distinguishable by appearance at all, and checking
appearance on every frame (rather than only when ambiguous) trades some
risk of *over*-rejecting the correct player under a real lighting/angle
change partway through a clip for the ability to actually catch a drifted
identity - if `frames_with_player` (in the "Dribbling-hand debug" log)
drops noticeably compared to `appearance_mismatches_rejected` being high,
that trade is biting and `MIN_APPEARANCE_SIMILARITY` is the first knob to
loosen.

### Kinetics score as corroboration, not an override

An earlier version of `fusion.py` treated *any* nonzero Kinetics score for
"dribbling basketball"/"shooting basketball" (`dribble_boost`/`shoot_boost`)
as a hard bypass for the branch's real evidence checks - e.g. the shooting
branch's `fraction_wrist_high >= 0.3 or shoot_boost > 0`. That looks like a
threshold but isn't one: VideoMAE's softmax spreads a sliver of probability
across most of its 400 Kinetics classes, so a window with zero real shooting
evidence (`fraction_wrist_high: 0.0`, `ball_released: False`) could still
carry `kinetics_shoot_score: 0.088` - noise, not a real "this is a shot"
signal - and get labeled SHOOTING purely from that. This was diagnosed from
a real clip's `Window` logs showing exactly that combination, and explained
a narrative that had a player pass then immediately shoot - both windows had
near-zero direct evidence but a nonzero Kinetics score. `KINETICS_OVERRIDE_MIN`
(0.3, matching the existing `fraction_wrist_high >= 0.3` bar) is the fix: a
Kinetics-only override now has to be as convincing as the direct-evidence
threshold it's standing in for, not merely nonzero. A *strong* Kinetics score
can still corroborate a call the direct checks alone wouldn't quite make -
the fix narrows the override, it doesn't remove it.

### Overlapping fusion windows (catching releases at a window boundary)

Fusion windows used to tile the clip back-to-back with no overlap
(`frames[start_idx:end_idx]`, stride == window size). That's fine as long as
a whole event fits inside one window, but a real clip's logs showed it
silently swallowing a pass: one window still had the ball
(`fraction_possessed: 0.6`, dribbling), and the very next window had lost it
entirely (`fraction_possessed: 0.0`) - the release itself happened right at
the boundary between them. Neither window ever saw the "starts with the
ball, ends without it" shape that the shooting/passing branches in
`fusion.py` require within a *single* window, so the pass just fell through
to `dribbling` followed by `moving_without_ball`, with no PASSING segment at
all.

`FUSION_WINDOW_STRIDE_SECONDS` (0.4s, half of the default 0.8s
`FUSION_WINDOW_SECONDS`) fixes this by making consecutive fusion windows
overlap, the same way `ACTION_WINDOW_STRIDE < ACTION_WINDOW_FRAMES` already
makes the coarser Kinetics windows overlap. `_fusion_window_bounds` in
`pipeline.py` builds the (start, end) frame ranges; with 50% overlap, any
transition that happens anywhere in the clip is guaranteed to be fully
contained in at least one window, not split across two. `merge_adjacent_segments`
already handled overlapping windows correctly (it was written for the
Kinetics case), so no change was needed there beyond updating its docstring.

This does **not** fix every missed pass by itself, and it's worth being
explicit about the difference: it fixes windowing *cutting a visible
transition in half*. It cannot manufacture a ball position that was never
detected in the first place - if the ball drops out of detection entirely
during the release (fast motion blur, occlusion, leaving frame) rather than
being detected at a "far" position at least once, there's still no evidence
for `score_window` to reason about, no matter how the windows are drawn.
`ball_frames_detected` was added to the IDLE and MOVING_WITHOUT_BALL evidence
dicts specifically so this can be told apart in the logs from a genuinely-far
reading (`fraction_possessed: 0.0` with `ball_frames_detected: 0` means total
dropout, not "detected but far") - if that shows up around a missed
event, the next step is improving ball-detection recall during fast motion
rather than further windowing changes.

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
