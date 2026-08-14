# Basketball Movement Analyzer — Backend

A FastAPI service that takes a basketball video clip and returns a timeline
of what the tracked player was doing: **dribbling**, **shooting**,
**passing**, **receiving**, or **moving without the ball**.

## How it works

There is no single Hugging Face model that classifies exactly these five
basketball-specific actions, so the pipeline combines several off-the-shelf
models with a small rule-based fusion layer:

| Stage | Model | Purpose |
|---|---|---|
| Detection | [`hustvl/yolos-small`](https://huggingface.co/hustvl/yolos-small) | Finds the ball ("sports ball") and players ("person") in each sampled frame (COCO classes). Upgraded from `yolos-tiny`, which under-detected the ball on real footage (~40% of frames in one test clip) — `yolos-small` trades slower CPU inference for meaningfully better recall on a small, fast-moving object; `yolos-tiny` is a drop-in fallback via `DETECTION_MODEL` if that trade-off doesn't work for you. |
| Pose | [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) | Estimates the tracked player's keypoints (wrists, shoulders, ...) per frame. Optional — the pipeline degrades gracefully if it's unavailable. |
| Action context | [`microsoft/xclip-base-patch32`](https://huggingface.co/microsoft/xclip-base-patch32) | X-CLIP: scores short clip windows via zero-shot video-text similarity against `settings.ACTION_CANDIDATE_LABELS` (e.g. "dribbling a basketball", "passing a basketball to a teammate"). Swapped in for `MCG-NJU/videomae-base-finetuned-kinetics` — see "Why X-CLIP, not VideoMAE-Kinetics" below. |
| Jersey number | [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) | Best-effort OCR on the tracked player's torso, majority-voted across a sample of frames, to personalize the narrative ("Number 23 dribbled...") when a number is legible. Falls back to "Player 1" otherwise — see `app/models/jersey_ocr.py`. |

`app/pipeline/fusion.py` combines ball-to-player distance/trajectory, wrist
height relative to the shoulder, and the action-classifier label scores into
one of the five target labels per analysis window (`app/pipeline/pipeline.py`
orchestrates the whole thing). Adjacent windows with the same label are
merged into segments with a `start_time`/`end_time`/`confidence`, and
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

### Why X-CLIP, not VideoMAE-Kinetics

The action-context model was originally `MCG-NJU/videomae-base-finetuned-
kinetics`, fine-tuned on Kinetics-400. Two problems with keeping it:

1. **License.** VideoMAE-Kinetics is CC-BY-NC-4.0 — non-commercial only,
   and that restriction extends to fine-tuned derivatives of it too, not
   just the checkpoint as distributed. That's a real blocker if this app
   is ever meant to be a commercial product.
2. **Fit.** Kinetics-400's 400 classes don't include a "passing basketball"
   concept at all, which is part of why `fusion.py`'s PASSING branch has
   historically had to guess entirely from ball-trajectory heuristics
   rather than any classifier corroboration (see its own history of
   passing/crossover-dribble mixups).

`microsoft/xclip-base-patch32` (X-CLIP, MIT licensed) fixes both: it's a
CLIP-style model trained contrastively on (video, text) pairs, so instead of
reading from a fixed classification head it scores zero-shot similarity
against whatever candidate phrases we hand it
(`settings.ACTION_CANDIDATE_LABELS`) — including a real
`"passing a basketball to a teammate"` candidate, wired into `fusion.py`'s
PASSING branch as `pass_boost` the same way `dribble_boost`/`shoot_boost`
already corroborate their branches.

One thing this swap leaves **unverified**: `KINETICS_OVERRIDE_MIN` (0.3,
fusion.py) was tuned against Kinetics-400's ~400-way softmax, where an
unrelated class sits near a ~0.0025 noise floor — so even a small nonzero
score was already meaningfully above chance. X-CLIP's softmax here runs
over only `len(ACTION_CANDIDATE_LABELS)` candidates (6 by default), so
chance-level is much higher (~0.17), and 0.3 may no longer be a meaningful
bar. This wasn't re-tuned blind — it needs a real clip's logged
`kinetics_*_score` values under the new model first, the same evidence-
driven approach every other threshold in this file has been tuned with.
If dribbling/shooting/passing calls look too eager to trust weak classifier
corroboration after this change, that constant is the first thing to
re-check against fresh logs.

### Narrative granularity

The rule-based scoring in `fusion.py` runs on much finer windows
(`FUSION_WINDOW_SECONDS`, default 0.8s) than the coarse action classifier
(`ACTION_WINDOW_FRAMES`/`ACTION_WINDOW_STRIDE`, ~2.5s per inference call).
These are deliberately decoupled: the fine windows only need already-cheap
per-frame ball/pose signals, so `pipeline.py` runs the expensive classifier
inference at its normal (coarser) cadence and has each fine window borrow
the label scores from whichever coarse window is temporally closest
(`_nearest_kinetics_labels` — name is a holdover from the VideoMAE-Kinetics
days, see "Why X-CLIP" above). This is what lets quick individual events
show up as their own segments instead of getting smoothed into one long
block — without adding more model inference calls.

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

(This section describes the bug and fix as diagnosed against the original
VideoMAE-Kinetics model. The classifier has since been swapped to X-CLIP -
see "Why X-CLIP, not VideoMAE-Kinetics" above - which changes the softmax's
noise floor enough that `KINETICS_OVERRIDE_MIN`'s calibration is flagged as
unverified there; the *shape* of this fix - a real threshold instead of a
bare `> 0` - still holds regardless of which model produces the score.)

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

### Interpolating brief ball-detection gaps

Even after fusion windows overlap (above), a real clip's pass was still
missed - because the ball wasn't detected in *any* frame of the window
covering the actual release (`ball_frames_detected: 0`, added as a
diagnostic field for exactly this). No amount of window-boundary
adjustment can fix that: there's simply no position data for that stretch
for `score_window` to reason about. This is a detection-recall gap, not a
windowing or fusion-logic one - the ball is the least reliable detection
in the pipeline (small, fast, and most prone to motion blur exactly when
it's flying across the frame during a release).

`_interpolate_ball_gaps` in `pipeline.py` runs once after per-frame
tracking finishes, over the whole clip's frame signals: for any run of up
to `BALL_GAP_INTERPOLATION_MAX_FRAMES` (default 8, ~1.3s at
`ANALYSIS_FPS=6`) consecutive frames where the player was tracked but the
ball wasn't detected, it linearly interpolates the ball's position (and
wrist distance, if available) from the nearest real detection immediately
before and after the gap. Deliberately conservative: it only fills a gap
that has a real ball reading on *both* sides (never extrapolates past the
start/end of a clip or a longer genuine absence), and only when the player
itself was tracked continuously through the gap (a tracking loss isn't
papered over with an invented ball position for an unknown player
location). Longer gaps - the ball genuinely leaving the frame, sustained
occlusion - are left alone rather than fabricating a multi-second
trajectory.

The bound started at 2 frames but wasn't enough: a real clip's pass still
went completely undetected at that setting, confirmed (not guessed) from
the log line described below rather than inferred from the segments alone.
`_interpolate_ball_gaps` returns (and `"Ball gap interpolation: %d frame(s)
filled (max_gap=%d), longest gap seen=%d frames"` logs) the *actual*
longest gap encountered in the clip, whether or not it was short enough to
fill - this is what makes the bound checkable against real footage
instead of a guess: "0 filled" alone can't tell a 1-frame-too-long miss
from a 20-frame one, but `longest_ball_gap_frames` can.

Raised to 4, then to 8 - both evidence-motivated from that log line, not
re-guesses. The second raise came after ruling out "wrong player tracked"
as the cause (a separate bug, fixed by seeding tracking from a tapped
player rather than the default heuristic - see "Focusing on a specific
player" above): with the *correct* player confirmed tracked, the same
clip's actual release still logged `longest gap seen=8 frames`. That's
real ball detection dropout, not a tracking artifact - consistent with the
mechanical expectation that a release is exactly when the ball moves
fastest, and therefore exactly when motion blur is most likely to defeat
the detector.

8 frames (~1.3s) is a real stretch to trust a straight-line interpolation
for - if `longest_ball_gap_frames` still exceeds this bound on a future
clip, that's the signal to stop raising it and address detection recall
directly instead (e.g. a lower `BALL_SCORE_THRESHOLD`, at the cost of more
false-positive ball hits), rather than keep widening a window that's
already covering more of the ball's real trajectory than a straight line
can respect.

### A short pass never counted as "released"

Even with full ball-position data (interpolation covering the whole gap,
no more dropout), a confirmed pass in a real clip still fell through to
IDLE. `ball_distances` - added to IDLE's evidence dict specifically to
debug this, since fraction_possessed alone can't show *where* in a window
the ball crossed either threshold - showed why:
`[0.11, 1.0, 1.07, 1.09, 1.1]`. The ball leaves the player's hand cleanly
(0.11 -> 1.0 in one frame) but only ever reaches about 1.1-1.21 across the
whole sequence, never anywhere near `BALL_RELEASE_MIN_DIST` (1.6 at the
time) - the threshold PASSING/SHOOTING require to call something
"released." That value was set early in this project without a real short
pass to check it against; it was simply calibrated for a longer, more
dramatic release than a short pass to a nearby teammate produces.

Lowered to 1.0 - comfortably above `BALL_POSSESSION_MAX_DIST` (0.9, so
"released" still means something distinct from possession-threshold
noise) and comfortably below this real pass's observed range, not a
guess. Affects SHOOTING too (shares the same constant) - no real evidence
yet either way on shots specifically, worth watching for regressions
there (a shot mislabeled from a not-fully-released ball) the same
evidence-first way everything else in this file gets tuned.

### Telling a crossover dribble apart from a pass

The PASSING branch calls a window a pass when the ball is released, moves
laterally more than vertically, and the wrist never raises (no shooting
motion). A crossover or hesitation dribble - the ball swinging from one
hand across the body to the other - has exactly that same signature: it
reads as "released" because the ball moves well away from the player's
bbox center, it's almost entirely lateral, and there's no wrist raise. A
real clip confirmed this: a crossover got scored as a pass, with
`ball_distances` climbing smoothly from 0.19 to 1.95 over the window -
genuinely released-looking - and then the ball back in close possession
(`fraction_possessed: 0.6`) in the very next fusion window, because it
never actually left the player's hand.

That "comes right back" behavior is the actual tell, so
`_ball_returns_to_possession_soon` (`pipeline.py`) checks the frames just
past a window's end - `PASSING_RETURN_CHECK_SECONDS` (0.5s) - for the ball
being back within `BALL_POSSESSION_MAX_DIST` of the same tracked player. A
real pass to a teammate doesn't boomerang back to the passer that fast; a
crossover's does. This needs context beyond a single window's frame slice
(`score_window` only sees its own `WindowSignals`), so it's computed in
`pipeline.py` from the full per-clip frame signal array and passed in as
`WindowSignals.ball_returns_to_possession_soon` - when true, the PASSING
branch doesn't fire and the window falls through to whatever else the
evidence supports (typically DRIBBLING, correctly, for a crossover).

Known trade-off: a genuinely fast give-and-go (pass out, immediate pass
back) would look the same as a crossover under this check and could get
misread as a retained-possession move instead of two real passes. That's
accepted as the less common case on casual footage - if it turns out to
matter, `PASSING_RETURN_CHECK_SECONDS` is the first knob to shorten.

### Receiving

RECEIVING is the mirror image of passing/shooting: the player did *not*
have the ball at the start of a window but does by the end. Detected the
same way passing is, just inverted - a window that starts with the ball
farther than `BALL_POSSESSION_MAX_DIST` away and ends with it close
(`distances[0] > BALL_POSSESSION_MAX_DIST`, `distances[-1] <=
BALL_POSSESSION_MAX_DIST`) is a reception, optionally corroborated by the
action classifier's `"catching or receiving a basketball pass"` candidate
the same `KINETICS_OVERRIDE_MIN` way every other branch here works.

The ball-distance signal alone can't tell a caught pass apart from picking
up a loose ball or grabbing a rebound - both look identical as "the ball
went from far to close for this player" - so the narrative deliberately
uses neutral wording ("received the ball") rather than presuming a
teammate threw it.

This shipped without the kind of refinement PASSING went through this
session (the crossover-dribble veto, the ball-gap interpolation) because
there's no real-clip evidence yet of what actually breaks it. The most
likely analogous failure, by symmetry with passing's `ball_returns_to_
possession_soon` veto, would be a false positive from the ball merely
rolling or bouncing past the player without them actually gaining control
- if that shows up, a mirrored "stays possessed afterward" check
(confirming the ball doesn't immediately leave again) would be the fix,
following the same evidence-first pattern as everything else in this
file - not added speculatively ahead of a real failure.

### Mentioning other players in a pass/reception

The pipeline only ever *tracks* one player - the one selected (or the
default heuristic's pick) - so it has no persistent notion of who anyone
else in the frame is, and can't say "passed to Player 2" and have that
number mean the same physical person the next time they show up. Building
that would mean tracking every visible player continuously, each with the
same jump-rejection/appearance-verification machinery the primary player
already gets - a substantially bigger, riskier lift than what's here now.

What's here instead is best-effort and moment-only:
`app/models/detection.py`'s object detector already finds every person in
each sampled frame, not just the tracked one - `other_people` (everyone
except the tracked player) gets normalized into
`FrameSignals.other_people_centers` the same way the ball/player positions
are (`_build_frame_signals` in `pipeline.py`). When PASSING/RECEIVING
fires, `_has_other_player_near_ball` checks whether anyone else was within
`BALL_POSSESSION_MAX_DIST` of the ball at the relevant single frame - the
last frame it was seen for a pass (where it ended up), the first frame for
a reception (where it came from) - and sets `nearby_other_player` in the
evidence dict. `narration.py` uses that to say "passed the ball to a
nearby player" / "received the ball from a nearby player" instead of the
plain phrasing.

Deliberately worded to not claim an identity: "a nearby player," not
"Player 2." Two different PASSING segments both flagging
`nearby_other_player: True` are not necessarily the same physical person -
there's no tracking connecting them. If that turns out to matter enough to
be worth the cost, full multi-player tracking (described above) is the
real fix, not a bigger version of this heuristic.

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
substitute for a model trained specifically on this task - it improves by
hand-deriving a new threshold or heuristic each time real footage exposes a
gap, which doesn't scale to basketball's full variety of movement (see
git history for several rounds of exactly this). Two paths forward, in
order of effort:

1. **Train a small classifier on the signals `fusion.py` already computes**
   (`fraction_possessed`, wrist height, `kinetics_*_score`, ball trajectory,
   `ball_returns_to_possession_soon`, ...) instead of hand-picked thresholds
   over them. Needs comparatively little labeled data - a few dozen labeled
   clips, segment-level ground truth mapped onto fusion windows - and no
   GPU to train (e.g. scikit-learn logistic regression / gradient-boosted
   trees). The lowest-effort real fix to the "every edge case needs a new
   manual threshold" problem.
2. **Fine-tune the action classifier itself** (X-CLIP, or a similar video
   model - see "Why X-CLIP, not VideoMAE-Kinetics" above for why the base
   model matters here) directly on labeled basketball footage. The
   stronger long-term answer, but needs substantially more labeled data
   (hundreds to thousands of clips) and training infrastructure than (1).

Either way, keep the license of whatever base model gets fine-tuned in mind
- a restrictively-licensed base (like the CC-BY-NC-4.0 VideoMAE-Kinetics
checkpoint this project moved away from) constrains what you can do with
the fine-tuned result too, regardless of how the fine-tuning data itself is
licensed.

### Checking labeled clips against the current pipeline

Before investing in either path above, `scripts/compare_labels.py` runs
hand-labeled ground truth against the *current* rule-based pipeline, using
`run_pipeline`'s `debug_scored_windows` hook to get the raw per-window
predictions (not just the merged/narrated output) - the same
tracking/windowing/scoring a real `/analyze` call uses, no duplicated
logic to drift out of sync.

```bash
cd backend
python scripts/compare_labels.py --csv labels.csv --clips-dir ./clips
```

`labels.csv` uses the same format as the labeling spreadsheet: one row per
labeled segment (`clip_filename,segment_start_sec,segment_end_sec,label,
dominant_hand,notes`), `label` matching `ActionLabel`'s values exactly.
Writes a per-window comparison (`label_comparison.csv` by default:
ground truth vs. predicted vs. confidence vs. raw evidence) and prints an
agreement rate plus a confusion breakdown.

If the clip was analyzed in the app with a tapped player selection rather
than the default heuristic, pass the same box/timestamp so the comparison
tracks the same person you labeled: `--clip my_clip.mov --selected-box
"[120,80,340,420]" --selected-timestamp 2.1` (same `[x1,y1,x2,y2]`/seconds
format the API takes - `--selected-box`/`--selected-timestamp` only apply
with `--clip`, since they're specific to one clip at a time). Without a
matching selection, a mismatch might mean the heuristic tracked a
different player than the one you labeled, not that the fusion logic is
wrong - worth ruling out before reading too much into a low agreement
rate. This isn't hypothetical: a real run against a multi-player clip
showed `ambiguous_frames_disambiguated_by_appearance: 47` out of 48
frames (logged in the `Tracking debug:` line - see "Focusing on a
specific player" above) - i.e. nearly every frame had several people
close together, exactly the situation where the default "largest in
frame 0" heuristic is most likely to lock onto the wrong person for the
whole clip.

Getting `--selected-box` coordinates normally means going through the
app's tap-to-select screen, which isn't convenient from the command line -
`scripts/preview_frame.py` gets you the same information without it:

```bash
python scripts/preview_frame.py --video ./clips/my_clip.mov --timestamp 2.1
```

Saves an annotated frame (`preview.jpg` by default) with every detected
person's box drawn and numbered, and prints each one's `[x1,y1,x2,y2]` -
open the image, find the number on the player you want, and use that
box's coordinates as `--selected-box` above.

This alone won't fix anything - it's a diagnostic, not a training step -
but it's the fastest way to see concretely where the current heuristics
agree or disagree with real ground truth before deciding whether labeling
more clips (and building on path 1 or 2 above) is worth the time
investment.
