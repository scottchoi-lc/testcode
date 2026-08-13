# Basketball Movement Analyzer

A mobile app that analyzes a basketball video and breaks down a player's
movement into **dribbling**, **shooting**, **passing**, and **moving
without the ball**.

```
mobile/     Expo (React Native + TypeScript) app: record/upload a clip,
            show upload/analysis progress, display the results.
backend/    FastAPI service: runs the Hugging Face models and the
            rule-based fusion logic that turns them into an action timeline.
```

## Architecture

```
 ┌──────────────┐   upload video    ┌──────────────────────────────────────┐
 │  Mobile app   │ ────────────────▶│  FastAPI backend                     │
 │  (Expo/RN)    │                  │                                      │
 │               │  poll /jobs/:id  │  1. sample frames                    │
 │               │◀──────────────── │  2. YOLOS: detect ball + players     │
 │  video player │   timeline JSON  │  3. ViTPose: player keypoints        │
 │  + timeline   │◀──────────────── │  4. VideoMAE: Kinetics action label  │
 └──────────────┘                   │  5. fusion.py: rule-based label per  │
                                     │     window -> merged segments        │
                                     └──────────────────────────────────────┘
```

There isn't an off-the-shelf Hugging Face model that classifies exactly
"dribbling vs. shooting vs. passing vs. moving without the ball" for
basketball, so the backend combines four general-purpose HF models (object
detection, pose estimation, Kinetics action classification, OCR) with a
small, unit-tested rule-based fusion layer specific to this task. See
[`backend/README.md`](backend/README.md) for the full breakdown of models
used, and [`backend/app/pipeline/fusion.py`](backend/app/pipeline/fusion.py)
for the fusion heuristics themselves — that file is the place to improve
accuracy for a specific dataset/camera setup, or eventually replace with a
fine-tuned classifier trained on labeled basketball clips.

Results also include a plain-English play-by-play narrative
(`app/pipeline/narration.py`) that calls out which hand a player is
dribbling with when detectable. You can also focus analysis on one player
by tapping them on a preview frame before analyzing (`POST /videos` →
`POST /videos/{id}/preview-frame` → tap → `POST /analyze`) instead of
leaving it to the default "largest person in frame 0" heuristic — see
`backend/README.md`'s "Focusing on a specific player" for how the
preview/selection flow and the resulting bidirectional tracking work.

## Quickstart

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Mobile (in another terminal)
cd mobile
npm install
EXPO_PUBLIC_API_BASE_URL=http://<your-machine-ip>:8000 npx expo start
```

Then open the app in Expo Go, record or pick a basketball clip, and wait for
the analysis to finish — you'll get a color-coded timeline plus a
per-action time breakdown.

## Tests

```bash
cd backend && pip install -r requirements.txt && pytest
cd mobile && npm install && npx tsc --noEmit
```
