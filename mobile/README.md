# Basketball Movement Analyzer — Mobile

Expo (React Native + TypeScript) app for recording or picking a basketball
clip, sending it to the [backend](../backend) for analysis, and displaying
the resulting dribbling/shooting/passing/moving-without-ball timeline.

## Running locally

```bash
npm install
npx expo start
```

Scan the QR code with Expo Go (iOS/Android), or press `i`/`a` for a
simulator/emulator.

By default the app talks to `http://localhost:8000`. Point it at your
backend with:

```bash
EXPO_PUBLIC_API_BASE_URL=http://<your-machine-ip>:8000 npx expo start
```

Use your machine's LAN IP (not `localhost`) when testing on a physical
device via Expo Go, since the phone can't resolve the dev machine's
`localhost`.

## Structure

- `App.tsx` — top-level screen state machine (home → analyzing → results). No
  navigation library is used since the flow is strictly linear.
- `src/screens/HomeScreen.tsx` — record via camera or pick a video from the
  library (`expo-image-picker`), with an optional jersey number field to
  focus analysis on a specific player.
- `src/screens/AnalyzeScreen.tsx` — uploads the video + jersey number
  (`src/api/client.ts`) and polls `/jobs/{id}` for progress until the
  analysis finishes.
- `src/screens/ResultsScreen.tsx` — video playback (`expo-av`), a warning
  banner if a requested jersey number couldn't be confidently matched
  (`result.player_identification_note`), plus the action timeline and
  per-action breakdown.
- `src/components/ActionTimeline.tsx` / `SummaryBreakdown.tsx` — presentation
  components shared by the results screen.
- `src/types.ts` — mirrors `backend/app/schemas.py`; keep in sync if the API
  changes shape.

## Type checking

```bash
npx tsc --noEmit
```
