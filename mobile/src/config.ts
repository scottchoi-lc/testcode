// Point this at your backend (see /backend/README.md). For local development
// with Expo Go, use your machine's LAN IP rather than "localhost" so a
// physical device can reach it, e.g. "http://192.168.1.23:8000".
export const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const JOB_POLL_INTERVAL_MS = 1500;
