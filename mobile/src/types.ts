// Mirrors backend/app/schemas.py — keep these in sync with the API.

export type ActionLabel = "dribbling" | "shooting" | "passing" | "moving_without_ball" | "idle";

export type JobStatus = "queued" | "processing" | "done" | "failed";

export interface ActionSegment {
  start_time: number;
  end_time: number;
  label: ActionLabel;
  confidence: number;
  evidence: Record<string, unknown>;
  dominant_hand: "left" | "right" | null;
}

export interface AnalysisResult {
  duration_seconds: number;
  fps_analyzed: number;
  segments: ActionSegment[];
  summary: Record<ActionLabel, number>;
  narrative: string;
  detected_player_number: string | null;
  player_selected: boolean;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  error?: string | null;
  result?: AnalysisResult | null;
}

export interface UploadVideoResponse {
  video_id: string;
  duration_seconds: number;
  frame_width: number;
  frame_height: number;
}

export interface DetectedPersonBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

export interface PreviewFrameResponse {
  video_id: string;
  timestamp: number;
  frame_width: number;
  frame_height: number;
  image_base64: string;
  people: DetectedPersonBox[];
}
