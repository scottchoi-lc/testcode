import axios from "axios";

import { API_BASE_URL, JOB_POLL_INTERVAL_MS } from "@/config";
import type {
  CombineNarrativeRequest,
  CombineNarrativeResponse,
  DetectedPersonBox,
  JobResponse,
  PreviewFrameResponse,
  UploadVideoResponse,
} from "@/types";

const client = axios.create({ baseURL: API_BASE_URL, timeout: 60_000 });

export async function uploadVideo(videoUri: string, fileName: string): Promise<UploadVideoResponse> {
  const form = new FormData();
  // React Native's FormData accepts this { uri, name, type } shape for files.
  form.append("video", {
    uri: videoUri,
    name: fileName,
    type: "video/mp4",
  } as unknown as Blob);

  const response = await client.post<UploadVideoResponse>("/videos", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120_000,
  });
  return response.data;
}

export async function getPreviewFrame(videoId: string, timestamp: number): Promise<PreviewFrameResponse> {
  const form = new FormData();
  form.append("timestamp", String(timestamp));

  const response = await client.post<PreviewFrameResponse>(
    `/videos/${videoId}/preview-frame`,
    form,
    { headers: { "Content-Type": "multipart/form-data" }, timeout: 30_000 }
  );
  return response.data;
}

export async function submitVideoForAnalysis(
  videoId: string,
  selectedBox?: DetectedPersonBox,
  selectedTimestamp?: number
): Promise<JobResponse> {
  const form = new FormData();
  form.append("video_id", videoId);
  if (selectedBox && selectedTimestamp != null) {
    form.append(
      "selected_box",
      JSON.stringify([selectedBox.x1, selectedBox.y1, selectedBox.x2, selectedBox.y2])
    );
    form.append("selected_timestamp", String(selectedTimestamp));
  }

  const response = await client.post<JobResponse>("/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30_000,
  });
  return response.data;
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const response = await client.get<JobResponse>(`/jobs/${jobId}`);
  return response.data;
}

/**
 * Merge several already-completed single-player analyses of the same clip
 * (one per tapped player) into one chronological "sequence of events."
 * Not multi-player tracking — see backend/README.md's "Combining several
 * single-player analyses into one sequence of events".
 */
export async function combineNarratives(
  request: CombineNarrativeRequest
): Promise<CombineNarrativeResponse> {
  const response = await client.post<CombineNarrativeResponse>("/combine-narratives", request, {
    timeout: 30_000,
  });
  return response.data;
}

/**
 * Poll a job until it reaches a terminal state (done/failed), invoking
 * `onUpdate` after every poll so the UI can show live progress.
 */
export async function pollJobUntilDone(
  jobId: string,
  onUpdate: (job: JobResponse) => void,
  signal?: { cancelled: boolean }
): Promise<JobResponse> {
  while (true) {
    const job = await getJob(jobId);
    onUpdate(job);
    if (job.status === "done" || job.status === "failed") {
      return job;
    }
    if (signal?.cancelled) {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL_MS));
  }
}
