import axios from "axios";

import { API_BASE_URL, JOB_POLL_INTERVAL_MS } from "@/config";
import type { JobResponse } from "@/types";

const client = axios.create({ baseURL: API_BASE_URL, timeout: 60_000 });

export async function submitVideoForAnalysis(videoUri: string, fileName: string): Promise<JobResponse> {
  const form = new FormData();
  // React Native's FormData accepts this { uri, name, type } shape for files.
  form.append("video", {
    uri: videoUri,
    name: fileName,
    type: "video/mp4",
  } as unknown as Blob);

  const response = await client.post<JobResponse>("/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120_000,
  });
  return response.data;
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const response = await client.get<JobResponse>(`/jobs/${jobId}`);
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
