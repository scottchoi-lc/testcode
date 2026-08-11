"""In-memory job store for tracking analysis jobs.

Good enough for a single-process prototype deployment. For multi-worker /
multi-instance deployments, swap this for a shared store (Redis, a DB
table, etc.) behind the same `JobStore` interface.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from app.schemas import AnalysisResult, JobStatus


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error: str | None = None
    result: AnalysisResult | None = None
    video_path: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_path: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), video_path=video_path)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)


job_store = JobStore()
