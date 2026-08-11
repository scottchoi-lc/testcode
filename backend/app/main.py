"""FastAPI application: upload a basketball clip, get back an action timeline."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.jobs.store import job_store
from app.pipeline.pipeline import run_pipeline
from app.schemas import JobResponse, JobStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Basketball Movement Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _process_job(job_id: str, video_path: str) -> None:
    job_store.update(job_id, status=JobStatus.PROCESSING, progress=0.0)

    def on_progress(fraction: float) -> None:
        job_store.update(job_id, progress=fraction)

    try:
        result = run_pipeline(video_path, progress_cb=on_progress)
        job_store.update(job_id, status=JobStatus.DONE, progress=1.0, result=result)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        logger.exception("Analysis failed for job %s", job_id)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))
    finally:
        Path(video_path).unlink(missing_ok=True)


@app.post("/analyze", response_model=JobResponse)
async def analyze(video: UploadFile, background_tasks: BackgroundTasks) -> JobResponse:
    if video.content_type is None or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    dest_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    size = 0
    with dest_path.open("wb") as dest:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_UPLOAD_BYTES:
                dest.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds maximum upload size")
            dest.write(chunk)

    job = job_store.create(video_path=str(dest_path))
    background_tasks.add_task(_process_job, job.job_id, str(dest_path))
    return JobResponse(job_id=job.job_id, status=job.status, progress=job.progress)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        result=job.result,
    )
