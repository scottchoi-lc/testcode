"""FastAPI application: upload a basketball clip, get back an action timeline.

Unlike the rest of the backend, this file avoids the `X | None` union
syntax (PEP 604) in favor of `typing.Optional[X]` - FastAPI eagerly
resolves every route handler's parameter types at startup, and `X | None`
as a real runtime expression only works on Python 3.10+. A jersey_number
parameter using `str | None` crashed uvicorn at import time on a Python 3.9
venv even though the rest of the codebase's `X | None` annotations (never
eagerly evaluated - plain functions/dataclasses, not FastAPI routes) were
fine. Keep new route handler parameters on `Optional[X]`.
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.jobs.store import job_store
from app.models.detection import get_detection_model
from app.pipeline.pipeline import run_pipeline
from app.pipeline.video_utils import extract_frame_at, probe_video
from app.schemas import (
    DetectedPersonBox,
    JobResponse,
    JobStatus,
    PreviewFrameResponse,
    UploadVideoResponse,
)
from app.videos.store import video_store

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


async def _save_upload(video: UploadFile) -> Path:
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
    return dest_path


@app.post("/videos", response_model=UploadVideoResponse)
async def upload_video(video: UploadFile) -> UploadVideoResponse:
    """Upload a clip once, ahead of analysis, so the client can request
    preview frames (see /videos/{id}/preview-frame) to pick a player to
    focus on before kicking off the real analysis job via /analyze."""
    dest_path = await _save_upload(video)
    try:
        duration, width, height = probe_video(str(dest_path))
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = video_store.create(dest_path)
    return UploadVideoResponse(
        video_id=session.video_id, duration_seconds=duration, frame_width=width, frame_height=height
    )


@app.post("/videos/{video_id}/preview-frame", response_model=PreviewFrameResponse)
def preview_frame(video_id: str, timestamp: float = Form(...)) -> PreviewFrameResponse:
    """Extract the frame nearest `timestamp` and return it (as a JPEG) along
    with every detected person's box, so the client can render tappable
    overlays for the select-a-player screen."""
    session = video_store.get(video_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown video_id (upload may have expired)")

    try:
        image = extract_frame_at(str(session.path), timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, people = get_detection_model().detect_ball_and_players(image)

    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, settings.PREVIEW_FRAME_JPEG_QUALITY]
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode preview frame")

    height, width = image.shape[:2]
    return PreviewFrameResponse(
        video_id=video_id,
        timestamp=timestamp,
        frame_width=width,
        frame_height=height,
        image_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        people=[
            DetectedPersonBox(x1=p.box[0], y1=p.box[1], x2=p.box[2], y2=p.box[3], score=p.score)
            for p in people
        ],
    )


def _parse_selected_box(raw: Optional[str]) -> Optional[tuple]:
    """`raw` is a JSON-encoded [x1, y1, x2, y2] array from the client (a
    plain form field can't carry structured data, so it's stringified on
    the mobile side). Malformed/missing input is treated as "no selection"
    rather than a hard error, the same tolerant handling as other optional
    inputs in this file."""
    if not raw:
        return None
    try:
        values = json.loads(raw)
        box = tuple(float(v) for v in values)
    except (ValueError, TypeError):
        return None
    return box if len(box) == 4 else None


def _process_job(
    job_id: str,
    video_id: str,
    video_path: str,
    selected_box: Optional[tuple],
    selected_timestamp: Optional[float],
) -> None:
    job_store.update(job_id, status=JobStatus.PROCESSING, progress=0.0)

    def on_progress(fraction: float) -> None:
        job_store.update(job_id, progress=fraction)

    try:
        result = run_pipeline(
            video_path,
            progress_cb=on_progress,
            selected_player_box=selected_box,
            selected_timestamp=selected_timestamp,
        )
        job_store.update(job_id, status=JobStatus.DONE, progress=1.0, result=result)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        logger.exception("Analysis failed for job %s", job_id)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))
    finally:
        Path(video_path).unlink(missing_ok=True)
        video_store.remove(video_id)


@app.post("/analyze", response_model=JobResponse)
def analyze(
    background_tasks: BackgroundTasks,
    video_id: str = Form(...),
    selected_box: Optional[str] = Form(None),
    selected_timestamp: Optional[float] = Form(None),
) -> JobResponse:
    session = video_store.get(video_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown video_id (upload may have expired)")

    box = _parse_selected_box(selected_box)
    timestamp = selected_timestamp if box is not None else None

    job = job_store.create(video_path=str(session.path))
    background_tasks.add_task(
        _process_job, job.job_id, video_id, str(session.path), box, timestamp
    )
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
