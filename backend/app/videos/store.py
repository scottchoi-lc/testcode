"""In-memory registry for uploaded-but-not-yet-analyzed videos.

Supports the select-a-player flow: a video is uploaded once via
``POST /videos``, the client can request preview frames (with detected
person boxes) at different timestamps via ``POST /videos/{id}/preview-frame``
without re-uploading, and finally kicks off the real analysis job against
the same file via ``POST /analyze``.

Same "good enough for a single-process prototype" caveat as
`app.jobs.store.JobStore`: swap for a shared store behind this interface for
multi-worker deployments. Sessions that are uploaded but never analyzed are
not automatically cleaned up - a known limitation for this prototype.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoSession:
    video_id: str
    path: Path
    created_at: float


class VideoStore:
    def __init__(self):
        self._sessions: dict[str, VideoSession] = {}
        self._lock = threading.Lock()

    def create(self, path: Path) -> VideoSession:
        session = VideoSession(video_id=str(uuid.uuid4()), path=path, created_at=time.time())
        with self._lock:
            self._sessions[session.video_id] = session
        return session

    def get(self, video_id: str) -> VideoSession | None:
        with self._lock:
            return self._sessions.get(video_id)

    def remove(self, video_id: str) -> None:
        with self._lock:
            self._sessions.pop(video_id, None)


video_store = VideoStore()
