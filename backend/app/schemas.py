"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionLabel(str, Enum):
    DRIBBLING = "dribbling"
    SHOOTING = "shooting"
    PASSING = "passing"
    MOVING_WITHOUT_BALL = "moving_without_ball"
    IDLE = "idle"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ActionSegment(BaseModel):
    start_time: float = Field(..., description="Segment start, seconds from clip start")
    end_time: float = Field(..., description="Segment end, seconds from clip start")
    label: ActionLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: dict = Field(default_factory=dict, description="Debug signals behind the label")


class AnalysisResult(BaseModel):
    duration_seconds: float
    fps_analyzed: float
    segments: list[ActionSegment]
    summary: dict[str, float] = Field(
        default_factory=dict, description="Total seconds spent per action label"
    )
    narrative: str = Field(
        "", description="Plain-English play-by-play generated from the segments"
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None
