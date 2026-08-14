"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ActionLabel(str, Enum):
    DRIBBLING = "dribbling"
    SHOOTING = "shooting"
    PASSING = "passing"
    RECEIVING = "receiving"
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
    dominant_hand: Optional[Literal["left", "right"]] = Field(
        None,
        description=(
            "For DRIBBLING segments, the wrist that was nearest the ball most often "
            "(majority vote across frames). None if pose data wasn't available/confident "
            "enough, or the segment isn't a dribbling segment."
        ),
    )


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
    detected_player_number: Optional[str] = Field(
        None,
        description=(
            "Jersey number read off the tracked player via OCR, if enough frames agreed. "
            "None means no confident reading - the narrative falls back to \"Player 1\"."
        ),
    )
    player_selected: bool = Field(
        False,
        description=(
            "True if the caller tapped a specific player in the preview frame (POST /analyze's "
            "selected_box/selected_timestamp) and tracking was seeded from that exact detection. "
            "False if the default heuristic (largest person in frame 0) picked the tracked player."
        ),
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None


class UploadVideoResponse(BaseModel):
    video_id: str
    duration_seconds: float
    frame_width: int
    frame_height: int


class DetectedPersonBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class PreviewFrameResponse(BaseModel):
    video_id: str
    timestamp: float
    frame_width: int
    frame_height: int
    image_base64: str = Field(..., description="JPEG-encoded preview frame, base64-encoded")
    people: list[DetectedPersonBox]
