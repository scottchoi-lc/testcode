"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

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
            "None means no confident reading - the narrative falls back to generic wording."
        ),
    )
    requested_player_number: Optional[str] = Field(
        None, description="Jersey number the caller asked to focus on, if any (POST /analyze field)."
    )
    player_match_found: Optional[bool] = Field(
        None,
        description=(
            "None if no player was requested. True if requested_player_number was confidently "
            "matched to a specific player before tracking began, and the rest of this result "
            "reflects that player. False if requested but not found - the tracked player is "
            "whoever the default heuristic picked instead; see player_identification_note."
        ),
    )
    player_identification_note: Optional[str] = Field(
        None,
        description="User-facing caveat when player_match_found is False. None otherwise.",
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None
