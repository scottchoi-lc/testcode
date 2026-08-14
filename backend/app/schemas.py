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
    player_label: Optional[str] = Field(
        None,
        description=(
            "Which tracked player this segment belongs to, e.g. \"Player 2\" or \"Number 23\". "
            "Only set on segments returned by POST /combine-narratives, which merges several "
            "separate single-player analyses of the same clip (each one only ever tracks the "
            "one player it was seeded with) into one shared timeline - see that endpoint. None "
            "on segments from a normal single-player POST /analyze result."
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


class CombinePlayerInput(BaseModel):
    label: Optional[str] = Field(
        None,
        description=(
            "How to refer to this player in the combined narrative, e.g. a jersey number "
            "(\"Number 23\") if you know it. If omitted, auto-numbered by position in "
            "`players` (1st entry -> \"Player 1\", 2nd -> \"Player 2\", ...)."
        ),
    )
    segments: list[ActionSegment] = Field(
        ..., description="The `segments` from a completed single-player AnalysisResult for this player"
    )


class CombineNarrativeRequest(BaseModel):
    players: list[CombinePlayerInput] = Field(
        ...,
        min_length=1,
        description=(
            "One entry per player, each carrying the `segments` from separately analyzing the "
            "same clip with that player selected (POST /analyze's selected_box/selected_timestamp, "
            "run once per player). Not multi-player tracking - each entry is a completely "
            "independent single-player analysis; this endpoint only merges the results."
        ),
    )


class CombineNarrativeResponse(BaseModel):
    narrative: str = Field(
        "",
        description=(
            "Plain-English sequence of events across all players, in chronological order. "
            "Only DRIBBLING/SHOOTING/PASSING/RECEIVING segments are mentioned - IDLE and "
            "MOVING_WITHOUT_BALL are noise once multiple players are interleaved, unlike in a "
            "single player's own narrative where they're still meaningful."
        ),
    )
    segments: list[ActionSegment] = Field(
        default_factory=list,
        description=(
            "The same mentionable segments the narrative is built from, merged across all "
            "players and sorted chronologically, each with `player_label` set to say who it "
            "belongs to - for a combined timeline UI, not just the narrative text."
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
