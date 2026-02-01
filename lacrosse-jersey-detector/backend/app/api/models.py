"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime


class PlayerDetection(BaseModel):
    """Single player detection: one bbox, one jersey number, one confidence."""
    player_bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    jersey_number: Union[int, str] = Field(..., description="0-99 or 'unknown'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ImageDetectionResponse(BaseModel):
    """Response for single-image jersey detection: all players with at most one number each."""
    detections: List[PlayerDetection]


class UploadResponse(BaseModel):
    """Response after video upload."""
    video_id: str
    filename: str
    message: str


class AnalyzeRequest(BaseModel):
    """Request to start analysis."""
    video_id: Optional[str] = Field(None, description="ID of the uploaded video (when using Render disk storage)")
    video_url: Optional[str] = Field(None, description="URL of the video (when using Vercel Blob; used with job_id)")
    job_id: Optional[str] = Field(None, description="Job ID from Vercel (required when video_url is set)")
    jersey_number: str = Field(..., description="Jersey number to detect", min_length=1, max_length=10)
    fast_mode: bool = Field(False, description="Use fast mode: fewer frames/crops/OCR strategies (faster, slightly lower recall)")


class AnalyzeResponse(BaseModel):
    """Response after starting analysis."""
    job_id: str
    message: str
    status: str


class TimestampInterval(BaseModel):
    """A time interval where the jersey number was detected."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")


class AnalysisResult(BaseModel):
    """Analysis result for a job."""
    job_id: str
    status: str  # "processing", "completed", "failed"
    jersey_number: str
    video_id: str
    intervals: List[TimestampInterval] = Field(default_factory=list)
    total_detections: int = 0
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
