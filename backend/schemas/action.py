"""
Action Recognition Schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class DetectedAction(BaseModel):
    """Single detected action"""
    action: str = Field(..., description="Action name (e.g., chopping, stirring)")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    start_frame: int = Field(..., description="Start frame number")
    end_frame: int = Field(..., description="End frame number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    duration: float = Field(..., description="Duration in seconds")

class ActionSequence(BaseModel):
    """Complete action sequence from video"""
    total_actions: int = Field(..., description="Total number of actions detected")
    actions: List[DetectedAction] = Field(..., description="List of detected actions")
    action_summary: Dict[str, Dict] = Field(..., description="Summary of actions by type")
    processed_at: str = Field(..., description="Processing timestamp")

class ActionRecognitionRequest(BaseModel):
    """Request to process video for action recognition"""
    video_id: str = Field(..., description="Video ID to process")
    video_url: Optional[str] = Field(None, description="Video URL (S3 or local path)")

class ActionRecognitionResponse(BaseModel):
    """Response from action recognition"""
    video_id: str
    status: str
    action_sequence: Optional[ActionSequence] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
