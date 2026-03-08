"""
Object Detection Schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DetectedObject(BaseModel):
    """Single detected object"""
    name: str = Field(..., description="Object name")
    type: str = Field(..., description="Object type (ingredient or utensil)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    first_seen: Optional[float] = Field(None, description="First seen timestamp")
    last_seen: Optional[float] = Field(None, description="Last seen timestamp")
    detections: Optional[int] = Field(None, description="Number of detections")

class IngredientReport(BaseModel):
    """Ingredient comparison report"""
    total_expected: int = Field(..., description="Total expected ingredients")
    total_detected: int = Field(..., description="Total detected ingredients")
    matched: List[str] = Field(..., description="Matched ingredients")
    missing: List[str] = Field(..., description="Missing ingredients")
    extra: List[str] = Field(..., description="Extra ingredients")
    accuracy: float = Field(..., description="Detection accuracy percentage")
    status: str = Field(..., description="Status (complete/incomplete)")

class ObjectDetectionResult(BaseModel):
    """Object detection results"""
    ingredients: List[DetectedObject] = Field(..., description="Detected ingredients")
    utensils: List[DetectedObject] = Field(..., description="Detected utensils")
    total_objects: int = Field(..., description="Total objects detected")
    ingredient_report: Optional[IngredientReport] = Field(None, description="Ingredient report")

class ObjectDetectionRequest(BaseModel):
    """Request to detect objects in video"""
    video_id: str = Field(..., description="Video ID to process")
    video_url: Optional[str] = Field(None, description="Video URL")
    dish_id: Optional[int] = Field(None, description="Dish ID for ingredient comparison")

class ObjectDetectionResponse(BaseModel):
    """Response from object detection"""
    video_id: str
    status: str
    result: Optional[ObjectDetectionResult] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
