"""
Dish Schemas for API validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DishBase(BaseModel):
    """Base dish schema"""
    name: str = Field(..., description="Name of the dish")
    description: Optional[str] = Field(None, description="Description of the dish")
    cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Indian, Chinese)")
    difficulty_level: Optional[str] = Field("medium", description="Difficulty level: easy, medium, hard")
    prep_time: Optional[int] = Field(None, description="Preparation time in minutes")
    cook_time: Optional[int] = Field(None, description="Cooking time in minutes")
    servings: Optional[int] = Field(None, description="Number of servings")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients")
    steps: List[str] = Field(default_factory=list, description="Cooking steps")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    expected_duration: Optional[int] = Field(None, description="Expected cooking duration in seconds from expert video")
    expected_steps: Optional[List] = Field(default_factory=list, description="Expected action sequence from expert video")

class DishCreate(DishBase):
    """Schema for creating a new dish"""
    pass

class DishUpdate(BaseModel):
    """Schema for updating a dish"""
    name: Optional[str] = None
    description: Optional[str] = None
    cuisine_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    prep_time: Optional[int] = None
    cook_time: Optional[int] = None
    servings: Optional[int] = None
    ingredients: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    expected_duration: Optional[int] = None
    expected_steps: Optional[List] = None

class DishResponse(DishBase):
    """Schema for dish response"""
    id: int
    dish_id: str
    expert_video_id: Optional[str] = None  # Changed from int to str to match model
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DishList(BaseModel):
    """Schema for list of dishes"""
    dishes: List[DishResponse]
    total: int
