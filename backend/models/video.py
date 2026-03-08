"""
Video Model
Stores uploaded video information
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base
import enum


class CameraType(str, enum.Enum):
    """Camera type enum"""
    OVERHEAD = "OVERHEAD"
    SIDE = "SIDE"
    FRONT = "FRONT"


class ProcessingStatus(str, enum.Enum):
    """Processing status enum"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VideoType(str, enum.Enum):
    """Video type enum"""
    EXPERT = "EXPERT"
    TRAINEE = "TRAINEE"


class Video(Base):
    """Video model for storing uploaded videos"""
    __tablename__ = 'videos'
    
    # Primary key
    video_id = Column(String, primary_key=True, index=True, nullable=False)
    
    # Video metadata
    trainee_id = Column(String, nullable=True)
    dish_id = Column(String, nullable=False)
    video_type = Column(String, nullable=False, default='TRAINEE')  # EXPERT or TRAINEE
    camera_type = Column(String, nullable=False, default='overhead')
    upload_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    cloud_url = Column(String, nullable=False)
    format = Column(String, nullable=False, default='mp4')
    duration = Column(Float, nullable=False, default=0.0)
    file_size = Column(Integer, nullable=False, default=0)
    processing_status = Column(String, nullable=False, default='pending')
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to evaluations
    evaluations = relationship("Evaluation", foreign_keys="[Evaluation.trainee_video_id]", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Video(video_id={self.video_id}, status={self.processing_status})>"
