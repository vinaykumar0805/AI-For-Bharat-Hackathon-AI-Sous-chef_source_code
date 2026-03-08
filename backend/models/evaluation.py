"""
Evaluation Model
Stores AI analysis results for videos
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base


class Evaluation(Base):
    """Evaluation model for storing AI analysis results"""
    __tablename__ = 'evaluations'
    
    id = Column(Integer, primary_key=True, index=True)
    trainee_video_id = Column(String, ForeignKey('videos.video_id'), nullable=False)
    dish_id = Column(String, ForeignKey('dishes.dish_id'), nullable=True)
    
    # Scoring
    overall_score = Column(Float, nullable=True)
    action_score = Column(Float, nullable=True)
    timing_score = Column(Float, nullable=True)
    technique_score = Column(Float, nullable=True)
    visual_score = Column(Float, nullable=True)
    
    # Detailed results (stored as JSON)
    results = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", foreign_keys=[trainee_video_id], back_populates="evaluations")
    dish = relationship("Dish", back_populates="evaluations")
    
    def __repr__(self):
        return f"<Evaluation(id={self.id}, trainee_video_id={self.trainee_video_id}, score={self.overall_score})>"
