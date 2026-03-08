"""
Trainee Model
Stores user/trainee information and skill level
"""
from sqlalchemy import Column, String, DateTime, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .base import Base

class SkillLevel(enum.Enum):
    """Skill level enum"""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

class Trainee(Base):
    """
    Trainee entity - stores user information
    
    Attributes:
        trainee_id: Unique identifier (UUID)
        name: Trainee's full name
        email: Trainee's email address
        skill_level: Current skill level (Beginner/Intermediate/Advanced)
        evaluation_history: JSON array of evaluation IDs
    """
    __tablename__ = "trainees"
    
    # Primary key
    trainee_id = Column(String(36), primary_key=True, index=True)
    
    # Trainee information
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=False, unique=True, index=True)
    skill_level = Column(Enum(SkillLevel), default=SkillLevel.BEGINNER, nullable=False)
    
    # Evaluation history (array of evaluation IDs)
    evaluation_history = Column(JSON, default=list, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    videos = relationship("Video", back_populates="trainee")
    evaluations = relationship("Evaluation", back_populates="trainee")
    
    def __repr__(self):
        return f"<Trainee(trainee_id={self.trainee_id}, name={self.name}, skill_level={self.skill_level.value})>"
