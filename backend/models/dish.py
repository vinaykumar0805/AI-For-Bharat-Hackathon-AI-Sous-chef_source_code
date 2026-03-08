"""
Dish Model
Stores dish recipes and expected ingredients/actions
"""
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from models.base import Base


class Dish(Base):
    """Dish model for storing recipes and expected cooking steps"""
    __tablename__ = 'dishes'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dish_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, unique=False, nullable=False, index=True)  # Allow duplicate names (expert vs trainee)
    description = Column(Text, nullable=True)
    cuisine_type = Column(String, nullable=True)
    difficulty_level = Column(String, nullable=True, default='medium')
    prep_time = Column(Integer, nullable=True)
    cook_time = Column(Integer, nullable=True)
    servings = Column(Integer, nullable=True)
    
    # Ingredients and steps (stored as JSON arrays)
    ingredients = Column(JSON, nullable=True, default=list)
    steps = Column(JSON, nullable=True, default=list)
    tags = Column(JSON, nullable=True, default=list)
    
    # Expected values from expert video
    expected_duration = Column(Integer, nullable=True)  # Expected cooking duration in seconds
    expected_steps = Column(JSON, nullable=True, default=list)  # Expected cooking steps sequence
    
    # Expert video reference
    expert_video_id = Column(String, ForeignKey('videos.video_id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="dish", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dish(id={self.id}, dish_id={self.dish_id}, name={self.name})>"
