"""
Conversation Model
Stores chat conversation state and history
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text
from sqlalchemy.sql import func
from models.base import Base


class Conversation(Base):
    """Conversation model for storing chat sessions"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    workflow = Column(String, nullable=False)  # 'expert' or 'trainee'
    stage = Column(String, nullable=False, default='initial')  # Current conversation stage
    state = Column(JSON, nullable=False, default=dict)  # Conversation state
    history = Column(JSON, nullable=False, default=list)  # Message history
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    def __repr__(self):
        return f"<Conversation(session_id={self.session_id}, workflow={self.workflow}, stage={self.stage})>"
