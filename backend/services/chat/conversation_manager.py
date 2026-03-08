"""
Conversation Manager
Manages chat conversation state and context
"""
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from models.conversation import Conversation
import uuid
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation state and history"""
    
    @staticmethod
    def create_session(db: Session, workflow: str) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        
        conversation = Conversation(
            session_id=session_id,
            workflow=workflow,
            stage='initial',
            state={
                'workflow': workflow,
                'video_uploaded': False,
                'dish_confirmed': False,
                'expert_selected': False
            },
            history=[]
        )
        
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Created conversation session: {session_id} ({workflow})")
        return session_id
    
    @staticmethod
    def get_conversation(db: Session, session_id: str) -> Optional[Conversation]:
        """Get conversation by session ID"""
        return db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).first()
    
    @staticmethod
    def update_state(db: Session, session_id: str, updates: Dict[str, Any]):
        """Update conversation state"""
        conversation = ConversationManager.get_conversation(db, session_id)
        if not conversation:
            raise ValueError(f"Conversation {session_id} not found")
        
        # Merge updates into existing state
        current_state = conversation.state or {}
        current_state.update(updates)
        conversation.state = current_state
        
        db.commit()
        logger.info(f"Updated conversation {session_id} state: {updates}")
    
    @staticmethod
    def update_stage(db: Session, session_id: str, stage: str):
        """Update conversation stage"""
        conversation = ConversationManager.get_conversation(db, session_id)
        if not conversation:
            raise ValueError(f"Conversation {session_id} not found")
        
        conversation.stage = stage
        db.commit()
        logger.info(f"Updated conversation {session_id} stage: {stage}")
    
    @staticmethod
    def add_message(
        db: Session,
        session_id: str,
        role: str,  # 'user' or 'bot'
        message: str,
        data: Optional[Dict] = None
    ):
        """Add message to conversation history"""
        conversation = ConversationManager.get_conversation(db, session_id)
        if not conversation:
            raise ValueError(f"Conversation {session_id} not found")
        
        history = conversation.history or []
        history.append({
            'role': role,
            'message': message,
            'data': data,
            'timestamp': None
        })
        conversation.history = history
        
        db.commit()
    
    @staticmethod
    def get_context(db: Session, session_id: str) -> Dict[str, Any]:
        """Get conversation context for AI"""
        conversation = ConversationManager.get_conversation(db, session_id)
        if not conversation:
            return {}
        
        return {
            'session_id': session_id,
            'workflow': conversation.workflow,
            'stage': conversation.stage,
            'state': conversation.state,
            'recent_history': conversation.history[-5:] if conversation.history else []
        }
