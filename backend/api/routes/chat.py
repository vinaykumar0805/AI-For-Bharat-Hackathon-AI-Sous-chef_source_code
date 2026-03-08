"""
Chat API
Interactive chatbot for expert upload and trainee evaluation
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

from api.dependencies import get_db
from services.chat.conversation_manager import ConversationManager
from services.chat.ai_assistant import AIAssistant
from services.dish.dish_service import DishService
from services.cv.multi_agent_analyzer import MultiAgentAnalyzer
from services.cv.video_utils import extract_frames
from services.cv.action_recognition_ai import ActionRecognizer
from services.cv.object_detection_ai import ObjectDetector
from services.cv.heat_analysis_ai import HeatAnalyzer
from models.dish import Dish
from models.video import Video
from schemas.dish import DishCreate
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# Initialize services
ai_assistant = AIAssistant(region='us-east-1')
multi_agent = MultiAgentAnalyzer(region='us-east-1')
action_recognizer = ActionRecognizer(use_ai=True)
object_detector = ObjectDetector(use_ai=True)
heat_analyzer = HeatAnalyzer(use_ai=True)


# Request/Response Models
class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    action: Optional[str] = None
    data: Optional[Dict] = None


@router.post("/session/create")
async def create_chat_session(
    workflow: str = Query(..., description="Workflow type: 'expert' or 'trainee'"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create new chat session
    
    Parameters:
    - workflow: 'expert' or 'trainee'
    
    Returns:
    - session_id: Unique session identifier
    - welcome_message: Initial bot message
    """
    try:
        if workflow not in ['expert', 'trainee']:
            raise HTTPException(status_code=400, detail="Workflow must be 'expert' or 'trainee'")
        
        # Create session
        session_id = ConversationManager.create_session(db, workflow)
        
        # Generate welcome message
        if workflow == 'expert':
            welcome_message = "👋 Welcome to Expert Video Upload! Upload your cooking video and I'll help you create an expert dish."
        else:
            welcome_message = "👋 Welcome to Trainee Evaluation! Upload your cooking video and I'll evaluate it against expert dishes."
        
        # Add welcome message to history
        ConversationManager.add_message(db, session_id, 'bot', welcome_message)
        
        return {
            "session_id": session_id,
            "workflow": workflow,
            "welcome_message": welcome_message
        }
        
    except Exception as e:
        logger.error(f"Session creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@router.post("/message")
async def send_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Send message to chatbot and get AI response
    
    Handles:
    - Dish name confirmation
    - Ingredient editing
    - Expert dish selection
    - General Q&A
    """
    try:
        session_id = request.session_id
        user_message = request.message
        
        # Get conversation context
        context = ConversationManager.get_context(db, session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user message to history
        ConversationManager.add_message(db, session_id, 'user', user_message)
        
        # Get current state
        stage = context.get('stage', 'initial')
        state = context.get('state', {})
        workflow = context.get('workflow', 'expert')
        
        # Prepare database data if needed
        db_data = None
        if stage == 'select_expert':
            # Get expert dishes for selection
            expert_dishes = DishService.get_dishes(db)
            expert_dishes_list = [
                {
                    'dish_id': d.dish_id,
                    'name': d.name,
                    'cuisine_type': d.cuisine_type
                }
                for d in expert_dishes
                if d.expert_video_id is not None
            ]
            db_data = {'expert_dishes': expert_dishes_list}
        
        # Generate AI response
        ai_response = await ai_assistant.generate_response(
            user_message,
            context,
            db_data
        )
        
        bot_message = ai_response.get('message', '')
        action = ai_response.get('action')
        
        # Handle actions
        response_data = None
        
        if action == 'dish_confirmed':
            # Move to next stage based on workflow
            if workflow == 'expert':
                ConversationManager.update_stage(db, session_id, 'save_expert')
                ConversationManager.update_state(db, session_id, {'dish_confirmed': True})
                bot_message += "\n\n✅ Great! Saving as expert dish..."
                
                # Save expert dish
                response_data = await save_expert_dish(db, session_id, state)
                bot_message += f"\n\n✅ Expert dish saved! Dish ID: {response_data.get('dish_id')}"
                
            else:  # trainee
                ConversationManager.update_stage(db, session_id, 'select_expert')
                ConversationManager.update_state(db, session_id, {'dish_confirmed': True})
                bot_message += "\n\nNow let's compare with an expert dish. Fetching expert dishes..."
        
        elif action == 'edit_ingredients':
            ConversationManager.update_stage(db, session_id, 'edit_ingredients')
            bot_message = "What would you like to change? (e.g., 'add kasuri methi', 'remove pork')"
        
        elif action == 'ingredients_updated':
            # Parse ingredient changes
            current_ingredients = state.get('ingredients', [])
            edit_result = await ai_assistant.handle_ingredient_edit(user_message, current_ingredients)
            
            updated_ingredients = edit_result['updated_ingredients']
            ConversationManager.update_state(db, session_id, {'ingredients': updated_ingredients})
            
            bot_message = f"✅ Updated!\n"
            if edit_result['added']:
                bot_message += f"Added: {', '.join(edit_result['added'])}\n"
            if edit_result['removed']:
                bot_message += f"Removed: {', '.join(edit_result['removed'])}\n"
            bot_message += f"\nNew ingredients: {', '.join(updated_ingredients[:10])}"
            if len(updated_ingredients) > 10:
                bot_message += f" and {len(updated_ingredients) - 10} more"
            bot_message += "\n\nConfirm and save? (Yes/No)"
            
            ConversationManager.update_stage(db, session_id, 'confirm_dish')
        
        elif action == 'expert_selected':
            # Parse expert dish selection
            expert_dish_id = parse_expert_selection(user_message, db_data.get('expert_dishes', []))
            if expert_dish_id:
                ConversationManager.update_state(db, session_id, {'expert_dish_id': expert_dish_id})
                ConversationManager.update_stage(db, session_id, 'evaluate')
                
                # Trigger evaluation
                trainee_dish_id = state.get('dish_id')
                bot_message = f"🔍 Evaluating your trainee video against expert dish...\n\nThis may take a moment..."
                
                # Note: Actual evaluation would be triggered here
                response_data = {'expert_dish_id': expert_dish_id, 'trainee_dish_id': trainee_dish_id}
        
        # Add bot message to history
        ConversationManager.add_message(db, session_id, 'bot', bot_message, response_data)
        
        return ChatResponse(
            session_id=session_id,
            message=bot_message,
            action=action,
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message handling failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Message handling failed: {str(e)}")


@router.post("/upload-video")
async def upload_video_chat(
    session_id: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload video in chat context and get AI analysis
    
    Returns AI-suggested dish name and ingredients
    """
    try:
        # Get conversation context
        context = ConversationManager.get_context(db, session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        workflow = context.get('workflow', 'expert')
        
        logger.info(f"Processing video upload for session {session_id} ({workflow})")
        
        # Read video content
        video_content = await video.read()
        
        # Extract frames
        frames_data = extract_frames(video_content, max_frames=30)
        logger.info(f"Extracted {len(frames_data)} frames")
        
        # Run AI analysis
        actions_result = await action_recognizer.recognize_actions(video_content)
        objects_result = await object_detector.detect_objects(video_content)
        heat_result = await heat_analyzer.analyze_heat(video_content)
        
        # Run multi-agent analysis
        dish_info = await multi_agent.analyze_video_multi_agent(
            frames_data=frames_data,
            actions_result=actions_result,
            objects_result=objects_result,
            heat_result=heat_result,
            filename=video.filename
        )
        
        logger.info(f"AI detected: {dish_info['name']} ({dish_info.get('confidence', 0):.0%})")
        
        # Update conversation state
        ConversationManager.update_state(db, session_id, {
            'video_uploaded': True,
            'video_filename': video.filename,
            'suggested_dish': dish_info['name'],
            'confidence': dish_info.get('confidence', 0),
            'ingredients': dish_info.get('ingredients', []),
            'steps': dish_info.get('steps', []),
            'analysis': dish_info
        })
        
        ConversationManager.update_stage(db, session_id, 'confirm_dish')
        
        # Generate response message
        response_message = await ai_assistant.handle_video_analysis(dish_info)
        
        # Add to history
        ConversationManager.add_message(db, session_id, 'bot', response_message, dish_info)
        
        return {
            "session_id": session_id,
            "message": response_message,
            "analysis": {
                "dish_name": dish_info['name'],
                "confidence": dish_info.get('confidence', 0),
                "ingredients": dish_info.get('ingredients', [])[:10],
                "cuisine_type": dish_info.get('cuisine_type', 'Unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


async def save_expert_dish(db: Session, session_id: str, state: Dict) -> Dict[str, Any]:
    """Save expert dish from conversation state"""
    
    analysis = state.get('analysis', {})
    
    # Create dish
    dish_data = DishCreate(**analysis)
    dish = DishService.create_dish(db, dish_data)
    db.flush()
    
    # Create video record
    video_record = Video(
        video_id=f"expert_{uuid.uuid4().hex[:8]}",
        dish_id=str(dish.dish_id),
        video_type="EXPERT",
        camera_type="OVERHEAD",
        cloud_url=f"s3://bharatchef-videos/expert/{state.get('video_filename', 'video.mp4')}",
        format="mp4",
        duration=0.0,
        file_size=0,
        processing_status="COMPLETED"
    )
    db.add(video_record)
    db.flush()
    
    # Associate expert video with dish
    dish.expert_video_id = video_record.video_id
    db.commit()
    db.refresh(dish)
    
    return {
        "dish_id": dish.dish_id,
        "name": dish.name
    }


def parse_expert_selection(user_input: str, expert_dishes: list) -> Optional[str]:
    """Parse user's expert dish selection"""
    
    user_input_lower = user_input.lower()
    
    # Try to match by number
    for i, dish in enumerate(expert_dishes, 1):
        if str(i) in user_input:
            return dish['dish_id']
    
    # Try to match by name
    for dish in expert_dishes:
        if dish['name'].lower() in user_input_lower:
            return dish['dish_id']
    
    return None


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get chat history for a session"""
    
    conversation = ConversationManager.get_conversation(db, session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "workflow": conversation.workflow,
        "stage": conversation.stage,
        "history": conversation.history
    }
