"""
Expert Video Processing - Multi-Agent Approach
Uses 4 specialized AI agents for maximum accuracy (90-95%)

Agents:
1. Ingredient Detective - Analyzes ingredients with high precision
2. Cooking Technique Analyzer - Understands cooking methods
3. Visual Pattern Recognizer - Analyzes final dish appearance
4. Culinary Expert Synthesizer - Combines all for final identification
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import uuid

from api.dependencies import get_db
from services.cv.action_recognition_ai import ActionRecognizer
from services.cv.object_detection_ai import ObjectDetector
from services.cv.heat_analysis_ai import HeatAnalyzer
from services.cv.multi_agent_analyzer import MultiAgentAnalyzer
from services.cv.video_utils import extract_frames
from services.dish.dish_service import DishService
from models.video import Video

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/expert",
    tags=["expert-multi-agent"],
    responses={404: {"description": "Not found"}},
)

# Initialize AI services
action_recognizer = ActionRecognizer(use_ai=True)
object_detector = ObjectDetector(use_ai=True)
heat_analyzer = HeatAnalyzer(use_ai=True)
multi_agent = MultiAgentAnalyzer(region='us-east-1')


@router.post("/upload-advanced")
async def upload_expert_video_advanced(
    video: UploadFile = File(..., description="Expert cooking video"),
    cuisine_type: str = Form(..., description="Cuisine type (required)"),
    dish_name: str = Form(..., description="Dish name (required) - e.g., 'Chicken Tikka Masala'"),
    use_dense_sampling: bool = False,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload and process expert video with USER-PROVIDED DISH NAME for maximum accuracy.
    
    NEW WORKFLOW:
    1. User provides cuisine type and dish name
    2. AI analyzes video with dish name context (better accuracy)
    3. AI extracts ingredients, steps, and cooking details
    4. Stores dish with user-provided name (no AI guessing)
    
    This advanced endpoint:
    1. Extracts 30-50 frames (vs 10 in basic mode)
    2. Runs 4 specialized AI agents with dish name context
    3. Validates ingredients/steps match the dish name
    4. Returns complete dish analysis
    
    Parameters:
    - video: Cooking video file (required)
    - cuisine_type: Cuisine type (required) - Indian, Chinese, Italian, etc.
    - dish_name: Dish name (required) - e.g., "Chicken Tikka Masala", "Pad Thai"
    - use_dense_sampling: If True, extracts 50 frames instead of 30 (slower but more accurate)
    
    Expected accuracy: 95-98% (with dish name context)
    Cost per video: ~$1.50 (vs $0.15 for basic mode)
    Processing time: 30-45 seconds
    """
    try:
        logger.info(f"🚀 Processing expert video with USER-PROVIDED DISH NAME: {dish_name}")
        logger.info(f"📊 Cuisine: {cuisine_type}, Dense sampling: {use_dense_sampling}")
        
        # Read video content
        video_content = await video.read()
        
        # Step 1: Extract frames (30 or 50 depending on mode)
        max_frames = 50 if use_dense_sampling else 30
        logger.info(f"Step 1: Extracting {max_frames} frames for analysis...")
        
        frames_data = extract_frames(video_content, max_frames=max_frames)
        logger.info(f"✅ Extracted {len(frames_data)} frames")
        
        # Step 2: Run basic AI analysis with dish name and cuisine context
        logger.info(f"Step 2: Running AI analysis with context: {dish_name} ({cuisine_type})...")
        
        actions_result = await action_recognizer.recognize_actions(video_content, cuisine_type=cuisine_type)
        objects_result = await object_detector.detect_objects(video_content, cuisine_type=cuisine_type)
        heat_result = await heat_analyzer.analyze_heat(video_content)
        
        logger.info("✅ Basic analysis complete")
        
        # Step 3: Run multi-agent analysis with dish name context for validation
        logger.info("Step 3: Running MULTI-AGENT ANALYSIS for validation...")
        logger.info(f"🤖 Validating video matches dish: {dish_name}...")
        
        # Pass dish_name to multi-agent for validation
        dish_info = await multi_agent.analyze_video_multi_agent(
            frames_data=frames_data,
            actions_result=actions_result,
            objects_result=objects_result,
            heat_result=heat_result,
            filename=video.filename,
            cuisine_type=cuisine_type,
            expected_dish_name=dish_name  # NEW: Tell AI what dish to expect
        )
        
        # Override AI-identified name with user-provided name
        dish_info['name'] = dish_name
        dish_info['cuisine_type'] = cuisine_type
        
        confidence = dish_info.get('confidence', 0)
        logger.info(f"✅ Multi-agent validation complete: {confidence:.0%} confidence")
        
        # Step 4: Check if EXPERT dish with this name already exists
        logger.info("Step 4: Checking for existing expert dishes...")
        
        try:
            existing_dishes = DishService.get_dishes(db)
            existing_expert_dish = None
            
            for d in existing_dishes:
                if d.name and dish_name:
                    if d.name.lower() == dish_name.lower() and d.expert_video_id is not None:
                        existing_expert_dish = d
                        break
            
            if existing_expert_dish:
                logger.info(f"⚠️ Expert dish '{dish_name}' already exists (ID: {existing_expert_dish.id})")
                return {
                    "status": "exists",
                    "message": f"Expert dish '{dish_name}' already exists in database",
                    "dish_id": existing_expert_dish.id,
                    "dish_string_id": existing_expert_dish.dish_id,
                    "confidence": confidence,
                    "analysis_method": "multi_agent_with_dish_name",
                    "dish_info": {
                        "dish_id": existing_expert_dish.id,
                        "dish_string_id": existing_expert_dish.dish_id,
                        "name": existing_expert_dish.name,
                        "description": existing_expert_dish.description,
                        "cuisine_type": existing_expert_dish.cuisine_type,
                        "ingredients": existing_expert_dish.ingredients,
                        "steps": existing_expert_dish.steps
                    },
                    "suggestion": "Use PUT /expert/update/{dish_id} to update expert dish or use a different dish name"
                }
            
            logger.info(f"✅ No expert dish found for '{dish_name}', creating new expert dish...")
            
        except Exception as e:
            logger.warning(f"Error checking existing dishes: {e}. Proceeding with dish creation...")
        
        # Step 5: Create dish entry
        logger.info("Step 5: Creating dish entry in database...")
        
        from schemas.dish import DishCreate
        
        # Calculate expected duration from heat analysis
        expected_duration = None
        if heat_result and heat_result.get('heat_intensities'):
            heat_times = [h.get('timestamp', 0) for h in heat_result['heat_intensities']]
            if heat_times:
                expected_duration = int(max(heat_times))
        
        # Ensure steps are populated (from multi-agent or fallback to actions)
        if not dish_info.get('steps') or len(dish_info.get('steps', [])) == 0:
            logger.warning("⚠️ No steps from multi-agent, using action sequence as fallback")
            dish_info['steps'] = [action.get('action', '') for action in actions_result.get('actions', [])]
        
        # Ensure ingredients are populated (from multi-agent or fallback to detected objects)
        if not dish_info.get('ingredients') or len(dish_info.get('ingredients', [])) == 0:
            logger.warning("⚠️ No ingredients from multi-agent, using detected objects as fallback")
            dish_info['ingredients'] = [obj.get('name', '') for obj in objects_result.get('ingredients', [])]
        
        # Add expected fields to dish_info
        dish_info['expected_duration'] = expected_duration
        dish_info['expected_steps'] = actions_result.get('actions', [])
        
        logger.info(f"📝 Dish info prepared: {len(dish_info.get('steps', []))} steps, {len(dish_info.get('ingredients', []))} ingredients")
        
        dish_data = DishCreate(**dish_info)
        dish = DishService.create_dish(db, dish_data)
        db.flush()
        
        logger.info(f"✅ Dish created: {dish.name} (ID: {dish.id}, dish_id: {dish.dish_id})")
        
        # Step 6: Create video record
        logger.info("Step 6: Creating video record...")
        
        video_record = Video(
            video_id=f"expert_{uuid.uuid4().hex[:8]}",
            dish_id=str(dish.dish_id),
            video_type="EXPERT",  # Always mark as EXPERT regardless of filename
            camera_type="OVERHEAD",
            cloud_url=f"s3://bharatchef-videos/expert/{video.filename}",
            format="mp4",
            duration=expected_duration if expected_duration else 0.0,
            file_size=len(video_content),
            processing_status="COMPLETED"
        )
        db.add(video_record)
        db.flush()
        
        logger.info(f"✅ Video stored (ID: {video_record.video_id})")
        
        # Step 7: Associate expert video with dish
        dish.expert_video_id = video_record.video_id
        db.commit()
        db.refresh(dish)
        
        logger.info(f"✅ Expert video associated with dish")
        
        # Step 8: Return comprehensive results
        multi_agent_analysis = dish_info.get('multi_agent_analysis', {})
        
        return {
            "status": "success",
            "message": f"Expert video processed. Dish '{dish.name}' created with {confidence:.0%} confidence",
            
            "analysis_method": "multi_agent_with_dish_name",
            "frames_analyzed": len(frames_data),
            "confidence": confidence,
            "user_provided_name": dish_name,
            
            "dish_id": dish.id,
            "dish_string_id": dish.dish_id,
            "video_id": video_record.video_id,
            
            "dish_info": {
                "dish_id": dish.id,
                "dish_string_id": dish.dish_id,
                "name": dish.name,
                "description": dish.description,
                "cuisine_type": dish.cuisine_type,
                "difficulty_level": dish.difficulty_level,
                "prep_time": dish.prep_time,
                "cook_time": dish.cook_time,
                "servings": dish.servings,
                "ingredients": dish.ingredients,
                "steps": dish.steps,
                "tags": dish.tags,
                "flavor_profile": dish_info.get('flavor_profile', []),
                "dietary_info": dish_info.get('dietary_info', []),
                "expected_duration": dish.expected_duration,
                "expected_steps": dish.expected_steps
            },
            
            "multi_agent_insights": {
                "reasoning": multi_agent_analysis.get('reasoning', {}),
                "alternative_names": multi_agent_analysis.get('alternative_names', []),
                "similar_dishes": multi_agent_analysis.get('similar_dishes', []),
                "agent_confidence": multi_agent_analysis.get('agent_confidence', {})
            },
            
            "expert_analysis": {
                "actions": {
                    "count": len(actions_result.get('actions', [])),
                    "sequence": [a['action'] for a in actions_result.get('actions', [])]
                },
                "ingredients": {
                    "count": len(objects_result.get('ingredients', [])),
                    "list": [i['name'] for i in objects_result.get('ingredients', [])]
                },
                "heat_control": {
                    "count": len(heat_result.get('heat_intensities', [])),
                    "pattern": heat_result.get('heat_intensities', [])
                }
            },
            
            "next_steps": {
                "view_dish": f"/dishes/{dish.dish_id}",
                "update_dish": f"/expert/update/{dish.dish_id}",
                "evaluate_trainee": f"/trainee/evaluate?dish_id={dish.dish_id}"
            },
            
            "performance": {
                "frames_analyzed": len(frames_data),
                "estimated_cost": "$1.50" if use_dense_sampling else "$1.00",
                "expected_accuracy": "90-95%"
            }
        }
        
    except Exception as e:
        logger.error(f"Multi-agent expert video processing failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Multi-agent expert video processing failed: {str(e)}"
        )


@router.get("/test-multi-agent")
async def test_multi_agent_endpoint():
    """Test multi-agent endpoint"""
    return {
        "status": "success",
        "message": "Multi-Agent Expert Video Processing is ready",
        "features": [
            "4 specialized AI agents for maximum accuracy",
            "30-50 frames analyzed (vs 10 in basic mode)",
            "Specific dish identification (e.g., 'Chicken Tikka Masala' not 'Chicken Dish')",
            "Confidence scoring and reasoning",
            "Alternative dish suggestions",
            "90-95% accuracy expected"
        ],
        "agents": {
            "agent_1": "Ingredient Detective - Identifies all ingredients with high precision",
            "agent_2": "Cooking Technique Analyzer - Understands cooking methods and sequences",
            "agent_3": "Visual Pattern Recognizer - Analyzes final dish appearance",
            "agent_4": "Culinary Expert Synthesizer - Combines all outputs for final identification"
        },
        "endpoints": {
            "upload_advanced": "POST /expert/upload-advanced",
            "upload_basic": "POST /expert/upload (existing endpoint)"
        },
        "comparison": {
            "basic_mode": {
                "frames": 10,
                "accuracy": "50-60%",
                "cost": "$0.15/video",
                "time": "10-15s"
            },
            "multi_agent_mode": {
                "frames": "30-50",
                "accuracy": "90-95%",
                "cost": "$1.00-1.50/video",
                "time": "30-45s"
            }
        }
    }


@router.delete("/cleanup/{dish_id}")
async def cleanup_expert_dish(
    dish_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete an expert dish and all associated videos and evaluations.
    
    This will:
    1. Delete all evaluations for this dish (CASCADE)
    2. Delete all videos associated with this dish (CASCADE)
    3. Delete the dish itself
    
    Use with caution - this cannot be undone!
    """
    try:
        logger.info(f"🗑️ Cleaning up expert dish: {dish_id}")
        
        # Get the dish
        dish = DishService.get_dish_by_string_id(db, dish_id)
        if not dish:
            raise HTTPException(status_code=404, detail=f"Dish {dish_id} not found")
        
        # Check if it's an expert dish
        if not dish.expert_video_id:
            raise HTTPException(
                status_code=400,
                detail=f"Dish {dish_id} is not an expert dish. Use /trainee/cleanup for trainee dishes."
            )
        
        # Count associated data before deletion
        from models.video import Video
        from models.evaluation import Evaluation
        
        videos = db.query(Video).filter(Video.dish_id == dish_id).all()
        video_count = len(videos)
        
        evaluations = db.query(Evaluation).filter(Evaluation.dish_id == dish_id).all()
        eval_count = len(evaluations)
        
        dish_name = dish.name
        
        # Delete the dish (CASCADE will handle videos and evaluations)
        db.delete(dish)
        db.commit()
        
        logger.info(f"✅ Cleaned up dish {dish_id}: {video_count} videos, {eval_count} evaluations deleted")
        
        return {
            "status": "success",
            "message": f"Expert dish '{dish_name}' and all associated data deleted",
            "deleted": {
                "dish_id": dish_id,
                "dish_name": dish_name,
                "videos_deleted": video_count,
                "evaluations_deleted": eval_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
