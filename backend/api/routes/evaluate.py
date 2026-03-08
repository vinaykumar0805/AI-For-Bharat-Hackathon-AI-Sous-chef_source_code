"""
Comprehensive Evaluation API - AI-powered end-to-end workflow
Analyzes videos, extracts dish info, stores everything in database
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import json
from datetime import datetime

from api.dependencies import get_db
from services.cv.action_recognition_ai import ActionRecognizer
from services.cv.object_detection_ai import ObjectDetector
from services.cv.heat_analysis_ai import HeatAnalyzer
from services.dish.dish_service import DishService
from models.video import Video
from models.evaluation import Evaluation
from utils.bedrock_utils import BedrockClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluate",
    tags=["evaluate"],
    responses={404: {"description": "Not found"}},
)

# Initialize AI services
action_recognizer = ActionRecognizer(use_ai=True)
object_detector = ObjectDetector(use_ai=True)
heat_analyzer = HeatAnalyzer(use_ai=True)
bedrock_client = BedrockClient(region='us-east-1')

@router.post("/complete")
async def complete_evaluation(
    trainee_video: UploadFile = File(..., description="Trainee's cooking video"),
    expert_video: UploadFile = File(..., description="Expert's cooking video"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Complete AI-powered evaluation workflow:
    1. Analyzes both videos (actions, objects, heat)
    2. Extracts dish information from expert video using AI
    3. Creates dish entry in database
    4. Stores video records in database
    5. Saves evaluation results in database
    6. Returns comprehensive results with database IDs
    
    This is a one-stop endpoint that does everything automatically!
    """
    try:
        logger.info(f"Starting complete evaluation: trainee={trainee_video.filename}, expert={expert_video.filename}")
        
        # Read video contents
        trainee_content = await trainee_video.read()
        expert_content = await expert_video.read()
        
        # Step 1: Analyze both videos
        logger.info("Step 1: Analyzing videos with AI...")
        
        # Action recognition
        trainee_actions = await action_recognizer.recognize_actions(trainee_content)
        expert_actions = await action_recognizer.recognize_actions(expert_content)
        action_comparison = action_recognizer.compare_action_sequences(
            trainee_actions['actions'],
            expert_actions['actions']
        )
        
        # Object detection
        trainee_objects = await object_detector.detect_objects(trainee_content)
        expert_objects = await object_detector.detect_objects(expert_content)
        object_comparison = object_detector.compare_ingredient_usage(
            trainee_objects['ingredients'],
            expert_objects['ingredients']
        )
        
        # Heat analysis
        trainee_heat = await heat_analyzer.analyze_heat(trainee_content)
        expert_heat = await heat_analyzer.analyze_heat(expert_content)
        heat_comparison = heat_analyzer.calculate_heat_control_score(
            trainee_heat['heat_intensities'],
            expert_heat['heat_intensities']
        )
        
        logger.info("✅ Video analysis complete")
        
        # Step 2: Extract dish information from expert video using AI
        logger.info("Step 2: Extracting dish information using AI...")
        
        dish_info = await extract_dish_info_from_video(
            expert_actions,
            expert_objects,
            expert_video.filename
        )
        
        logger.info(f"✅ Extracted dish: {dish_info['name']}")
        
        # Step 3: Create or find dish in database
        logger.info("Step 3: Creating dish entry in database...")
        
        # Check if dish already exists
        existing_dishes = DishService.get_dishes(db)
        existing_dish = next((d for d in existing_dishes if d.name.lower() == dish_info['name'].lower()), None)
        
        if existing_dish:
            dish = existing_dish
            logger.info(f"✅ Found existing dish: {dish.name} (ID: {dish.dish_id})")
        else:
            from schemas.dish import DishCreate
            dish_data = DishCreate(**dish_info)
            dish = DishService.create_dish(db, dish_data)
            logger.info(f"✅ Created new dish: {dish.name} (ID: {dish.dish_id})")
        
        # Step 4: Store video records in database
        logger.info("Step 4: Storing video records...")
        
        # Create expert video record
        import uuid
        expert_video_record = Video(
            video_id=f"expert_{uuid.uuid4().hex[:8]}",
            video_url=f"s3://bharatchef-videos/expert/{expert_video.filename}",
            processing_status="completed"
        )
        db.add(expert_video_record)
        db.flush()
        
        # Create trainee video record
        trainee_video_record = Video(
            video_id=f"trainee_{uuid.uuid4().hex[:8]}",
            video_url=f"s3://bharatchef-videos/trainee/{trainee_video.filename}",
            processing_status="completed"
        )
        db.add(trainee_video_record)
        db.flush()
        
        # Associate expert video with dish
        dish.expert_video_id = expert_video_record.id
        
        logger.info(f"✅ Stored videos: expert_id={expert_video_record.id}, trainee_id={trainee_video_record.id}")
        
        # Step 5: Calculate overall scores
        logger.info("Step 5: Calculating overall scores...")
        
        overall_score = (
            action_comparison['score'] * 0.4 +
            object_comparison['score'] * 0.3 +
            heat_comparison['score'] * 0.3
        )
        
        # Determine skill level
        if overall_score >= 85:
            skill_level = "Advanced"
        elif overall_score >= 70:
            skill_level = "Intermediate"
        else:
            skill_level = "Beginner"
        
        # Step 6: Save evaluation results
        logger.info("Step 6: Saving evaluation results...")
        
        evaluation = Evaluation(
            dish_id=dish.dish_id,
            trainee_video_id=trainee_video_record.id,
            overall_score=overall_score,
            action_score=action_comparison['score'],
            timing_score=action_comparison.get('scoring_breakdown', {}).get('timing_score', 0),
            technique_score=object_comparison['score'],
            visual_score=heat_comparison['score'],
            results={
                "action_comparison": action_comparison,
                "object_comparison": object_comparison,
                "heat_comparison": heat_comparison,
                "skill_level": skill_level
            }
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)
        
        logger.info(f"✅ Saved evaluation: ID={evaluation.id}, Score={overall_score:.2f}")
        
        # Step 7: Return comprehensive results
        return {
            "status": "success",
            "message": "Complete evaluation finished successfully",
            
            "database_ids": {
                "dish_id": dish.dish_id,
                "expert_video_id": expert_video_record.id,
                "trainee_video_id": trainee_video_record.id,
                "evaluation_id": evaluation.id
            },
            
            "dish_info": {
                "id": dish.dish_id,
                "name": dish.name,
                "description": dish.description,
                "cuisine_type": dish.cuisine_type,
                "ingredients": dish.ingredients,
                "steps": dish.steps
            },
            
            "overall_results": {
                "overall_score": round(overall_score, 2),
                "skill_level": skill_level,
                "action_score": round(action_comparison['score'], 2),
                "ingredient_score": round(object_comparison['score'], 2),
                "heat_score": round(heat_comparison['score'], 2)
            },
            
            "detailed_comparison": {
                "actions": {
                    "score": action_comparison['score'],
                    "performance_level": action_comparison.get('performance_level'),
                    "insights": action_comparison.get('insights', [])[:3]  # Top 3
                },
                "ingredients": {
                    "score": object_comparison['score'],
                    "performance_level": object_comparison.get('performance_level'),
                    "matched": object_comparison.get('ingredients', {}).get('matched', []),
                    "missing": object_comparison.get('ingredients', {}).get('missing', []),
                    "insights": object_comparison.get('insights', [])[:3]
                },
                "heat": {
                    "score": heat_comparison['score'],
                    "performance_level": heat_comparison.get('performance_level'),
                    "insights": heat_comparison.get('insights', [])[:3]
                }
            },
            
            "next_steps": {
                "view_dish": f"/dishes/{dish.dish_id}",
                "view_evaluation": f"/evaluations/{evaluation.id}",
                "view_expert_video": f"/videos/{expert_video_record.id}",
                "view_trainee_video": f"/videos/{trainee_video_record.id}"
            }
        }
        
    except Exception as e:
        logger.error(f"Complete evaluation failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Complete evaluation failed: {str(e)}"
        )


async def extract_dish_info_from_video(
    actions_result: Dict[str, Any],
    objects_result: Dict[str, Any],
    filename: str
) -> Dict[str, Any]:
    """
    Use AI to extract dish information from video analysis results
    """
    try:
        # Prepare data for AI
        actions_list = [a['action'] for a in actions_result.get('actions', [])]
        ingredients_list = [i['name'] for i in objects_result.get('ingredients', [])]
        
        # Create prompt for AI
        prompt = f"""Based on this cooking video analysis, identify the dish being prepared:

Filename: {filename}
Actions performed: {', '.join(actions_list)}
Ingredients used: {', '.join(ingredients_list)}

Please identify:
1. Dish name
2. Brief description
3. Cuisine type (e.g., Indian, Chinese, Italian)
4. Difficulty level (easy, medium, hard)
5. Estimated prep time (minutes)
6. Estimated cook time (minutes)
7. Typical servings
8. Main cooking steps (3-5 steps)
9. Tags for categorization

Respond in JSON format:
{{
    "name": "Dish Name",
    "description": "Brief description",
    "cuisine_type": "Cuisine",
    "difficulty_level": "medium",
    "prep_time": 20,
    "cook_time": 30,
    "servings": 4,
    "ingredients": {ingredients_list},
    "steps": ["Step 1", "Step 2", "Step 3"],
    "tags": ["tag1", "tag2"]
}}"""

        # Call Bedrock AI
        response = bedrock_client.invoke_model(
            prompt=prompt,
            model='nova-lite',
            max_tokens=800
        )
        
        # Parse response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            dish_info = json.loads(json_str)
            
            # Ensure ingredients list is correct
            dish_info['ingredients'] = ingredients_list if ingredients_list else dish_info.get('ingredients', [])
            
            return dish_info
        else:
            # Fallback if AI doesn't return proper JSON
            return {
                "name": filename.replace('.mp4', '').replace('_', ' ').title(),
                "description": "Cooking dish",
                "cuisine_type": "Unknown",
                "difficulty_level": "medium",
                "prep_time": 20,
                "cook_time": 30,
                "servings": 4,
                "ingredients": ingredients_list,
                "steps": actions_list[:5] if actions_list else ["Prepare ingredients", "Cook", "Serve"],
                "tags": ["cooking"]
            }
            
    except Exception as e:
        logger.error(f"Failed to extract dish info: {e}")
        # Return fallback
        return {
            "name": filename.replace('.mp4', '').replace('_', ' ').title(),
            "description": "Cooking dish",
            "cuisine_type": "Unknown",
            "difficulty_level": "medium",
            "prep_time": 20,
            "cook_time": 30,
            "servings": 4,
            "ingredients": [i['name'] for i in objects_result.get('ingredients', [])],
            "steps": [a['action'] for a in actions_result.get('actions', [])][:5],
            "tags": ["cooking"]
        }


@router.get("/test")
async def test_evaluate_endpoint():
    """Test evaluation endpoint"""
    return {
        "status": "success",
        "message": "Comprehensive evaluation endpoint is ready",
        "features": [
            "AI-powered video analysis",
            "Automatic dish extraction",
            "Database storage",
            "Complete evaluation results"
        ],
        "endpoint": "POST /evaluate/complete"
    }
