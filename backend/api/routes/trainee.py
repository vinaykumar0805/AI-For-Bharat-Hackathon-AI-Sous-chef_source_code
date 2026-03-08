"""
Trainee Evaluation API
Evaluate trainee cooking videos against expert references
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
import uuid

from api.dependencies import get_db
from services.cv.action_recognition_ai import ActionRecognizer
from services.cv.object_detection_ai import ObjectDetector
from services.cv.heat_analysis_ai import HeatAnalyzer
from services.dish.dish_service import DishService
from models.video import Video
from models.evaluation import Evaluation

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trainee",
    tags=["trainee"],
    responses={404: {"description": "Not found"}},
)

# Initialize AI services
action_recognizer = ActionRecognizer(use_ai=True)
object_detector = ObjectDetector(use_ai=True)
heat_analyzer = HeatAnalyzer(use_ai=True)


@router.post("/evaluate")
async def evaluate_trainee_video(
    video: UploadFile = File(..., description="Trainee cooking video"),
    dish_id: int = Query(..., description="ID of the expert dish to compare against"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Evaluate a trainee's cooking video against an expert reference.
    
    This endpoint:
    1. Retrieves the expert dish and video from database
    2. Analyzes the trainee video with AI
    3. Compares trainee performance against expert reference
    4. Calculates scores and provides detailed feedback
    5. Stores evaluation results in database
    6. Returns comprehensive evaluation with recommendations
    
    Use this to evaluate how close the trainee is to the expert's technique.
    """
    try:
        logger.info(f"Evaluating trainee video: {video.filename} against dish ID: {dish_id}")
        
        # Step 1: Get expert dish reference
        logger.info("Step 1: Retrieving expert dish reference...")
        
        dish = DishService.get_dish(db, dish_id)
        if not dish:
            raise HTTPException(
                status_code=404,
                detail=f"Dish with ID {dish_id} not found. Please upload expert video first using POST /expert/upload"
            )
        
        if not dish.expert_video_id:
            raise HTTPException(
                status_code=400,
                detail=f"Dish '{dish.name}' has no expert video reference. Please upload expert video first."
            )
        
        logger.info(f"✅ Found expert dish: {dish.name}")
        
        # Step 2: Get expert video data (we'll use stored dish info as reference)
        # In a full implementation, you'd retrieve and re-analyze the expert video
        # For now, we use the stored dish information as the expert reference
        
        expert_reference = {
            'actions': dish.steps if dish.steps else [],
            'ingredients': dish.ingredients if dish.ingredients else [],
            'heat_pattern': []  # Would be stored with expert video in full implementation
        }
        
        logger.info("✅ Expert reference loaded")
        
        # Step 3: Analyze trainee video
        logger.info("Step 2: Analyzing trainee video with AI...")
        
        video_content = await video.read()
        
        trainee_actions = await action_recognizer.recognize_actions(video_content)
        trainee_objects = await object_detector.detect_objects(video_content)
        trainee_heat = await heat_analyzer.analyze_heat(video_content)
        
        logger.info("✅ Trainee video analysis complete")
        
        # Step 4: Compare trainee vs expert
        logger.info("Step 3: Comparing trainee performance against expert...")
        
        # Action comparison
        action_comparison = action_recognizer.compare_action_sequences(
            trainee_actions['actions'],
            [{'action': step, 'timestamp': 0} for step in expert_reference['actions']]
        )
        
        # Ingredient comparison
        ingredient_comparison = object_detector.compare_ingredient_usage(
            trainee_objects['ingredients'],
            [{'name': ing, 'confidence': 1.0} for ing in expert_reference['ingredients']]
        )
        
        # Heat comparison (if expert heat pattern available)
        if expert_reference['heat_pattern']:
            heat_comparison = heat_analyzer.calculate_heat_control_score(
                trainee_heat['heat_intensities'],
                expert_reference['heat_pattern']
            )
        else:
            # Provide basic heat analysis without comparison
            heat_comparison = {
                'score': 75.0,  # Default score
                'performance_level': 'Good',
                'insights': [{
                    'type': 'info',
                    'message': 'Heat analysis completed (no expert reference available)',
                    'recommendation': 'Maintain consistent heat control'
                }]
            }
        
        logger.info("✅ Comparison complete")
        
        # Step 5: Calculate overall score
        logger.info("Step 4: Calculating overall score...")
        
        overall_score = (
            action_comparison['score'] * 0.4 +
            ingredient_comparison['score'] * 0.3 +
            heat_comparison['score'] * 0.3
        )
        
        # Determine skill level
        if overall_score >= 85:
            skill_level = "Advanced"
            performance_summary = "Excellent! Very close to expert technique."
        elif overall_score >= 70:
            skill_level = "Intermediate"
            performance_summary = "Good performance with room for improvement."
        elif overall_score >= 50:
            skill_level = "Beginner"
            performance_summary = "Decent attempt, but needs more practice."
        else:
            skill_level = "Novice"
            performance_summary = "Needs significant improvement."
        
        # Step 6: Store trainee video record
        logger.info("Step 5: Storing trainee video record...")
        
        trainee_video_record = Video(
            video_id=f"trainee_{uuid.uuid4().hex[:8]}",
            dish_id=str(dish.dish_id),
            video_type="TRAINEE",  # Mark as trainee video
            camera_type="OVERHEAD",
            cloud_url=f"s3://bharatchef-videos/trainee/{video.filename}",
            format="mp4",
            duration=0.0,
            file_size=len(video_content),
            processing_status="COMPLETED"
        )
        db.add(trainee_video_record)
        db.flush()
        
        logger.info(f"✅ Trainee video stored (ID: {trainee_video_record.video_id})")
        
        # Step 7: Save evaluation results
        logger.info("Step 6: Saving evaluation results...")
        
        evaluation = Evaluation(
            dish_id=dish.dish_id,
            trainee_video_id=trainee_video_record.video_id,
            overall_score=overall_score,
            action_score=action_comparison['score'],
            timing_score=action_comparison.get('scoring_breakdown', {}).get('timing_score', 0),
            technique_score=ingredient_comparison['score'],
            visual_score=heat_comparison['score'],
            results={
                "action_comparison": action_comparison,
                "ingredient_comparison": ingredient_comparison,
                "heat_comparison": heat_comparison,
                "skill_level": skill_level,
                "performance_summary": performance_summary
            }
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)
        
        logger.info(f"✅ Evaluation saved (ID: {evaluation.id}, Score: {overall_score:.2f})")
        
        # Step 8: Return comprehensive evaluation
        return {
            "status": "success",
            "message": f"Trainee evaluation complete for dish '{dish.name}'",
            
            "evaluation_id": evaluation.id,
            "trainee_video_id": trainee_video_record.video_id,
            "dish_id": dish.dish_id,
            
            "dish_info": {
                "name": dish.name,
                "description": dish.description,
                "cuisine_type": dish.cuisine_type,
                "difficulty_level": dish.difficulty_level
            },
            
            "overall_results": {
                "overall_score": round(overall_score, 2),
                "skill_level": skill_level,
                "performance_summary": performance_summary,
                "closeness_to_expert": f"{round(overall_score, 1)}% match"
            },
            
            "detailed_scores": {
                "action_sequence": {
                    "score": round(action_comparison['score'], 2),
                    "performance_level": action_comparison.get('performance_level', 'N/A'),
                    "expert_steps": len(expert_reference['actions']),
                    "trainee_steps": len(trainee_actions['actions']),
                    "match_rate": f"{round(action_comparison.get('scoring_breakdown', {}).get('exact_matches', 0) / max(len(expert_reference['actions']), 1) * 100, 1)}%"
                },
                "ingredients": {
                    "score": round(ingredient_comparison['score'], 2),
                    "performance_level": ingredient_comparison.get('performance_level', 'N/A'),
                    "expert_ingredients": len(expert_reference['ingredients']),
                    "trainee_ingredients": len(trainee_objects['ingredients']),
                    "matched": ingredient_comparison.get('ingredients', {}).get('matched', []),
                    "missing": ingredient_comparison.get('ingredients', {}).get('missing', []),
                    "extra": ingredient_comparison.get('ingredients', {}).get('extra', [])
                },
                "heat_control": {
                    "score": round(heat_comparison['score'], 2),
                    "performance_level": heat_comparison.get('performance_level', 'N/A')
                }
            },
            
            "recommendations": {
                "actions": action_comparison.get('insights', [])[:3],
                "ingredients": ingredient_comparison.get('insights', [])[:3],
                "heat": heat_comparison.get('insights', [])[:3]
            },
            
            "next_steps": {
                "view_evaluation": f"/evaluations/{evaluation.id}",
                "view_dish": f"/dishes/{dish.dish_id}",
                "retry_evaluation": f"/trainee/evaluate?dish_id={dish.dish_id}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trainee evaluation failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Trainee evaluation failed: {str(e)}"
        )


@router.get("/history")
async def get_trainee_history(
    dish_id: Optional[int] = Query(None, description="Filter by dish ID"),
    limit: int = Query(10, description="Number of evaluations to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get trainee evaluation history.
    Optionally filter by dish_id to see progress on a specific dish.
    """
    try:
        query = db.query(Evaluation)
        
        if dish_id:
            query = query.filter(Evaluation.dish_id == dish_id)
        
        evaluations = query.order_by(Evaluation.created_at.desc()).limit(limit).all()
        
        return {
            "status": "success",
            "total": len(evaluations),
            "evaluations": [
                {
                    "evaluation_id": eval.id,
                    "dish_id": eval.dish_id,
                    "overall_score": round(eval.overall_score, 2),
                    "skill_level": eval.results.get('skill_level', 'N/A'),
                    "created_at": eval.created_at.isoformat() if eval.created_at else None,
                    "action_score": round(eval.action_score, 2),
                    "technique_score": round(eval.technique_score, 2),
                    "visual_score": round(eval.visual_score, 2)
                }
                for eval in evaluations
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/test")
async def test_trainee_endpoint():
    """Test trainee endpoint"""
    return {
        "status": "success",
        "message": "Trainee evaluation endpoint is ready",
        "features": [
            "Evaluate trainee videos against expert references",
            "Compare actions, ingredients, and heat control",
            "Calculate closeness to expert technique",
            "Provide detailed feedback and recommendations",
            "Track evaluation history"
        ],
        "endpoints": {
            "evaluate": "POST /trainee/evaluate?dish_id={id}",
            "history": "GET /trainee/history?dish_id={id}"
        }
    }
