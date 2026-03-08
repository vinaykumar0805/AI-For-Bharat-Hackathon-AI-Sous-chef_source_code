from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.cv.object_detection_ai import ObjectDetector

router = APIRouter(prefix="/objects", tags=["objects"])

# Initialize detector with AI enabled
detector = ObjectDetector(use_ai=True)

@router.get("/test")
async def test_object_endpoint() -> Dict[str, Any]:
    """Test endpoint to verify object detection is working"""
    return {
        "status": "success",
        "message": "Object detection endpoint is working",
        "ai_enabled": detector.use_ai,
        "available_ingredients": [
            "tomato",
            "onion",
            "garlic",
            "ginger",
            "potato",
            "carrot",
            "bell_pepper",
            "chili",
            "cilantro",
            "cumin"
        ],
        "available_utensils": [
            "knife",
            "cutting_board",
            "pan",
            "pot",
            "spatula",
            "spoon",
            "bowl",
            "plate"
        ]
    }

@router.post("/detect")
async def detect_objects(
    video: UploadFile = File(...),
    expected_ingredients: List[str] = None
) -> Dict[str, Any]:
    """
    Detect ingredients and utensils in a video
    
    Args:
        video: Video file to analyze
        expected_ingredients: Optional list of expected ingredients for comparison
        
    Returns:
        Dictionary with detected objects and ingredient report
    """
    try:
        # Read video content
        video_content = await video.read()
        
        # Detect objects
        result = await detector.detect_objects(video_content)
        
        # Generate ingredient report if expected ingredients provided
        ingredient_report = None
        if expected_ingredients:
            ingredient_report = detector.generate_ingredient_report(
                result["ingredients"],
                expected_ingredients
            )
        
        return {
            "status": "success",
            "ingredients": result["ingredients"],
            "utensils": result["utensils"],
            "total_ingredients": len(result["ingredients"]),
            "total_utensils": len(result["utensils"]),
            "ingredient_report": ingredient_report,
            "method": result.get("method", "unknown"),
            "ai_enabled": detector.use_ai
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@router.post("/compare")
async def compare_objects(
    trainee_video: UploadFile = File(..., description="Trainee video"),
    expert_video: UploadFile = File(..., description="Expert video")
) -> Dict[str, Any]:
    """
    Compare ingredient and utensil usage between trainee and expert videos
    
    Args:
        trainee_video: Trainee's cooking video
        expert_video: Expert's cooking video
        
    Returns:
        Comprehensive comparison with:
        - Ingredient usage score and analysis
        - Utensil usage comparison
        - Missing/extra items
        - Insights and recommendations
    """
    try:
        # Read video contents
        trainee_content = await trainee_video.read()
        expert_content = await expert_video.read()
        
        # Detect objects in both videos
        trainee_result = await detector.detect_objects(trainee_content)
        expert_result = await detector.detect_objects(expert_content)
        
        # Compare ingredients
        ingredient_comparison = detector.compare_ingredient_usage(
            trainee_ingredients=trainee_result['ingredients'],
            expert_ingredients=expert_result['ingredients']
        )
        
        # Compare utensils
        utensil_comparison = detector.compare_utensil_usage(
            trainee_utensils=trainee_result['utensils'],
            expert_utensils=expert_result['utensils']
        )
        
        return {
            "status": "success",
            "trainee_filename": trainee_video.filename,
            "expert_filename": expert_video.filename,
            
            "ingredient_comparison": {
                "score": ingredient_comparison['score'],
                "performance_level": ingredient_comparison.get('performance_level'),
                "performance_description": ingredient_comparison.get('performance_description'),
                "scoring_breakdown": ingredient_comparison.get('scoring_breakdown'),
                "comparison_stats": ingredient_comparison.get('comparison_stats'),
                "ingredients": ingredient_comparison.get('ingredients'),
                "quantity_analysis": ingredient_comparison.get('quantity_analysis'),
                "insights": ingredient_comparison.get('insights')
            },
            
            "utensil_comparison": utensil_comparison,
            
            "trainee_detection": {
                "ingredients": trainee_result['ingredients'],
                "utensils": trainee_result['utensils'],
                "frames_analyzed": trainee_result['total_frames_analyzed']
            },
            
            "expert_detection": {
                "ingredients": expert_result['ingredients'],
                "utensils": expert_result['utensils'],
                "frames_analyzed": expert_result['total_frames_analyzed']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object comparison failed: {str(e)}")
