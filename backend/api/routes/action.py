from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.cv.action_recognition_ai import ActionRecognizer

router = APIRouter(prefix="/actions", tags=["actions"])

# Initialize recognizer with AI enabled
recognizer = ActionRecognizer(use_ai=True)

@router.get("/test")
async def test_action_endpoint() -> Dict[str, Any]:
    """Test endpoint to verify action recognition is working"""
    return {
        "status": "success",
        "message": "Action recognition endpoint is working",
        "ai_enabled": recognizer.use_ai,
        "available_actions": [
            "chopping",
            "stirring", 
            "pouring",
            "mixing",
            "seasoning",
            "frying",
            "boiling",
            "sauteing"
        ]
    }

@router.post("/recognize")
async def recognize_actions(
    video: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Recognize cooking actions in a video
    
    Args:
        video: Video file to analyze
        
    Returns:
        Dictionary with recognized actions and timestamps
    """
    try:
        # Read video content
        video_content = await video.read()
        
        # Recognize actions
        result = await recognizer.recognize_actions(video_content)
        
        return {
            "status": "success",
            "actions": result["actions"],
            "total_actions": len(result["actions"]),
            "video_duration": result.get("duration", 0),
            "method": result.get("method", "unknown"),
            "ai_enabled": recognizer.use_ai
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action recognition failed: {str(e)}")

@router.post("/compare")
async def compare_actions(
    trainee_video: UploadFile = File(..., description="Trainee video"),
    expert_video: UploadFile = File(..., description="Expert video")
) -> Dict[str, Any]:
    """
    Compare action sequences between trainee and expert videos with detailed analysis
    
    Args:
        trainee_video: Trainee's cooking video
        expert_video: Expert's cooking video
        
    Returns:
        Comprehensive comparison with:
        - Action sequence score (0-100)
        - Performance level and description
        - Sequence similarity analysis
        - Missing/extra/out-of-order actions
        - Timing analysis
        - Insights and recommendations
    """
    try:
        # Read video contents
        trainee_content = await trainee_video.read()
        expert_content = await expert_video.read()
        
        # Recognize actions in both videos
        trainee_result = await recognizer.recognize_actions(trainee_content)
        expert_result = await recognizer.recognize_actions(expert_content)
        
        # Compare action sequences
        comparison = recognizer.compare_action_sequences(
            trainee_actions=trainee_result['actions'],
            expert_actions=expert_result['actions']
        )
        
        return {
            "status": "success",
            "trainee_filename": trainee_video.filename,
            "expert_filename": expert_video.filename,
            
            "comparison": {
                "score": comparison['score'],
                "performance_level": comparison.get('performance_level'),
                "performance_description": comparison.get('performance_description')
            },
            
            "scoring_breakdown": comparison.get('scoring_breakdown'),
            "sequence_analysis": comparison.get('sequence_analysis'),
            "action_comparison": comparison.get('action_comparison'),
            "timing_analysis": comparison.get('timing_analysis'),
            "out_of_order_actions": comparison.get('out_of_order_actions'),
            "insights": comparison.get('insights'),
            
            "trainee_recognition": {
                "actions": trainee_result['actions'],
                "total_actions": len(trainee_result['actions']),
                "duration": trainee_result.get('duration', 0)
            },
            
            "expert_recognition": {
                "actions": expert_result['actions'],
                "total_actions": len(expert_result['actions']),
                "duration": expert_result.get('duration', 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action comparison failed: {str(e)}")
