"""
Heat and Flame Analysis API Routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import logging

from services.cv.heat_analysis_ai import HeatAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/heat",
    tags=["heat"],
    responses={404: {"description": "Not found"}},
)

# Initialize heat analyzer with AI
heat_analyzer = HeatAnalyzer(use_ai=True)

@router.post("/analyze")
async def analyze_heat(
    video: UploadFile = File(..., description="Video file to analyze for heat and flame")
) -> Dict[str, Any]:
    """
    Analyze heat and flame in cooking video using AI
    
    Args:
        video: Video file (MP4, MOV, AVI)
        
    Returns:
        Dictionary with flame detections and heat intensities
    """
    try:
        # Validate file type
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {video.content_type}. Must be a video file."
            )
        
        logger.info(f"Analyzing heat for video: {video.filename}")
        
        # Read video content
        video_content = await video.read()
        
        if len(video_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty video file"
            )
        
        logger.info(f"Video size: {len(video_content)} bytes")
        
        # Analyze heat
        result = await heat_analyzer.analyze_heat(video_content)
        
        logger.info(f"Heat analysis complete: {result['total_frames_analyzed']} frames analyzed")
        
        return {
            "status": "success",
            "filename": video.filename,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heat analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Heat analysis failed: {str(e)}"
        )

@router.post("/compare")
async def compare_heat_control(
    trainee_video: UploadFile = File(..., description="Trainee video"),
    expert_video: UploadFile = File(..., description="Expert video")
) -> Dict[str, Any]:
    """
    Compare heat control between trainee and expert videos with detailed analysis
    
    Args:
        trainee_video: Trainee's cooking video
        expert_video: Expert's cooking video
        
    Returns:
        Comprehensive heat control comparison with:
        - Overall heat control score (0-100)
        - Performance level and description
        - Detailed deviation analysis (critical, moderate, minor)
        - Insights and recommendations
        - Heat pattern comparison
        - Scoring breakdown
    """
    try:
        logger.info(f"Comparing heat control: trainee={trainee_video.filename}, expert={expert_video.filename}")
        
        # Read video contents
        trainee_content = await trainee_video.read()
        expert_content = await expert_video.read()
        
        # Analyze both videos
        logger.info("Analyzing trainee video...")
        trainee_result = await heat_analyzer.analyze_heat(trainee_content)
        
        logger.info("Analyzing expert video...")
        expert_result = await heat_analyzer.analyze_heat(expert_content)
        
        # Calculate heat control score with detailed analysis
        logger.info("Calculating heat control comparison...")
        comparison = heat_analyzer.calculate_heat_control_score(
            trainee_heat=trainee_result['heat_intensities'],
            expert_heat=expert_result['heat_intensities']
        )
        
        logger.info(f"Heat control comparison complete: score={comparison['score']}, level={comparison.get('performance_level', 'N/A')}")
        
        return {
            "status": "success",
            "trainee_filename": trainee_video.filename,
            "expert_filename": expert_video.filename,
            
            # Overall comparison results
            "comparison": {
                "score": comparison['score'],
                "performance_level": comparison.get('performance_level'),
                "performance_description": comparison.get('performance_description'),
                "consistency_score": comparison.get('comparison_stats', {}).get('consistency_score')
            },
            
            # Detailed breakdown
            "scoring_breakdown": comparison.get('scoring_breakdown', {}),
            "comparison_stats": comparison.get('comparison_stats', {}),
            "deviation_summary": comparison.get('deviation_summary', {}),
            
            # Deviations by severity
            "critical_deviations": comparison.get('deviations', {}).get('critical', []),
            "moderate_deviations": comparison.get('deviations', {}).get('moderate', []),
            
            # Insights and recommendations
            "insights": comparison.get('insights', []),
            
            # Heat patterns
            "heat_patterns": comparison.get('heat_patterns', {}),
            
            # Full analysis results (optional, for detailed review)
            "trainee_analysis": {
                "flame_summary": trainee_result.get('flame_summary'),
                "heat_summary": trainee_result.get('heat_summary'),
                "frames_analyzed": trainee_result.get('total_frames_analyzed')
            },
            "expert_analysis": {
                "flame_summary": expert_result.get('flame_summary'),
                "heat_summary": expert_result.get('heat_summary'),
                "frames_analyzed": expert_result.get('total_frames_analyzed')
            }
        }
        
    except Exception as e:
        logger.error(f"Heat control comparison failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Heat control comparison failed: {str(e)}"
        )

@router.get("/test")
async def test_heat_analyzer() -> Dict[str, Any]:
    """
    Test heat analyzer initialization
    
    Returns:
        Status of heat analyzer
    """
    return {
        "status": "success",
        "message": "Heat analyzer initialized with Bedrock AI",
        "ai_enabled": heat_analyzer.use_ai,
        "endpoints": {
            "analyze": "POST /heat/analyze - Analyze heat in a single video",
            "compare": "POST /heat/compare - Compare heat control between trainee and expert"
        }
    }
