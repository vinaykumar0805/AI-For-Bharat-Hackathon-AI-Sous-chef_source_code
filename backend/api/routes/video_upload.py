"""
Video Upload API with Automated Processing
Handles video upload and triggers automatic AI processing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Optional
import uuid
import logging
from datetime import datetime

from api.dependencies import get_db
from models import Video
from services.video.upload_service import VideoUploadService
from tasks.video_processing_tasks import process_video_complete

router = APIRouter(prefix="/videos", tags=["videos"])
logger = logging.getLogger(__name__)


@router.post("/upload-and-process")
async def upload_and_process_video(
    file: UploadFile = File(...),
    dish_id: Optional[int] = None,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Upload video and automatically trigger AI processing
    
    This is the main endpoint - just upload and everything happens automatically!
    
    Args:
        file: Video file to upload
        dish_id: Optional dish ID for ingredient comparison
        
    Returns:
        video_id, task_id, and status
    """
    try:
        logger.info(f"Received video upload: {file.filename}")
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Initialize upload service
        upload_service = VideoUploadService()
        
        # Validate video
        validation_result = upload_service.validate_video(file)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Upload to S3
        logger.info(f"Uploading video {video_id} to S3...")
        upload_result = await upload_service.upload_to_s3(file, video_id)
        
        if not upload_result["success"]:
            raise HTTPException(status_code=500, detail=upload_result["error"])
        
        video_url = upload_result["video_url"]
        
        # Save video record to database
        video = Video(
            video_id=video_id,
            video_url=video_url,
            processing_status='pending',
            uploaded_at=datetime.utcnow()
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        
        logger.info(f"Video {video_id} saved to database")
        
        # Trigger automated processing task
        logger.info(f"Triggering automated processing for video {video_id}")
        task = process_video_complete.delay(video_id, video_url, dish_id)
        
        logger.info(f"Processing task {task.id} started for video {video_id}")
        
        return {
            "status": "success",
            "message": "Video uploaded successfully. Processing started automatically.",
            "video_id": video_id,
            "task_id": task.id,
            "video_url": video_url,
            "processing_status": "pending",
            "check_status_url": f"/videos/{video_id}/status",
            "get_results_url": f"/videos/{video_id}/results"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{video_id}/status")
async def get_video_status(video_id: str, db: Session = Depends(get_db)) -> Dict:
    """
    Check processing status of a video
    
    Args:
        video_id: Video identifier
        
    Returns:
        Processing status and progress
    """
    try:
        # Get video from database
        video = db.query(Video).filter(Video.video_id == video_id).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get task status from Celery
        from celery.result import AsyncResult
        from celery_app import celery_app
        
        # Find task ID (stored in video or search recent tasks)
        # For now, return database status
        response = {
            "video_id": video_id,
            "status": video.processing_status,
            "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None
        }
        
        if video.processing_status == 'completed':
            response["message"] = "Processing complete! Retrieve results using /videos/{video_id}/results"
            response["results_url"] = f"/videos/{video_id}/results"
        elif video.processing_status == 'failed':
            response["message"] = "Processing failed. Please try uploading again."
        elif video.processing_status == 'pending':
            response["message"] = "Video is being processed. Check back in a few moments."
        else:
            response["message"] = f"Status: {video.processing_status}"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{video_id}/results")
async def get_video_results(video_id: str, db: Session = Depends(get_db)) -> Dict:
    """
    Get complete processing results for a video
    
    Args:
        video_id: Video identifier
        
    Returns:
        Complete evaluation results including actions, objects, and reports
    """
    try:
        # Get video from database
        video = db.query(Video).filter(Video.video_id == video_id).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video.processing_status != 'completed':
            raise HTTPException(
                status_code=400, 
                detail=f"Video processing not complete. Status: {video.processing_status}"
            )
        
        # Get evaluation results
        from models import Evaluation
        evaluation = db.query(Evaluation).filter(Evaluation.video_id == video.id).first()
        
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation results not found")
        
        # Return complete results
        return {
            "video_id": video_id,
            "video_url": video.video_url,
            "processing_status": video.processing_status,
            "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None,
            "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None,
            "results": {
                "evaluation_id": evaluation.id,
                "dish_id": evaluation.dish_id,
                "actions": evaluation.action_sequence,
                "objects": evaluation.detected_objects,
                "ingredient_report": evaluation.ingredient_report,
                "overall_score": evaluation.overall_score
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test")
async def test_video_endpoint():
    """Test endpoint to verify video routes are working"""
    return {
        "status": "success",
        "message": "Video upload endpoint is working",
        "endpoints": {
            "upload": "POST /videos/upload-and-process",
            "status": "GET /videos/{video_id}/status",
            "results": "GET /videos/{video_id}/results"
        }
    }
