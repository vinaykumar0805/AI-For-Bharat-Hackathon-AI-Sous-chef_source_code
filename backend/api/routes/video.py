"""
Video Upload API Routes
Handles video upload endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging

from services.video.validation_service import validate_video_file
from services.video.upload_service import (
    generate_video_id,
    upload_video_to_s3,
    get_upload_progress
)
from tasks.video_tasks import process_video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload a video file
    
    Args:
        file: Video file to upload
        background_tasks: FastAPI background tasks
    
    Returns:
        Upload result with video_id and URLs
    """
    try:
        # Validate video file
        is_valid, error_message = await validate_video_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Generate video ID
        video_id = generate_video_id()
        
        logger.info(f"Starting video upload: {video_id}, filename: {file.filename}")
        
        # Upload to S3
        upload_result = await upload_video_to_s3(file, video_id)
        
        # Submit video processing task in background
        background_tasks.add_task(
            process_video.delay,
            video_id,
            upload_result['s3_url']
        )
        
        logger.info(f"Video uploaded successfully: {video_id}")
        
        return {
            "status": "success",
            "message": "Video uploaded successfully",
            "video_id": video_id,
            "s3_url": upload_result['s3_url'],
            "file_size": upload_result['file_size'],
            "filename": file.filename,
            "processing_status": "queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/upload/progress/{video_id}")
async def get_video_upload_progress(video_id: str) -> Dict[str, Any]:
    """
    Get upload progress for a video
    
    Args:
        video_id: Video identifier
    
    Returns:
        Upload progress data
    """
    progress = get_upload_progress(video_id)
    
    if not progress:
        return {
            "status": "not_found",
            "message": "No upload in progress for this video"
        }
    
    return {
        "status": "in_progress",
        "video_id": video_id,
        "uploaded_bytes": progress['uploaded_bytes'],
        "parts_uploaded": progress['parts_uploaded']
    }


@router.get("/test-upload")
async def test_upload_endpoint():
    """Test endpoint to verify video routes are working"""
    return {
        "status": "success",
        "message": "Video upload endpoints are ready",
        "endpoints": {
            "upload": "POST /videos/upload",
            "progress": "GET /videos/upload/progress/{video_id}"
        }
    }
