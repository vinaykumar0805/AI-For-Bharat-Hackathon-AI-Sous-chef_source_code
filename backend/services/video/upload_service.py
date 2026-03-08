"""
Video Upload Service
Handles video uploads to S3 with resumable support
"""
import os
import uuid
import json
from typing import Dict, Any, Optional
from fastapi import UploadFile
import logging
from datetime import datetime, timedelta

from utils.aws_utils import get_s3_client, generate_presigned_url
from config import settings
import redis

logger = logging.getLogger(__name__)

# Redis client for upload checkpoints
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

# Multipart upload settings
CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB chunks
CHECKPOINT_EXPIRY = 86400  # 24 hours


def generate_video_id() -> str:
    """Generate unique video ID"""
    return str(uuid.uuid4())


def get_s3_key(video_id: str, filename: str) -> str:
    """
    Generate S3 key for video
    
    Args:
        video_id: Unique video identifier
        filename: Original filename
    
    Returns:
        S3 key path
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else 'mp4'
    timestamp = datetime.utcnow().strftime('%Y/%m/%d')
    return f"videos/{timestamp}/{video_id}.{ext}"


async def upload_video_to_s3(
    file: UploadFile,
    video_id: str,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Upload video to S3 using multipart upload
    
    Args:
        file: Uploaded file
        video_id: Unique video identifier
        progress_callback: Optional callback for progress updates
    
    Returns:
        Upload result dictionary
    """
    s3_client = get_s3_client()
    s3_key = get_s3_key(video_id, file.filename)
    
    try:
        # Get file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        logger.info(f"Starting upload: {video_id}, size: {file_size / (1024*1024):.2f} MB")
        
        # For small files, use simple upload
        if file_size < CHUNK_SIZE:
            content = await file.read()
            s3_client.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                Body=content,
                ContentType=file.content_type
            )
            
            s3_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"Upload completed: {video_id}")
            return {
                'video_id': video_id,
                's3_key': s3_key,
                's3_url': s3_url,
                'file_size': file_size,
                'status': 'completed'
            }
        
        # For large files, use multipart upload
        return await multipart_upload(
            s3_client,
            file,
            video_id,
            s3_key,
            file_size,
            progress_callback
        )
        
    except Exception as e:
        logger.error(f"Upload failed for {video_id}: {str(e)}")
        raise


async def multipart_upload(
    s3_client,
    file: UploadFile,
    video_id: str,
    s3_key: str,
    file_size: int,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Perform multipart upload to S3
    
    Args:
        s3_client: S3 client
        file: Uploaded file
        video_id: Video identifier
        s3_key: S3 key
        file_size: File size in bytes
        progress_callback: Progress callback
    
    Returns:
        Upload result
    """
    # Check for existing upload
    checkpoint_key = f"upload:{video_id}"
    checkpoint = redis_client.get(checkpoint_key)
    
    if checkpoint:
        checkpoint_data = json.loads(checkpoint)
        upload_id = checkpoint_data['upload_id']
        uploaded_parts = checkpoint_data['parts']
        logger.info(f"Resuming upload: {video_id}, parts: {len(uploaded_parts)}")
    else:
        # Start new multipart upload
        response = s3_client.create_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            ContentType=file.content_type
        )
        upload_id = response['UploadId']
        uploaded_parts = []
        logger.info(f"Started multipart upload: {video_id}, upload_id: {upload_id}")
    
    try:
        # Upload parts
        part_number = len(uploaded_parts) + 1
        uploaded_bytes = sum(part['Size'] for part in uploaded_parts)
        
        # Skip already uploaded bytes
        if uploaded_bytes > 0:
            file.file.seek(uploaded_bytes)
        
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            
            # Upload part
            response = s3_client.upload_part(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=chunk
            )
            
            part_info = {
                'PartNumber': part_number,
                'ETag': response['ETag'],
                'Size': len(chunk)
            }
            uploaded_parts.append(part_info)
            uploaded_bytes += len(chunk)
            
            # Save checkpoint
            checkpoint_data = {
                'upload_id': upload_id,
                'parts': uploaded_parts,
                's3_key': s3_key
            }
            redis_client.setex(
                checkpoint_key,
                CHECKPOINT_EXPIRY,
                json.dumps(checkpoint_data)
            )
            
            # Progress callback
            progress = int((uploaded_bytes / file_size) * 100)
            if progress_callback:
                progress_callback(progress)
            
            logger.info(f"Uploaded part {part_number}: {video_id}, progress: {progress}%")
            part_number += 1
        
        # Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={
                'Parts': [
                    {'PartNumber': p['PartNumber'], 'ETag': p['ETag']}
                    for p in uploaded_parts
                ]
            }
        )
        
        # Delete checkpoint
        redis_client.delete(checkpoint_key)
        
        s3_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
        
        logger.info(f"Multipart upload completed: {video_id}")
        return {
            'video_id': video_id,
            's3_key': s3_key,
            's3_url': s3_url,
            'file_size': file_size,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Multipart upload failed: {video_id}, error: {str(e)}")
        # Abort multipart upload
        try:
            s3_client.abort_multipart_upload(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                UploadId=upload_id
            )
        except:
            pass
        raise


def get_upload_progress(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get upload progress from checkpoint
    
    Args:
        video_id: Video identifier
    
    Returns:
        Progress data or None
    """
    checkpoint_key = f"upload:{video_id}"
    checkpoint = redis_client.get(checkpoint_key)
    
    if not checkpoint:
        return None
    
    checkpoint_data = json.loads(checkpoint)
    uploaded_bytes = sum(part['Size'] for part in checkpoint_data['parts'])
    
    return {
        'video_id': video_id,
        'uploaded_bytes': uploaded_bytes,
        'parts_uploaded': len(checkpoint_data['parts']),
        's3_key': checkpoint_data['s3_key']
    }
