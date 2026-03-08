"""
Video Validation Service
Validates video files before upload
"""
import os
from typing import Tuple, Optional
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

# Allowed video formats
ALLOWED_FORMATS = ['mp4', 'mov', 'avi']
ALLOWED_MIME_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo']

# Size limits
MIN_FILE_SIZE = 1024 * 1024  # 1 MB
MAX_FILE_SIZE = 1024 * 1024 * 1024 * 2  # 2 GB

# Duration limits (in seconds)
MIN_DURATION = 30  # 30 seconds
MAX_DURATION = 1800  # 30 minutes


def validate_video_format(filename: str, content_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate video file format
    
    Args:
        filename: Name of the file
        content_type: MIME type of the file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_ext not in ALLOWED_FORMATS:
        return False, f"Invalid video format. Allowed formats: {', '.join(ALLOWED_FORMATS)}"
    
    # Check MIME type
    if content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Unexpected MIME type: {content_type} for file: {filename}")
        # Don't fail on MIME type mismatch, just log it
    
    return True, None


def validate_file_size(file_size: int) -> Tuple[bool, Optional[str]]:
    """
    Validate video file size
    
    Args:
        file_size: Size of the file in bytes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size < MIN_FILE_SIZE:
        return False, f"File too small. Minimum size: {MIN_FILE_SIZE / (1024*1024):.1f} MB"
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024*1024):.1f} GB"
    
    return True, None


async def validate_video_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
    """
    Validate video file before upload
    
    Args:
        file: Uploaded file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate format
    is_valid, error = validate_video_format(file.filename, file.content_type)
    if not is_valid:
        return False, error
    
    # Get file size
    # Read file to get size (FastAPI doesn't provide size directly)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    # Validate size
    is_valid, error = validate_file_size(file_size)
    if not is_valid:
        return False, error
    
    logger.info(f"Video validation passed: {file.filename} ({file_size / (1024*1024):.2f} MB)")
    return True, None


def get_video_extension(filename: str) -> str:
    """Get video file extension"""
    return filename.lower().split('.')[-1] if '.' in filename else 'mp4'
