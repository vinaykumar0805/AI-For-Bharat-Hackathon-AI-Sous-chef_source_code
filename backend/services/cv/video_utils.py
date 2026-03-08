"""
Video utilities for extracting frames from various video formats
"""
import logging
import tempfile
import os
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def extract_frames_moviepy(video_content: bytes, max_frames: int = 30) -> List[Dict[str, Any]]:
    """Extract frames using moviepy (better codec support)
    
    Default increased to 30 frames for better temporal analysis
    """
    try:
        from moviepy.editor import VideoFileClip
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_content)
            temp_path = temp_file.name
        
        logger.info(f"Extracting frames with moviepy from: {temp_path}")
        
        # Load video
        clip = VideoFileClip(temp_path)
        duration = clip.duration
        fps = clip.fps
        total_frames = int(duration * fps)
        
        logger.info(f"Video: {duration}s, {fps} fps, {total_frames} frames")
        
        # Calculate frame times
        frame_times = []
        if total_frames <= max_frames:
            # Use all frames
            frame_times = [i / fps for i in range(total_frames)]
        else:
            # Sample evenly
            interval = duration / max_frames
            frame_times = [i * interval for i in range(max_frames)]
        
        # Extract frames
        frames = []
        for idx, t in enumerate(frame_times):
            try:
                frame = clip.get_frame(t)
                frames.append({
                    'frame': frame,
                    'timestamp': round(t, 2),
                    'frame_number': int(t * fps)
                })
            except Exception as e:
                logger.warning(f"Could not extract frame at {t}s: {e}")
                continue
        
        clip.close()
        os.unlink(temp_path)
        
        logger.info(f"Extracted {len(frames)} frames with moviepy")
        return frames
        
    except Exception as e:
        logger.error(f"moviepy extraction failed: {e}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

def extract_frames_opencv(video_content: bytes, max_frames: int = 30) -> List[Dict[str, Any]]:
    """Extract frames using OpenCV (fallback)
    
    Default increased to 30 frames for better temporal analysis
    """
    try:
        import cv2
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_content)
            temp_path = temp_file.name
        
        logger.info(f"Extracting frames with OpenCV from: {temp_path}")
        
        # Try to open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            logger.error("OpenCV failed to open video")
            cap = cv2.VideoCapture(temp_path, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                os.unlink(temp_path)
                raise ValueError("OpenCV could not open video file")
        
        # Get properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            cap.release()
            os.unlink(temp_path)
            raise ValueError("Invalid video properties")
        
        logger.info(f"Video: {total_frames} frames, {fps} fps")
        
        # Extract frames
        frame_interval = max(1, total_frames // max_frames)
        frames = []
        frame_idx = 0
        
        while len(frames) < max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps
                frames.append({
                    'frame': frame_rgb,
                    'timestamp': round(timestamp, 2),
                    'frame_number': frame_idx
                })
            
            frame_idx += frame_interval
        
        cap.release()
        os.unlink(temp_path)
        
        logger.info(f"Extracted {len(frames)} frames with OpenCV")
        return frames
        
    except Exception as e:
        logger.error(f"OpenCV extraction failed: {e}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

def extract_frames(video_content: bytes, max_frames: int = 30) -> List[Dict[str, Any]]:
    """
    Extract frames from video using best available method
    Tries moviepy first (better codec support), falls back to OpenCV
    
    Default: 30 frames for multi-agent analysis (was 10)
    For dense analysis, use max_frames=50
    """
    # Try moviepy first
    try:
        return extract_frames_moviepy(video_content, max_frames)
    except Exception as e:
        logger.warning(f"moviepy failed, trying OpenCV: {e}")
    
    # Fallback to OpenCV
    try:
        return extract_frames_opencv(video_content, max_frames)
    except Exception as e:
        logger.error(f"Both moviepy and OpenCV failed: {e}")
        raise ValueError(
            "Could not extract frames from video. "
            "Video format may not be supported. "
            "Please try converting to standard MP4 (H.264 codec)."
        )
