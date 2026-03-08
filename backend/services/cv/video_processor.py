"""
Video Processing Service
Handles video frame extraction and preprocessing
"""
import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos for action recognition"""
    
    def __init__(self, target_fps: int = 8):
        """
        Initialize video processor
        
        Args:
            target_fps: Target frames per second for extraction (default: 8)
        """
        self.target_fps = target_fps
    
    def extract_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video at target FPS
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None = all)
            
        Returns:
            List of frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
            
            # Calculate frame skip to achieve target FPS
            frame_skip = max(1, int(fps / self.target_fps))
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at target FPS
                if frame_count % frame_skip == 0:
                    # Resize frame for efficiency (224x224 is standard for CV models)
                    frame_resized = cv2.resize(frame, (224, 224))
                    frames.append(frame_resized)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame (RGB, normalized)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_seconds": duration
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
