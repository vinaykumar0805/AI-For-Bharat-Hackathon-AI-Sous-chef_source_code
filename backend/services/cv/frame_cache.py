"""
Frame-based caching for AI analysis results
Caches AI analysis based on video frame hashes for perfect consistency
"""
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FrameCache:
    """
    Cache AI analysis results based on frame hashes
    
    When the same video is uploaded twice:
    1. Extract frames
    2. Generate hash from frames
    3. Check cache
    4. If found: Return cached results (100% match guaranteed!)
    5. If not found: Call AI and cache results
    """
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize frame cache
        
        Args:
            cache_ttl_hours: How long to keep cached results (default: 24 hours)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.hits = 0
        self.misses = 0
        logger.info(f"FrameCache initialized with TTL={cache_ttl_hours}h")
    
    def generate_frame_hash(self, frames: List[Dict[str, Any]]) -> str:
        """
        Generate unique hash from video frames
        
        Args:
            frames: List of frame dictionaries with 'frame' (numpy array) and 'timestamp'
            
        Returns:
            SHA256 hash string representing the frames
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available, using basic frame hashing")
            cv2 = None
        
        hasher = hashlib.sha256()
        
        for frame_data in frames:
            frame = frame_data['frame']
            
            if cv2 is not None:
                # Resize frame to small size for consistent hashing
                # (handles minor encoding differences)
                small_frame = cv2.resize(frame, (64, 64))
                frame_bytes = small_frame.tobytes()
            else:
                # Fallback: use frame as-is
                frame_bytes = frame.tobytes()
            
            # Convert to bytes and hash
            hasher.update(frame_bytes)
            
            # Also include timestamp for ordering
            timestamp_bytes = str(frame_data['timestamp']).encode()
            hasher.update(timestamp_bytes)
        
        frame_hash = hasher.hexdigest()
        logger.debug(f"Generated frame hash: {frame_hash[:16]}...")
        return frame_hash
    
    def get_cached_analysis(
        self,
        frames: List[Dict[str, Any]],
        analysis_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached AI analysis results if available
        
        Args:
            frames: List of frame dictionaries
            analysis_type: Type of analysis ('actions', 'objects', 'heat')
            
        Returns:
            Cached analysis results or None if not found
        """
        frame_hash = self.generate_frame_hash(frames)
        cache_key = f"{frame_hash}:{analysis_type}"
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            
            # Check if cache is still valid (TTL)
            cached_time = cached_data.get('cached_at')
            if cached_time:
                age = datetime.now() - cached_time
                if age > self.cache_ttl:
                    # Cache expired
                    logger.info(f"Cache EXPIRED for {analysis_type} (age: {age})")
                    del self._cache[cache_key]
                    self.misses += 1
                    return None
            
            # Cache hit!
            self.hits += 1
            hit_rate = (self.hits / (self.hits + self.misses)) * 100
            logger.info(f"✅ Cache HIT for {analysis_type} (hit rate: {hit_rate:.1f}%)")
            logger.info(f"   Frame hash: {frame_hash[:16]}...")
            logger.info(f"   Returning cached results (no AI call needed!)")
            
            return cached_data.get('results')
        
        # Cache miss
        self.misses += 1
        hit_rate = (self.hits / (self.hits + self.misses)) * 100 if (self.hits + self.misses) > 0 else 0
        logger.info(f"❌ Cache MISS for {analysis_type} (hit rate: {hit_rate:.1f}%)")
        logger.info(f"   Frame hash: {frame_hash[:16]}...")
        return None
    
    def cache_analysis(
        self,
        frames: List[Dict[str, Any]],
        analysis_type: str,
        results: Dict[str, Any]
    ) -> None:
        """
        Cache AI analysis results for future use
        
        Args:
            frames: List of frame dictionaries
            analysis_type: Type of analysis ('actions', 'objects', 'heat')
            results: Analysis results to cache
        """
        frame_hash = self.generate_frame_hash(frames)
        cache_key = f"{frame_hash}:{analysis_type}"
        
        self._cache[cache_key] = {
            'results': results,
            'cached_at': datetime.now(),
            'frame_hash': frame_hash,
            'analysis_type': analysis_type
        }
        
        logger.info(f"💾 Cached {analysis_type} results")
        logger.info(f"   Frame hash: {frame_hash[:16]}...")
        logger.info(f"   Cache size: {len(self._cache)} entries")
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        count = len(self._cache)
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info(f"🗑️ Cache cleared ({count} entries removed)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "total_requests": total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_hours": self.cache_ttl.total_seconds() / 3600
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = []
        
        for key, data in self._cache.items():
            cached_time = data.get('cached_at')
            if cached_time:
                age = now - cached_time
                if age > self.cache_ttl:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


# Global cache instance (shared across all requests)
_global_cache = FrameCache(cache_ttl_hours=24)


def get_frame_cache() -> FrameCache:
    """Get the global frame cache instance"""
    return _global_cache
