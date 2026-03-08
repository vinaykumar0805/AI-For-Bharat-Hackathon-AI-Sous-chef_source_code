"""
Heat and Flame Analysis Service using AWS Bedrock - AI ONLY with Frame Caching
"""
from typing import Dict, Any, List
import json
import logging
from utils.bedrock_utils import BedrockClient
from services.cv.video_utils import extract_frames
from services.cv.frame_cache import get_frame_cache

logger = logging.getLogger(__name__)

class HeatAnalyzer:
    """Analyzes heat and flame in cooking videos using AWS Bedrock AI - NO FALLBACK"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        if not self.use_ai:
            raise ValueError("HeatAnalyzer requires AI to be enabled. No fallback available.")
        
        try:
            self.bedrock_client = BedrockClient(region='us-east-1')
            logger.info("HeatAnalyzer initialized with Bedrock AI (us-east-1)")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    async def analyze_heat(self, video_content: bytes) -> Dict[str, Any]:
        """
        Analyze heat and flame in cooking video using AI with frame caching
        
        Args:
            video_content: Video file content as bytes
            
        Returns:
            Dictionary with flame detections and heat intensities
        """
        try:
            # Extract frames from video
            frames = extract_frames(video_content, max_frames=10)
            
            if not frames:
                logger.error("No frames extracted from video")
                raise ValueError("Could not extract frames from video")
            
            logger.info(f"Extracted {len(frames)} frames for heat analysis")
            
            # Check cache first
            cache = get_frame_cache()
            cached_result = cache.get_cached_analysis(frames, 'heat')
            
            if cached_result:
                logger.info("✅ Using CACHED heat analysis results (100% consistent!)")
                return cached_result
            
            logger.info("❌ Cache miss - calling AI for heat analysis")
            
            flame_detections = []
            heat_intensities = []
            
            for idx, frame_data in enumerate(frames):
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                logger.info(f"Analyzing frame {idx+1}/{len(frames)} at timestamp {timestamp}s")
                
                # Analyze frame for heat and flame
                prompt = """Analyze this cooking video frame for heat and flame indicators:

1. Flame Detection:
   - Is there visible flame or fire? (yes/no)
   - Flame level: none, low, medium, high
   - Confidence level (0.0 to 1.0)

2. Heat Intensity Analysis:
   - Visual cues present: steam, bubbling, sizzling, color changes, smoke
   - Overall heat level: none, low, medium, high
   - Confidence level (0.0 to 1.0)

Respond in JSON format:
{
    "flame": {
        "present": true,
        "level": "medium",
        "confidence": 0.95,
        "description": "brief description"
    },
    "heat": {
        "level": "high",
        "visual_cues": ["steam", "bubbling"],
        "confidence": 0.90,
        "description": "brief description"
    }
}

If no flame or heat indicators visible, set level to "none" and confidence accordingly."""

                try:
                    response = self.bedrock_client.analyze_image(
                        image=frame,
                        prompt=prompt,
                        model='nova-lite',
                        max_tokens=500,
                        temperature=0.0  # DETERMINISTIC
                    )
                    
                    logger.info(f"Bedrock response for frame {idx}: {response[:100]}...")
                    
                    # Parse response
                    analysis = self._parse_heat_response(response)
                    
                    # Add timestamp to flame detection
                    if analysis.get('flame'):
                        flame_data = analysis['flame'].copy()
                        flame_data['timestamp'] = timestamp
                        flame_data['frame_index'] = idx
                        flame_detections.append(flame_data)
                        logger.info(f"Flame detected: {flame_data['level']} (confidence: {flame_data['confidence']})")
                    
                    # Add timestamp to heat intensity
                    if analysis.get('heat'):
                        heat_data = analysis['heat'].copy()
                        heat_data['timestamp'] = timestamp
                        heat_data['frame_index'] = idx
                        heat_intensities.append(heat_data)
                        logger.info(f"Heat detected: {heat_data['level']} (confidence: {heat_data['confidence']})")
                        
                except Exception as e:
                    logger.error(f"Bedrock API error on frame {idx}: {e}")
                    # Don't fail the whole request, just skip this frame
                    continue
            
            # Aggregate results
            flame_summary = self._summarize_flame_detections(flame_detections)
            heat_summary = self._summarize_heat_intensities(heat_intensities)
            
            logger.info(f"Heat analysis complete: {len(flame_detections)} flame detections, {len(heat_intensities)} heat readings")
            
            result = {
                "flame_detections": flame_detections,
                "heat_intensities": heat_intensities,
                "flame_summary": flame_summary,
                "heat_summary": heat_summary,
                "total_frames_analyzed": len(frames),
                "method": "bedrock_ai",
                "cached": False
            }
            
            # Cache the result for future use
            cache.cache_analysis(frames, 'heat', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Heat analysis failed: {e}", exc_info=True)
            raise Exception(f"AI heat analysis failed: {str(e)}")
    
    def _parse_heat_response(self, response: str) -> Dict[str, Any]:
        """Parse Bedrock response to extract heat and flame data"""
        try:
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in Bedrock response")
                return {
                    "flame": {"present": False, "level": "none", "confidence": 0.0},
                    "heat": {"level": "none", "visual_cues": [], "confidence": 0.0}
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "flame": {"present": False, "level": "none", "confidence": 0.0},
                "heat": {"level": "none", "visual_cues": [], "confidence": 0.0}
            }
    
    def _summarize_flame_detections(self, detections: List[Dict]) -> Dict[str, Any]:
        """Summarize flame detections across all frames"""
        if not detections:
            return {
                "flame_present": False,
                "average_level": "none",
                "max_level": "none",
                "flame_duration": 0
            }
        
        # Count flame levels
        levels = [d['level'] for d in detections if d.get('present', False)]
        
        if not levels:
            return {
                "flame_present": False,
                "average_level": "none",
                "max_level": "none",
                "flame_duration": 0
            }
        
        # Determine max level
        level_order = ['none', 'low', 'medium', 'high']
        max_level = max(levels, key=lambda x: level_order.index(x) if x in level_order else 0)
        
        # Calculate average confidence
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        return {
            "flame_present": True,
            "average_level": max_level,  # Simplified
            "max_level": max_level,
            "flame_duration": len(levels),
            "average_confidence": round(avg_confidence, 2)
        }
    
    def _summarize_heat_intensities(self, intensities: List[Dict]) -> Dict[str, Any]:
        """Summarize heat intensities across all frames"""
        if not intensities:
            return {
                "average_heat": "none",
                "max_heat": "none",
                "common_visual_cues": []
            }
        
        # Count heat levels
        levels = [h['level'] for h in intensities]
        
        # Determine max level
        level_order = ['none', 'low', 'medium', 'high']
        max_level = max(levels, key=lambda x: level_order.index(x) if x in level_order else 0)
        
        # Collect all visual cues
        all_cues = []
        for h in intensities:
            all_cues.extend(h.get('visual_cues', []))
        
        # Count cue frequency
        cue_counts = {}
        for cue in all_cues:
            cue_counts[cue] = cue_counts.get(cue, 0) + 1
        
        # Get most common cues
        common_cues = sorted(cue_counts.items(), key=lambda x: x[1], reverse=True)
        common_cues = [cue for cue, count in common_cues[:3]]
        
        # Calculate average confidence
        avg_confidence = sum(h['confidence'] for h in intensities) / len(intensities)
        
        return {
            "average_heat": max_level,  # Simplified
            "max_heat": max_level,
            "common_visual_cues": common_cues,
            "average_confidence": round(avg_confidence, 2)
        }
    
    def calculate_heat_control_score(
        self,
        trainee_heat: List[Dict],
        expert_heat: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compare trainee and expert heat patterns with detailed analysis
        Calculate heat control score (0-100) with comprehensive insights
        
        Args:
            trainee_heat: Trainee's heat intensity data
            expert_heat: Expert's heat intensity data
            
        Returns:
            Dictionary with heat control score, deviations, and detailed analysis
        """
        if not expert_heat:
            return {
                "score": 0,
                "message": "No expert heat data available for comparison",
                "analysis": "Cannot perform comparison without expert reference data"
            }
        
        if not trainee_heat:
            return {
                "score": 0,
                "message": "No trainee heat data available",
                "analysis": "Cannot perform comparison without trainee data"
            }
        
        level_order = ['none', 'low', 'medium', 'high']
        level_values = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        
        trainee_levels = [h['level'] for h in trainee_heat]
        expert_levels = [h['level'] for h in expert_heat]
        
        # Align sequences (handle different lengths)
        min_length = min(len(trainee_levels), len(expert_levels))
        max_length = max(len(trainee_levels), len(expert_levels))
        
        # Calculate exact matches
        exact_matches = sum(1 for t, e in zip(trainee_levels, expert_levels) if t == e)
        
        # Calculate close matches (within 1 level)
        close_matches = 0
        for t, e in zip(trainee_levels, expert_levels):
            t_val = level_values.get(t, 0)
            e_val = level_values.get(e, 0)
            if abs(t_val - e_val) == 1:
                close_matches += 1
        
        # Calculate deviation severity
        total_deviation = 0
        overheating_count = 0
        underheating_count = 0
        critical_deviations = []
        moderate_deviations = []
        minor_deviations = []
        
        for i, (t_level, e_level) in enumerate(zip(trainee_levels, expert_levels)):
            t_val = level_values.get(t_level, 0)
            e_val = level_values.get(e_level, 0)
            deviation = t_val - e_val
            
            if deviation != 0:
                total_deviation += abs(deviation)
                
                deviation_info = {
                    "frame_index": i,
                    "timestamp": trainee_heat[i].get('timestamp', 0),
                    "trainee_level": t_level,
                    "expert_level": e_level,
                    "trainee_confidence": trainee_heat[i].get('confidence', 0),
                    "expert_confidence": expert_heat[i].get('confidence', 0),
                    "trainee_cues": trainee_heat[i].get('visual_cues', []),
                    "expert_cues": expert_heat[i].get('visual_cues', []),
                    "severity": abs(deviation)
                }
                
                if deviation > 0:
                    deviation_info["type"] = "overheating"
                    deviation_info["impact"] = f"Heat is {abs(deviation)} level(s) too high"
                    overheating_count += 1
                else:
                    deviation_info["type"] = "underheating"
                    deviation_info["impact"] = f"Heat is {abs(deviation)} level(s) too low"
                    underheating_count += 1
                
                # Categorize by severity
                if abs(deviation) >= 2:
                    deviation_info["severity_label"] = "critical"
                    critical_deviations.append(deviation_info)
                elif abs(deviation) == 1:
                    deviation_info["severity_label"] = "moderate"
                    moderate_deviations.append(deviation_info)
        
        # Calculate scoring components
        exact_match_score = (exact_matches / min_length * 100) if min_length > 0 else 0
        close_match_bonus = (close_matches / min_length * 20) if min_length > 0 else 0
        length_penalty = abs(len(trainee_levels) - len(expert_levels)) * 5
        deviation_penalty = (total_deviation / min_length * 30) if min_length > 0 else 0
        
        # Final score calculation
        raw_score = exact_match_score + close_match_bonus - length_penalty - deviation_penalty
        final_score = max(0, min(100, raw_score))
        
        # Determine performance level
        if final_score >= 90:
            performance_level = "Excellent"
            performance_description = "Heat control is nearly perfect, matching expert technique closely"
        elif final_score >= 75:
            performance_level = "Good"
            performance_description = "Heat control is solid with minor deviations from expert technique"
        elif final_score >= 60:
            performance_level = "Fair"
            performance_description = "Heat control shows understanding but needs improvement in consistency"
        elif final_score >= 40:
            performance_level = "Needs Improvement"
            performance_description = "Heat control has significant deviations from expert technique"
        else:
            performance_level = "Poor"
            performance_description = "Heat control requires major improvement and practice"
        
        # Generate insights
        insights = []
        
        if overheating_count > underheating_count:
            insights.append({
                "type": "pattern",
                "message": f"Tendency to overheat ({overheating_count} instances)",
                "recommendation": "Try reducing heat levels and being more patient with temperature control"
            })
        elif underheating_count > overheating_count:
            insights.append({
                "type": "pattern",
                "message": f"Tendency to underheat ({underheating_count} instances)",
                "recommendation": "Increase heat levels to match expert technique for better results"
            })
        
        if critical_deviations:
            insights.append({
                "type": "critical",
                "message": f"{len(critical_deviations)} critical heat deviations detected",
                "recommendation": "Focus on these moments - they have the highest impact on cooking quality"
            })
        
        if len(trainee_levels) > len(expert_levels):
            insights.append({
                "type": "timing",
                "message": "Video is longer than expert reference",
                "recommendation": "Work on efficiency - expert completes the dish faster"
            })
        elif len(trainee_levels) < len(expert_levels):
            insights.append({
                "type": "timing",
                "message": "Video is shorter than expert reference",
                "recommendation": "Ensure all cooking steps are captured in the video"
            })
        
        # Calculate consistency score
        trainee_variance = self._calculate_heat_variance(trainee_levels)
        expert_variance = self._calculate_heat_variance(expert_levels)
        consistency_score = max(0, 100 - abs(trainee_variance - expert_variance) * 20)
        
        return {
            "score": round(final_score, 2),
            "performance_level": performance_level,
            "performance_description": performance_description,
            
            "scoring_breakdown": {
                "exact_match_score": round(exact_match_score, 2),
                "close_match_bonus": round(close_match_bonus, 2),
                "length_penalty": round(length_penalty, 2),
                "deviation_penalty": round(deviation_penalty, 2)
            },
            
            "comparison_stats": {
                "exact_matches": exact_matches,
                "close_matches": close_matches,
                "total_comparisons": min_length,
                "trainee_frames": len(trainee_levels),
                "expert_frames": len(expert_levels),
                "consistency_score": round(consistency_score, 2)
            },
            
            "deviation_summary": {
                "total_deviations": overheating_count + underheating_count,
                "overheating_count": overheating_count,
                "underheating_count": underheating_count,
                "critical_count": len(critical_deviations),
                "moderate_count": len(moderate_deviations),
                "average_deviation": round(total_deviation / min_length, 2) if min_length > 0 else 0
            },
            
            "deviations": {
                "critical": critical_deviations,
                "moderate": moderate_deviations,
                "all": critical_deviations + moderate_deviations
            },
            
            "insights": insights,
            
            "heat_patterns": {
                "trainee": {
                    "levels": trainee_levels,
                    "average_level": self._calculate_average_level(trainee_levels),
                    "variance": round(trainee_variance, 2),
                    "dominant_level": max(set(trainee_levels), key=trainee_levels.count)
                },
                "expert": {
                    "levels": expert_levels,
                    "average_level": self._calculate_average_level(expert_levels),
                    "variance": round(expert_variance, 2),
                    "dominant_level": max(set(expert_levels), key=expert_levels.count)
                }
            }
        }
    
    def _calculate_heat_variance(self, levels: List[str]) -> float:
        """Calculate variance in heat levels"""
        level_values = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        values = [level_values.get(l, 0) for l in levels]
        
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_average_level(self, levels: List[str]) -> str:
        """Calculate average heat level"""
        level_values = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        level_names = ['none', 'low', 'medium', 'high']
        
        values = [level_values.get(l, 0) for l in levels]
        if not values:
            return 'none'
        
        avg_value = sum(values) / len(values)
        avg_index = round(avg_value)
        return level_names[min(avg_index, 3)]
