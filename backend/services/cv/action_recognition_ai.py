"""
Action Recognition Service using AWS Bedrock - AI ONLY with Frame Caching
"""
from typing import Dict, Any, List
import json
import logging
from utils.bedrock_utils import BedrockClient
from services.cv.video_utils import extract_frames
from services.cv.frame_cache import get_frame_cache

logger = logging.getLogger(__name__)

class ActionRecognizer:
    """Recognizes cooking actions in videos using AWS Bedrock AI - NO FALLBACK"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        if not self.use_ai:
            raise ValueError("ActionRecognizer requires AI to be enabled. No fallback available.")
        
        try:
            self.bedrock_client = BedrockClient(region='us-east-1')
            logger.info("ActionRecognizer initialized with Bedrock AI (us-east-1)")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    async def recognize_actions(self, video_content: bytes, cuisine_type: str = None) -> Dict[str, Any]:
        """
        Recognize cooking actions in a video using AI with frame caching
        
        Args:
            video_content: Video file content as bytes
            cuisine_type: Optional cuisine type hint for better accuracy (e.g., "Indian", "Chinese")
            
        Returns:
            Dictionary with recognized actions and timestamps
        """
        try:
            # Extract frames from video
            frames = extract_frames(video_content, max_frames=10)
            
            if not frames:
                logger.error("No frames extracted from video")
                raise ValueError("Could not extract frames from video")
            
            logger.info(f"Extracted {len(frames)} frames for analysis")
            if cuisine_type:
                logger.info(f"🍽️ Using cuisine context: {cuisine_type}")
            
            # Check cache first
            cache = get_frame_cache()
            cached_result = cache.get_cached_analysis(frames, 'actions')
            
            if cached_result:
                logger.info("✅ Using CACHED action recognition results (100% consistent!)")
                return cached_result
            
            logger.info("❌ Cache miss - calling AI for action recognition")
            
            # Build cuisine-specific context
            cuisine_context = ""
            if cuisine_type:
                cuisine_context = f"\n\nIMPORTANT: This is a {cuisine_type} dish. Focus on {cuisine_type}-specific cooking techniques."
                
                if cuisine_type.lower() == "indian":
                    cuisine_context += "\nLook for: tempering (tadka), grinding spices, kneading dough, making roti/naan, pressure cooking dal, sautéing with ghee."
                elif cuisine_type.lower() == "chinese":
                    cuisine_context += "\nLook for: wok cooking, stir-frying at high heat, steaming, making dumplings, using chopsticks for mixing."
                elif cuisine_type.lower() == "italian":
                    cuisine_context += "\nLook for: making pasta, kneading pizza dough, sautéing in olive oil, making risotto, tossing pasta."
                elif cuisine_type.lower() == "thai":
                    cuisine_context += "\nLook for: pounding curry paste, stir-frying in wok, making pad thai, using mortar and pestle."
                elif cuisine_type.lower() == "mexican":
                    cuisine_context += "\nLook for: making tortillas, grinding spices, sautéing with cumin, making salsa, grilling."
            
            # Analyze frames with Bedrock
            all_actions = []
            video_duration = len(frames) * 2.0
            
            for idx, frame_data in enumerate(frames):
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                logger.info(f"Analyzing frame {idx+1}/{len(frames)} at timestamp {timestamp}s")
                
                # Analyze frame for cooking actions
                prompt = f"""Analyze this cooking video frame and identify any cooking actions being performed.
{cuisine_context}

For each action you detect, provide:
1. Action name (e.g., chopping, stirring, pouring, mixing, seasoning, frying, boiling, etc.)
2. Confidence level (0.0 to 1.0)
3. Brief description

Respond in JSON format:
{{
    "actions": [
        {{
            "action": "action_name",
            "confidence": 0.95,
            "description": "brief description"
        }}
    ]
}}

If no cooking action is visible, return empty actions array."""

                try:
                    response = self.bedrock_client.analyze_image(
                        image=frame,
                        prompt=prompt,
                        model='nova-lite',  # Use Nova Lite (we know it's accessible)
                        max_tokens=500,
                        temperature=0.0  # DETERMINISTIC - same video = same results
                    )
                    
                    logger.info(f"Bedrock response for frame {idx}: {response[:100]}...")
                    
                    # Parse response
                    actions_data = self._parse_action_response(response)
                    
                    # Add timestamp to each action
                    for action in actions_data.get('actions', []):
                        action['timestamp'] = timestamp
                        action['frame_index'] = idx
                        all_actions.append(action)
                        logger.info(f"Detected action: {action['action']} (confidence: {action['confidence']})")
                        
                except Exception as e:
                    logger.error(f"Bedrock API error on frame {idx}: {e}")
                    # Don't fail the whole request, just skip this frame
                    continue
            
            if not all_actions:
                logger.warning("No actions detected in any frame")
                return {
                    "actions": [],
                    "duration": video_duration,
                    "total_frames_analyzed": len(frames),
                    "method": "bedrock_ai",
                    "message": "No actions detected in video"
                }
            
            # Merge similar consecutive actions
            merged_actions = self._merge_consecutive_actions(all_actions)
            
            logger.info(f"Final result: {len(merged_actions)} actions detected")
            
            result = {
                "actions": merged_actions,
                "duration": video_duration,
                "total_frames_analyzed": len(frames),
                "method": "bedrock_ai",
                "cached": False
            }
            
            # Cache the result for future use
            cache.cache_analysis(frames, 'actions', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Action recognition failed: {e}", exc_info=True)
            raise Exception(f"AI action recognition failed: {str(e)}")
    
    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """Parse Bedrock response to extract actions"""
        try:
            # Ensure response is a string
            if not isinstance(response, str):
                logger.error(f"Response is not a string, got type: {type(response)}")
                return {"actions": []}
            
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in Bedrock response")
                return {"actions": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"actions": []}
    
    def _merge_consecutive_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar consecutive actions into single actions with duration"""
        if not actions:
            return []
        
        # Sort by timestamp
        sorted_actions = sorted(actions, key=lambda x: x['timestamp'])
        
        merged = []
        current_action = None
        
        for action in sorted_actions:
            if current_action is None:
                current_action = action.copy()
                current_action['duration'] = 2.0
            elif (action['action'] == current_action['action'] and 
                  action['timestamp'] - current_action['timestamp'] < 5.0):
                # Extend duration
                current_action['duration'] = action['timestamp'] - current_action['timestamp'] + 2.0
                current_action['confidence'] = (current_action['confidence'] + action['confidence']) / 2
            else:
                merged.append(current_action)
                current_action = action.copy()
                current_action['duration'] = 2.0
        
        if current_action:
            merged.append(current_action)
        
        return merged
    
    def compare_action_sequences(
        self,
        trainee_actions: List[Dict[str, Any]],
        expert_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare action sequences between trainee and expert with detailed analysis
        
        Args:
            trainee_actions: Trainee's detected actions
            expert_actions: Expert's detected actions
            
        Returns:
            Comprehensive comparison with score, sequence analysis, and insights
        """
        if not expert_actions:
            return {
                "score": 0,
                "message": "No expert action data available for comparison",
                "analysis": "Cannot perform comparison without expert reference data"
            }
        
        if not trainee_actions:
            return {
                "score": 0,
                "message": "No trainee action data available",
                "analysis": "Cannot perform comparison without trainee data"
            }
        
        # Extract action sequences
        trainee_sequence = [a['action'].lower() for a in trainee_actions]
        expert_sequence = [a['action'].lower() for a in expert_actions]
        
        # Calculate sequence similarity using Longest Common Subsequence (LCS)
        lcs_length = self._longest_common_subsequence(trainee_sequence, expert_sequence)
        sequence_similarity = (lcs_length / len(expert_sequence) * 100) if expert_sequence else 0
        
        # Identify action differences
        trainee_set = set(trainee_sequence)
        expert_set = set(expert_sequence)
        
        matched_actions = trainee_set & expert_set
        missing_actions = expert_set - trainee_set
        extra_actions = trainee_set - matched_actions
        
        # Calculate order correctness
        order_score = self._calculate_order_score(trainee_sequence, expert_sequence)
        
        # Calculate timing differences
        timing_analysis = self._analyze_timing_differences(trainee_actions, expert_actions)
        
        # Calculate final score
        sequence_score = sequence_similarity * 0.4
        action_match_score = (len(matched_actions) / len(expert_set) * 100 * 0.3) if expert_set else 0
        order_score_weighted = order_score * 0.2
        timing_score = timing_analysis['timing_score'] * 0.1
        
        raw_score = sequence_score + action_match_score + order_score_weighted + timing_score
        missing_penalty = len(missing_actions) * 10
        extra_penalty = len(extra_actions) * 5
        
        final_score = max(0, min(100, raw_score - missing_penalty - extra_penalty))
        
        # Determine performance level
        if final_score >= 90:
            performance_level = "Excellent"
            performance_description = "Action sequence matches expert technique nearly perfectly"
        elif final_score >= 75:
            performance_level = "Good"
            performance_description = "Action sequence is mostly correct with minor deviations"
        elif final_score >= 60:
            performance_level = "Fair"
            performance_description = "Action sequence shows understanding but needs improvement"
        elif final_score >= 40:
            performance_level = "Needs Improvement"
            performance_description = "Significant action sequence differences from expert"
        else:
            performance_level = "Poor"
            performance_description = "Major action sequence issues requiring practice"
        
        # Generate insights
        insights = []
        
        if missing_actions:
            insights.append({
                "type": "critical",
                "message": f"{len(missing_actions)} essential action(s) missing",
                "details": sorted(list(missing_actions)),
                "recommendation": f"Include these actions: {', '.join(sorted(missing_actions))}"
            })
        
        if extra_actions:
            insights.append({
                "type": "warning",
                "message": f"{len(extra_actions)} unexpected action(s) performed",
                "details": sorted(list(extra_actions)),
                "recommendation": "Review if these actions are necessary for the dish"
            })
        
        if order_score < 70:
            insights.append({
                "type": "warning",
                "message": "Action sequence order differs significantly from expert",
                "recommendation": "Pay attention to the correct order of cooking steps"
            })
        
        if timing_analysis['average_timing_difference'] > 5.0:
            insights.append({
                "type": "info",
                "message": f"Average timing difference: {timing_analysis['average_timing_difference']:.1f}s",
                "recommendation": "Work on timing to match expert pace"
            })
        
        if len(trainee_sequence) > len(expert_sequence) * 1.5:
            insights.append({
                "type": "info",
                "message": "Video contains significantly more actions than expert",
                "recommendation": "Focus on efficiency and essential actions only"
            })
        
        # Identify out-of-order actions
        out_of_order = self._identify_out_of_order_actions(trainee_sequence, expert_sequence)
        
        return {
            "score": round(final_score, 2),
            "performance_level": performance_level,
            "performance_description": performance_description,
            
            "scoring_breakdown": {
                "sequence_similarity": round(sequence_similarity, 2),
                "action_match_score": round(action_match_score, 2),
                "order_score": round(order_score, 2),
                "timing_score": round(timing_score, 2),
                "missing_penalty": round(missing_penalty, 2),
                "extra_penalty": round(extra_penalty, 2)
            },
            
            "sequence_analysis": {
                "lcs_length": lcs_length,
                "sequence_similarity": round(sequence_similarity, 2),
                "order_correctness": round(order_score, 2),
                "trainee_length": len(trainee_sequence),
                "expert_length": len(expert_sequence)
            },
            
            "action_comparison": {
                "matched_actions": sorted(list(matched_actions)),
                "missing_actions": sorted(list(missing_actions)),
                "extra_actions": sorted(list(extra_actions)),
                "match_percentage": round(len(matched_actions) / len(expert_set) * 100, 2) if expert_set else 0
            },
            
            "timing_analysis": timing_analysis,
            
            "out_of_order_actions": out_of_order,
            
            "insights": insights,
            
            "sequences": {
                "trainee": trainee_sequence,
                "expert": expert_sequence
            },
            
            "detailed_actions": {
                "trainee": trainee_actions,
                "expert": expert_actions
            }
        }
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_order_score(self, trainee_seq: List[str], expert_seq: List[str]) -> float:
        """Calculate how well the order of actions matches"""
        if not trainee_seq or not expert_seq:
            return 0.0
        
        # Find common actions and check their relative order
        common_actions = set(trainee_seq) & set(expert_seq)
        if not common_actions:
            return 0.0
        
        correct_order_count = 0
        total_pairs = 0
        
        for i, action1 in enumerate(expert_seq):
            if action1 not in common_actions:
                continue
            for j in range(i + 1, len(expert_seq)):
                action2 = expert_seq[j]
                if action2 not in common_actions:
                    continue
                
                total_pairs += 1
                
                # Check if this pair appears in same order in trainee sequence
                try:
                    trainee_idx1 = trainee_seq.index(action1)
                    trainee_idx2 = trainee_seq.index(action2)
                    if trainee_idx1 < trainee_idx2:
                        correct_order_count += 1
                except ValueError:
                    pass
        
        return (correct_order_count / total_pairs * 100) if total_pairs > 0 else 0.0
    
    def _analyze_timing_differences(
        self,
        trainee_actions: List[Dict[str, Any]],
        expert_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze timing differences between trainee and expert"""
        timing_diffs = []
        
        # Match actions by name and compare timestamps
        for expert_action in expert_actions:
            expert_name = expert_action['action'].lower()
            expert_time = expert_action['timestamp']
            
            # Find corresponding trainee action
            trainee_action = next(
                (a for a in trainee_actions if a['action'].lower() == expert_name),
                None
            )
            
            if trainee_action:
                time_diff = abs(trainee_action['timestamp'] - expert_time)
                timing_diffs.append({
                    "action": expert_name,
                    "expert_time": expert_time,
                    "trainee_time": trainee_action['timestamp'],
                    "difference": round(time_diff, 2)
                })
        
        avg_diff = sum(d['difference'] for d in timing_diffs) / len(timing_diffs) if timing_diffs else 0
        max_diff = max((d['difference'] for d in timing_diffs), default=0)
        
        # Calculate timing score (inverse of average difference)
        timing_score = max(0, 100 - (avg_diff * 2))
        
        return {
            "timing_differences": timing_diffs,
            "average_timing_difference": round(avg_diff, 2),
            "max_timing_difference": round(max_diff, 2),
            "timing_score": round(timing_score, 2)
        }
    
    def _identify_out_of_order_actions(
        self,
        trainee_seq: List[str],
        expert_seq: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify actions that are out of order"""
        out_of_order = []
        
        for i, action in enumerate(trainee_seq):
            if action in expert_seq:
                expert_idx = expert_seq.index(action)
                expected_position = expert_idx
                actual_position = i
                
                if abs(expected_position - actual_position) > 1:
                    out_of_order.append({
                        "action": action,
                        "expected_position": expected_position + 1,
                        "actual_position": actual_position + 1,
                        "deviation": actual_position - expected_position
                    })
        
        return out_of_order
