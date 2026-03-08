"""
Object Detection Service using AWS Bedrock - AI ONLY with Frame Caching
"""
from typing import Dict, Any, List
import json
import logging
from utils.bedrock_utils import BedrockClient
from services.cv.video_utils import extract_frames
from services.cv.frame_cache import get_frame_cache

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Detects ingredients and utensils in videos using AWS Bedrock AI - NO FALLBACK"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        if not self.use_ai:
            raise ValueError("ObjectDetector requires AI to be enabled. No fallback available.")
        
        try:
            self.bedrock_client = BedrockClient(region='us-east-1')
            logger.info("ObjectDetector initialized with Bedrock AI (us-east-1)")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    async def detect_objects(self, video_content: bytes, cuisine_type: str = None) -> Dict[str, Any]:
        """
        Detect ingredients and utensils in a video using AI with frame caching
        
        Args:
            video_content: Video file content as bytes
            cuisine_type: Optional cuisine type hint for better accuracy (e.g., "Indian", "Chinese")
            
        Returns:
            Dictionary with detected ingredients and utensils
        """
        try:
            # Extract frames from video
            frames = extract_frames(video_content, max_frames=8)
            
            if not frames:
                logger.error("No frames extracted from video")
                raise ValueError("Could not extract frames from video")
            
            logger.info(f"Extracted {len(frames)} frames for analysis")
            if cuisine_type:
                logger.info(f"🍽️ Using cuisine context: {cuisine_type}")
            
            # Check cache first
            cache = get_frame_cache()
            cached_result = cache.get_cached_analysis(frames, 'objects')
            
            if cached_result:
                logger.info("✅ Using CACHED object detection results (100% consistent!)")
                return cached_result
            
            logger.info("❌ Cache miss - calling AI for object detection")
            
            # Build cuisine-specific context
            cuisine_context = ""
            if cuisine_type:
                cuisine_context = f"\n\nIMPORTANT: This is a {cuisine_type} dish. Focus on {cuisine_type}-specific ingredients and utensils."
                
                if cuisine_type.lower() == "indian":
                    cuisine_context += "\nLook for: spices (turmeric, cumin, coriander, garam masala), paneer, ghee, curry leaves, mustard seeds, dal, rice, naan, roti, kadai, tawa."
                elif cuisine_type.lower() == "chinese":
                    cuisine_context += "\nLook for: soy sauce, oyster sauce, rice wine, bok choy, tofu, wok, bamboo steamer, chopsticks, rice vinegar, sesame oil."
                elif cuisine_type.lower() == "italian":
                    cuisine_context += "\nLook for: pasta, olive oil, tomatoes, basil, parmesan, mozzarella, garlic, pizza dough, pasta machine, pizza stone."
                elif cuisine_type.lower() == "thai":
                    cuisine_context += "\nLook for: fish sauce, lemongrass, galangal, coconut milk, Thai basil, lime, chili, curry paste, mortar and pestle, wok."
                elif cuisine_type.lower() == "mexican":
                    cuisine_context += "\nLook for: tortillas, beans, corn, avocado, cilantro, lime, chili peppers, cumin, salsa, comal, molcajete."
            
            # Analyze frames with Bedrock
            all_ingredients = []
            all_utensils = []
            
            for idx, frame_data in enumerate(frames):
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                logger.info(f"Analyzing frame {idx+1}/{len(frames)} at timestamp {timestamp}s")
                
                # Analyze frame for ingredients and utensils
                prompt = f"""Analyze this cooking video frame and identify:
1. All visible ingredients (vegetables, spices, liquids, etc.)
2. All visible cooking utensils and equipment
{cuisine_context}

For each item, provide:
- Name
- Confidence level (0.0 to 1.0)
- Quantity (for ingredients, if countable)

Respond in JSON format:
{{
    "ingredients": [
        {{
            "name": "ingredient_name",
            "confidence": 0.95,
            "quantity": 2,
            "description": "brief description"
        }}
    ],
    "utensils": [
        {{
            "name": "utensil_name",
            "confidence": 0.90,
            "description": "brief description"
        }}
    ]
}}

If nothing is visible, return empty arrays."""

                try:
                    response = self.bedrock_client.analyze_image(
                        image=frame,
                        prompt=prompt,
                        model='nova-lite',  # Use Nova Lite (we know it's accessible)
                        max_tokens=800,
                        temperature=0.0  # DETERMINISTIC
                    )
                    
                    logger.info(f"Bedrock response for frame {idx}: {response[:100]}...")
                    
                    # Parse response
                    objects_data = self._parse_objects_response(response)
                    
                    # Add timestamp to each object
                    for ingredient in objects_data.get('ingredients', []):
                        ingredient['timestamp'] = timestamp
                        ingredient['frame_index'] = idx
                        all_ingredients.append(ingredient)
                        logger.info(f"Detected ingredient: {ingredient['name']} (confidence: {ingredient['confidence']})")
                    
                    for utensil in objects_data.get('utensils', []):
                        utensil['timestamp'] = timestamp
                        utensil['frame_index'] = idx
                        all_utensils.append(utensil)
                        logger.info(f"Detected utensil: {utensil['name']} (confidence: {utensil['confidence']})")
                        
                except Exception as e:
                    logger.error(f"Bedrock API error on frame {idx}: {e}")
                    # Don't fail the whole request, just skip this frame
                    continue
            
            if not all_ingredients and not all_utensils:
                logger.warning("No objects detected in any frame")
                return {
                    "ingredients": [],
                    "utensils": [],
                    "total_frames_analyzed": len(frames),
                    "method": "bedrock_ai",
                    "message": "No objects detected in video"
                }
            
            # Deduplicate and consolidate objects
            unique_ingredients = self._consolidate_objects(all_ingredients, 'ingredient')
            unique_utensils = self._consolidate_objects(all_utensils, 'utensil')
            
            logger.info(f"Final result: {len(unique_ingredients)} ingredients, {len(unique_utensils)} utensils")
            
            result = {
                "ingredients": unique_ingredients,
                "utensils": unique_utensils,
                "total_frames_analyzed": len(frames),
                "method": "bedrock_ai",
                "cached": False
            }
            
            # Cache the result for future use
            cache.cache_analysis(frames, 'objects', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}", exc_info=True)
            raise Exception(f"AI object detection failed: {str(e)}")
    
    def _parse_objects_response(self, response: str) -> Dict[str, Any]:
        """Parse Bedrock response to extract objects"""
        try:
            # Ensure response is a string
            if not isinstance(response, str):
                logger.error(f"Response is not a string, got type: {type(response)}")
                return {"ingredients": [], "utensils": []}
            
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in Bedrock response")
                return {"ingredients": [], "utensils": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"ingredients": [], "utensils": []}
    
    def _consolidate_objects(self, objects: List[Dict[str, Any]], object_type: str) -> List[Dict[str, Any]]:
        """Consolidate duplicate objects detected across multiple frames"""
        if not objects:
            return []
        
        # Group by name
        grouped = {}
        for obj in objects:
            name = obj['name'].lower()
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(obj)
        
        # Consolidate each group
        consolidated = []
        for name, group in grouped.items():
            # Calculate average confidence (ensure all are floats)
            confidences = []
            for o in group:
                conf = o.get('confidence', 0)
                if isinstance(conf, (int, float)):
                    confidences.append(float(conf))
                else:
                    try:
                        confidences.append(float(conf))
                    except:
                        confidences.append(0.5)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Get first timestamp (ensure all are numbers)
            timestamps = []
            for o in group:
                ts = o.get('timestamp', 0)
                if isinstance(ts, (int, float)):
                    timestamps.append(float(ts))
                else:
                    try:
                        timestamps.append(float(ts))
                    except:
                        timestamps.append(0.0)
            
            first_timestamp = min(timestamps) if timestamps else 0.0
            
            # For ingredients, get max quantity
            if object_type == 'ingredient':
                quantities = []
                for o in group:
                    qty = o.get('quantity', 1)
                    if isinstance(qty, (int, float)):
                        quantities.append(int(qty))
                    else:
                        try:
                            quantities.append(int(qty))
                        except:
                            quantities.append(1)
                
                max_quantity = max(quantities) if quantities else 1
                
                consolidated.append({
                    "name": name,
                    "confidence": round(avg_confidence, 2),
                    "quantity": max_quantity,
                    "timestamp": first_timestamp,
                    "detections": len(group)
                })
            else:
                consolidated.append({
                    "name": name,
                    "confidence": round(avg_confidence, 2),
                    "timestamp": first_timestamp,
                    "detections": len(group)
                })
        
        # Sort by confidence
        consolidated.sort(key=lambda x: x['confidence'], reverse=True)
        
        return consolidated
    
    def generate_ingredient_report(
        self,
        detected_ingredients: List[Dict[str, Any]],
        expected_ingredients: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a report comparing detected vs expected ingredients
        
        Args:
            detected_ingredients: List of detected ingredients
            expected_ingredients: List of expected ingredient names
            
        Returns:
            Dictionary with comparison report
        """
        detected_names = [ing["name"].lower() for ing in detected_ingredients]
        expected_names = [ing.lower() for ing in expected_ingredients if ing]  # Filter empty strings
        
        missing = [ing for ing in expected_names if ing not in detected_names]
        extra = [ing for ing in detected_names if ing not in expected_names]
        matched = [ing for ing in detected_names if ing in expected_names]
        
        accuracy = len(matched) / len(expected_names) if expected_names else 0
        
        return {
            "expected_count": len(expected_names),
            "detected_count": len(detected_ingredients),
            "matched_count": len(matched),
            "missing": missing,
            "extra": extra,
            "matched": matched,
            "accuracy": round(accuracy * 100, 2)
        }
    
    def compare_ingredient_usage(
        self,
        trainee_ingredients: List[Dict[str, Any]],
        expert_ingredients: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare ingredient usage between trainee and expert with detailed analysis
        
        Args:
            trainee_ingredients: Trainee's detected ingredients
            expert_ingredients: Expert's detected ingredients
            
        Returns:
            Comprehensive comparison with score, deviations, and insights
        """
        if not expert_ingredients:
            return {
                "score": 0,
                "message": "No expert ingredient data available for comparison",
                "analysis": "Cannot perform comparison without expert reference data"
            }
        
        if not trainee_ingredients:
            return {
                "score": 0,
                "message": "No trainee ingredient data available",
                "analysis": "Cannot perform comparison without trainee data"
            }
        
        # Extract ingredient names
        trainee_names = set(ing['name'].lower() for ing in trainee_ingredients)
        expert_names = set(ing['name'].lower() for ing in expert_ingredients)
        
        # Calculate matches and differences
        matched = trainee_names & expert_names
        missing = expert_names - trainee_names
        extra = trainee_names - expert_names
        
        # Calculate base score
        match_score = (len(matched) / len(expert_names) * 100) if expert_names else 0
        missing_penalty = len(missing) * 15
        extra_penalty = len(extra) * 5
        
        raw_score = match_score - missing_penalty - extra_penalty
        final_score = max(0, min(100, raw_score))
        
        # Determine performance level
        if final_score >= 90:
            performance_level = "Excellent"
            performance_description = "Ingredient usage matches expert technique perfectly"
        elif final_score >= 75:
            performance_level = "Good"
            performance_description = "Ingredient usage is mostly correct with minor differences"
        elif final_score >= 60:
            performance_level = "Fair"
            performance_description = "Ingredient usage shows understanding but has some gaps"
        elif final_score >= 40:
            performance_level = "Needs Improvement"
            performance_description = "Significant ingredient differences from expert technique"
        else:
            performance_level = "Poor"
            performance_description = "Major ingredient usage issues requiring attention"
        
        # Generate insights
        insights = []
        
        if missing:
            insights.append({
                "type": "critical",
                "message": f"{len(missing)} essential ingredient(s) missing",
                "details": list(missing),
                "recommendation": f"Add these ingredients: {', '.join(missing)}"
            })
        
        if extra:
            insights.append({
                "type": "warning",
                "message": f"{len(extra)} unexpected ingredient(s) used",
                "details": list(extra),
                "recommendation": "Verify if these ingredients are necessary for the dish"
            })
        
        if len(matched) == len(expert_names) and not extra:
            insights.append({
                "type": "success",
                "message": "Perfect ingredient match with expert",
                "recommendation": "Excellent ingredient selection!"
            })
        
        # Compare quantities for matched ingredients
        quantity_analysis = []
        for ing_name in matched:
            trainee_ing = next((i for i in trainee_ingredients if i['name'].lower() == ing_name), None)
            expert_ing = next((i for i in expert_ingredients if i['name'].lower() == ing_name), None)
            
            if trainee_ing and expert_ing:
                trainee_qty = trainee_ing.get('quantity', 1)
                expert_qty = expert_ing.get('quantity', 1)
                
                if trainee_qty != expert_qty:
                    quantity_analysis.append({
                        "ingredient": ing_name,
                        "trainee_quantity": trainee_qty,
                        "expert_quantity": expert_qty,
                        "difference": trainee_qty - expert_qty
                    })
        
        if quantity_analysis:
            insights.append({
                "type": "info",
                "message": f"Quantity differences detected for {len(quantity_analysis)} ingredient(s)",
                "details": quantity_analysis,
                "recommendation": "Review ingredient quantities for accuracy"
            })
        
        # Calculate confidence comparison
        trainee_avg_confidence = sum(i['confidence'] for i in trainee_ingredients) / len(trainee_ingredients)
        expert_avg_confidence = sum(i['confidence'] for i in expert_ingredients) / len(expert_ingredients)
        
        return {
            "score": round(final_score, 2),
            "performance_level": performance_level,
            "performance_description": performance_description,
            
            "scoring_breakdown": {
                "match_score": round(match_score, 2),
                "missing_penalty": round(missing_penalty, 2),
                "extra_penalty": round(extra_penalty, 2)
            },
            
            "comparison_stats": {
                "matched_count": len(matched),
                "missing_count": len(missing),
                "extra_count": len(extra),
                "trainee_total": len(trainee_ingredients),
                "expert_total": len(expert_ingredients),
                "match_percentage": round(len(matched) / len(expert_names) * 100, 2) if expert_names else 0
            },
            
            "ingredients": {
                "matched": sorted(list(matched)),
                "missing": sorted(list(missing)),
                "extra": sorted(list(extra))
            },
            
            "quantity_analysis": quantity_analysis,
            
            "confidence_comparison": {
                "trainee_average": round(trainee_avg_confidence, 2),
                "expert_average": round(expert_avg_confidence, 2),
                "difference": round(trainee_avg_confidence - expert_avg_confidence, 2)
            },
            
            "insights": insights,
            
            "ingredient_details": {
                "trainee": trainee_ingredients,
                "expert": expert_ingredients
            }
        }
    
    def compare_utensil_usage(
        self,
        trainee_utensils: List[Dict[str, Any]],
        expert_utensils: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare utensil usage between trainee and expert
        
        Args:
            trainee_utensils: Trainee's detected utensils
            expert_utensils: Expert's detected utensils
            
        Returns:
            Comparison with score and insights
        """
        if not expert_utensils:
            return {
                "score": 0,
                "message": "No expert utensil data available"
            }
        
        if not trainee_utensils:
            return {
                "score": 0,
                "message": "No trainee utensil data available"
            }
        
        trainee_names = set(u['name'].lower() for u in trainee_utensils)
        expert_names = set(u['name'].lower() for u in expert_utensils)
        
        matched = trainee_names & expert_names
        missing = expert_names - trainee_names
        extra = trainee_names - expert_names
        
        match_score = (len(matched) / len(expert_names) * 100) if expert_names else 0
        missing_penalty = len(missing) * 10
        extra_penalty = len(extra) * 3
        
        final_score = max(0, min(100, match_score - missing_penalty - extra_penalty))
        
        insights = []
        if missing:
            insights.append({
                "type": "warning",
                "message": f"{len(missing)} essential utensil(s) not detected",
                "details": list(missing),
                "recommendation": "Ensure proper utensils are visible in video"
            })
        
        if extra:
            insights.append({
                "type": "info",
                "message": f"{len(extra)} additional utensil(s) used",
                "details": list(extra),
                "recommendation": "Extra utensils may indicate different technique"
            })
        
        return {
            "score": round(final_score, 2),
            "matched": sorted(list(matched)),
            "missing": sorted(list(missing)),
            "extra": sorted(list(extra)),
            "match_percentage": round(len(matched) / len(expert_names) * 100, 2) if expert_names else 0,
            "insights": insights
        }
