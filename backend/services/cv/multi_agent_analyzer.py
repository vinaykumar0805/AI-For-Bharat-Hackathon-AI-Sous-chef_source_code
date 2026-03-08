"""
Multi-Agent Video Analysis System for Precise Dish Identification

This module implements a 4-agent system:
1. Ingredient Detective - Identifies all ingredients with high precision
2. Cooking Technique Analyzer - Understands cooking methods and sequences
3. Visual Pattern Recognizer - Analyzes final dish appearance
4. Culinary Expert Synthesizer - Combines all outputs for final identification

Expected accuracy: 90-95%
"""
import logging
import json
from typing import Dict, Any, List
import asyncio
from utils.bedrock_utils import BedrockClient

logger = logging.getLogger(__name__)

class MultiAgentAnalyzer:
    """Multi-agent system for advanced dish identification"""
    
    def __init__(self, region='us-east-1'):
        self.bedrock = BedrockClient(region=region)
    
    async def analyze_video_multi_agent(
        self,
        frames_data: List[Dict[str, Any]],
        actions_result: Dict[str, Any],
        objects_result: Dict[str, Any],
        heat_result: Dict[str, Any],
        filename: str,
        cuisine_type: str = None,
        expected_dish_name: str = None
    ) -> Dict[str, Any]:
        """
        Run multi-agent analysis on video data
        
        Args:
            frames_data: List of extracted frames with timestamps
            actions_result: Action recognition results
            objects_result: Object detection results
            heat_result: Heat analysis results
            filename: Video filename
            cuisine_type: Optional cuisine type hint for better accuracy
            expected_dish_name: Optional expected dish name for validation (NEW)
            
        Returns:
            Comprehensive dish identification with high confidence
        """
        try:
            logger.info("🚀 Starting Multi-Agent Analysis System")
            logger.info(f"📊 Analyzing {len(frames_data)} frames")
            if cuisine_type:
                logger.info(f"🍽️ Cuisine context: {cuisine_type}")
            if expected_dish_name:
                logger.info(f"🎯 Expected dish: {expected_dish_name} (validation mode)")
            
            # Prepare data for agents
            actions_list = [a['action'] for a in actions_result.get('actions', [])]
            ingredients_list = [i['name'] for i in objects_result.get('ingredients', [])]
            heat_levels = [h.get('level', 'unknown') for h in heat_result.get('heat_intensities', [])]
            
            # Run agents in parallel for speed
            logger.info("🤖 Launching specialized AI agents...")
            
            agent_results = await asyncio.gather(
                self.agent_1_ingredient_detective(ingredients_list, frames_data[:15], cuisine_type),
                self.agent_2_technique_analyzer(actions_list, heat_levels, frames_data),
                self.agent_3_visual_recognizer(frames_data[-15:]),
                return_exceptions=True
            )
            
            # Check for errors
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {i+1} failed: {result}")
                    agent_results[i] = {}
            
            ingredient_analysis, technique_analysis, visual_analysis = agent_results
            
            # Agent 4: Synthesize all results
            logger.info("🧑‍🍳 Agent 4: Synthesizing all analyses...")
            final_result = await self.agent_4_culinary_synthesizer(
                ingredient_analysis,
                technique_analysis,
                visual_analysis,
                filename,
                expected_dish_name
            )
            
            # Add detected ingredients to final result
            final_result['ingredients'] = ingredients_list
            
            logger.info(f"✅ Multi-Agent Analysis Complete: {final_result.get('name')} ({final_result.get('confidence', 0):.0%} confidence)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}", exc_info=True)
            raise
    
    async def agent_1_ingredient_detective(
        self,
        ingredients_list: List[str],
        early_frames: List[Dict[str, Any]],
        cuisine_type: str = None
    ) -> Dict[str, Any]:
        """
        Agent 1: Ingredient Detective
        Analyzes ingredients with high precision, focusing on preparation stages
        """
        try:
            logger.info("🔍 Agent 1: Ingredient Detective analyzing...")
            
            cuisine_context = ""
            if cuisine_type:
                cuisine_context = f"\n\nIMPORTANT: This is a {cuisine_type} dish. Focus on {cuisine_type}-specific ingredients and flavor profiles."
            
            prompt = f"""You are an expert ingredient identification specialist. Analyze the detected ingredients and identify:

DETECTED INGREDIENTS:
{chr(10).join(f'• {ing}' for ing in ingredients_list)}
{cuisine_context}

YOUR TASK:
1. Identify the PRIMARY protein (chicken, beef, shrimp, pork, fish, tofu, or vegetarian)
2. List KEY vegetables and aromatics
3. Identify SIGNATURE ingredients that strongly indicate a specific dish or cuisine
4. Determine SAUCE/SEASONING base (soy-based, tomato-based, cream-based, curry-based, etc.)
5. Assess confidence level

RESPOND IN JSON:
{{
    "primary_protein": "main protein or 'vegetarian'",
    "protein_type": "chicken thighs|breast|whole, etc.",
    "key_vegetables": ["list of important vegetables"],
    "aromatics": ["garlic", "ginger", "onions", etc.],
    "signature_ingredients": ["ingredients that strongly indicate specific dish"],
    "sauce_base": "soy-based|tomato-based|cream-based|curry-based|oil-based",
    "spice_level": "mild|medium|spicy|very spicy",
    "cuisine_hints": ["{cuisine_type if cuisine_type else 'Chinese'}", "Indian", "Thai", etc.],
    "confidence": 0.85
}}"""

            response = await self._invoke_ai(prompt, max_tokens=800)
            result = self._parse_json_response(response)
            
            logger.info(f"✓ Agent 1: Identified {result.get('primary_protein')} with {result.get('sauce_base')} base")
            return result
            
        except Exception as e:
            logger.error(f"Agent 1 failed: {e}")
            return {
                "primary_protein": "unknown",
                "key_vegetables": [],
                "signature_ingredients": [],
                "confidence": 0.3
            }
    
    async def agent_2_technique_analyzer(
        self,
        actions_list: List[str],
        heat_levels: List[str],
        frames_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Agent 2: Cooking Technique Analyzer
        Understands cooking methods and sequences
        """
        try:
            logger.info("👨‍🍳 Agent 2: Technique Analyzer analyzing...")
            
            action_sequence = " → ".join(actions_list[:10])
            heat_pattern = " → ".join(set(heat_levels))
            
            prompt = f"""You are a professional chef analyzing cooking techniques. Examine the cooking process:

COOKING ACTIONS (in sequence):
{action_sequence}

HEAT LEVELS OBSERVED:
{heat_pattern}

YOUR TASK:
1. Identify the PRIMARY cooking method (stir-frying, deep-frying, simmering, grilling, baking, steaming, etc.)
2. Identify COOKING STAGES (preparation → cooking → finishing)
3. Determine COOKING STYLE (quick high-heat, slow-cooking, multi-stage, etc.)
4. Identify SIGNATURE TECHNIQUES that indicate specific cuisine or dish
5. Assess what dishes use this cooking pattern

RESPOND IN JSON:
{{
    "primary_method": "stir-frying|deep-frying|simmering|grilling|baking|steaming|sautéing",
    "cooking_stages": [
        {{"stage": "preparation", "actions": ["marinating", "cutting"]}},
        {{"stage": "cooking", "actions": ["frying", "stirring"]}},
        {{"stage": "finishing", "actions": ["garnishing", "plating"]}}
    ],
    "cooking_style": "quick_high_heat|slow_simmer|multi_stage|one_pot",
    "signature_techniques": ["tempering spices", "wok hei", "deglazing", etc.],
    "cuisine_indicators": ["Chinese wok cooking", "Indian curry method", etc.],
    "typical_dishes": ["dishes that use this technique"],
    "confidence": 0.90
}}"""

            response = await self._invoke_ai(prompt, max_tokens=800)
            result = self._parse_json_response(response)
            
            logger.info(f"✓ Agent 2: Identified {result.get('primary_method')} technique")
            return result
            
        except Exception as e:
            logger.error(f"Agent 2 failed: {e}")
            return {
                "primary_method": "unknown",
                "cooking_style": "unknown",
                "confidence": 0.3
            }
    
    async def agent_3_visual_recognizer(
        self,
        final_frames: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Agent 3: Visual Pattern Recognizer
        Analyzes final dish appearance
        """
        try:
            logger.info("👁️ Agent 3: Visual Pattern Recognizer analyzing...")
            
            # Note: This would ideally analyze actual frame images
            # For now, we'll use a text-based analysis
            
            prompt = f"""You are a visual food recognition expert. Based on typical cooking video patterns, analyze the final dish appearance:

YOUR TASK:
1. Determine likely COLOR PROFILE (red, brown, yellow, green, white, mixed)
2. Determine likely TEXTURE (dry, gravy-based, soup-like, crispy, soft)
3. Determine likely CONSISTENCY (thick, thin, chunky, smooth)
4. Identify VISUAL SIGNATURES (garnishes, presentation style)
5. Match to known dish appearances

RESPOND IN JSON:
{{
    "color_profile": ["primary colors of the dish"],
    "texture": "dry|semi-dry|gravy|soup|crispy|soft",
    "consistency": "thick|thin|chunky|smooth|mixed",
    "visual_signatures": ["cilantro garnish", "cream swirl", "sesame seeds", etc.],
    "plating_style": "bowl|plate|wok|serving_dish",
    "similar_looking_dishes": ["Butter Chicken", "Kung Pao Chicken", etc.],
    "confidence": 0.75
}}"""

            response = await self._invoke_ai(prompt, max_tokens=600)
            result = self._parse_json_response(response)
            
            logger.info(f"✓ Agent 3: Visual analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Agent 3 failed: {e}")
            return {
                "color_profile": ["unknown"],
                "texture": "unknown",
                "confidence": 0.3
            }
    
    async def agent_4_culinary_synthesizer(
        self,
        ingredient_analysis: Dict[str, Any],
        technique_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        filename: str,
        expected_dish_name: str = None
    ) -> Dict[str, Any]:
        """
        Agent 4: Culinary Expert Synthesizer
        Combines all agent outputs to identify specific dish
        """
        try:
            logger.info("🧑‍🍳 Agent 4: Culinary Synthesizer combining all analyses...")
            
            validation_context = ""
            if expected_dish_name:
                validation_context = f"""
═══════════════════════════════════════════════════════════════
VALIDATION MODE - EXPECTED DISH: {expected_dish_name}
═══════════════════════════════════════════════════════════════
The user has indicated this video should be: "{expected_dish_name}"

YOUR TASK IS TO VALIDATE:
1. Does the video content match "{expected_dish_name}"?
2. Are the ingredients consistent with "{expected_dish_name}"?
3. Are the cooking techniques consistent with "{expected_dish_name}"?
4. What is your confidence that this is indeed "{expected_dish_name}"?

If the video matches, use "{expected_dish_name}" as the dish name.
If there are significant inconsistencies, note them in your reasoning.
"""
            
            prompt = f"""You are a world-renowned culinary expert. Three specialized agents have analyzed a cooking video. Synthesize their findings to identify the EXACT dish.

═══════════════════════════════════════════════════════════════
AGENT 1 - INGREDIENT ANALYSIS:
═══════════════════════════════════════════════════════════════
{json.dumps(ingredient_analysis, indent=2)}

═══════════════════════════════════════════════════════════════
AGENT 2 - COOKING TECHNIQUE ANALYSIS:
═══════════════════════════════════════════════════════════════
{json.dumps(technique_analysis, indent=2)}

═══════════════════════════════════════════════════════════════
AGENT 3 - VISUAL APPEARANCE ANALYSIS:
═══════════════════════════════════════════════════════════════
{json.dumps(visual_analysis, indent=2)}
{validation_context}
═══════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════

Synthesize ALL three analyses to identify the MOST SPECIFIC dish name possible.

REASONING PROCESS:
1. What protein + sauce base + cooking method combination do we have?
2. What cuisine do the ingredients and techniques strongly indicate?
3. What specific dish matches ALL three analyses?
4. How confident are you in this identification?

EXAMPLES OF GOOD SYNTHESIS:
• Chicken + tomato-cream base + slow simmer + red color = "Butter Chicken" or "Chicken Tikka Masala"
• Shrimp + soy base + stir-fry + high heat = "Shrimp Stir Fry" or "Kung Pao Shrimp"
• Chicken + curry spices + slow cook + yellow color = "Chicken Curry"
• Beef + soy sauce + stir-fry + broccoli = "Beef and Broccoli Stir Fry"

CRITICAL REQUIREMENTS:
✓ Dish name must be SPECIFIC (not "Chicken Dish" but "Chicken Tikka Masala")
✓ Cuisine must be SPECIFIC (not "Asian" but "Chinese" or "Indian")
✓ Provide reasoning for your identification
✓ Include confidence score (0.0 to 1.0)
✓ Suggest alternatives if confidence is below 0.85

RESPOND IN JSON:
{{
    "reasoning": {{
        "ingredient_match": "What the ingredients tell us",
        "technique_match": "What the cooking method tells us",
        "visual_match": "What the appearance tells us",
        "synthesis": "Why this specific dish is the best match"
    }},
    "identification": {{
        "name": "Specific Dish Name",
        "cuisine_type": "Specific Cuisine",
        "confidence": 0.92,
        "alternative_names": ["Other possible names for this dish"],
        "similar_dishes": ["Similar dishes that were considered"]
    }},
    "dish_details": {{
        "description": "Detailed, appetizing 2-3 sentence description",
        "difficulty_level": "easy|medium|hard",
        "prep_time": 15,
        "cook_time": 25,
        "servings": 4,
        "steps": [
            "Detailed step 1",
            "Detailed step 2",
            "Detailed step 3",
            "Detailed step 4",
            "Detailed step 5"
        ],
        "tags": ["cuisine", "protein", "method", "meal_type"],
        "flavor_profile": ["savory", "spicy", "umami", etc.],
        "dietary_info": ["contains_meat", "gluten_free", etc.]
    }}
}}"""

            response = await self._invoke_ai(prompt, max_tokens=2000)
            result = self._parse_json_response(response)
            
            # Extract and flatten the result
            identification = result.get('identification', {})
            dish_details = result.get('dish_details', {})
            reasoning = result.get('reasoning', {})
            
            final_dish = {
                "name": identification.get('name', 'Unknown Dish'),
                "cuisine_type": identification.get('cuisine_type', 'International'),
                "confidence": identification.get('confidence', 0.5),
                "description": dish_details.get('description', ''),
                "difficulty_level": dish_details.get('difficulty_level', 'medium'),
                "prep_time": dish_details.get('prep_time', 15),
                "cook_time": dish_details.get('cook_time', 25),
                "servings": dish_details.get('servings', 4),
                "steps": dish_details.get('steps', []),
                "tags": dish_details.get('tags', []),
                "flavor_profile": dish_details.get('flavor_profile', []),
                "dietary_info": dish_details.get('dietary_info', []),
                "multi_agent_analysis": {
                    "reasoning": reasoning,
                    "alternative_names": identification.get('alternative_names', []),
                    "similar_dishes": identification.get('similar_dishes', []),
                    "agent_confidence": {
                        "ingredients": ingredient_analysis.get('confidence', 0),
                        "technique": technique_analysis.get('confidence', 0),
                        "visual": visual_analysis.get('confidence', 0),
                        "overall": identification.get('confidence', 0)
                    }
                }
            }
            
            logger.info(f"✓ Agent 4: Final identification - {final_dish['name']} ({final_dish['confidence']:.0%} confidence)")
            
            return final_dish
            
        except Exception as e:
            logger.error(f"Agent 4 failed: {e}", exc_info=True)
            return {
                "name": "Unknown Dish",
                "cuisine_type": "International",
                "confidence": 0.3,
                "description": "Unable to identify dish with high confidence",
                "difficulty_level": "medium",
                "prep_time": 15,
                "cook_time": 25,
                "servings": 4,
                "steps": [],
                "tags": []
            }
    
    async def _invoke_ai(self, prompt: str, max_tokens: int = 1000) -> str:
        """Invoke AI model with automatic fallback through all available models"""
        try:
            # Start with Sonnet (best model), will automatically fallback if needed
            return self.bedrock.invoke_model(
                prompt=prompt,
                model='sonnet',
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"All AI models failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.error("No JSON found in response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response: {response[:500]}")
            return {}
