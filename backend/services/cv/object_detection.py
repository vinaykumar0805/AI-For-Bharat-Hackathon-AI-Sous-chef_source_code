"""AI-Powered Object Detection using AWS Bedrock Claude Vision"""
import cv2
import numpy as np
import json
from typing import List, Dict
import logging
from utils.bedrock_utils import BedrockClient

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        self.bedrock = BedrockClient(region='us-east-1')
        logger.info("ObjectDetector initialized with Bedrock Claude Vision")
    
    def detect_objects(self, frames: List[np.ndarray], fps: float = 8.0) -> Dict:
        logger.info(f"Starting AI object detection on {len(frames)} frames")
        sample_interval = max(1, int(fps * 5))
        sample_frames = frames[::sample_interval]
        all_ingredients = {}
        all_utensils = {}
        
        prompt = """Analyze this cooking frame and identify ALL visible objects.

Respond ONLY with valid JSON (no other text):
{
  "ingredients": [{"name": "tomato", "confidence": 0.95}],
  "utensils": [{"name": "knife", "confidence": 0.92}]
}

If nothing: {"ingredients": [], "utensils": []}"""

        for i, frame in enumerate(sample_frames):
            try:
                frame_index = i * sample_interval
                timestamp = frame_index / fps
                
                response = self.bedrock.analyze_image(image=frame, prompt=prompt, model='haiku', max_tokens=500)
                response_clean = response.strip()
                if response_clean.startswith('```'):
                    lines = response_clean.split('\n')
                    response_clean = '\n'.join([l for l in lines if not l.startswith('```')])
                
                data = json.loads(response_clean)
                
                for ing in data.get("ingredients", []):
                    name = ing["name"].lower().strip()
                    if name not in all_ingredients:
                        all_ingredients[name] = {
                            "name": name,
                            "type": "ingredient",
                            "confidence": float(ing.get("confidence", 0.8)),
                            "first_seen": round(timestamp, 2),
                            "detections": 1
                        }
                    else:
                        all_ingredients[name]["detections"] += 1
                
                for utensil in data.get("utensils", []):
                    name = utensil["name"].lower().strip()
                    if name not in all_utensils:
                        all_utensils[name] = {
                            "name": name,
                            "type": "utensil",
                            "confidence": float(utensil.get("confidence", 0.8)),
                            "first_seen": round(timestamp, 2)
                        }
                
                logger.info(f"✅ Frame {i}: {len(data.get('ingredients', []))} ingredients, {len(data.get('utensils', []))} utensils")
            except Exception as e:
                logger.error(f"Error frame {i}: {e}")
                continue
        
        logger.info(f"✅ Detected {len(all_ingredients)} ingredients, {len(all_utensils)} utensils with AI")
        
        return {
            "ingredients": list(all_ingredients.values()),
            "utensils": list(all_utensils.values()),
            "total_objects": len(all_ingredients) + len(all_utensils)
        }
    
    def generate_ingredient_report(self, detected: List[Dict], expected: List[Dict]) -> Dict:
        detected_names = {ing["name"].lower() for ing in detected}
        expected_names = {ing["name"].lower() for ing in expected}
        missing = expected_names - detected_names
        extra = detected_names - expected_names
        matched = detected_names & expected_names
        accuracy = len(matched) / len(expected_names) * 100 if expected_names else 0
        
        return {
            "total_expected": len(expected_names),
            "total_detected": len(detected_names),
            "matched": list(matched),
            "missing": list(missing),
            "extra": list(extra),
            "accuracy": round(accuracy, 2),
            "status": "complete" if not missing else "incomplete"
        }
