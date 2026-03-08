"""AI-Powered Action Recognition using AWS Bedrock Claude Vision"""
import cv2
import numpy as np
import json
from typing import List, Dict
import logging
from utils.bedrock_utils import BedrockClient

logger = logging.getLogger(__name__)

class ActionRecognizer:
    def __init__(self):
        self.bedrock = BedrockClient(region='us-east-1')
        logger.info("ActionRecognizer initialized with Bedrock Claude Vision")
    
    def recognize_actions(self, frames: List[np.ndarray], fps: float = 8.0) -> List[Dict]:
        logger.info(f"Starting AI action recognition on {len(frames)} frames")
        sample_interval = max(1, int(fps * 3))
        sample_frames = frames[::sample_interval]
        actions = []
        
        prompt = """Analyze this cooking video frame and identify the cooking action.

Respond ONLY with valid JSON (no other text):
{
  "action": "specific action (e.g., 'chopping onions', 'stirring curry')",
  "confidence": 0.85,
  "details": "brief description"
}

If no action: {"action": "no action", "confidence": 0.0, "details": ""}"""

        for i, frame in enumerate(sample_frames):
            try:
                frame_index = i * sample_interval
                timestamp = frame_index / fps
                
                response = self.bedrock.analyze_image(image=frame, prompt=prompt, model='haiku', max_tokens=300)
                response_clean = response.strip()
                if response_clean.startswith('```'):
                    lines = response_clean.split('\n')
                    response_clean = '\n'.join([l for l in lines if not l.startswith('```')])
                
                action_data = json.loads(response_clean)
                
                if action_data.get('confidence', 0) > 0.5:
                    actions.append({
                        "action": action_data["action"],
                        "confidence": float(action_data["confidence"]),
                        "start_time": round(timestamp, 2),
                        "end_time": round(timestamp + 3, 2),
                        "details": action_data.get("details", "")
                    })
                    logger.info(f"✅ Frame {i}: {action_data['action']}")
            except Exception as e:
                logger.error(f"Error frame {i}: {e}")
                continue
        
        logger.info(f"✅ Detected {len(actions)} actions with AI")
        return actions
    
    def get_action_summary(self, actions: List[Dict]) -> Dict:
        if not actions:
            return {"total_actions": 0, "unique_actions": 0, "actions": []}
        
        unique = {}
        for action in actions:
            name = action["action"].lower()
            if name not in unique:
                unique[name] = {
                    "action": action["action"],
                    "count": 1,
                    "first_seen": action["start_time"],
                    "last_seen": action["end_time"],
                    "avg_confidence": action["confidence"]
                }
            else:
                unique[name]["count"] += 1
                unique[name]["last_seen"] = action["end_time"]
        
        return {
            "total_actions": len(actions),
            "unique_actions": len(unique),
            "actions": list(unique.values())
        }
