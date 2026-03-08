"""
AI Assistant - Simplified Version
"""
from typing import Dict, Any, List, Optional
from utils.bedrock_utils import BedrockClient
import json
import logging

logger = logging.getLogger(__name__)


class AIAssistant:
    """AI-powered chat assistant"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.bedrock_client = BedrockClient(region=region)
    
    async def generate_response(
        self,
        user_message: str,
        context: Dict[str, Any],
        db_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate AI response"""
        
        # Simple response for now
        return {
            'message': f"I received your message: {user_message}. This is a test response.",
            'action': None
        }
    
    async def handle_video_analysis(self, analysis: Dict[str, Any]) -> str:
        """Generate response for video analysis"""
        
        dish_name = analysis.get('name', 'Unknown Dish')
        confidence = analysis.get('confidence', 0)
        ingredients = analysis.get('ingredients', [])
        
        message = f"I've analyzed your video!\n\n"
        message += f"I detected: {dish_name} ({confidence:.0%} confidence)\n\n"
        message += f"Ingredients: {', '.join(ingredients[:8])}"
        if len(ingredients) > 8:
            message += f" and {len(ingredients) - 8} more"
        message += "\n\nIs this correct? (Yes/No/Edit)"
        
        return message
    
    async def handle_ingredient_edit(
        self,
        user_input: str,
        current_ingredients: List[str]
    ) -> Dict[str, Any]:
        """Parse and handle ingredient editing"""
        
        # Simple parsing
        to_add = []
        to_remove = []
        
        return {
            'updated_ingredients': current_ingredients,
            'added': to_add,
            'removed': to_remove
        }
