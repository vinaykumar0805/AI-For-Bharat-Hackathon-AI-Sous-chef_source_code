"""
Expert Video Processing API
Upload and process expert cooking videos to create dish references
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import json
import uuid

from api.dependencies import get_db
from services.cv.action_recognition_ai import ActionRecognizer
from services.cv.object_detection_ai import ObjectDetector
from services.cv.heat_analysis_ai import HeatAnalyzer
from services.dish.dish_service import DishService
from models.video import Video
from utils.bedrock_utils import BedrockClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/expert",
    tags=["expert"],
    responses={404: {"description": "Not found"}},
)

# Initialize AI services
action_recognizer = ActionRecognizer(use_ai=True)
object_detector = ObjectDetector(use_ai=True)
heat_analyzer = HeatAnalyzer(use_ai=True)
bedrock_client = BedrockClient(region='us-east-1')


@router.post("/upload")
async def upload_expert_video(
    video: UploadFile = File(..., description="Expert cooking video"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload and process an expert cooking video.
    
    This endpoint:
    1. Analyzes the expert video with AI (actions, objects, heat)
    2. Extracts dish information using AI
    3. Creates dish entry in database
    4. Stores expert video record
    5. Returns dish details with database ID
    
    Use this to build your library of expert cooking references.
    """
    try:
        logger.info(f"Processing expert video: {video.filename}")
        
        # Read video content
        video_content = await video.read()
        
        # Step 1: Analyze video with AI
        logger.info("Step 1: Analyzing expert video with AI...")
        
        actions_result = await action_recognizer.recognize_actions(video_content)
        objects_result = await object_detector.detect_objects(video_content)
        heat_result = await heat_analyzer.analyze_heat(video_content)
        
        logger.info("✅ Video analysis complete")
        
        # Step 2: Extract dish information using AI
        logger.info("Step 2: Extracting dish information...")
        
        dish_info = await extract_dish_info_from_video(
            actions_result,
            objects_result,
            video.filename
        )
        
        logger.info(f"✅ Extracted dish: {dish_info['name']}")
        
        # Step 3: Check if dish already exists
        existing_dishes = DishService.get_dishes(db)
        existing_dish = next((d for d in existing_dishes if d.name.lower() == dish_info['name'].lower()), None)
        
        if existing_dish:
            logger.info(f"⚠️ Dish '{dish_info['name']}' already exists (ID: {existing_dish.id})")
            return {
                "status": "exists",
                "message": f"Dish '{dish_info['name']}' already exists in database",
                "dish_id": existing_dish.id,
                "dish_string_id": existing_dish.dish_id,
                "dish_info": {
                    "dish_id": existing_dish.id,
                    "dish_string_id": existing_dish.dish_id,
                    "name": existing_dish.name,
                    "description": existing_dish.description,
                    "cuisine_type": existing_dish.cuisine_type,
                    "ingredients": existing_dish.ingredients,
                    "steps": existing_dish.steps
                },
                "suggestion": "Use PUT /expert/update/{dish_id} to update, or use different dish name"
            }
        
        # Step 4: Create dish entry FIRST
        logger.info("Step 3: Creating dish entry...")
        
        from schemas.dish import DishCreate
        
        # Calculate expected duration from video analysis
        expected_duration = None
        if heat_result and heat_result.get('heat_intensities'):
            # Estimate duration from heat analysis timestamps
            heat_times = [h.get('timestamp', 0) for h in heat_result['heat_intensities']]
            if heat_times:
                expected_duration = int(max(heat_times))
        
        # Add expected fields to dish_info
        dish_info['expected_duration'] = expected_duration
        dish_info['expected_steps'] = actions_result.get('actions', [])
        
        dish_data = DishCreate(**dish_info)
        dish = DishService.create_dish(db, dish_data)
        db.flush()  # Flush to get the id
        
        logger.info(f"✅ Dish created: {dish.name} (ID: {dish.id}, dish_id: {dish.dish_id})")
        
        # Step 5: Create video record with dish_id
        logger.info("Step 4: Creating video record...")
        
        video_record = Video(
            video_id=f"expert_{uuid.uuid4().hex[:8]}",
            dish_id=str(dish.dish_id),
            video_type="EXPERT",  # Mark as expert video
            camera_type="OVERHEAD",
            cloud_url=f"s3://bharatchef-videos/expert/{video.filename}",
            format="mp4",
            duration=expected_duration if expected_duration else 0.0,
            file_size=len(video_content),
            processing_status="COMPLETED"
        )
        db.add(video_record)
        db.flush()
        
        logger.info(f"✅ Video stored (ID: {video_record.video_id})")
        
        # Step 6: Associate expert video with dish
        dish.expert_video_id = video_record.video_id
        db.commit()
        db.refresh(dish)
        
        logger.info(f"✅ Expert video associated with dish")
        
        # Step 6: Return comprehensive results
        return {
            "status": "success",
            "message": f"Expert video processed and dish '{dish.name}' created successfully",
            
            "dish_id": dish.id,
            "dish_string_id": dish.dish_id,
            "video_id": video_record.video_id,
            
            "dish_info": {
                "dish_id": dish.id,
                "dish_string_id": dish.dish_id,
                "name": dish.name,
                "description": dish.description,
                "cuisine_type": dish.cuisine_type,
                "difficulty_level": dish.difficulty_level,
                "prep_time": dish.prep_time,
                "cook_time": dish.cook_time,
                "servings": dish.servings,
                "ingredients": dish.ingredients,
                "steps": dish.steps,
                "tags": dish.tags
            },
            
            "expert_analysis": {
                "actions": {
                    "count": len(actions_result.get('actions', [])),
                    "sequence": [a['action'] for a in actions_result.get('actions', [])]
                },
                "ingredients": {
                    "count": len(objects_result.get('ingredients', [])),
                    "list": [i['name'] for i in objects_result.get('ingredients', [])]
                },
                "heat_control": {
                    "count": len(heat_result.get('heat_intensities', [])),
                    "pattern": heat_result.get('heat_intensities', [])
                }
            },
            
            "next_steps": {
                "view_dish": f"/dishes/{dish.dish_id}",
                "update_dish": f"/expert/update/{dish.dish_id}",
                "evaluate_trainee": f"/trainee/evaluate?dish_id={dish.dish_id}"
            }
        }
        
    except Exception as e:
        logger.error(f"Expert video processing failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Expert video processing failed: {str(e)}"
        )


@router.put("/update/{dish_id}")
async def update_expert_video(
    dish_id: int,
    video: UploadFile = File(..., description="Updated expert cooking video"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update an existing dish with a new expert video.
    
    This re-analyzes the video and updates the dish information.
    """
    try:
        # Check if dish exists
        dish = DishService.get_dish(db, dish_id)
        if not dish:
            raise HTTPException(status_code=404, detail=f"Dish with ID {dish_id} not found")
        
        logger.info(f"Updating dish '{dish.name}' (ID: {dish_id}) with new expert video")
        
        # Read and analyze video
        video_content = await video.read()
        
        actions_result = await action_recognizer.recognize_actions(video_content)
        objects_result = await object_detector.detect_objects(video_content)
        heat_result = await heat_analyzer.analyze_heat(video_content)
        
        # Extract updated dish info
        dish_info = await extract_dish_info_from_video(
            actions_result,
            objects_result,
            video.filename
        )
        
        # Create new video record
        video_record = Video(
            video_id=f"expert_{uuid.uuid4().hex[:8]}",
            dish_id=str(dish_id),
            camera_type="OVERHEAD",
            cloud_url=f"s3://bharatchef-videos/expert/{video.filename}",
            format="mp4",
            duration=0.0,
            file_size=len(video_content),
            processing_status="COMPLETED"
        )
        db.add(video_record)
        db.flush()
        
        # Update dish
        from schemas.dish import DishUpdate
        update_data = DishUpdate(**dish_info)
        updated_dish = DishService.update_dish(db, dish_id, update_data)
        
        # Update expert video reference
        updated_dish.expert_video_id = video_record.video_id
        db.commit()
        
        return {
            "status": "success",
            "message": f"Dish '{updated_dish.name}' updated successfully",
            "dish_id": updated_dish.dish_id,
            "video_id": video_record.video_id,
            "dish_info": {
                "dish_id": updated_dish.dish_id,
                "name": updated_dish.name,
                "description": updated_dish.description,
                "ingredients": updated_dish.ingredients,
                "steps": updated_dish.steps
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.get("/list")
async def list_expert_dishes(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    List all expert dishes available for trainee evaluation.
    """
    try:
        dishes = DishService.get_dishes(db)
        
        return {
            "status": "success",
            "total": len(dishes),
            "dishes": [
                {
                    "dish_id": dish.dish_id,
                    "name": dish.name,
                    "description": dish.description,
                    "cuisine_type": dish.cuisine_type,
                    "difficulty_level": dish.difficulty_level,
                    "ingredients_count": len(dish.ingredients) if dish.ingredients else 0,
                    "steps_count": len(dish.steps) if dish.steps else 0,
                    "has_expert_video": dish.expert_video_id is not None
                }
                for dish in dishes
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list dishes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list dishes: {str(e)}")


@router.get("/test")
async def test_expert_endpoint():
    """Test expert endpoint"""
    return {
        "status": "success",
        "message": "Expert video processing endpoint is ready",
        "features": [
            "Upload expert cooking videos",
            "AI extracts dish information automatically",
            "Creates dish reference in database",
            "Stores expert video for comparison",
            "Update existing dishes with new videos"
        ],
        "endpoints": {
            "upload": "POST /expert/upload",
            "update": "PUT /expert/update/{dish_id}",
            "list": "GET /expert/list"
        }
    }


async def extract_dish_info_from_video(
    actions_result: Dict[str, Any],
    objects_result: Dict[str, Any],
    filename: str
) -> Dict[str, Any]:
    """
    Use AI to extract detailed dish information from video analysis results
    """
    try:
        actions_list = [a['action'] for a in actions_result.get('actions', [])]
        ingredients_list = [i['name'] for i in objects_result.get('ingredients', [])]
        
        # Enhanced prompt for better dish identification
        prompt = f"""You are a professional chef and culinary expert. Analyze this cooking video data and identify the specific dish being prepared.

VIDEO ANALYSIS DATA:
- Filename: {filename}
- Cooking Actions: {', '.join(actions_list)}
- Ingredients Detected: {', '.join(ingredients_list)}

TASK: Based on the ingredients and cooking techniques, identify the EXACT dish name and provide complete details.

INSTRUCTIONS:
1. Dish Name: Provide the SPECIFIC dish name (not generic like "Cook" or "Cooking"). Examples: "Chicken Stir Fry", "Shrimp Fried Rice", "Vegetable Lo Mein"
2. Description: Write a 2-3 sentence description of the dish
3. Cuisine Type: Identify the specific cuisine (Chinese, Indian, Italian, Thai, Japanese, Mexican, American, etc.) based on ingredients and cooking style
4. Difficulty Level: Rate as "easy", "medium", or "hard"
5. Prep Time: Realistic preparation time in minutes
6. Cook Time: Realistic cooking time in minutes
7. Servings: Typical number of servings
8. Cooking Steps: List 4-6 clear, sequential cooking steps (not just actions like "frying")
9. Tags: Add 3-5 relevant tags (cuisine type, main protein, cooking method, meal type)

IMPORTANT: 
- Be SPECIFIC with the dish name based on the ingredients
- If you see chicken + broccoli + soy sauce = likely "Chicken and Broccoli Stir Fry"
- If you see shrimp + vegetables + soy sauce = likely "Shrimp Stir Fry" or "Shrimp Fried Rice"
- Infer the cuisine from ingredients (soy sauce = Asian, curry = Indian, etc.)

Respond ONLY with valid JSON in this exact format:
{{
    "name": "Specific Dish Name Here",
    "description": "Detailed description of the dish, its flavors, and characteristics",
    "cuisine_type": "Specific Cuisine Type",
    "difficulty_level": "easy|medium|hard",
    "prep_time": 15,
    "cook_time": 20,
    "servings": 4,
    "steps": [
        "Prepare and cut all vegetables and proteins",
        "Heat oil in wok or pan over high heat",
        "Stir fry proteins until cooked through",
        "Add vegetables and sauce, stir fry until tender",
        "Season and serve hot"
    ],
    "tags": ["cuisine_type", "main_ingredient", "cooking_method", "meal_type"]
}}"""

        response = bedrock_client.invoke_model(
            prompt=prompt,
            model='nova-lite',
            max_tokens=1000
        )
        
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            dish_info = json.loads(json_str)
            
            # Always use detected ingredients
            dish_info['ingredients'] = ingredients_list if ingredients_list else dish_info.get('ingredients', [])
            
            # Validate and set defaults if needed
            if not dish_info.get('name') or dish_info['name'] in ['Cook', 'Cooking', 'Dish']:
                dish_info['name'] = infer_dish_name_from_ingredients(ingredients_list, actions_list)
            
            if not dish_info.get('cuisine_type') or dish_info['cuisine_type'] == 'Unknown':
                dish_info['cuisine_type'] = infer_cuisine_from_ingredients(ingredients_list)
            
            return dish_info
        else:
            # Fallback with intelligent inference
            return create_fallback_dish_info(filename, ingredients_list, actions_list)
            
    except Exception as e:
        logger.error(f"Failed to extract dish info: {e}")
        return create_fallback_dish_info(filename, ingredients_list, actions_list)


def infer_dish_name_from_ingredients(ingredients: list, actions: list) -> str:
    """Infer dish name from ingredients"""
    ingredients_lower = [i.lower() for i in ingredients]
    
    # Check for common dish patterns
    has_chicken = any('chicken' in i for i in ingredients_lower)
    has_shrimp = any('shrimp' in i for i in ingredients_lower)
    has_beef = any('beef' in i for i in ingredients_lower)
    has_pork = any('pork' in i for i in ingredients_lower)
    has_broccoli = any('broccoli' in i for i in ingredients_lower)
    has_rice = any('rice' in i for i in ingredients_lower)
    has_noodles = any('noodle' in i for i in ingredients_lower)
    has_soy_sauce = any('soy' in i for i in ingredients_lower)
    
    # Infer dish name
    if has_soy_sauce or any('stir' in a.lower() for a in actions):
        if has_chicken and has_broccoli:
            return "Chicken and Broccoli Stir Fry"
        elif has_shrimp and has_broccoli:
            return "Shrimp and Broccoli Stir Fry"
        elif has_chicken:
            return "Chicken Stir Fry"
        elif has_shrimp:
            return "Shrimp Stir Fry"
        elif has_beef:
            return "Beef Stir Fry"
        else:
            return "Vegetable Stir Fry"
    
    if has_rice and any('fry' in a.lower() for a in actions):
        if has_shrimp:
            return "Shrimp Fried Rice"
        elif has_chicken:
            return "Chicken Fried Rice"
        else:
            return "Fried Rice"
    
    # Generic fallback
    main_protein = next((i for i in ['chicken', 'shrimp', 'beef', 'pork', 'fish'] if any(i in ing for ing in ingredients_lower)), 'vegetable')
    return f"{main_protein.title()} Dish"


def infer_cuisine_from_ingredients(ingredients: list) -> str:
    """Infer cuisine type from ingredients"""
    ingredients_lower = [i.lower() for i in ingredients]
    
    # Asian cuisine indicators
    asian_indicators = ['soy sauce', 'rice', 'noodle', 'wok', 'sesame', 'ginger', 'bok choy']
    if any(ind in ' '.join(ingredients_lower) for ind in asian_indicators):
        return "Asian"
    
    # Indian cuisine indicators
    indian_indicators = ['curry', 'turmeric', 'cumin', 'coriander', 'garam masala', 'naan', 'paneer']
    if any(ind in ' '.join(ingredients_lower) for ind in indian_indicators):
        return "Indian"
    
    # Italian cuisine indicators
    italian_indicators = ['pasta', 'tomato sauce', 'parmesan', 'basil', 'mozzarella', 'olive oil']
    if any(ind in ' '.join(ingredients_lower) for ind in italian_indicators):
        return "Italian"
    
    # Mexican cuisine indicators
    mexican_indicators = ['tortilla', 'salsa', 'avocado', 'cilantro', 'lime', 'jalapeño']
    if any(ind in ' '.join(ingredients_lower) for ind in mexican_indicators):
        return "Mexican"
    
    return "International"


def create_fallback_dish_info(filename: str, ingredients: list, actions: list) -> Dict[str, Any]:
    """Create fallback dish info with intelligent inference"""
    dish_name = infer_dish_name_from_ingredients(ingredients, actions)
    cuisine_type = infer_cuisine_from_ingredients(ingredients)
    
    return {
        "name": dish_name,
        "description": f"A delicious {cuisine_type.lower()} dish featuring {', '.join(ingredients[:3])}",
        "cuisine_type": cuisine_type,
        "difficulty_level": "medium",
        "prep_time": 15,
        "cook_time": 25,
        "servings": 4,
        "ingredients": ingredients,
        "steps": [
            "Prepare and wash all ingredients",
            "Cut vegetables and proteins into appropriate sizes",
            "Heat cooking oil in pan or wok",
            "Cook proteins until done, then add vegetables",
            "Add seasonings and sauces, stir well",
            "Serve hot"
        ],
        "tags": [cuisine_type.lower(), "main course", "homemade"]
    }
