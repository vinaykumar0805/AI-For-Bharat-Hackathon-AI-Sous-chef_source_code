"""
Dish Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import logging

from api.dependencies import get_db
from schemas.dish import DishCreate, DishUpdate, DishResponse, DishList
from services.dish.dish_service import DishService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dishes",
    tags=["dishes"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=DishResponse, status_code=status.HTTP_201_CREATED)
async def create_dish(
    dish: DishCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new dish
    
    Args:
        dish: Dish data
        db: Database session
        
    Returns:
        Created dish
    """
    try:
        return DishService.create_dish(db, dish)
    except Exception as e:
        logger.error(f"Failed to create dish: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dish: {str(e)}"
        )

@router.get("/", response_model=DishList)
async def list_dishes(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all dishes with pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of dishes
    """
    try:
        dishes = DishService.get_dishes(db, skip=skip, limit=limit)
        total = DishService.count_dishes(db)
        return DishList(dishes=dishes, total=total)
    except Exception as e:
        logger.error(f"Failed to list dishes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list dishes: {str(e)}"
        )

@router.get("/{dish_id}", response_model=DishResponse)
async def get_dish(
    dish_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific dish by ID
    
    Args:
        dish_id: Dish ID
        db: Database session
        
    Returns:
        Dish details
    """
    dish = DishService.get_dish(db, dish_id)
    if not dish:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dish with ID {dish_id} not found"
        )
    return dish

@router.put("/{dish_id}", response_model=DishResponse)
async def update_dish(
    dish_id: int,
    dish_data: DishUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a dish
    
    Args:
        dish_id: Dish ID
        dish_data: Updated dish data
        db: Database session
        
    Returns:
        Updated dish
    """
    dish = DishService.update_dish(db, dish_id, dish_data)
    if not dish:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dish with ID {dish_id} not found"
        )
    return dish

@router.delete("/{dish_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dish(
    dish_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a dish
    
    Args:
        dish_id: Dish ID
        db: Database session
    """
    success = DishService.delete_dish(db, dish_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dish with ID {dish_id} not found"
        )

@router.post("/{dish_id}/expert-video/{video_id}", response_model=DishResponse)
async def associate_expert_video(
    dish_id: int,
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Associate an expert video with a dish
    
    Args:
        dish_id: Dish ID
        video_id: Video ID
        db: Database session
        
    Returns:
        Updated dish
    """
    dish = DishService.associate_expert_video(db, dish_id, video_id)
    if not dish:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dish with ID {dish_id} not found"
        )
    return dish

@router.get("/count/total")
async def count_dishes(db: Session = Depends(get_db)):
    """
    Get total count of dishes
    
    Args:
        db: Database session
        
    Returns:
        Total count
    """
    return {"total": DishService.count_dishes(db)}


@router.get("/by-cuisine/{cuisine_type}")
async def get_dishes_by_cuisine(
    cuisine_type: str,
    expert_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get all dishes for a specific cuisine type
    
    Args:
        cuisine_type: Cuisine type (e.g., Indian, Chinese, Italian)
        expert_only: If True, return only expert dishes (default: True)
        db: Database session
        
    Returns:
        List of dishes for the cuisine type
    """
    try:
        all_dishes = DishService.get_dishes(db)
        
        # Filter by cuisine type
        filtered_dishes = [
            d for d in all_dishes 
            if d.cuisine_type and d.cuisine_type.lower() == cuisine_type.lower()
        ]
        
        # Filter expert dishes only if requested
        if expert_only:
            filtered_dishes = [d for d in filtered_dishes if d.expert_video_id is not None]
        
        # Sort by creation date (newest first)
        filtered_dishes.sort(key=lambda x: x.created_at if x.created_at else '', reverse=True)
        
        return {
            "cuisine_type": cuisine_type,
            "expert_only": expert_only,
            "count": len(filtered_dishes),
            "dishes": [
                {
                    "dish_id": d.dish_id,
                    "name": d.name,
                    "description": d.description[:150] + "..." if d.description and len(d.description) > 150 else d.description,
                    "cuisine_type": d.cuisine_type,
                    "difficulty_level": d.difficulty_level,
                    "prep_time": d.prep_time,
                    "cook_time": d.cook_time,
                    "ingredients_count": len(d.ingredients) if d.ingredients else 0,
                    "steps_count": len(d.steps) if d.steps else 0,
                    "has_expert_video": d.expert_video_id is not None,
                    "created_at": d.created_at.isoformat() if d.created_at else None
                }
                for d in filtered_dishes
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get dishes by cuisine: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dishes by cuisine: {str(e)}"
        )


@router.get("/cuisines/list")
async def list_available_cuisines(db: Session = Depends(get_db)):
    """
    Get list of all available cuisine types with dish counts
    
    Args:
        db: Database session
        
    Returns:
        List of cuisine types with counts
    """
    try:
        all_dishes = DishService.get_dishes(db)
        
        # Count dishes by cuisine
        cuisine_counts = {}
        expert_counts = {}
        
        for dish in all_dishes:
            if dish.cuisine_type:
                cuisine = dish.cuisine_type
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
                
                if dish.expert_video_id:
                    expert_counts[cuisine] = expert_counts.get(cuisine, 0) + 1
        
        # Build response
        cuisines = [
            {
                "cuisine_type": cuisine,
                "total_dishes": count,
                "expert_dishes": expert_counts.get(cuisine, 0),
                "trainee_dishes": count - expert_counts.get(cuisine, 0)
            }
            for cuisine, count in cuisine_counts.items()
        ]
        
        # Sort by expert dish count (most first)
        cuisines.sort(key=lambda x: x['expert_dishes'], reverse=True)
        
        return {
            "total_cuisines": len(cuisines),
            "cuisines": cuisines
        }
    except Exception as e:
        logger.error(f"Failed to list cuisines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cuisines: {str(e)}"
        )


@router.get("/test/connection")
async def test_dish_endpoint(db: Session = Depends(get_db)):
    """
    Test dish endpoint and database connection
    
    Args:
        db: Database session
        
    Returns:
        Status message
    """
    try:
        count = DishService.count_dishes(db)
        return {
            "status": "success",
            "message": "Dish endpoint is working",
            "database_connected": True,
            "total_dishes": count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database connection failed: {str(e)}",
            "database_connected": False
        }
