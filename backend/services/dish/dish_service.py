"""
Dish Service - Business logic for dish management
"""
from sqlalchemy.orm import Session
from models.dish import Dish
from schemas.dish import DishCreate, DishUpdate
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DishService:
    """Service for managing dishes"""
    
    @staticmethod
    def create_dish(db: Session, dish_data: DishCreate) -> Dish:
        """Create a new dish"""
        try:
            import uuid
            dish = Dish(
                dish_id=f"dish_{uuid.uuid4().hex[:12]}",  # Generate unique dish_id
                name=dish_data.name,
                description=dish_data.description,
                cuisine_type=dish_data.cuisine_type,
                difficulty_level=dish_data.difficulty_level,
                prep_time=dish_data.prep_time,
                cook_time=dish_data.cook_time,
                servings=dish_data.servings,
                ingredients=dish_data.ingredients,
                steps=dish_data.steps,
                tags=dish_data.tags,
                expected_duration=dish_data.expected_duration,
                expected_steps=dish_data.expected_steps
            )
            db.add(dish)
            db.commit()
            db.refresh(dish)
            logger.info(f"Created dish: {dish.name} (ID: {dish.id}, dish_id: {dish.dish_id})")
            return dish
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create dish: {e}")
            raise
    
    @staticmethod
    def get_dish(db: Session, dish_id: int) -> Optional[Dish]:
        """Get a dish by integer ID"""
        return db.query(Dish).filter(Dish.id == dish_id).first()
    
    @staticmethod
    def get_dish_by_string_id(db: Session, dish_string_id: str) -> Optional[Dish]:
        """Get a dish by string dish_id (e.g., 'dish_abc123')"""
        return db.query(Dish).filter(Dish.dish_id == dish_string_id).first()
    
    @staticmethod
    def get_dishes(db: Session, skip: int = 0, limit: int = 100) -> List[Dish]:
        """Get all dishes with pagination"""
        return db.query(Dish).offset(skip).limit(limit).all()
    
    @staticmethod
    def update_dish(db: Session, dish_id: int, dish_data: DishUpdate) -> Optional[Dish]:
        """Update a dish"""
        try:
            dish = db.query(Dish).filter(Dish.dish_id == dish_id).first()
            if not dish:
                return None
            
            # Update only provided fields
            update_data = dish_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(dish, field, value)
            
            db.commit()
            db.refresh(dish)
            logger.info(f"Updated dish: {dish.name} (ID: {dish.dish_id})")
            return dish
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update dish: {e}")
            raise
    
    @staticmethod
    def delete_dish(db: Session, dish_id: int) -> bool:
        """Delete a dish"""
        try:
            dish = db.query(Dish).filter(Dish.dish_id == dish_id).first()
            if not dish:
                return False
            
            db.delete(dish)
            db.commit()
            logger.info(f"Deleted dish ID: {dish_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete dish: {e}")
            raise
    
    @staticmethod
    def associate_expert_video(db: Session, dish_id: int, video_id: int) -> Optional[Dish]:
        """Associate an expert video with a dish"""
        try:
            dish = db.query(Dish).filter(Dish.dish_id == dish_id).first()
            if not dish:
                return None
            
            dish.expert_video_id = video_id
            db.commit()
            db.refresh(dish)
            logger.info(f"Associated video {video_id} with dish {dish_id}")
            return dish
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to associate video: {e}")
            raise
    
    @staticmethod
    def count_dishes(db: Session) -> int:
        """Count total dishes"""
        return db.query(Dish).count()
