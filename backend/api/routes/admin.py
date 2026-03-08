"""
Admin API - Database migrations and maintenance
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any
import logging

from api.dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)


@router.post("/migrate/add-video-url")
async def add_video_url_column(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Add video_url column to videos table if it doesn't exist
    """
    try:
        logger.info("Checking if video_url column exists...")
        
        # Check if column exists
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='videos' AND column_name='video_url'
        """))
        
        if result.fetchone():
            return {
                "status": "success",
                "message": "video_url column already exists",
                "action": "none"
            }
        
        logger.info("Adding video_url column...")
        
        # Add the column
        db.execute(text("""
            ALTER TABLE videos 
            ADD COLUMN video_url VARCHAR
        """))
        
        db.commit()
        logger.info("✅ video_url column added successfully!")
        
        # Show current columns
        result = db.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name='videos'
            ORDER BY ordinal_position
        """))
        
        columns = []
        for row in result:
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2]
            })
        
        return {
            "status": "success",
            "message": "video_url column added successfully",
            "action": "added",
            "columns": columns
        }
            
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Migration failed: {str(e)}"
        )


@router.get("/check/videos-table")
async def check_videos_table(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Check the current structure of the videos table
    """
    try:
        result = db.execute(text("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name='videos'
            ORDER BY ordinal_position
        """))
        
        columns = []
        for row in result:
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2],
                "default": row[3]
            })
        
        return {
            "status": "success",
            "table": "videos",
            "columns": columns
        }
            
    except Exception as e:
        logger.error(f"Check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Check failed: {str(e)}"
        )


@router.get("/check/dishes-table")
async def check_dishes_table(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Check the current structure of the dishes table
    """
    try:
        result = db.execute(text("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name='dishes'
            ORDER BY ordinal_position
        """))
        
        columns = []
        for row in result:
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2],
                "default": row[3]
            })
        
        return {
            "status": "success",
            "table": "dishes",
            "columns": columns
        }
            
    except Exception as e:
        logger.error(f"Check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Check failed: {str(e)}"
        )


@router.get("/test")
async def test_admin_endpoint():
    """Test admin endpoint"""
    return {
        "status": "success",
        "message": "Admin endpoint is ready",
        "endpoints": {
            "migrate": "POST /admin/migrate/add-video-url",
            "check_videos": "GET /admin/check/videos-table",
            "check_dishes": "GET /admin/check/dishes-table"
        }
    }
