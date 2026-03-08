"""
Database Initialization Script
Creates all tables in the PostgreSQL database
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from models.base import Base, engine
from models import Video, Evaluation, Dish
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize database by creating all tables"""
    try:
        logger.info("Creating database tables...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✓ Database tables created successfully!")
        logger.info(f"Tables created: {', '.join(Base.metadata.tables.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
