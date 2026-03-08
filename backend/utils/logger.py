"""
Logging Configuration
Sets up logging for the entire application
"""
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application"""
    log_dir = Path("/opt/bharatchef/logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "bharatchef.log")
        ]
    )
    
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()
