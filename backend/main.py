"""
BharatChef AI Coach - Main Application
This is the entry point for the FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="BharatChef AI Coach API",
    description="AI-powered cooking skill evaluation platform",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows web browsers to access your API
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
logger.info(f"CORS origins: {cors_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in cors_origins else cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from api.routes import action, object, heat, dish, evaluate, expert, trainee, admin, expert_multi_agent, trainee_multi_agent, evaluation
# from api.routes import chat  # Temporarily disabled - will enable after fixing

# Include routers
app.include_router(action.router)
app.include_router(object.router)
app.include_router(heat.router)
app.include_router(dish.router)
app.include_router(evaluate.router)
app.include_router(expert.router)
app.include_router(expert_multi_agent.router)  # Multi-agent advanced analysis for expert
app.include_router(trainee.router)
app.include_router(trainee_multi_agent.router)  # Multi-agent advanced analysis for trainee
app.include_router(evaluation.router)  # New evaluation endpoint
# app.include_router(chat.router)  # Temporarily disabled - will enable after fixing
app.include_router(admin.router)

# Serve frontend if it exists
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    @app.get("/ui")
    async def serve_ui():
        """Serve the frontend UI"""
        index_path = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Frontend not found"}

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - returns basic API information
    """
    return {
        "message": "Welcome to BharatChef AI Coach API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/ui"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint - verifies the API is running
    """
    return {
        "status": "healthy",
        "service": "BharatChef AI Coach",
        "environment": os.getenv("ENVIRONMENT", "dev")
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts
    """
    logger.info("🚀 BharatChef AI Coach API starting up...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'dev')}")
    logger.info(f"AWS Region: {os.getenv('AWS_REGION', 'not set')}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the application shuts down
    """
    logger.info("👋 BharatChef AI Coach API shutting down...")

# For running directly with python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
