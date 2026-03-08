"""
Configuration Management
Loads all settings from environment variables
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

# Get the directory where this config file is located
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "BharatChef AI Coach"
    ENVIRONMENT: str = "dev"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # AWS
    AWS_REGION: str
    AWS_ACCOUNT_ID: str
    
    # VPC and Security Groups
    VPC_ID: Optional[str] = None
    RDS_SECURITY_GROUP_ID: Optional[str] = None
    REDIS_SECURITY_GROUP_ID: Optional[str] = None
    EC2_SECURITY_GROUP_ID: Optional[str] = None
    LAMBDA_SECURITY_GROUP_ID: Optional[str] = None
    
    # EC2
    EC2_INSTANCE_ID: str
    EC2_PUBLIC_IP: str
    EC2_PRIVATE_IP: str
    
    # Database
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USERNAME: str
    DB_PASSWORD: str
    DATABASE_URL: str
    
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_URL: str
    
    # S3
    VIDEO_BUCKET_NAME: str
    S3_BUCKET_ARN: str
    
    # SQS
    PROCESSING_QUEUE_URL: str
    PROCESSING_QUEUE_ARN: str
    
    # Cognito
    COGNITO_USER_POOL_ID: str
    COGNITO_USER_POOL_ARN: Optional[str] = None
    COGNITO_CLIENT_ID: str
    COGNITO_REGION: str
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # Aliases for AWS utilities (for backward compatibility)
    @property
    def S3_BUCKET_NAME(self) -> str:
        """Alias for VIDEO_BUCKET_NAME"""
        return self.VIDEO_BUCKET_NAME
    
    @property
    def SQS_QUEUE_URL(self) -> str:
        """Alias for PROCESSING_QUEUE_URL"""
        return self.PROCESSING_QUEUE_URL
    
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()
