# BharatChef AI Coach - Prerequisites & Setup Guide

## System Requirements

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS 10.15+
- Windows 10/11 with WSL2

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space
- GPU: Optional (for faster video processing)

---

## Software Prerequisites

### 1. Python
- Version: Python 3.9 or 3.10
- Check version: `python3 --version`
- Install: https://www.python.org/downloads/

### 2. pip (Python Package Manager)
- Usually comes with Python
- Check version: `pip3 --version`
- Upgrade: `python3 -m pip install --upgrade pip`

### 3. Redis (for Celery task queue)
- Version: 6.0+
- Install on Ubuntu: `sudo apt-get install redis-server`
- Install on macOS: `brew install redis`
- Start service: `redis-server`

### 4. FFmpeg (for video processing)
- Required for OpenCV video operations
- Install on Ubuntu: `sudo apt-get install ffmpeg`
- Install on macOS: `brew install ffmpeg`
- Install on Windows: Download from https://ffmpeg.org/

### 5. AWS Account & Credentials
- AWS Account with Bedrock access
- IAM user with permissions for:
  - Amazon Bedrock (nova-lite model)
  - S3 (for video storage)
- AWS Access Key ID and Secret Access Key

---

## Python Dependencies

All Python packages are listed in `requirements.txt`. Install with:

```bash
pip3 install -r requirements.txt
```

### Key Dependencies:
- **FastAPI**: Web framework for API
- **Uvicorn**: ASGI server
- **SQLAlchemy**: Database ORM
- **Boto3**: AWS SDK for Python
- **Celery**: Distributed task queue
- **Redis**: Message broker for Celery
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing

---

## AWS Configuration

### 1. Enable Bedrock Models
You need to enable the following models in AWS Bedrock console:
- `amazon.nova-lite-v1:0` (primary model)

Steps:
1. Go to AWS Console → Bedrock → Model access
2. Request access to Amazon Nova Lite
3. Wait for approval (usually instant)

### 2. IAM Permissions
Your IAM user/role needs these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```

### 3. Configure Credentials
Edit `backend/config.py` and set:

```python
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_REGION = "us-east-1"  # or your preferred region
```

Or use environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

---

## Database Setup

The application uses SQLite by default (included in download).

### Initialize Database
```bash
cd backend
python3 init_database.py
```

This creates:
- `bharatchef.db` - SQLite database file
- All required tables (dishes, evaluations, videos, etc.)

---

## Installation Steps

### 1. Download Code
Code downloaded to: `C:\AI-Bharat-finalcode`

### 2. Create Virtual Environment (Recommended)
```bash
cd C:\AI-Bharat-finalcode
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure AWS
Edit `backend/config.py` with your AWS credentials

### 5. Initialize Database
```bash
cd backend
python init_database.py
```

### 6. Start Redis (in separate terminal)
```bash
redis-server
```

### 7. Start Celery Worker (in separate terminal)
```bash
cd backend
celery -A celery_app worker --loglevel=info
```

### 8. Start API Server
```bash
cd backend
python main.py
```

The API will start on: http://localhost:8000

### 9. Open Frontend
Open `frontend/index.html` in your browser, or access:
http://localhost:8000/ui

---

## Verification

### Check API is Running
```bash
curl http://localhost:8000/
```

Should return: `{"message": "BharatChef AI Coach API"}`

### Check Bedrock Access
```bash
curl http://localhost:8000/admin/test-bedrock
```

Should return success message if AWS is configured correctly.

---

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'cv2'"
Solution: Install OpenCV
```bash
pip install opencv-python opencv-python-headless
```

### Issue: "Redis connection refused"
Solution: Start Redis server
```bash
redis-server
```

### Issue: "Bedrock access denied"
Solution: 
1. Check AWS credentials in config.py
2. Verify IAM permissions
3. Enable Bedrock models in AWS console

### Issue: "Database locked"
Solution: Close any other processes using the database
```bash
pkill -9 python3
rm bharatchef.db-journal  # if exists
```

---

## Production Deployment

For production deployment on EC2:

1. Use a proper web server (Nginx + Gunicorn)
2. Use PostgreSQL instead of SQLite
3. Set up SSL/TLS certificates
4. Configure firewall rules
5. Use systemd for process management
6. Set up monitoring and logging

---

## Architecture Overview

```
BharatChef AI Coach
├── Backend (FastAPI)
│   ├── API Routes (evaluation, expert, trainee, chat)
│   ├── Services (CV, AI, video processing)
│   ├── Models (SQLAlchemy ORM)
│   └── Tasks (Celery background jobs)
├── Frontend (HTML/JS)
│   └── Single-page application
├── Database (SQLite/PostgreSQL)
│   └── Dishes, evaluations, videos
└── AWS Services
    ├── Bedrock (AI models)
    └── S3 (video storage - optional)
```

---

## Features

- Expert video upload and analysis
- Trainee video upload and analysis
- AI-powered evaluation and comparison
- Ingredient matching (fuzzy + AI)
- Step-by-step action comparison
- Heat analysis
- Detailed recommendations
- AI chatbot assistant
- Frame caching for performance

---

## Technology Stack

- **Backend**: Python 3.9, FastAPI, SQLAlchemy
- **Frontend**: HTML5, JavaScript, CSS
- **AI**: AWS Bedrock (Amazon Nova Lite)
- **Computer Vision**: OpenCV, PyTorch
- **Database**: SQLite (dev), PostgreSQL (prod)
- **Task Queue**: Celery + Redis
- **Deployment**: EC2, Uvicorn, Systemd

---

Last Updated: March 8, 2026
Version: 1.0 (Production)
