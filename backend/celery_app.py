"""
Celery Application Configuration
Handles asynchronous task processing for video analysis
"""
from celery import Celery
from config import settings
import logging

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'bharatchef',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['tasks.video_tasks']  # Import task modules
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
    
    # Task execution settings
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    
    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,
    
    # Queue settings
    task_default_queue='default',
    task_queues={
        'default': {
            'exchange': 'default',
            'routing_key': 'default',
        },
        'video_processing': {
            'exchange': 'video_processing',
            'routing_key': 'video.process',
        },
    },
    task_routes={
        'tasks.video_tasks.*': {'queue': 'video_processing'},
    },
)

logger.info("Celery app configured successfully")
logger.info(f"Broker: {settings.REDIS_URL}")
logger.info(f"Backend: {settings.REDIS_URL}")

# Optional: Add task events for monitoring
celery_app.conf.worker_send_task_events = True
celery_app.conf.task_send_sent_event = True
