"""
Automated Video Processing Tasks
Celery tasks for end-to-end video processing
"""
from celery import Task
from celery_app import celery_app
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.process_video_complete')
async def process_video_complete(self: Task, video_id: str, video_url: str, dish_id: int = None) -> Dict:
    """
    Complete automated video processing pipeline
    
    Args:
        video_id: Unique video identifier
        video_url: S3 URL or local path to video
        dish_id: Optional dish ID for ingredient comparison
        
    Returns:
        Complete processing results
    """
    try:
        logger.info(f"Starting automated processing for video: {video_id}")
        
        # Update task state
        self.update_state(state='PROCESSING', meta={'step': 'initializing', 'progress': 0})
        
        # Step 1: Download video from S3 or read from local path
        logger.info("Step 1: Loading video...")
        self.update_state(state='PROCESSING', meta={'step': 'loading_video', 'progress': 10})
        
        # Download video content
        import requests
        if video_url.startswith('http'):
            response = requests.get(video_url)
            video_content = response.content
        else:
            with open(video_url, 'rb') as f:
                video_content = f.read()
        
        logger.info(f"Video loaded: {len(video_content)} bytes")
        
        # Step 2: Action Recognition with AI
        logger.info("Step 2: Recognizing actions with Bedrock AI...")
        self.update_state(state='PROCESSING', meta={'step': 'action_recognition', 'progress': 30})
        
        from services.cv.action_recognition_ai import ActionRecognizer
        action_recognizer = ActionRecognizer(use_ai=True)
        action_result = await action_recognizer.recognize_actions(video_content)
        actions = action_result.get('actions', [])
        
        logger.info(f"Detected {len(actions)} actions using {action_result.get('method', 'unknown')}")
        
        # Step 3: Object Detection with AI
        logger.info("Step 3: Detecting objects with Bedrock AI...")
        self.update_state(state='PROCESSING', meta={'step': 'object_detection', 'progress': 60})
        
        from services.cv.object_detection_ai import ObjectDetector
        object_detector = ObjectDetector(use_ai=True)
        object_result = await object_detector.detect_objects(video_content)
        
        logger.info(f"Detected {len(object_result.get('ingredients', []))} ingredients and {len(object_result.get('utensils', []))} utensils using {object_result.get('method', 'unknown')}")
        
        # Step 4: Generate ingredient report (if dish_id provided)
        ingredient_report = None
        if dish_id:
            logger.info("Step 4: Generating ingredient report...")
            self.update_state(state='PROCESSING', meta={'step': 'ingredient_report', 'progress': 80})
            
            # Get expected ingredients from dish
            from models import Dish
            from models.base import SessionLocal
            
            db = SessionLocal()
            try:
                dish = db.query(Dish).filter(Dish.id == dish_id).first()
                if dish and dish.expected_ingredients:
                    expected_ingredients = dish.expected_ingredients
                    ingredient_report = object_detector.generate_ingredient_report(
                        object_result['ingredients'],
                        expected_ingredients
                    )
            finally:
                db.close()
        
        # Step 5: Save results to database
        logger.info("Step 5: Saving results to database...")
        self.update_state(state='PROCESSING', meta={'step': 'saving_results', 'progress': 90})
        
        from models import Video, Evaluation
        from models.base import SessionLocal
        from datetime import datetime
        
        db = SessionLocal()
        try:
            # Save or update video record
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if not video:
                video = Video(
                    video_id=video_id,
                    video_url=video_url,
                    processing_status='completed',
                    uploaded_at=datetime.utcnow()
                )
                db.add(video)
            else:
                video.processing_status = 'completed'
            
            db.commit()
            db.refresh(video)
            
            # Save evaluation results
            evaluation = Evaluation(
                video_id=video.id,
                dish_id=dish_id,
                action_sequence=actions,
                detected_objects=object_result,
                ingredient_report=ingredient_report,
                overall_score=None,  # Will be calculated in scoring task
                evaluated_at=datetime.utcnow()
            )
            db.add(evaluation)
            db.commit()
            db.refresh(evaluation)
            
            evaluation_id = evaluation.id
            
        finally:
            db.close()
        
        # Step 6: Complete
        logger.info("Processing complete!")
        self.update_state(state='SUCCESS', meta={'step': 'complete', 'progress': 100})
        
        # Return complete results
        result = {
            'video_id': video_id,
            'status': 'completed',
            'evaluation_id': evaluation_id,
            'actions': {
                'total_actions': len(actions),
                'actions': actions,
                'method': action_result.get('method', 'unknown')
            },
            'objects': {
                'ingredients': object_result.get('ingredients', []),
                'utensils': object_result.get('utensils', []),
                'method': object_result.get('method', 'unknown')
            },
            'ingredient_report': ingredient_report,
            'processing_time': time.time()
        }
        
        logger.info(f"Video {video_id} processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        
        # Update video status to failed
        from models import Video
        from models.base import SessionLocal
        
        db = SessionLocal()
        try:
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if video:
                video.processing_status = 'failed'
                db.commit()
        finally:
            db.close()
        
        raise


@celery_app.task(name='tasks.get_processing_status')
def get_processing_status(task_id: str) -> Dict:
    """
    Get status of video processing task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and progress
    """
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state == 'PENDING':
        response = {
            'state': result.state,
            'status': 'Task is waiting to be processed'
        }
    elif result.state == 'PROCESSING':
        response = {
            'state': result.state,
            'step': result.info.get('step', ''),
            'progress': result.info.get('progress', 0),
            'status': 'Processing...'
        }
    elif result.state == 'SUCCESS':
        response = {
            'state': result.state,
            'result': result.result,
            'status': 'Processing complete'
        }
    elif result.state == 'FAILURE':
        response = {
            'state': result.state,
            'error': str(result.info),
            'status': 'Processing failed'
        }
    else:
        response = {
            'state': result.state,
            'status': str(result.info)
        }
    
    return response
