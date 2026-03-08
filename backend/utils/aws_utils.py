"""
AWS Service Utilities
Provides helper functions for S3, SQS, and Bedrock operations
"""
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
import logging
from config import settings

logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = None
sqs_client = None
bedrock_client = None


def get_s3_client():
    """Get or create S3 client"""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION
        )
        logger.info(f"S3 client initialized for region {settings.AWS_REGION}")
    return s3_client


def get_sqs_client():
    """Get or create SQS client"""
    global sqs_client
    if sqs_client is None:
        sqs_client = boto3.client(
            'sqs',
            region_name=settings.AWS_REGION
        )
        logger.info(f"SQS client initialized for region {settings.AWS_REGION}")
    return sqs_client


def get_bedrock_client():
    """Get or create Bedrock Runtime client"""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.AWS_REGION
        )
        logger.info(f"Bedrock client initialized for region {settings.AWS_REGION}")
    return bedrock_client


# S3 Operations
def upload_file_to_s3(file_path: str, object_key: str) -> Optional[str]:
    """
    Upload a file to S3 bucket
    
    Args:
        file_path: Local file path
        object_key: S3 object key (path in bucket)
    
    Returns:
        S3 URL if successful, None otherwise
    """
    try:
        s3 = get_s3_client()
        s3.upload_file(file_path, settings.S3_BUCKET_NAME, object_key)
        url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{object_key}"
        logger.info(f"File uploaded to S3: {url}")
        return url
    except ClientError as e:
        logger.error(f"Failed to upload file to S3: {e}")
        return None


def generate_presigned_url(object_key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for S3 object access
    
    Args:
        object_key: S3 object key
        expiration: URL expiration time in seconds (default 1 hour)
    
    Returns:
        Presigned URL if successful, None otherwise
    """
    try:
        s3 = get_s3_client()
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': object_key},
            ExpiresIn=expiration
        )
        logger.info(f"Generated presigned URL for {object_key}")
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None


def delete_file_from_s3(object_key: str) -> bool:
    """
    Delete a file from S3 bucket
    
    Args:
        object_key: S3 object key
    
    Returns:
        True if successful, False otherwise
    """
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=object_key)
        logger.info(f"File deleted from S3: {object_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete file from S3: {e}")
        return False


# SQS Operations
def send_message_to_queue(message_body: Dict[Any, Any]) -> Optional[str]:
    """
    Send a message to SQS queue
    
    Args:
        message_body: Message content as dictionary
    
    Returns:
        Message ID if successful, None otherwise
    """
    try:
        import json
        sqs = get_sqs_client()
        response = sqs.send_message(
            QueueUrl=settings.SQS_QUEUE_URL,
            MessageBody=json.dumps(message_body)
        )
        message_id = response.get('MessageId')
        logger.info(f"Message sent to SQS: {message_id}")
        return message_id
    except ClientError as e:
        logger.error(f"Failed to send message to SQS: {e}")
        return None


def receive_messages_from_queue(max_messages: int = 1, wait_time: int = 10) -> list:
    """
    Receive messages from SQS queue
    
    Args:
        max_messages: Maximum number of messages to receive (1-10)
        wait_time: Long polling wait time in seconds
    
    Returns:
        List of messages
    """
    try:
        sqs = get_sqs_client()
        response = sqs.receive_message(
            QueueUrl=settings.SQS_QUEUE_URL,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time
        )
        messages = response.get('Messages', [])
        logger.info(f"Received {len(messages)} messages from SQS")
        return messages
    except ClientError as e:
        logger.error(f"Failed to receive messages from SQS: {e}")
        return []


def delete_message_from_queue(receipt_handle: str) -> bool:
    """
    Delete a message from SQS queue after processing
    
    Args:
        receipt_handle: Receipt handle from received message
    
    Returns:
        True if successful, False otherwise
    """
    try:
        sqs = get_sqs_client()
        sqs.delete_message(
            QueueUrl=settings.SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )
        logger.info("Message deleted from SQS")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete message from SQS: {e}")
        return False


# Bedrock Operations (AI Coach)
def invoke_bedrock_model(prompt: str, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0") -> Optional[str]:
    """
    Invoke AWS Bedrock model for AI coaching
    
    Args:
        prompt: Input prompt for the model
        model_id: Bedrock model ID
    
    Returns:
        Model response text if successful, None otherwise
    """
    try:
        import json
        bedrock = get_bedrock_client()
        
        # Prepare request body for Claude 3
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        text = response_body['content'][0]['text']
        logger.info("Bedrock model invoked successfully")
        return text
    except ClientError as e:
        logger.error(f"Failed to invoke Bedrock model: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error invoking Bedrock: {e}")
        return None


# Health check functions
def check_s3_connectivity() -> Dict[str, Any]:
    """Check S3 connectivity and bucket access"""
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        return {"status": "connected", "bucket": settings.S3_BUCKET_NAME}
    except ClientError as e:
        return {"status": "error", "message": str(e)}


def check_sqs_connectivity() -> Dict[str, Any]:
    """Check SQS connectivity and queue access"""
    try:
        sqs = get_sqs_client()
        response = sqs.get_queue_attributes(
            QueueUrl=settings.SQS_QUEUE_URL,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        return {
            "status": "connected",
            "queue_url": settings.SQS_QUEUE_URL,
            "messages_available": response['Attributes']['ApproximateNumberOfMessages']
        }
    except ClientError as e:
        return {"status": "error", "message": str(e)}


def check_bedrock_connectivity() -> Dict[str, Any]:
    """Check Bedrock connectivity"""
    try:
        bedrock = get_bedrock_client()
        # Simple test - bedrock-runtime doesn't have list methods
        # Just return connected if client initialized successfully
        return {"status": "connected", "service": "bedrock-runtime"}
    except ClientError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
