"""AWS Bedrock Utilities for Claude Vision API"""
import boto3
import json
import base64
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class BedrockClient:
    """Client for AWS Bedrock Vision API (Claude and Nova models)"""
    
    def __init__(self, region='us-east-1'):
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        # Claude models
        self.sonnet_model = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
        self.haiku_model = 'anthropic.claude-3-haiku-20240307-v1:0'
        # Amazon Nova models (alternative)
        self.nova_pro_model = 'amazon.nova-pro-v1:0'
        self.nova_lite_model = 'amazon.nova-lite-v1:0'
        
        # Model fallback order (best to cheapest)
        self.model_fallback_order = [
            ('sonnet', self.sonnet_model, False),
            ('haiku', self.haiku_model, False),
            ('nova-pro', self.nova_pro_model, True),
            ('nova-lite', self.nova_lite_model, True)
        ]
    
    def invoke_model(self, prompt: str, model: str = 'haiku', max_tokens: int = 1000, temperature: float = 0.0) -> str:
        """Invoke model with text-only prompt (no image) with automatic fallback
        
        Args:
            prompt: Text prompt
            model: Model to use ('sonnet', 'haiku', 'nova-pro', 'nova-lite')
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0.0 = deterministic, 1.0 = creative)
        """
        errors = []
        
        # Determine starting point based on requested model
        start_index = 0
        if model == 'sonnet' or model == 'claude-3-5-sonnet':
            start_index = 0
        elif model == 'haiku':
            start_index = 1
        elif model == 'nova-pro':
            start_index = 2
        elif model == 'nova-lite':
            start_index = 3
        
        # Try models in fallback order starting from requested model
        for i in range(start_index, len(self.model_fallback_order)):
            model_name, model_id, use_nova = self.model_fallback_order[i]
            
            try:
                logger.info(f"Trying model: {model_name} ({model_id}) with temperature={temperature}")
                
                # Prepare request based on model type
                if use_nova:
                    # Amazon Nova format
                    body = json.dumps({
                        "messages": [{
                            "role": "user",
                            "content": [{"text": prompt}]
                        }],
                        "inferenceConfig": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature
                        }
                    })
                else:
                    # Claude format
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }]
                    })
                
                response = self.bedrock.invoke_model(
                    modelId=model_id,
                    body=body
                )
                
                result = json.loads(response['body'].read())
                
                # Parse response based on model type
                if use_nova:
                    text_response = result['output']['message']['content'][0]['text']
                else:
                    text_response = result['content'][0]['text']
                
                logger.info(f"✅ Successfully used model: {model_name}")
                return text_response
                
            except Exception as e:
                error_msg = f"{model_name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"❌ Model {model_name} failed: {e}")
                
                # If this is the last model, raise error
                if i == len(self.model_fallback_order) - 1:
                    logger.error(f"All models failed. Errors: {errors}")
                    raise Exception(f"All Bedrock models failed. Tried: {', '.join(errors)}")
                
                # Otherwise, continue to next model
                logger.info(f"Trying next model in fallback chain...")
                continue
        
        # Should never reach here, but just in case
        raise Exception(f"All Bedrock models failed. Errors: {errors}")
    
    def analyze_image(self, image: np.ndarray, prompt: str, model: str = 'haiku', max_tokens: int = 1000, temperature: float = 0.0) -> str:
        """Analyze image with Bedrock Vision (Claude or Nova) with automatic fallback
        
        Args:
            image: Image as numpy array
            prompt: Text prompt
            model: Model to use ('sonnet', 'haiku', 'nova-pro', 'nova-lite')
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0.0 = deterministic, 1.0 = creative)
        """
        errors = []
        
        # Prepare image once
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Determine starting point based on requested model
        start_index = 0
        if model == 'sonnet' or model == 'claude-3-5-sonnet':
            start_index = 0
        elif model == 'haiku':
            start_index = 1
        elif model == 'nova-pro':
            start_index = 2
        elif model == 'nova-lite':
            start_index = 3
        
        # Try models in fallback order starting from requested model
        for i in range(start_index, len(self.model_fallback_order)):
            model_name, model_id, use_nova = self.model_fallback_order[i]
            
            try:
                logger.info(f"Trying model: {model_name} ({model_id}) with temperature={temperature}")
                
                # Prepare request based on model type
                if use_nova:
                    # Amazon Nova format
                    body = json.dumps({
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": "jpeg",
                                        "source": {"bytes": image_base64}
                                    }
                                },
                                {"text": prompt}
                            ]
                        }],
                        "inferenceConfig": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature
                        }
                    })
                else:
                    # Claude format
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                                {"type": "text", "text": prompt}
                            ]
                        }]
                    })
                
                response = self.bedrock.invoke_model(
                    modelId=model_id,
                    body=body
                )
                
                result = json.loads(response['body'].read())
                
                # Parse response based on model type
                if use_nova:
                    text_response = result['output']['message']['content'][0]['text']
                else:
                    text_response = result['content'][0]['text']
                
                logger.info(f"✅ Successfully used model: {model_name}")
                return text_response
                
            except Exception as e:
                error_msg = f"{model_name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"❌ Model {model_name} failed: {e}")
                
                # If this is the last model, raise error
                if i == len(self.model_fallback_order) - 1:
                    logger.error(f"All models failed. Errors: {errors}")
                    raise Exception(f"All Bedrock models failed. Tried: {', '.join(errors)}")
                
                # Otherwise, continue to next model
                logger.info(f"Trying next model in fallback chain...")
                continue
        
        # Should never reach here, but just in case
        raise Exception(f"All Bedrock models failed. Errors: {errors}")
