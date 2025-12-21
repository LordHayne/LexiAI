"""
Models endpoints for the Lexi API.
"""
import logging
import requests
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
from pydantic import BaseModel

from backend.config.middleware_config import MiddlewareConfig
from backend.api.v1.models.response_models import ModelsResponse, ModelInfo

# Setup logging
logger = logging.getLogger("lexi_middleware.routes.models")

# Create router
router = APIRouter(tags=["models"])

# Request/Response models
class ModelPullRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = True

class ModelDeleteRequest(BaseModel):
    name: str

class ModelDeleteResponse(BaseModel):
    success: bool
    message: str

class RecommendedModelsResponse(BaseModel):
    success: bool
    models: List[Dict[str, Any]]

@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    Get a list of available models from the Ollama API.
    
    Returns:
        ModelsResponse: List of available models and default configurations
    """
    try:
        # Get models from Ollama API
        ollama_url = f"{MiddlewareConfig.get_ollama_base_url()}/api/tags"
        response = requests.get(ollama_url, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Failed to get models from Ollama API: {response.status_code}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to get models from Ollama API: {response.status_code}"
            )
        
        # Parse response
        data = response.json()
        models_data = data.get('models', [])
        
        # Convert to ModelInfo objects
        models: List[ModelInfo] = []
        for model_data in models_data:
            # Extract model name and details
            name = model_data.get('name', '')
            
            # Try to parse model details from name (format often like: model:size-quantization)
            size = None
            family = None
            quantization = None
            
            # Extract additional details
            details: Dict[str, Any] = {}
            for key, value in model_data.items():
                if key not in ['name', 'size', 'family', 'quantization', 'modified_at', 'digest']:
                    details[key] = value
            
            # Handle size conversion (Ollama may return size as an integer in bytes)
            raw_size = model_data.get('size')
            size_str = None
            if isinstance(raw_size, int):
                # Convert bytes to a human-readable format
                if raw_size >= 1_000_000_000:
                    size_str = f"{raw_size / 1_000_000_000:.1f}GB"
                elif raw_size >= 1_000_000:
                    size_str = f"{raw_size / 1_000_000:.1f}MB"
                else:
                    size_str = str(raw_size)
            elif raw_size is not None:
                size_str = str(raw_size)
            
            # Create ModelInfo object
            model_info = ModelInfo(
                name=name,
                size=size_str,
                family=family,
                quantization=quantization,
                modified_at=model_data.get('modified_at'),
                digest=model_data.get('digest'),
                details=details if details else None
            )
            
            models.append(model_info)
        
        # Get default models from config
        default_llm_model = MiddlewareConfig.get_default_llm_model()
        default_embedding_model = MiddlewareConfig.get_default_embedding_model()
        
        # Create response
        response = ModelsResponse(
            success=True,
            models=models,
            default_llm_model=default_llm_model,
            default_embedding_model=default_embedding_model
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting models: {str(e)}"
        )

# ========== NEW ENDPOINTS FOR OLLAMA MODEL MANAGEMENT ==========

@router.get("/models/pull")
async def pull_model(name: str, insecure: bool = False):
    """
    Pull a model from Ollama with streaming progress updates.

    Returns Server-Sent Events (SSE) stream with progress updates.
    Uses GET to support EventSource in browser.
    """
    async def generate_progress() -> AsyncGenerator[str, None]:
        try:
            ollama_url = f"{MiddlewareConfig.get_ollama_base_url()}/api/pull"
            payload = {"name": name, "stream": True, "insecure": insecure}

            with requests.post(ollama_url, json=payload, stream=True, timeout=600) as response:
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Failed to pull model: {response.status_code}'})}\n\n"
                    return
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            # Send progress update
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                        except json.JSONDecodeError:
                            continue
                            
                # Send completion message
                yield f"data: {json.dumps({'status': 'success', 'message': 'Model pulled successfully'})}\n\n"
                
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@router.delete("/models/{model_name}", response_model=ModelDeleteResponse)
async def delete_model(model_name: str):
    """Delete a model from Ollama."""
    try:
        ollama_url = f"{MiddlewareConfig.get_ollama_base_url()}/api/delete"
        response = requests.delete(ollama_url, json={"name": model_name}, timeout=30)
        
        if response.status_code == 200:
            return ModelDeleteResponse(
                success=True,
                message=f"Model '{model_name}' deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to delete model: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/recommended", response_model=RecommendedModelsResponse)
async def get_recommended_models():
    """Get a list of recommended models with descriptions."""
    recommended = [
        {
            "name": "gemma3:12b",
            "description": "Google Gemma 3 (12B) - Excellent instruction following, balanced speed/quality",
            "size": "~7GB",
            "use_case": "General LLM (Recommended)",
            "parameters": "12B",
            "type": "llm",
            "tags": ["recommended", "balanced", "instruction-following"]
        },
        {
            "name": "gemma3:4b",
            "description": "Google Gemma 3 (4B) - Fast but may hallucinate with complex instructions",
            "size": "~2.5GB",
            "use_case": "Fast LLM",
            "parameters": "4B",
            "type": "llm",
            "tags": ["fast", "lightweight"]
        },
        {
            "name": "qwen2.5:7b",
            "description": "Qwen 2.5 (7B) - Good balance, strong reasoning",
            "size": "~4.5GB",
            "use_case": "General LLM",
            "parameters": "7B",
            "type": "llm",
            "tags": ["balanced", "reasoning"]
        },
        {
            "name": "llama3.2:3b",
            "description": "Meta Llama 3.2 (3B) - Compact, efficient",
            "size": "~2GB",
            "use_case": "Lightweight LLM",
            "parameters": "3B",
            "type": "llm",
            "tags": ["lightweight", "efficient"]
        },
        {
            "name": "nomic-embed-text",
            "description": "Nomic Embeddings - 768 dimensions, excellent for semantic search",
            "size": "~275MB",
            "use_case": "Embeddings (Required)",
            "parameters": "137M",
            "type": "embedding",
            "tags": ["recommended", "embeddings", "required"]
        },
        {
            "name": "llama3.1:8b",
            "description": "Meta Llama 3.1 (8B) - Strong general performance",
            "size": "~4.7GB",
            "use_case": "General LLM",
            "parameters": "8B",
            "type": "llm",
            "tags": ["balanced", "general"]
        }
    ]
    
    return RecommendedModelsResponse(success=True, models=recommended)
