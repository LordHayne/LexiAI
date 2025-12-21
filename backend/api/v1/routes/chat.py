"""
Chat endpoints for the Lexi API.
"""
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from backend.api.v1.models.request_models import ChatRequest
from backend.api.v1.models.response_models import ChatResponse
from backend.models.feedback import FeedbackType
from backend.api.middleware.auth import verify_api_key
from backend.core.chat_logic import (
    process_chat_message_async,
    process_chat_message_streaming
)
from backend.core.chat_processing_with_tools import process_chat_with_tools
from backend.config.feature_flags import FeatureFlags
from backend.core.bootstrap import initialize_components, ConfigurationError
from typing import Optional, Dict, Any, AsyncGenerator
import logging
import json
import datetime
import asyncio
from contextlib import asynccontextmanager

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger("lexi_middleware.chat")

router = APIRouter(tags=["chat"])

# Initialize limiter for this module
limiter = Limiter(key_func=get_remote_address)

# Constants
MAX_MESSAGE_LENGTH = 10000
STREAMING_CHUNK_SIZE = 1024
DEFAULT_TIMEOUT = 300  # 5 minutes

class ChatError(Exception):
    """Custom exception for chat-related errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@asynccontextmanager
async def get_chat_components():
    """
    Context manager for getting cached chat components.
    """
    embeddings = vectorstore = memory = chat_client = None
    try:
        logger.debug("Getting cached chat components")
        from backend.core.component_cache import get_cached_components
        bundle = get_cached_components()

        embeddings = bundle.embeddings
        vectorstore = bundle.vectorstore
        memory = bundle.memory
        chat_client = bundle.chat_client
        config_warning = bundle.config_warning

        if config_warning:
            logger.warning(f"Component initialization warning: {config_warning}")

        if not all([embeddings, vectorstore, memory, chat_client]):
            missing = []
            if not embeddings: missing.append("embeddings")
            if not vectorstore: missing.append("vectorstore")
            if not memory: missing.append("memory")
            if not chat_client: missing.append("chat_client")
            raise ChatError(f"Failed to initialize components: {', '.join(missing)} not available")

        yield embeddings, vectorstore, memory, chat_client, config_warning
        
    except ConfigurationError as e:
        logger.error(f"Configuration error initializing components: {str(e)}")
        raise ChatError(f"Configuration error: {str(e)}", status_code=503)
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error initializing components: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ChatError(f"Component initialization failed: {str(e)}", status_code=503)
    finally:
        # Cleanup logic would go here if needed
        logger.debug("Chat components context closed")

def validate_chat_request(request: ChatRequest) -> None:
    """
    Validate chat request parameters with comprehensive security checks.

    SECURITY: Uses centralized InputValidator to prevent injection attacks.
    """
    from backend.utils.input_validation import InputValidator, ValidationError

    try:
        # Validate message
        request.message = InputValidator.validate_chat_message(request.message)

        # Validate user_id if provided
        if request.user_id:
            request.user_id = InputValidator.validate_user_id(request.user_id)

        # Validate session_id if provided
        if request.session_id:
            if len(request.session_id) > 255:
                raise ValidationError("Session ID too long")

    except ValidationError as e:
        logger.warning(f"Chat request validation failed: {e}")
        raise ChatError(str(e), status_code=400)

async def create_streaming_response(
    message: str,
    chat_client,
    vectorstore,
    memory,
    embeddings,
    collect_feedback: bool = False,
    user_id: str = "default"
) -> AsyncGenerator[str, None]:
    """
    Create streaming response with proper error handling and formatting.
    """
    try:
        logger.info(f"Starting streaming response for message (length: {len(message)})")
        
        # Add metadata at the start
        metadata = {
            "type": "metadata",
            "timestamp": datetime.datetime.now().isoformat(),
            "streaming": True
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Process the streaming response
        async for chunk in process_chat_message_streaming(
            message,
            chat_client=chat_client,
            vectorstore=vectorstore,
            memory=memory,
            embeddings=embeddings,
            collect_feedback=collect_feedback,
            user_id=user_id
        ):
            # Ensure chunk is properly formatted
            if isinstance(chunk, dict):
                yield f"data: {json.dumps(chunk)}\n\n"
            elif isinstance(chunk, str):
                # Wrap plain text in a structured format
                chunk_data = {
                    "type": "content",
                    "content": chunk,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            else:
                logger.warning(f"Unexpected chunk type: {type(chunk)}")
        
        # Send completion signal
        completion = {
            "type": "complete",
            "timestamp": datetime.datetime.now().isoformat()
        }
        yield f"data: {json.dumps(completion)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")  # SECURITY: 20 chat requests per minute per IP
async def chat_endpoint(
    request: Request,  # Required for slowapi (must be named 'request')
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Main chat endpoint that processes messages synchronously or with streaming.

    Args:
        request: HTTP request (required for slowapi rate limiter)
        chat_request: Chat request containing message and options
        background_tasks: FastAPI background tasks
        api_key: API key for authentication

    Returns:
        ChatResponse or StreamingResponse depending on chat_request.stream
    """
    start_time = datetime.datetime.now()

    try:
        # Use user_id from middleware if not provided or default in request
        if (not chat_request.user_id or chat_request.user_id == "default") and hasattr(request.state, 'user_id'):
            chat_request.user_id = request.state.user_id
            logger.debug(f"Using user_id from middleware: {chat_request.user_id}")

        # Validate request
        validate_chat_request(chat_request)

        logger.info(f"Processing chat request - Stream: {chat_request.stream}, Message length: {len(chat_request.message)}")
        
        # Check if streaming is requested and enabled
        if chat_request.stream:
            if not FeatureFlags.is_enabled("streaming"):
                raise ChatError("Streaming is currently disabled", status_code=400)

            # Initialize components and return streaming response
            async with get_chat_components() as (embeddings, vectorstore, memory, chat_client, config_warning):

                # Add warning to background tasks if present
                if config_warning:
                    logger.warning(f"Processing with configuration warning: {config_warning}")

                return StreamingResponse(
                    create_streaming_response(
                        chat_request.message,
                        chat_client=chat_client,
                        vectorstore=vectorstore,
                        memory=memory,
                        embeddings=embeddings,
                        collect_feedback=FeatureFlags.is_enabled("memory_feedback"),
                        user_id=chat_request.user_id
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Cache-Control"
                    }
                )
        
        # Process synchronous request
        async with get_chat_components() as (embeddings, vectorstore, memory, chat_client, config_warning):
            
            # Set up timeout for the chat processing
            try:
                # Phase 2.1: Use tool-calling system if enabled
                if FeatureFlags.is_enabled("llm_tool_calling"):
                    logger.info("🔧 Using LLM Tool-Calling System (Phase 2.1)")
                    response_data = await asyncio.wait_for(
                        process_chat_with_tools(
                            message=chat_request.message,
                            chat_client=chat_client,
                            vectorstore=vectorstore,
                            memory=memory,
                            embeddings=embeddings,
                            user_id=chat_request.user_id
                        ),
                        timeout=DEFAULT_TIMEOUT
                    )
                    # FIX: Tool-calling system now returns dict with memory entries
                    if isinstance(response_data, dict):
                        response_text = response_data.get("response", "")
                        memory_used = response_data.get("final", True)
                        source = response_data.get("source", "llm_tool_calling")
                        memory_entries = response_data.get("relevant_memory", [])
                        turn_id = response_data.get("turn_id")
                    else:
                        # Fallback for old string format (backwards compatibility)
                        logger.warning("Received string response - using fallback")
                        response_text = response_data
                        memory_used = True
                        source = "llm_tool_calling"
                        memory_entries = []
                        turn_id = None
                else:
                    logger.info("Using traditional chat processing")
                    response_data = await asyncio.wait_for(
                        process_chat_message_async(
                            chat_request.message,
                            chat_client=chat_client,
                            vectorstore=vectorstore,
                            memory=memory,
                            embeddings=embeddings,
                            user_id=chat_request.user_id
                        ),
                        timeout=DEFAULT_TIMEOUT
                    )

                    # FIXED: response_data is now a dict, not a tuple
                    if isinstance(response_data, dict):
                        response_text = response_data.get("response", "")
                        memory_used = response_data.get("final", True)
                        source = response_data.get("source", "llm")
                        memory_entries = response_data.get("relevant_memory", [])
                        turn_id = response_data.get("turn_id")
                    else:
                        # Fallback for old tuple format (backwards compatibility)
                        logger.warning("Received tuple response - this format is deprecated")
                        response_text, memory_used, source, memory_entries, turn_id = response_data
                
            except asyncio.TimeoutError:
                logger.error(f"Chat processing timed out after {DEFAULT_TIMEOUT} seconds")
                raise ChatError("Request timed out. Please try again with a shorter message.", status_code=408)
            
            # Process memory entries safely - convert to Pydantic MemoryEntry
            from backend.api.v1.models.response_models import MemoryEntry as PydanticMemoryEntry

            processed_memory_entries = []
            if memory_entries:
                for entry in memory_entries:
                    try:
                        # Convert backend.models.memory_entry.MemoryEntry to Pydantic MemoryEntry
                        if hasattr(entry, "content"):
                            processed_memory_entries.append(
                                PydanticMemoryEntry(
                                    id=str(entry.id) if hasattr(entry, "id") else "unknown",
                                    content=str(entry.content),
                                    tag=entry.category if hasattr(entry, "category") else None,
                                    timestamp=entry.timestamp.isoformat() if hasattr(entry, "timestamp") and entry.timestamp else "",
                                    relevance=float(entry.relevance) if hasattr(entry, "relevance") and entry.relevance else None
                                )
                            )
                        elif isinstance(entry, dict):
                            processed_memory_entries.append(
                                PydanticMemoryEntry(
                                    id=entry.get("id", "unknown"),
                                    content=entry.get("content", ""),
                                    tag=entry.get("category") or entry.get("tag"),
                                    timestamp=entry.get("timestamp", ""),
                                    relevance=entry.get("relevance")
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Error processing memory entry: {str(e)}")
                        # Skip invalid entries instead of adding error placeholders
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            # turn_id already created in chat_processing.py during message processing

            # Create response
            response = ChatResponse(
                success=True,
                response=response_text,
                memory_used=memory_used,
                source=source,
                memory_entries=processed_memory_entries,
                processing_time=processing_time,
                timestamp=datetime.datetime.now().isoformat(),
                config_warning=config_warning,
                turn_id=turn_id
            )
            
            logger.info(f"Chat request processed successfully in {processing_time:.2f}s")
            return response
    
    except ChatError as e:
        logger.error(f"Chat error: {e.message}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "error": e.message,
                "timestamp": datetime.datetime.now().isoformat(),
                "processing_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error occurred while processing your message",
                "timestamp": datetime.datetime.now().isoformat(),
                "processing_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        )

from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    turn_id: str
    feedback_type: str
    comment: Optional[str] = None

@router.post("/chat/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Sammelt User-Feedback zu einer Response.

    Body:
    {
        "turn_id": "uuid",
        "feedback_type": "explicit_positive" | "explicit_negative",
        "comment": "optional comment"
    }
    """
    from backend.memory.conversation_tracker import record_user_feedback

    try:
        feedback_type = FeedbackType(request.feedback_type)

        record_user_feedback(
            turn_id=request.turn_id,
            feedback_type=feedback_type,
            user_comment=request.comment
        )

        return {"status": "success", "message": "Feedback recorded"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/chat/status")
async def chat_status():
    """
    Get chat service status and capabilities.
    
    Returns:
        Dict: Current chat service status and available features
    """
    try:
        # Check component health
        components_healthy = True
        component_status = {}
        
        try:
            async with get_chat_components() as (embeddings, vectorstore, memory, chat_client, config_warning):
                component_status = {
                    "embeddings": "healthy" if embeddings else "unavailable",
                    "vectorstore": "healthy" if vectorstore else "unavailable", 
                    "memory": "healthy" if memory else "unavailable",
                    "chat_client": "healthy" if chat_client else "unavailable"
                }
                if config_warning:
                    component_status["warning"] = config_warning
        except Exception as e:
            components_healthy = False
            component_status["error"] = str(e)
        
        # Get feature flags
        features = {
            "streaming": FeatureFlags.is_enabled("streaming"),
            "memory_feedback": FeatureFlags.is_enabled("memory_feedback"),
            "rate_limiting": FeatureFlags.is_enabled("rate_limiting"),
            "caching": FeatureFlags.is_enabled("caching")
        }
        
        return {
            "success": True,
            "status": "healthy" if components_healthy else "degraded",
            "timestamp": datetime.datetime.now().isoformat(),
            "components": component_status,
            "features": features,
            "limits": {
                "max_message_length": MAX_MESSAGE_LENGTH,
                "default_timeout": DEFAULT_TIMEOUT
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting chat status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
