"""
Main API server for Lexi Middleware.
Improved version with better error handling, security, and code organization.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from backend.config.cors_config import CORSConfig
from backend.config.middleware_config import MiddlewareConfig
from backend.config.auth_config import SecurityConfig as AuthSecurityConfig
from backend.config.security_config import SecurityConfig
from backend.config.feature_flags import FeatureFlags
from backend.api.middleware.auth import verify_api_key, verify_ui_auth
from backend.api.middleware.error_handler import ErrorHandler, LexiError
from backend.utils.version import get_version
from backend.utils.audit_logger import AuditLogger
from backend.utils.validators import InputValidator

# Rate limiting (optional)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _rate_limiting_available = True
except ImportError:  # pragma: no cover - fallback for missing dependency
    _rate_limiting_available = False

    class _NoOpLimiter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    Limiter = _NoOpLimiter  # type: ignore[assignment]
    RateLimitExceeded = Exception  # type: ignore[assignment]

    def get_remote_address(request):  # type: ignore[override]
        return "unknown"

    def _rate_limit_exceeded_handler(request, exc):  # type: ignore[override]
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limiting unavailable (slowapi not installed)"}
        )

# Import routes
from backend.api.v1.routes.health import router as health_router
from backend.api.v1.routes.chat import router as chat_router
from backend.api.v1.routes.memory import router as memory_router
from backend.api.v1.routes.config import router as config_router
from backend.api.v1.routes.models import router as models_router
from backend.api.v1.routes.debug import router as debug_router
from backend.api.v1.routes.performance import router as performance_router
from backend.api.v1.routes.audio import router as audio_router
from backend.api.v1.routes.feedback import router as feedback_router
from backend.api.v1.routes.goals import router as goals_router
from backend.api.v1.routes.patterns import router as patterns_router
from backend.api.v1.routes.knowledge_gaps import router as knowledge_gaps_router
from backend.api.v1.routes.cache import router as cache_router
from backend.api.v1.routes.users import router as users_router
from backend.api.v1.routes.auth import router as auth_router  # JWT Authentication
from backend.api.v1.routes.profile import router as profile_router  # User Profile Management
from backend.api.v1.routes.home_assistant import router as home_assistant_router



# Configure logging with better formatting and rotation
def setup_logging():
    """Setup logging configuration with proper formatting and file rotation."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "lexi_middleware.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    return logging.getLogger("lexi_middleware")


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting Lexi Middleware API v{get_version()}")

    # SECURITY: Validate security configuration on startup
    try:
        SecurityConfig.validate_security_config()
    except ValueError as e:
        logger.error(f"Security configuration validation failed: {e}")
        # In production, fail fast. In development, warn and continue.
        env = os.environ.get("ENV", "development").lower()
        if env in ["production", "prod"]:
            raise
        else:
            logger.warning("Continuing in development mode with security warnings...")

    logger.info(f"API Key Enabled: {SecurityConfig.API_KEY_ENABLED}")
    logger.info(f"JWT Auth Enabled: {SecurityConfig.JWT_ENABLED}")
    
    # Load persisted configuration if available
    try:
        from backend.config.persistence import ConfigPersistence
        config = ConfigPersistence.load_config()
        if config:
            logger.info("Loading persisted configuration...")
            ConfigPersistence.apply_config(config)
            logger.info("Persisted configuration applied successfully")
    except Exception as e:
        logger.error(f"Error loading persisted configuration: {str(e)}")
    
    # Initialize feature flags
    try:
        FeatureFlags.initialize()
        enabled_features = FeatureFlags.get_enabled_features()
        logger.info(f"Active Features: {enabled_features}")
    except Exception as e:
        logger.error(f"Error initializing feature flags: {str(e)}")
    
    # Pre-initialize components to check connections
    try:
        from backend.core.lexi_adapter import check_lexi_components_health
        components = check_lexi_components_health()

        for component, status in components.items():
            if status.status == "ok":
                logger.info(f"Component {component} initialized successfully")
            else:
                logger.warning(f"Component {component} status: {status.status} - {status.message}")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")

    # Start Heartbeat Service for intelligent memory management
    try:
        from backend.services.heartbeat_memory import (
            start_heartbeat_service,
            get_heartbeat_config_snapshot,
        )

        started = start_heartbeat_service()
        config_snapshot = get_heartbeat_config_snapshot()
        interval_seconds = config_snapshot["run_interval_seconds"]
        idle_minutes = config_snapshot["idle_threshold_minutes"]

        if started:
            logger.info("🫀 Heartbeat Service started - Memory consolidation active")
        else:
            logger.info("🫀 Heartbeat Service already running")

        if interval_seconds % 60 == 0:
            logger.info("   → Runs every %s minutes", interval_seconds // 60)
        else:
            logger.info("   → Runs every %s seconds", interval_seconds)
        logger.info("   → IDLE mode (intensive) after %s min inactivity", idle_minutes)
        logger.info("   → ACTIVE mode (lightweight) during user activity")
    except Exception as e:
        logger.error(f"Error starting Heartbeat Service: {str(e)}")

    # Start feedback worker (optional, controlled via env)
    try:
        from backend.services.feedback_worker import start_feedback_worker
        started = start_feedback_worker()
        if started:
            logger.info("🧠 Feedback worker started (immediate corrections)")
    except Exception as e:
        logger.error(f"Error starting Feedback Worker: {str(e)}")

    yield

    # Shutdown - Graceful cleanup
    logger.info("Shutting down Lexi Middleware API - Starting graceful cleanup...")

    try:
        # 1. Close embedding HTTP clients
        logger.info("Closing embedding clients...")
        from backend.core.component_cache import get_cached_components
        try:
            bundle = get_cached_components()
            if bundle and bundle.embeddings:
                if hasattr(bundle.embeddings, 'close'):
                    bundle.embeddings.close()
                    logger.info("✓ Embedding sync client closed")
        except Exception as e:
            logger.warning(f"Error closing embedding clients: {e}")

        # 2. Close Qdrant client connection
        logger.info("Closing Qdrant client...")
        try:
            from backend.qdrant.client_wrapper import _client as qdrant_client
            if qdrant_client:
                # Qdrant client doesn't need explicit close in most cases
                # but we can reset it
                from backend.qdrant.client_wrapper import reset_client
                reset_client()
                logger.info("✓ Qdrant client reset")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

        # 3. Clear component cache
        logger.info("Clearing component cache...")
        try:
            from backend.core.component_cache import clear_component_cache
            clear_component_cache()
            logger.info("✓ Component cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing component cache: {e}")

        logger.info("✅ Graceful shutdown completed successfully")

    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")
        logger.info("⚠️  Shutdown completed with errors")

    logger.info("Heartbeat Service will stop automatically (daemon thread)")


# Initialize rate limiter with security config
# SECURITY: Rate limiting prevents DoS attacks and API abuse
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[SecurityConfig.RATE_LIMITS["default"]],
    storage_uri=SecurityConfig.RATE_LIMIT_STORAGE
)

# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="Lexi Middleware API",
    description="OpenWebUI middleware for Lexi's intelligent memory system",
    version=get_version(),
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan
)

# Register rate limiter with app
app.state.limiter = limiter
if _rate_limiting_available:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORSConfig.ALLOW_ORIGINS,
    allow_credentials=CORSConfig.ALLOW_CREDENTIALS,
    allow_methods=CORSConfig.ALLOW_METHODS,
    allow_headers=CORSConfig.ALLOW_HEADERS,
)

# Add User middleware for automatic user_id injection
from backend.api.middleware.user_middleware import UserMiddleware
app.add_middleware(UserMiddleware)


# Security Headers Middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Add security headers to all responses.

    SECURITY: Protects against common web vulnerabilities:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Enables browser XSS filters
    - Strict-Transport-Security: Enforces HTTPS
    - Content-Security-Policy: Restricts resource loading
    """
    response = await call_next(request)

    # Add all security headers to the response
    for header_name, header_value in SecurityConfig.SECURITY_HEADERS.items():
        # Only add HSTS header if using HTTPS
        if header_name == "Strict-Transport-Security":
            if request.url.scheme == "https":
                response.headers[header_name] = header_value
        else:
            response.headers[header_name] = header_value

    return response


# Activity Tracking Middleware (Phase 1 & 3 Integration)
@app.middleware("http")
async def activity_tracking_middleware(request: Request, call_next):
    """
    Trackt User-Aktivität und unterbricht Lernprozesse bei Bedarf.

    Diese Middleware läuft bei JEDEM Request und:
    1. Zeichnet Aktivität auf (für Idle-Detection)
    2. Unterbricht laufende Lernprozesse falls User Request kommt
    3. Führt den eigentlichen Request aus

    Phase 1 Integration: Idle-Mode Learning
    Phase 3 Integration: Self-Correction während Idle
    """
    from backend.memory.activity_tracker import track_activity
    from backend.services.heartbeat_memory import is_learning_in_progress, stop_learning_processes

    # 1. Aktivität aufzeichnen (für Idle-Detection)
    track_activity()

    # 2. Wenn intensive Lernprozesse laufen, unterbrechen
    if is_learning_in_progress():
        logger.warning(f"⚠️ User request during learning - interrupting (path: {request.url.path})")
        stop_learning_processes()

    # 3. Request normal ausführen
    response = await call_next(request)

    return response


# Initialize templates with proper error handling
def setup_templates():
    """Setup Jinja2 templates with error handling."""
    template_dir = Path("frontend/pages")
    if not template_dir.exists():
        logger.warning(f"Template directory {template_dir} not found, creating it...")
        template_dir.mkdir(parents=True, exist_ok=True)
    return Jinja2Templates(directory=str(template_dir))


templates = setup_templates()

# Mount static files directory with error handling
def setup_static_files():
    """Setup static files with proper path handling."""
    static_dirs = [
        ("frontend", "frontend"),
        ("static", "static")
    ]
    optional_dirs = {"static"}
    
    for mount_path, directory in static_dirs:
        static_path = Path(directory)
        if static_path.exists():
            app.mount(f"/{mount_path}", StaticFiles(directory=str(static_path)), name=mount_path)
            logger.info(f"Mounted static files: /{mount_path} -> {static_path}")
        else:
            if directory in optional_dirs:
                logger.info(f"Static directory {static_path} not found (optional)")
            else:
                logger.warning(f"Static directory {static_path} not found")


setup_static_files()


# Enhanced request logging middleware
@app.middleware("http")
async def enhanced_request_logging(request: Request, call_next):
    """
    Enhanced middleware to log all requests and responses with better error handling.
    """
    start_time = time.time()
    
    # Get client information
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Create request ID for tracking
    request_id = f"{int(start_time * 1000)}-{hash(f'{client_host}{request.url.path}') % 1000:03d}"
    
    # Log the request with more details
    logger.info(
        f"[{request_id}] Request: {request.method} {request.url.path} "
        f"from {client_host} (UA: {user_agent[:50]}...)"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = round((time.time() - start_time) * 1000)
        
        # Log the response
        logger.info(
            f"[{request_id}] Response: {response.status_code} "
            f"for {request.method} {request.url.path} took {process_time}ms"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        process_time = round((time.time() - start_time) * 1000)
        
        # Log the error with more context
        logger.error(
            f"[{request_id}] Error processing request: {request.method} {request.url.path}, "
            f"Error: {str(e)}, Time: {process_time}ms"
        )
        
        # Enhanced audit logging
        if FeatureFlags.is_enabled("audit_logging"):
            try:
                AuditLogger.log_error(
                    error_code="INTERNAL_ERROR",
                    message=str(e),
                    resource=request.url.path,
                    ip_address=client_host,
                    additional_data={
                        "request_id": request_id,
                        "method": request.method,
                        "user_agent": user_agent,
                        "process_time": process_time
                    }
                )
            except Exception as audit_error:
                logger.error(f"Failed to log audit event: {audit_error}")
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
                "timestamp": int(time.time()),
                "error_type": type(e).__name__
            },
            headers={"X-Request-ID": request_id}
        )


# Enhanced exception handler for custom errors
@app.exception_handler(LexiError)
async def handle_lexi_error(request: Request, exc: LexiError):
    """
    Handle custom Lexi errors with enhanced logging.
    """
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    
    logger.warning(
        f"LexiError on {request.method} {request.url.path} from {client_host}: "
        f"{exc.error_code} - {exc.message}"
    )
    
    return ErrorHandler.handle_exception(exc)


# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """
    Custom 404 handler with logging.
    """
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    logger.warning(f"404 Not Found: {request.method} {request.url.path} from {client_host}")
    
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Resource not found",
            "path": request.url.path,
            "timestamp": int(time.time())
        }
    )


# Root route - redirect to Chat UI
@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint - redirects to Chat UI as the main interface.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/chat_ui.html")


# Custom docs route with authentication
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    """
    Swagger UI with custom configuration and optional authentication.
    """
    # Optional: Add authentication check here if needed
    # if SecurityConfig.API_KEY_ENABLED:
    #     await verify_api_key(request)

    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# Health check endpoint (always accessible)
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint that's always accessible.
    """
    return {
        "status": "healthy",
        "version": get_version(),
        "timestamp": int(time.time())
    }


# Register routes with proper error handling
def register_routes():
    """Register all API routes with proper error handling."""
    routes_config = [
        (health_router, "/v1", "Health endpoints"),
        (auth_router, "/v1", "Authentication endpoints (JWT)"),  # Auth routes FIRST
        (chat_router, "/v1", "Chat endpoints"),
        (memory_router, "/v1", "Memory endpoints"),
        (config_router, "/v1", "Configuration endpoints"),
        (models_router, "/v1", "Models endpoints"),
        (performance_router, "/v1", "Performance endpoints"),
        (debug_router, "/v1", "Debug endpoints"),
        (goals_router, "/v1", "Goals endpoints"),
        (patterns_router, "/v1", "Patterns endpoints"),
        (knowledge_gaps_router, "/v1", "Knowledge Gaps endpoints"),
        (cache_router, "/v1/cache", "Cache endpoints"),
        (profile_router, "/v1", "User Profile endpoints"),
        (home_assistant_router, "/v1", "Home Assistant event endpoints"),
        (users_router, "", "User management endpoints"),  # No prefix, already has /v1/users
        (feedback_router, "", "Feedback endpoints"),  # No prefix, already in router
        (audio_router, "", "Audio endpoints")
    ]

    for router, prefix, description in routes_config:
        try:
            app.include_router(router, prefix=prefix)
            logger.info(f"Registered {description} at {prefix}")
        except Exception as e:
            logger.error(f"Failed to register {description}: {str(e)}")


register_routes()


# Enhanced config UI endpoint with error handling
@app.get("/config", response_class=HTMLResponse)
async def config_ui(request: Request):
    """
    Configuration UI for the middleware with enhanced error handling.
    """
    try:
        context = {
            "request": request,
            "version": get_version(),
            "api_key_enabled": SecurityConfig.API_KEY_ENABLED,
            "features": FeatureFlags.get_enabled_features()
        }
        return templates.TemplateResponse("config_ui.html", context)
    except Exception as e:
        logger.error(f"Error rendering config UI: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Configuration UI Error</h1><p>Error: {str(e)}</p>",
            status_code=500
        )


# Metrics Dashboard UI endpoint
@app.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard(request: Request):
    """
    Performance Metrics Dashboard UI for real-time monitoring.
    """
    try:
        context = {
            "request": request,
            "version": get_version()
        }
        return templates.TemplateResponse("metrics_dashboard.html", context)
    except Exception as e:
        logger.error(f"Error rendering metrics dashboard: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Metrics Dashboard Error</h1><p>Error: {str(e)}</p>",
            status_code=500
        )


# Memory Management UI endpoint
@app.get("/memory", response_class=HTMLResponse)
async def memory_management_ui(request: Request):
    """
    Memory Management UI for consolidation, synthesis and heartbeat control.
    """
    try:
        context = {
            "request": request,
            "version": get_version()
        }
        return templates.TemplateResponse("memory_management_ui.html", context)
    except Exception as e:
        logger.error(f"Error rendering memory management UI: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Memory Management Error</h1><p>Error: {str(e)}</p>",
            status_code=500
        )


# Goals UI endpoint
@app.get("/goals", response_class=HTMLResponse)
async def goals_ui():
    """
    Goals dashboard UI.
    """
    try:
        with open("frontend/pages/goals_ui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error rendering goals UI: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Goals UI Error</h1><p>Error: {str(e)}</p>",
            status_code=500
        )


# Enhanced UI-specific endpoints with optional authentication
@app.get("/ui/config")
async def ui_get_config(
    request: Request,
    auth: bool = Depends(verify_ui_auth)
):
    """
    Get configuration for the UI with optional authentication.

    SECURITY: Authentication can be enforced by setting LEXI_UI_AUTH_REQUIRED=True
    """
    try:
        from backend.api.v1.routes.config import get_config
        return await get_config()
    except Exception as e:
        logger.error(f"Error getting UI config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")


@app.post("/ui/config")
async def ui_update_config(
    request: Request,
    auth: bool = Depends(verify_ui_auth)
):
    """
    Update configuration from the UI with optional authentication.

    SECURITY: Authentication can be enforced by setting LEXI_UI_AUTH_REQUIRED=True
    Configuration changes should always be protected in production.
    """
    try:
        from backend.api.v1.routes.config import update_config
        return await update_config(request)
    except Exception as e:
        logger.error(f"Error updating UI config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration update error: {str(e)}")


@app.post("/ui/chat")
@limiter.limit("30/minute")  # SECURITY: Rate limit UI chat endpoint
async def ui_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    auth: bool = Depends(verify_ui_auth)
):
    """
    Chat endpoint for the UI with optional authentication.

    SECURITY: Authentication can be enforced by setting LEXI_UI_AUTH_REQUIRED=True
    Rate limited to 30 requests per minute per IP.
    """
    import datetime
    import asyncio
    import json
    from backend.api.v1.models.request_models import ChatRequest
    from backend.api.v1.models.response_models import ChatResponse, MemoryEntry as PydanticMemoryEntry
    from backend.api.v1.routes.chat import get_chat_components, validate_chat_request, ChatError, create_streaming_response, DEFAULT_TIMEOUT
    from backend.core.chat_processing_with_tools import process_chat_with_tools
    from backend.core.chat_logic import process_chat_message_async, process_chat_message_streaming
    from backend.config.feature_flags import FeatureFlags
    from fastapi.responses import StreamingResponse, JSONResponse

    start_time = datetime.datetime.now()

    try:
        # Parse and validate the request body
        data = await request.json()
        chat_request = ChatRequest(**data)

        # Use user_id from middleware if not provided or default in request
        if (not chat_request.user_id or chat_request.user_id == "default") and hasattr(request.state, "user_id"):
            chat_request.user_id = request.state.user_id
            logger.debug(f"Using user_id from middleware: {chat_request.user_id}")

        # Validate request
        validate_chat_request(chat_request)

        logger.info(f"Processing UI chat request - Stream: {chat_request.stream}, Message length: {len(chat_request.message)}")

        # Check if streaming is requested and enabled
        if chat_request.stream:
            if not FeatureFlags.is_enabled("streaming"):
                raise ChatError("Streaming is currently disabled", status_code=400)

            # Initialize components and return streaming response
            async with get_chat_components() as (embeddings, vectorstore, memory, chat_client, config_warning):

                # Add warning to background tasks if present
                if config_warning:
                    logger.warning(f"Processing with configuration warning: {config_warning}")

                async def stream_with_history():
                    collected_chunks = []
                    turn_id = None

                    try:
                        metadata = {
                            "type": "metadata",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "streaming": True
                        }
                        yield f"data: {json.dumps(metadata)}\n\n"

                        async for chunk in process_chat_message_streaming(
                            chat_request.message,
                            chat_client=chat_client,
                            vectorstore=vectorstore,
                            memory=memory,
                            embeddings=embeddings,
                            collect_feedback=FeatureFlags.is_enabled("memory_feedback"),
                            user_id=chat_request.user_id
                        ):
                            if isinstance(chunk, dict):
                                chunk_text = chunk.get("chunk")
                                if chunk_text:
                                    collected_chunks.append(chunk_text)
                                if chunk.get("final_chunk"):
                                    turn_id = chunk.get("turn_id")
                                yield f"data: {json.dumps(chunk)}\n\n"
                            elif isinstance(chunk, str):
                                collected_chunks.append(chunk)
                                chunk_data = {
                                    "type": "content",
                                    "content": chunk,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                            else:
                                logger.warning(f"Unexpected chunk type: {type(chunk)}")

                        if chat_request.session_id:
                            from backend.qdrant.chat_history_store import store_chat_turn
                            await asyncio.to_thread(
                                store_chat_turn,
                                user_id=chat_request.user_id,
                                session_id=chat_request.session_id,
                                user_message=chat_request.message,
                                assistant_message=" ".join(collected_chunks).strip(),
                                turn_id=turn_id
                            )

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

                return StreamingResponse(
                    stream_with_history(),
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

                    if isinstance(response_data, dict):
                        response_text = response_data.get("response", "")
                        memory_used = response_data.get("final", True)
                        source = response_data.get("source", "llm")
                        memory_entries = response_data.get("relevant_memory", [])
                        turn_id = response_data.get("turn_id")
                    else:
                        response_text, memory_used, source, memory_entries, turn_id = response_data

            except asyncio.TimeoutError:
                logger.error(f"Chat processing timed out after {DEFAULT_TIMEOUT} seconds")
                raise ChatError("Request timed out. Please try again with a shorter message.", status_code=408)

            # Store chat history in the background (no embeddings)
            if chat_request.session_id:
                from backend.qdrant.chat_history_store import store_chat_turn

                background_tasks.add_task(
                    store_chat_turn,
                    user_id=chat_request.user_id,
                    session_id=chat_request.session_id,
                    user_message=chat_request.message,
                    assistant_message=response_text,
                    turn_id=turn_id,
                )
            else:
                logger.debug("No session_id provided; skipping chat history storage")

            # Process memory entries safely - convert to Pydantic MemoryEntry
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

            logger.info(f"UI chat request processed successfully in {processing_time:.2f}s")
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
        logger.error(f"Error in UI chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "processing_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        )


@app.get("/ui/chat/sessions")
async def ui_chat_sessions(
    request: Request,
    limit: int = 50,
    auth: bool = Depends(verify_ui_auth)
):
    """
    Return a list of chat sessions for the current user.
    """
    try:
        user_id = getattr(request.state, "user_id", None) or request.query_params.get("user_id") or "default"
        from backend.qdrant.chat_history_store import list_sessions

        sessions = list_sessions(user_id=user_id, limit=limit)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/ui/chat/history/{session_id}")
async def ui_chat_history(
    request: Request,
    session_id: str,
    limit: int = 500,
    auth: bool = Depends(verify_ui_auth)
):
    """
    Return messages for a specific session.
    """
    try:
        user_id = getattr(request.state, "user_id", None) or request.query_params.get("user_id") or "default"
        from backend.qdrant.chat_history_store import fetch_session_messages

        messages = fetch_session_messages(user_id=user_id, session_id=session_id, limit=limit)
        return {"success": True, "messages": messages}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


# Enhanced root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint that serves the config UI as the main interface.
    """
    return await config_ui(request)


# Enhanced models endpoint with better compatibility
@app.get("/models")
async def get_models_redirect():
    """
    Models endpoint in Ollama format for OpenWebUI compatibility.
    Enhanced with better error handling and logging.
    """
    try:
        # Import with fallback
        try:
            from backend.api.v1.routes.models import get_models
        except ImportError:
            from api.v1.routes.models import get_models
        
        # Get models from the backend
        lexi_models = await get_models()
        
        # Transform to Ollama format with validation
        models = []
        if hasattr(lexi_models, 'models') and lexi_models.models:
            for model in lexi_models.models:
                model_data = {
                    "name": getattr(model, 'name', 'unknown'),
                    "modified_at": getattr(model, 'modified_at', ''),
                    "size": getattr(model, 'size', 0),
                    "digest": getattr(model, 'digest', ''),
                    "details": getattr(model, 'details', {})
                }
                models.append(model_data)
        
        logger.info(f"Returning {len(models)} models for compatibility")
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error in /models endpoint: {str(e)}")
        # Return empty models list for compatibility instead of error
        return {"models": []}


# System info endpoint for debugging
@app.get("/system/info")
async def system_info():
    """
    System information endpoint for debugging and monitoring.
    """
    try:
        import psutil
        import platform
        
        return {
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "resources": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "application": {
                "version": get_version(),
                "features_enabled": FeatureFlags.get_enabled_features(),
                "security": {
                    "api_key_enabled": SecurityConfig.API_KEY_ENABLED,
                    "jwt_enabled": SecurityConfig.JWT_ENABLED
                }
            }
        }
    except ImportError:
        # If psutil is not available, return basic info
        return {
            "application": {
                "version": get_version(),
                "features_enabled": FeatureFlags.get_enabled_features()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Enhanced development server configuration
    logger.info("Starting development server...")
    uvicorn.run(
        "backend.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
        log_level="info",
        access_log=True
    )
