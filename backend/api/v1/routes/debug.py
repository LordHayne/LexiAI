"""
Debug endpoints for the Lexi API.
"""
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import logging
import os
import sys
import datetime
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import timezone

from backend.api.middleware.auth import verify_api_key

# Setup logging
logger = logging.getLogger("lexi_middleware.debug")

# Create router with authentication dependency for sensitive endpoints
router = APIRouter(tags=["debug"])

# Global lock and rate limiting for UI trigger
_ui_trigger_lock = asyncio.Lock()
_last_ui_trigger = None
UI_TRIGGER_COOLDOWN = 1  # Seconds between triggers (reduced to 1 second for development)

@router.post("/debug/echo")
async def debug_echo(request: Request):
    """
    Echo back the request details for debugging purposes.
    """
    try:
        # Get request body safely
        body = None
        try:
            body = await request.json()
        except Exception:
            # If JSON parsing fails, try to get raw body
            try:
                raw_body = await request.body()
                body = {"raw_body": raw_body.decode('utf-8') if raw_body else None}
            except Exception:
                body = {"error": "Could not parse request body"}
        
        # Get headers
        headers = dict(request.headers)
        # Remove sensitive information
        sensitive_headers = ["authorization", "x-api-key", "cookie", "x-auth-token"]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "***REDACTED***"
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Log the request details
        logger.info(f"DEBUG ECHO - Method: {request.method}, URL: {request.url}")
        logger.info(f"DEBUG ECHO - Headers: {headers}")
        logger.info(f"DEBUG ECHO - Body: {body}")
        
        # Return the request details
        return {
            "success": True,
            "message": "Request details echoed for debugging",
            "timestamp": datetime.datetime.now().isoformat(),
            "request": {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": query_params,
                "headers": headers,
                "body": body,
                "client_host": request.client.host if request.client else "unknown",
                "client_port": request.client.port if request.client else "unknown"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in debug echo: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.get("/debug/config", dependencies=[Depends(verify_api_key)])
async def debug_config():
    """
    Return the current configuration for debugging purposes.
    Requires API key authentication for security.
    """
    try:
        from backend.config.middleware_config import MiddlewareConfig
        from backend.config.feature_flags import FeatureFlags
        
        # Get configuration (with sensitive data masked)
        config = {
            "ollama_url": MiddlewareConfig.get_ollama_base_url(),
            "embedding_url": MiddlewareConfig.get_embedding_base_url(),
            "qdrant_host": MiddlewareConfig.get_qdrant_host(),
            "qdrant_port": MiddlewareConfig.get_qdrant_port(),
            "memory_collection": MiddlewareConfig.get_memory_collection(),
            "memory_dimension": MiddlewareConfig.get_memory_dimension(),
            "default_llm_model": MiddlewareConfig.get_default_llm_model(),
            "default_embedding_model": MiddlewareConfig.get_default_embedding_model(),
            "cache_enabled": MiddlewareConfig.get_cache_enabled(),
            "rate_limit_enabled": MiddlewareConfig.get_rate_limit_enabled(),
            "features": FeatureFlags.get_all()
        }
        
        # Add environment info (non-sensitive)
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "debug_mode": os.getenv("DEBUG", "False").lower() == "true",
            "environment": os.getenv("ENVIRONMENT", "unknown")
        }
        
        # Log the configuration access
        logger.info("DEBUG CONFIG accessed")
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "environment": env_info
        }
        
    except Exception as e:
        logger.error(f"Error in debug config: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.get("/debug/logs", dependencies=[Depends(verify_api_key)])
async def debug_logs(lines: int = 50):
    """
    Return recent log entries for debugging.
    Requires API key authentication.
    """
    try:
        # Get recent log entries (this is a simplified implementation)
        # In a real scenario, you might want to read from actual log files
        log_entries = []
        
        # Get all handlers for the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, 'buffer'):
                # If it's a memory handler, get recent entries
                log_entries.extend(handler.buffer[-lines:])
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "log_entries": [str(entry) for entry in log_entries],
            "note": "This shows recent log entries from memory handlers only"
        }
    
    except Exception as e:
        logger.error(f"Error in debug logs: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.get("/debug/system", dependencies=[Depends(verify_api_key)])
async def debug_system():
    """
    Return system information for debugging.
    Requires API key authentication.
    """
    try:
        import psutil
        import platform
        
        # Get system information
        system_info = {
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "usage": psutil.disk_usage('/')._asdict()
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:5]  # Show first 5 paths only
            }
        }
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "system_info": system_info
        }
        
    except Exception as e:
        logger.error(f"Error in debug system: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.post("/debug/test-component")
async def debug_test_component(component: str, action: str = "ping"):
    """
    Test connectivity to specific components.
    """
    try:
        result = {"component": component, "action": action, "status": "unknown"}
        
        if component.lower() == "qdrant":
            from backend.core.lexi_adapter import check_lexi_components_health
            components = check_lexi_components_health()
            if "qdrant" in components:
                result["status"] = components["qdrant"].status
                result["message"] = components["qdrant"].message
            else:
                result["status"] = "not_found"
                result["message"] = "Qdrant component not found in health check"
                
        elif component.lower() == "ollama":
            # Test Ollama connection
            from backend.config.middleware_config import MiddlewareConfig
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{MiddlewareConfig.get_ollama_base_url()}/api/tags")
                    if response.status_code == 200:
                        result["status"] = "ok"
                        result["message"] = "Ollama is responding"
                    else:
                        result["status"] = "error"
                        result["message"] = f"Ollama returned status {response.status_code}"
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"Failed to connect to Ollama: {str(e)}"
                
        else:
            result["status"] = "unsupported"
            result["message"] = f"Component '{component}' testing not implemented"
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "test_result": result
        }

    except Exception as e:
        logger.error(f"Error in debug test component: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.get("/debug/heartbeat/status")
async def heartbeat_status():
    """
    Get current status of the Heartbeat Service.
    Shows statistics about memory consolidation, cleanup, and current mode.
    Includes LIVE memory count from Qdrant.
    """
    try:
        from backend.services.heartbeat_memory import (
            get_heartbeat_status,
            get_heartbeat_config_snapshot,
        )
        from backend.core.component_cache import get_cached_components

        hb_status = get_heartbeat_status()
        config_snapshot = get_heartbeat_config_snapshot()

        # Get LIVE memory count from Qdrant (without vectors for performance)
        try:
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore
            all_memories = vectorstore.get_all_entries(with_vectors=False)
            live_memory_count = len(all_memories)

            # Override cached count with live count
            hb_status["total_memories"] = live_memory_count
        except Exception as e:
            logger.warning(f"Could not fetch live memory count: {e}")
            # Keep cached count if live fetch fails

        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "heartbeat": hb_status,
            "info": {
                "description": "Heartbeat service manages intelligent memory consolidation",
                "interval_seconds": config_snapshot["run_interval_seconds"],
                "idle_threshold_minutes": config_snapshot["idle_threshold_minutes"],
                "modes": {
                    "IDLE": "Intensive processing (synthesis, consolidation, cleanup)",
                    "ACTIVE": "Lightweight updates only"
                }
            }
        }

    except Exception as e:
        logger.error(f"Error getting heartbeat status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )


@router.get("/debug/feedback/status")
async def feedback_status():
    """
    Get feedback storage and hydration status.
    Useful for debugging feedback learning pipeline.
    """
    try:
        from backend.memory.conversation_tracker import get_conversation_tracker

        tracker = get_conversation_tracker()
        stats = tracker.get_feedback_stats()
        storage_status = tracker.get_feedback_storage_status()

        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "feedback": stats,
            "storage": storage_status,
        }
    except Exception as e:
        logger.error(f"Error getting feedback status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.post("/debug/heartbeat/trigger", dependencies=[Depends(verify_api_key)])
async def trigger_heartbeat():
    """
    Manually trigger memory maintenance cycle.
    Requires API key authentication.
    This will run the full maintenance cycle regardless of idle status.
    """
    try:
        from backend.services.heartbeat_memory import intelligent_memory_maintenance

        logger.info("Manual heartbeat trigger initiated (API)")

        # Run maintenance cycle (async function must be awaited)
        result = await intelligent_memory_maintenance()

        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Memory maintenance cycle completed",
            "result": result
        }

    except Exception as e:
        logger.error(f"Error triggering heartbeat: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

@router.post("/debug/heartbeat/ui-trigger")
async def ui_trigger_heartbeat(request: Request):
    """
    UI-friendly endpoint to trigger memory operations.
    Protected with lock and rate limiting to prevent parallel execution.

    Body Parameters:
    - mode: str - Operation mode ("consolidation", "synthesis", "cleanup", "deep_learning")
    - force_deep_learning: bool - Force full deep learning cycle (for backwards compatibility)
    """
    global _last_ui_trigger

    try:
        from backend.services.heartbeat_memory import (
            run_deep_learning_tasks,
            _synthesize_memories,
            _consolidate_memories,
            _cleanup_memories,
            _update_all_relevances,
            _consolidate_meta_knowledge,
            HeartbeatLimits,
            MemoryBudgetManager,
            _config,
            allow_learning_processes
        )
        from backend.core.component_cache import get_cached_components
        from backend.memory.memory_intelligence import get_usage_tracker

        # Get request body
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        mode = body.get("mode", "deep_learning")
        force_deep = body.get("force_deep_learning", False)
        cleanup_max_age_days = body.get("max_age_days")
        cleanup_min_relevance = body.get("min_relevance")
        cleanup_unused_after_days = body.get("unused_after_days")
        cleanup_max_unused_relevance = body.get("max_unused_relevance")
        consolidation_threshold = body.get("consolidation_threshold")
        consolidation_min_cluster_size = body.get("consolidation_min_cluster_size")
        consolidation_allowed_tags = body.get("consolidation_allowed_tags")
        consolidation_allowed_sources = body.get("consolidation_allowed_sources")
        consolidation_require_meta = body.get("consolidation_require_meta")

        if cleanup_max_age_days is not None:
            try:
                cleanup_max_age_days = int(cleanup_max_age_days)
            except (TypeError, ValueError):
                logger.warning(f"Invalid max_age_days: {cleanup_max_age_days}")
                cleanup_max_age_days = None

        if cleanup_min_relevance is not None:
            try:
                cleanup_min_relevance = float(cleanup_min_relevance)
            except (TypeError, ValueError):
                logger.warning(f"Invalid min_relevance: {cleanup_min_relevance}")
                cleanup_min_relevance = None

        if cleanup_unused_after_days is not None:
            try:
                cleanup_unused_after_days = int(cleanup_unused_after_days)
            except (TypeError, ValueError):
                logger.warning(f"Invalid unused_after_days: {cleanup_unused_after_days}")
                cleanup_unused_after_days = None

        if cleanup_max_unused_relevance is not None:
            try:
                cleanup_max_unused_relevance = float(cleanup_max_unused_relevance)
            except (TypeError, ValueError):
                logger.warning(f"Invalid max_unused_relevance: {cleanup_max_unused_relevance}")
                cleanup_max_unused_relevance = None

        if consolidation_threshold is not None:
            try:
                consolidation_threshold = float(consolidation_threshold)
            except (TypeError, ValueError):
                logger.warning(f"Invalid consolidation_threshold: {consolidation_threshold}")
                consolidation_threshold = None

        if consolidation_min_cluster_size is not None:
            try:
                consolidation_min_cluster_size = int(consolidation_min_cluster_size)
            except (TypeError, ValueError):
                logger.warning(f"Invalid consolidation_min_cluster_size: {consolidation_min_cluster_size}")
                consolidation_min_cluster_size = None

        if consolidation_allowed_tags is not None and not isinstance(consolidation_allowed_tags, list):
            logger.warning(f"Invalid consolidation_allowed_tags: {consolidation_allowed_tags}")
            consolidation_allowed_tags = None
        if consolidation_allowed_sources is not None and not isinstance(consolidation_allowed_sources, list):
            logger.warning(f"Invalid consolidation_allowed_sources: {consolidation_allowed_sources}")
            consolidation_allowed_sources = None
        if consolidation_require_meta is not None and not isinstance(consolidation_require_meta, bool):
            logger.warning(f"Invalid consolidation_require_meta: {consolidation_require_meta}")
            consolidation_require_meta = None

        # ✅ Rate Limiting Check
        if _last_ui_trigger:
            elapsed = (datetime.datetime.now(timezone.utc) - _last_ui_trigger).total_seconds()
            if elapsed < UI_TRIGGER_COOLDOWN:
                wait_time = UI_TRIGGER_COOLDOWN - elapsed
                logger.warning(f"⚠️ Rate limit: {wait_time:.0f}s remaining")
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "error": f"Rate limit: Please wait {wait_time:.0f} seconds",
                        "retry_after": int(wait_time),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )

        # ✅ Lock Check - Prevent parallel execution
        if _ui_trigger_lock.locked():
            logger.warning("⚠️ Operation already in progress")
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "error": "Memory operation already in progress",
                    "message": "Please wait for current operation to complete",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )

        # ✅ Execute with Lock
        async with _ui_trigger_lock:
            # Update last trigger time
            _last_ui_trigger = datetime.datetime.now(timezone.utc)

            logger.info(f"🎨 UI trigger initiated - Mode: {mode}")
            allow_learning_processes()

            # Get stats BEFORE operation
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore
            embeddings = bundle.embeddings
            memories_before = len(vectorstore.get_all_entries(with_vectors=False))

            # Execute based on mode
            if mode == "deep_learning" or force_deep:
                # Run FULL deep learning cycle (all 8 phases)
                result = await run_deep_learning_tasks()
            else:
                # Run individual operation
                limits = HeartbeatLimits(_config)
                budget_manager = MemoryBudgetManager(vectorstore, _config)
                usage_tracker = get_usage_tracker()
                all_memories = vectorstore.get_all_entries()

                result = {
                    "consolidated": 0,
                    "synthesized": 0,
                    "deleted": 0,
                    "updated": 0,
                    "corrections": 0,
                    "patterns_detected": 0,
                    "knowledge_gaps_found": 0,
                    "goals_reminded": 0
                }

                # Accept both "consolidate" and "consolidation"
                if mode in ["consolidation", "consolidate"]:
                    logger.info("🔗 Running Memory Consolidation")
                    if consolidation_require_meta:
                        result["consolidated"] = _consolidate_meta_knowledge(
                            vectorstore,
                            all_memories,
                            limits,
                            similarity_threshold=consolidation_threshold,
                            min_cluster_size=consolidation_min_cluster_size or 2
                        )
                    else:
                        result["consolidated"] = _consolidate_memories(
                            vectorstore,
                            embeddings,
                            all_memories,
                            limits,
                            similarity_threshold=consolidation_threshold,
                            min_cluster_size=consolidation_min_cluster_size or 2,
                            allowed_tags=consolidation_allowed_tags,
                            allowed_sources=consolidation_allowed_sources,
                            require_meta_knowledge=bool(consolidation_require_meta)
                        )

                # Accept both "synthesize" and "synthesis"
                elif mode in ["synthesis", "synthesize"]:
                    logger.info("🧪 Running Memory Synthesis")
                    result["synthesized"] = _synthesize_memories(bundle, limits, budget_manager)

                elif mode == "cleanup":
                    logger.info("🧹 Running Intelligent Cleanup")
                    result["deleted"] = _cleanup_memories(
                        vectorstore,
                        all_memories,
                        usage_tracker,
                        limits,
                        max_age_days=cleanup_max_age_days,
                        min_relevance=cleanup_min_relevance,
                        unused_after_days=cleanup_unused_after_days,
                        max_unused_relevance=cleanup_max_unused_relevance
                    )
                    result["updated"] = _update_all_relevances(vectorstore, all_memories, usage_tracker, limits)

                else:
                    raise ValueError(f"Unknown mode: {mode}. Supported modes: consolidation/consolidate, synthesis/synthesize, cleanup, deep_learning")

            # Get stats AFTER operation
            memories_after = len(vectorstore.get_all_entries(with_vectors=False))

            logger.info(f"✅ UI trigger completed: {result}")

            return {
                "success": True,
                "timestamp": datetime.datetime.now().isoformat(),
                "message": f"{mode.replace('_', ' ').title()} completed",
                "stats": {
                    "memories_before": memories_before,
                    "memories_after": memories_after,
                    "memories_reduced": memories_before - memories_after,
                    "consolidated": result.get("consolidated", 0),
                    "synthesized": result.get("synthesized", 0),
                    "deleted": result.get("deleted", 0),
                    "updated": result.get("updated", 0),
                    "corrections": result.get("corrections", 0),
                    "patterns_detected": result.get("patterns_detected", 0),
                    "knowledge_gaps_found": result.get("knowledge_gaps_found", 0),
                    "goals_reminded": result.get("goals_reminded", 0)
                }
            }

    except Exception as e:
        logger.error(f"Error in UI trigger: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
