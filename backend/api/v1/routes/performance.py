"""
Performance monitoring endpoints for the Lexi API.
"""
from fastapi import APIRouter, HTTPException, Query
import httpx
import logging
import time
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from backend.monitoring.performance_metrics import get_metrics_collector

# Setup logging
logger = logging.getLogger("lexi_middleware.performance")

# Create router
router = APIRouter(tags=["performance"])

# Status cache to reduce frequent requests to Ollama
status_cache = {
    "last_update": 0,
    "data": None,
    "cache_duration": 5  # Cache for 5 seconds
}

class LLMStatus(BaseModel):
    """Model for LLM status response"""
    status: str  # "ok", "warning", "error"
    queue_length: int
    active_requests: int
    last_update: float
    model: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

@router.get("/status/llm", response_model=LLMStatus)
async def get_llm_status():
    """
    Get the current status of the LLM service (Ollama)
    
    Returns:
    - status: Current status (ok, warning, error)
    - queue_length: Number of requests in queue
    - active_requests: Number of active requests being processed
    - details: Additional status details from Ollama
    """
    try:
        # Check if we have a fresh cache
        current_time = time.time()
        if status_cache["data"] and (current_time - status_cache["last_update"] < status_cache["cache_duration"]):
            logger.debug("Returning cached LLM status")
            return status_cache["data"]
            
        # Get fresh data from Ollama
        from backend.config.middleware_config import MiddlewareConfig
        
        ollama_url = MiddlewareConfig.get_llm_url()
        ollama_status_url = f"{ollama_url}/api/stats" 
        
        async with httpx.AsyncClient(timeout=2.0) as client:
            logger.info(f"Fetching Ollama status from {ollama_status_url}")
            response = await client.get(ollama_status_url)
            
            if response.status_code != 200:
                logger.warning(f"Ollama returned non-200 status: {response.status_code}")
                return LLMStatus(
                    status="error",
                    queue_length=0,
                    active_requests=0,
                    last_update=current_time,
                    message=f"Ollama returned status code {response.status_code}"
                )
                
            # Parse the Ollama response
            data = response.json()
            logger.debug(f"Ollama stats: {data}")
            
            # Extract queue length and active requests
            # Note: These fields may vary based on Ollama's API
            queue_length = data.get("queue_size", 0)
            active_requests = data.get("processing", 0)
            
            # Determine status based on queue length
            status = "ok"
            if queue_length > 5:
                status = "error"
                message = "Hohe Auslastung - Lange Wartezeiten erwartet"
            elif queue_length > 2:
                status = "warning"
                message = "Moderate Auslastung - Kurze Verzögerung möglich"
            else:
                message = "Normaler Betrieb"
                
            # Create status object
            llm_status = LLMStatus(
                status=status,
                queue_length=queue_length,
                active_requests=active_requests,
                last_update=current_time,
                model=MiddlewareConfig.get_llm_model(),
                details=data,
                message=message
            )
            
            # Update cache
            status_cache["data"] = llm_status
            status_cache["last_update"] = current_time
            
            return llm_status
            
    except httpx.ConnectError:
        logger.error(f"Connection error to Ollama service")
        return LLMStatus(
            status="error",
            queue_length=0,
            active_requests=0,
            last_update=time.time(),
            message="Verbindungsfehler zum LLM-Dienst"
        )
    except Exception as e:
        logger.exception(f"Error checking LLM status: {str(e)}")
        return LLMStatus(
            status="error",
            queue_length=0,
            active_requests=0,
            last_update=time.time(),
            message=f"Fehler bei Statusabfrage: {str(e)}"
        )


# ===========================
# Performance Metrics Endpoints
# ===========================

class MetricsResponse(BaseModel):
    """Response model for metrics endpoints"""
    summary: Dict[str, Any]
    components: Dict[str, Any]
    slow_queries: List[Dict[str, Any]]
    recent_errors: List[Dict[str, Any]]
    uptime_seconds: float
    history_size: int


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    window_minutes: Optional[int] = Query(None, description="Optional time window in minutes (None = all time)")
):
    """
    Get comprehensive performance metrics.

    **Parameters:**
    - window_minutes: Optional time window (e.g., 5, 15, 60 for last 5/15/60 minutes)

    **Returns:**
    - summary: Aggregated query metrics (latency, throughput, cache hits)
    - components: Component-specific metrics (embeddings, Qdrant, memory)
    - slow_queries: Last 10 slow queries (>1000ms)
    - recent_errors: Last 10 errors
    - uptime_seconds: System uptime
    - history_size: Number of queries in history

    **Usage:**
    ```
    GET /v1/metrics              # All-time metrics
    GET /v1/metrics?window_minutes=15  # Last 15 minutes
    ```
    """
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats(window_minutes=window_minutes)

        logger.info(f"Metrics retrieved (window={window_minutes}min, queries={stats.get('history_size', 0)})")

        return MetricsResponse(**stats)

    except Exception as e:
        logger.exception(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Metriken: {str(e)}")


@router.get("/metrics/summary")
async def get_metrics_summary(
    window_minutes: Optional[int] = Query(None, description="Optional time window in minutes")
):
    """
    Get simplified metrics summary.

    **Returns:**
    - total_queries: Total query count
    - avg_latency_ms: Average latency
    - p95_latency_ms: 95th percentile latency
    - cache_hit_rate: Cache hit rate (%)
    - queries_per_second: Throughput
    - success_rate: Success rate (%)
    """
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats(window_minutes=window_minutes)

        summary = stats.get("summary", {})

        # Compute success rate
        total_queries = summary.get("total_queries", 0)
        successful_queries = summary.get("successful_queries", 0)
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": summary.get("failed_queries", 0),
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(summary.get("avg_latency_ms", 0), 2),
            "p50_latency_ms": round(summary.get("p50_latency_ms", 0), 2),
            "p95_latency_ms": round(summary.get("p95_latency_ms", 0), 2),
            "p99_latency_ms": round(summary.get("p99_latency_ms", 0), 2),
            "cache_hit_rate": round(summary.get("cache_hit_rate", 0), 2),
            "queries_per_second": round(summary.get("queries_per_second", 0), 3),
            "uptime_seconds": stats.get("uptime_seconds", 0),
            "window_minutes": window_minutes or "all"
        }

    except Exception as e:
        logger.exception(f"Error retrieving metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Metrikzusammenfassung: {str(e)}")


@router.get("/metrics/slow-queries")
async def get_slow_queries(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of slow queries to return")
):
    """
    Get recent slow queries.

    **Parameters:**
    - limit: Max number of slow queries (1-100, default 10)

    **Returns:**
    List of slow queries with query_id, duration_ms, timestamp
    """
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats()

        slow_queries = stats.get("slow_queries", [])[:limit]

        return {
            "slow_queries": slow_queries,
            "count": len(slow_queries),
            "threshold_ms": metrics_collector.slow_query_threshold_ms
        }

    except Exception as e:
        logger.exception(f"Error retrieving slow queries: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen langsamer Abfragen: {str(e)}")


@router.get("/metrics/errors")
async def get_recent_errors(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of errors to return")
):
    """
    Get recent errors.

    **Parameters:**
    - limit: Max number of errors (1-100, default 10)

    **Returns:**
    List of recent errors with query_id, error, timestamp
    """
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats()

        recent_errors = stats.get("recent_errors", [])[:limit]

        return {
            "errors": recent_errors,
            "count": len(recent_errors)
        }

    except Exception as e:
        logger.exception(f"Error retrieving errors: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Fehler: {str(e)}")


@router.get("/metrics/components")
async def get_component_metrics():
    """
    Get component-specific metrics.

    **Returns:**
    - embeddings: Embedding service metrics (calls, cache hits, latency)
    - qdrant: Qdrant database metrics (queries, errors, latency)
    - memory: Memory operation metrics (stores, retrievals, cache hits)
    """
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats()

        components = stats.get("components", {})

        # Compute derived metrics
        embedding_calls = components.get("embedding_calls", 0)
        embedding_cache_hits = components.get("embedding_cache_hits", 0)
        embedding_hit_rate = (embedding_cache_hits / embedding_calls * 100) if embedding_calls > 0 else 0

        memory_retrievals = components.get("memory_retrievals", 0)
        memory_cache_hits = components.get("memory_cache_hits", 0)
        memory_hit_rate = (memory_cache_hits / memory_retrievals * 100) if memory_retrievals > 0 else 0

        qdrant_queries = components.get("qdrant_queries", 0)
        qdrant_errors = components.get("qdrant_errors", 0)
        qdrant_error_rate = (qdrant_errors / qdrant_queries * 100) if qdrant_queries > 0 else 0

        return {
            "embeddings": {
                "total_calls": embedding_calls,
                "cache_hits": embedding_cache_hits,
                "cache_misses": components.get("embedding_cache_misses", 0),
                "cache_hit_rate": round(embedding_hit_rate, 2),
                "avg_latency_ms": round(components.get("embedding_avg_latency_ms", 0), 2)
            },
            "qdrant": {
                "total_queries": qdrant_queries,
                "errors": qdrant_errors,
                "error_rate": round(qdrant_error_rate, 2),
                "avg_latency_ms": round(components.get("qdrant_avg_latency_ms", 0), 2)
            },
            "memory": {
                "stores": components.get("memory_stores", 0),
                "retrievals": memory_retrievals,
                "cache_hits": memory_cache_hits,
                "cache_misses": components.get("memory_cache_misses", 0),
                "cache_hit_rate": round(memory_hit_rate, 2)
            }
        }

    except Exception as e:
        logger.exception(f"Error retrieving component metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Komponentenmetriken: {str(e)}")


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all performance metrics.

    **WARNING:** This clears all collected metrics history.
    Use only for testing or maintenance.
    """
    try:
        metrics_collector = get_metrics_collector()
        metrics_collector.reset()

        logger.warning("Metrics reset via API endpoint")

        return {
            "status": "success",
            "message": "Alle Metriken wurden zurückgesetzt"
        }

    except Exception as e:
        logger.exception(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Zurücksetzen der Metriken: {str(e)}")
