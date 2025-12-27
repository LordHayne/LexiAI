"""
Health check endpoints for the Lexi API.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
import time
import datetime
import psutil
import platform
from typing import Dict, Optional

from backend.utils.version import get_version
from backend.config.middleware_config import MiddlewareConfig
from backend.api.v1.models.response_models import HealthResponse, ComponentStatus
from backend.core.lexi_adapter import check_lexi_components_health
from backend.monitoring.performance_metrics import get_metrics_collector

# Create router
router = APIRouter(tags=["health"])

# Start time of the server
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to monitor the API status.
    """
    # Calculate uptime
    uptime_seconds = time.time() - START_TIME
    uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
    
    # Get basic system info
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    
    # Check Lexi components
    try:
        components_status = check_lexi_components_health()
    except Exception as e:
        components_status = {
            "database": ComponentStatus(
                status="error",
                message=f"Failed to check components: {str(e)}"
            )
        }
    
    # Determine overall status
    statuses = [comp.status for comp in components_status.values()]
    if "error" in statuses:
        overall_status = "error"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "ok"
    
    # Get memory stats if available
    memory_stats: Optional[Dict] = None
    if components_status.get("database") and components_status["database"].status == "ok":
        try:
            from backend.memory.adapter import get_memory_stats
            stats = get_memory_stats()
            memory_stats = {
                "total": stats.get("total", 0),
                "categories": stats.get("categories", {}),
                "last_access": datetime.datetime.now().isoformat()
            }
        except Exception:
            # Don't fail the health check if we can't get memory stats
            pass
    
    # Construct response
    response = HealthResponse(
        status=overall_status,
        version=get_version(),
        uptime=uptime_str,
        components=components_status,
        memory_stats=memory_stats
    )
    
    return response

@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint that returns minimal information.
    Useful for load balancers and basic monitoring.
    """
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint with comprehensive system information.
    """
    # Get detailed system metrics
    cpu_info = {
        "percent": psutil.cpu_percent(interval=1),
        "count": psutil.cpu_count(),
        "count_logical": psutil.cpu_count(logical=True)
    }
    
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    
    system_metrics = {
        "cpu": cpu_info,
        "memory": {
            "total": memory_info.total,
            "available": memory_info.available,
            "percent": memory_info.percent,
            "used": memory_info.used,
            "free": memory_info.free
        },
        "disk": {
            "total": disk_info.total,
            "used": disk_info.used,
            "free": disk_info.free,
            "percent": (disk_info.used / disk_info.total) * 100
        }
    }
    
    # Calculate uptime
    uptime_seconds = time.time() - START_TIME
    uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
    
    # Check components
    try:
        components_status = check_lexi_components_health()
    except Exception as e:
        components_status = {
            "error": ComponentStatus(
                status="error",
                message=f"Failed to check components: {str(e)}"
            )
        }

    # Get performance metrics
    performance_summary = {}
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_stats()

        # Extract key metrics
        summary = stats.get("summary", {})
        components_metrics = stats.get("components", {})

        performance_summary = {
            "queries": {
                "total": summary.get("total_queries", 0),
                "successful": summary.get("successful_queries", 0),
                "failed": summary.get("failed_queries", 0),
                "qps": round(summary.get("queries_per_second", 0), 3)
            },
            "latency": {
                "avg_ms": round(summary.get("avg_latency_ms", 0), 2),
                "p50_ms": round(summary.get("p50_latency_ms", 0), 2),
                "p95_ms": round(summary.get("p95_latency_ms", 0), 2),
                "p99_ms": round(summary.get("p99_latency_ms", 0), 2)
            },
            "cache": {
                "query_hit_rate": round(summary.get("cache_hit_rate", 0), 2),
                "embedding_hit_rate": round(
                    (components_metrics.get("embedding_cache_hits", 0) /
                     components_metrics.get("embedding_calls", 1) * 100)
                    if components_metrics.get("embedding_calls", 0) > 0 else 0,
                    2
                ),
                "memory_hit_rate": round(
                    (components_metrics.get("memory_cache_hits", 0) /
                     components_metrics.get("memory_retrievals", 1) * 100)
                    if components_metrics.get("memory_retrievals", 0) > 0 else 0,
                    2
                )
            },
            "slow_queries": summary.get("slow_queries_count", 0),
            "uptime_seconds": stats.get("uptime_seconds", 0)
        }
    except Exception as e:
        performance_summary = {"error": f"Failed to get metrics: {str(e)}"}

    # Get embedding cache stats
    embedding_cache_stats = {}
    try:
        from backend.embeddings.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        stats = cache.get_stats()
        embedding_cache_stats = {
            "size": stats.get("size", 0),
            "maxsize": stats.get("maxsize", 0),
            "hit_rate": round(stats.get("hit_rate", 0), 2),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "evictions": stats.get("evictions", 0)
        }
    except Exception as e:
        embedding_cache_stats = {"error": f"Cache not initialized: {str(e)}"}

    # Get memory cache stats
    memory_cache_stats = {}
    try:
        from backend.memory.cache import get_memory_cache
        cache = get_memory_cache()
        stats = cache.get_stats()
        memory_cache_stats = {
            "size": stats.get("size", 0),
            "hit_rate": round(stats.get("hit_rate", 0), 2),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0)
        }
    except Exception as e:
        memory_cache_stats = {"error": f"Cache not initialized: {str(e)}"}

    # Get Qdrant collection stats
    qdrant_stats = {}
    try:
        from backend.core.component_cache import get_cached_components
        bundle = get_cached_components()
        if hasattr(bundle, 'vectorstore'):
            collection_info = bundle.vectorstore.client.get_collection(
                collection_name=MiddlewareConfig.get_memory_collection()
            )
            qdrant_stats = {
                "collection": MiddlewareConfig.get_memory_collection(),
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                "status": collection_info.status if hasattr(collection_info, 'status') else "unknown"
            }
    except Exception as e:
        qdrant_stats = {"error": f"Failed to get collection stats: {str(e)}"}

    return {
        "status": "ok",
        "version": get_version(),
        "uptime": uptime_str,
        "timestamp": datetime.datetime.now().isoformat(),
        "system_metrics": system_metrics,
        "components": components_status,
        "performance": performance_summary,
        "caches": {
            "embedding_cache": embedding_cache_stats,
            "memory_cache": memory_cache_stats
        },
        "qdrant": qdrant_stats,
        "platform_info": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor()
        }
    }
