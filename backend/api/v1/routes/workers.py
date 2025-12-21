"""
Worker Management API Routes

Provides endpoints for:
- Worker health status
- Manual worker execution
- Worker metrics
- Worker configuration
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.workers.qdrant_optimizer import WorkerCoordinator

logger = logging.getLogger("workers_api")
router = APIRouter(prefix="/workers", tags=["workers"])

# Global worker coordinator (initialized at startup)
_worker_coordinator: Optional[WorkerCoordinator] = None


def set_worker_coordinator(coordinator: WorkerCoordinator):
    """Set global worker coordinator instance."""
    global _worker_coordinator
    _worker_coordinator = coordinator


def get_worker_coordinator() -> WorkerCoordinator:
    """Get worker coordinator dependency."""
    if _worker_coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Worker coordinator not initialized"
        )
    return _worker_coordinator


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class WorkerStatus(BaseModel):
    """Status of a single worker."""
    name: str
    enabled: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_duration_seconds: Optional[float] = None
    last_status: str = "never_run"
    metrics: Dict[str, Any] = {}


class WorkersHealthResponse(BaseModel):
    """Overall workers health response."""
    overall_status: str
    workers: list[WorkerStatus]


class WorkerExecutionRequest(BaseModel):
    """Request to execute a worker."""
    worker_name: str


class WorkerExecutionResponse(BaseModel):
    """Response from worker execution."""
    success: bool
    duration: float
    timestamp: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/health", response_model=WorkersHealthResponse)
async def get_workers_health(
    coordinator: WorkerCoordinator = Depends(get_worker_coordinator)
):
    """
    Get health status of all workers.

    Returns:
        WorkersHealthResponse with overall status and per-worker details
    """
    try:
        status = await coordinator.get_worker_status()

        # Convert to response model
        workers = [
            WorkerStatus(
                name=w["name"],
                enabled=w["enabled"],
                last_run=w.get("last_run"),
                last_duration_seconds=w.get("last_duration"),
                last_status=w.get("last_status", "never_run"),
                metrics=w.get("metrics", {})
            )
            for w in status.get("workers", [])
        ]

        return WorkersHealthResponse(
            overall_status=status.get("overall_status", "unknown"),
            workers=workers
        )

    except Exception as e:
        logger.error(f"Failed to get worker health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=WorkerExecutionResponse)
async def execute_worker(
    request: WorkerExecutionRequest,
    coordinator: WorkerCoordinator = Depends(get_worker_coordinator)
):
    """
    Manually execute a specific worker.

    Args:
        request: Worker execution request with worker_name

    Returns:
        WorkerExecutionResponse with execution results
    """
    try:
        result = await coordinator.run_worker(request.worker_name)

        return WorkerExecutionResponse(
            success=result.get("success", False),
            duration=result.get("duration", 0.0),
            timestamp=result.get("timestamp", ""),
            metrics=result.get("metrics") if result.get("success") else None,
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Failed to execute worker {request.worker_name}: {e}",
                    exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{worker_name}")
async def get_worker_metrics(
    worker_name: str,
    coordinator: WorkerCoordinator = Depends(get_worker_coordinator)
):
    """
    Get detailed metrics for a specific worker.

    Args:
        worker_name: Name of the worker

    Returns:
        Dict with worker metrics from last run
    """
    try:
        status = await coordinator.get_worker_status()

        # Find worker
        worker = next(
            (w for w in status.get("workers", []) if w["name"] == worker_name),
            None
        )

        if not worker:
            raise HTTPException(
                status_code=404,
                detail=f"Worker {worker_name} not found"
            )

        return {
            "worker": worker_name,
            "metrics": worker.get("metrics", {}),
            "last_run": worker.get("last_run"),
            "last_duration": worker.get("last_duration"),
            "status": worker.get("last_status")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for {worker_name}: {e}",
                    exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_workers(
    coordinator: WorkerCoordinator = Depends(get_worker_coordinator)
):
    """
    List all available workers.

    Returns:
        Dict with worker names and enabled status
    """
    try:
        status = await coordinator.get_worker_status()

        workers = [
            {
                "name": w["name"],
                "enabled": w["enabled"],
                "description": _get_worker_description(w["name"])
            }
            for w in status.get("workers", [])
        ]

        return {
            "total": len(workers),
            "workers": workers
        }

    except Exception as e:
        logger.error(f"Failed to list workers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _get_worker_description(worker_name: str) -> str:
    """Get human-readable description of worker."""
    descriptions = {
        "deduplication": "Finds and merges duplicate/similar memories",
        "index_optimization": "Automatically tunes HNSW index parameters",
        "relevance_reranking": "Updates relevance scores based on usage",
        "data_quality": "Detects and repairs data integrity issues",
        "collection_balancing": "Implements HOT/WARM/COLD tiered storage"
    }

    return descriptions.get(worker_name, "Unknown worker")
