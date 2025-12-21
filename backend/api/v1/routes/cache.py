"""
Cache Statistics Endpoints
==========================

Provides cache performance metrics and management.
"""

import logging
from fastapi import APIRouter, Depends
from backend.api.middleware.auth import verify_api_key
from backend.api.middleware.response_cache import get_response_cache

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_cache_stats():
    """
    Get cache performance statistics.

    Returns:
        Dict with cache metrics:
        - size: Current number of cached responses
        - max_size: Maximum cache capacity
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate_percent: Cache hit rate percentage
        - evictions: Number of LRU evictions
        - total_saved_time_ms: Total time saved by cache hits
        - avg_saved_time_per_hit_ms: Average time saved per cache hit
    """
    cache = get_response_cache()
    stats = cache.get_stats()

    logger.info(f"Cache stats requested: {stats}")

    return {
        "status": "success",
        "cache": stats
    }


@router.post("/clear", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """
    Clear all cached responses.

    Returns:
        Dict with number of entries cleared
    """
    cache = get_response_cache()

    # Get size before clearing
    size_before = len(cache.cache)

    # Clear cache
    cache.clear()

    logger.info(f"Cache cleared: {size_before} entries removed")

    return {
        "status": "success",
        "message": f"Cache cleared ({size_before} entries removed)",
        "entries_removed": size_before
    }


@router.post("/cleanup", dependencies=[Depends(verify_api_key)])
async def cleanup_expired():
    """
    Remove expired cache entries.

    Returns:
        Dict with number of expired entries removed
    """
    cache = get_response_cache()

    # Cleanup expired entries
    removed_count = cache.cleanup_expired()

    logger.info(f"Cache cleanup: {removed_count} expired entries removed")

    return {
        "status": "success",
        "message": f"Removed {removed_count} expired entries",
        "entries_removed": removed_count
    }


@router.post("/invalidate/{user_id}", dependencies=[Depends(verify_api_key)])
async def invalidate_user_cache(user_id: str):
    """
    Invalidate all cached responses for a specific user.

    Args:
        user_id: User identifier

    Returns:
        Dict with number of entries invalidated
    """
    cache = get_response_cache()

    # Invalidate user's cache
    removed_count = cache.invalidate_user(user_id)

    logger.info(f"Cache invalidated for user={user_id}: {removed_count} entries removed")

    return {
        "status": "success",
        "message": f"Invalidated {removed_count} entries for user {user_id}",
        "entries_removed": removed_count,
        "user_id": user_id
    }
