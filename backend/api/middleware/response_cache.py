"""
Response Caching Middleware
===========================

Caches LLM responses to avoid expensive re-computation for identical queries.

Performance Impact:
- Cache Hit: <50ms (vs 24s for LLM call)
- ~480x faster for repeated queries
- Significantly reduces Ollama load
"""

import hashlib
import time
import logging
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU-based response cache with TTL support.

    Features:
    - Per-user caching (user_id isolation)
    - TTL-based expiration (default: 1 hour)
    - LRU eviction when max size reached
    - Cache statistics tracking
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of cached responses (LRU eviction)
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        # Cache storage: {cache_key: (response, expiry_timestamp)}
        self.cache: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_saved_time_ms": 0.0
        }

        logger.info(f"✓ Response cache initialized (max_size={max_size}, ttl={default_ttl}s)")

    def _generate_cache_key(self, user_id: str, message: str, language: str = "de") -> str:
        """
        Generate unique cache key for a query.

        Args:
            user_id: User identifier
            message: User message
            language: Language flag

        Returns:
            MD5 hash as cache key
        """
        # Normalize message (lowercase, strip whitespace)
        normalized_message = message.lower().strip()

        # Create cache key string
        key_string = f"{user_id}:{language}:{normalized_message}"

        # Hash to fixed-length key
        cache_key = hashlib.md5(key_string.encode()).hexdigest()

        return cache_key

    def get(
        self,
        user_id: str,
        message: str,
        language: str = "de"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.

        Args:
            user_id: User identifier
            message: User message
            language: Language flag

        Returns:
            Cached response dict or None if not found/expired
        """
        cache_key = self._generate_cache_key(user_id, message, language)

        # Check if key exists
        if cache_key not in self.cache:
            self.stats["misses"] += 1
            logger.debug(f"✗ Cache miss for user={user_id}")
            return None

        # Get cached entry
        response, expiry = self.cache[cache_key]

        # Check if expired
        if time.time() > expiry:
            # Remove expired entry
            del self.cache[cache_key]
            self.stats["misses"] += 1
            logger.debug(f"✗ Cache expired for user={user_id}")
            return None

        # Move to end (LRU)
        self.cache.move_to_end(cache_key)

        # Update statistics
        self.stats["hits"] += 1
        self.stats["total_saved_time_ms"] += 24000  # Avg LLM response time

        logger.info(f"✓ Cache hit for user={user_id} (saved ~24s)")

        return response

    def set(
        self,
        user_id: str,
        message: str,
        response: Dict[str, Any],
        language: str = "de",
        ttl: Optional[int] = None
    ) -> None:
        """
        Store response in cache.

        Args:
            user_id: User identifier
            message: User message
            response: Response dict to cache
            language: Language flag
            ttl: Time-to-live in seconds (None = use default)
        """
        cache_key = self._generate_cache_key(user_id, message, language)

        # Calculate expiry timestamp
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl_seconds

        # Check if cache is full
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            # Evict oldest entry (LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
            logger.debug(f"⚠️  Cache full - evicted oldest entry")

        # Store in cache
        self.cache[cache_key] = (response, expiry)

        # Move to end (most recent)
        self.cache.move_to_end(cache_key)

        logger.debug(f"✓ Cached response for user={user_id} (ttl={ttl_seconds}s)")

    def invalidate_user(self, user_id: str) -> int:
        """
        Invalidate all cached responses for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of entries invalidated
        """
        # Find all keys for this user
        keys_to_remove = []

        for cache_key in self.cache.keys():
            # Cache key format: md5(user_id:language:message)
            # We need to check the actual stored data
            response, _ = self.cache[cache_key]
            if response.get("user_id") == user_id:
                keys_to_remove.append(cache_key)

        # Remove entries
        for key in keys_to_remove:
            del self.cache[key]

        count = len(keys_to_remove)
        logger.info(f"✓ Invalidated {count} cache entries for user={user_id}")

        return count

    def clear(self) -> None:
        """Clear all cached responses."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"✓ Cleared {count} cache entries")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of expired entries removed
        """
        current_time = time.time()
        keys_to_remove = []

        for cache_key, (_, expiry) in self.cache.items():
            if current_time > expiry:
                keys_to_remove.append(cache_key)

        for key in keys_to_remove:
            del self.cache[key]

        count = len(keys_to_remove)
        if count > 0:
            logger.info(f"✓ Cleaned up {count} expired cache entries")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.stats["evictions"],
            "total_saved_time_ms": self.stats["total_saved_time_ms"],
            "avg_saved_time_per_hit_ms": (
                self.stats["total_saved_time_ms"] / self.stats["hits"]
                if self.stats["hits"] > 0 else 0.0
            )
        }


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """
    Get global response cache instance (singleton).

    Returns:
        ResponseCache instance
    """
    global _response_cache

    if _response_cache is None:
        _response_cache = ResponseCache(
            max_size=1000,  # Store up to 1000 responses
            default_ttl=3600  # 1 hour default TTL
        )

    return _response_cache


def clear_response_cache() -> None:
    """Clear the global response cache."""
    global _response_cache

    if _response_cache is not None:
        _response_cache.clear()
