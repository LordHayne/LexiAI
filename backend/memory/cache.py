"""
Memory caching system for improving retrieval performance.
Enhanced with better error handling, configuration options, and monitoring.
"""

import logging
import time
import hashlib
import json
import copy
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
import gc

from backend.config.feature_flags import FeatureFlags

logger = logging.getLogger("lexi_middleware.memory_cache")


@dataclass
class CacheConfig:
    """Configuration for memory cache behavior."""
    default_ttl: int = 600  # seconds
    max_entries_per_user: int = 1000
    max_total_entries: int = 50000
    cleanup_interval: int = 300  # seconds
    enable_compression: bool = False
    enable_metrics: bool = True


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    timestamp: float
    data: List[Dict[str, Any]]
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size()

    def _estimate_size(self) -> int:
        """Rough estimation of memory usage."""
        try:
            return len(json.dumps(self.data).encode('utf-8'))
        except (TypeError, ValueError):
            return len(str(self.data).encode('utf-8'))

    def is_expired(self, ttl: int) -> bool:
        """Check if entry is expired based on TTL."""
        return time.time() - self.timestamp > ttl

    def touch(self):
        """Update last access time and increment counter."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """
    Enhanced in-memory cache for memory retrieval results.
    Features:
    - Thread-safe access with RLock
    - Per-user TTL and statistics
    - Configurable limits and cleanup
    - Memory usage tracking
    - LRU eviction when limits exceeded
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: Dict[str, Dict[str, CacheEntry]] = {}
        self._global_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
        self._user_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'hits': 0, 'misses': 0, 'entries': 0, 'size_bytes': 0
        })
        self._lock = RLock()
        self._last_cleanup = time.time()

    def get(self, user_id: str, query: str, tags: Optional[List[str]] = None, 
            ttl: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached results for a query.
        
        Args:
            user_id: User identifier
            query: Search query
            tags: Optional query tags
            ttl: Custom TTL override
            
        Returns:
            Cached results or None if not found/expired
        """
        if not FeatureFlags.is_enabled("memory_caching"):
            self._record_miss(user_id)
            return None

        self._maybe_cleanup()
        cache_key = self._generate_cache_key(query, tags)
        effective_ttl = ttl or self.config.default_ttl

        with self._lock:
            try:
                user_cache = self._cache.get(user_id)
                if not user_cache:
                    self._record_miss(user_id)
                    return None

                entry = user_cache.get(cache_key)
                if not entry:
                    self._record_miss(user_id)
                    return None

                if entry.is_expired(effective_ttl):
                    self._remove_entry(user_id, cache_key)
                    self._record_miss(user_id)
                    return None

                entry.touch()
                self._record_hit(user_id)
                return copy.deepcopy(entry.data)

            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
                self._record_miss(user_id)
                return None

    def store(self, user_id: str, query: str, results: List[Dict[str, Any]], 
              tags: Optional[List[str]] = None, ttl: Optional[int] = None) -> bool:
        """
        Store results in cache.
        
        Args:
            user_id: User identifier
            query: Search query
            results: Results to cache
            tags: Optional query tags
            ttl: Custom TTL override
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not FeatureFlags.is_enabled("memory_caching"):
            return False

        if not results:
            return False

        cache_key = self._generate_cache_key(query, tags)

        with self._lock:
            try:
                # Check global limits
                total_entries = sum(len(cache) for cache in self._cache.values())
                if total_entries >= self.config.max_total_entries:
                    self._evict_lru_global()

                user_cache = self._cache.setdefault(user_id, {})

                # Check per-user limits
                if len(user_cache) >= self.config.max_entries_per_user:
                    self._evict_lru_user(user_id)

                # Create and store entry
                entry = CacheEntry(
                    timestamp=time.time(),
                    data=copy.deepcopy(results)
                )

                user_cache[cache_key] = entry
                self._update_user_stats(user_id, entry.size_bytes)

                logger.debug(f"Cached {len(results)} results for user {user_id} "
                           f"(key: {cache_key[:8]}..., size: {entry.size_bytes} bytes)")
                return True

            except Exception as e:
                logger.error(f"Error storing in cache: {e}")
                return False

    def invalidate_user(self, user_id: str) -> int:
        """Invalidate all cache entries for a user."""
        with self._lock:
            if user_id in self._cache:
                count = len(self._cache[user_id])
                del self._cache[user_id]
                if user_id in self._user_stats:
                    del self._user_stats[user_id]
                logger.info(f"Invalidated {count} cache entries for user {user_id}")
                return count
            return 0

    def invalidate_pattern(self, user_id: str, query_pattern: str) -> int:
        """Invalidate entries matching a query pattern."""
        removed = 0
        with self._lock:
            user_cache = self._cache.get(user_id)
            if not user_cache:
                return 0

            keys_to_remove = []
            for cache_key, entry in user_cache.items():
                # Simple pattern matching - could be enhanced with regex
                if query_pattern.lower() in cache_key.lower():
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                self._remove_entry(user_id, key)
                removed += 1

        logger.info(f"Invalidated {removed} entries matching pattern '{query_pattern}' for user {user_id}")
        return removed

    def clear(self, user_id: Optional[str] = None) -> int:
        """Clear cache entries."""
        with self._lock:
            if user_id:
                return self.invalidate_user(user_id)
            else:
                count = sum(len(entries) for entries in self._cache.values())
                self._cache.clear()
                self._global_stats = {k: 0 for k in self._global_stats}
                self._user_stats.clear()
                logger.info(f"Cleared all cache entries ({count} total)")
                return count

    def cleanup(self, force: bool = False) -> int:
        """Remove expired entries."""
        if not force and time.time() - self._last_cleanup < self.config.cleanup_interval:
            return 0

        removed = 0
        now = time.time()

        with self._lock:
            users_to_remove = []
            for user_id in list(self._cache.keys()):
                user_cache = self._cache[user_id]
                expired_keys = [
                    k for k, entry in user_cache.items() 
                    if entry.is_expired(self.config.default_ttl)
                ]
                
                for key in expired_keys:
                    self._remove_entry(user_id, key)
                    removed += 1

                if not user_cache:
                    users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del self._cache[user_id]
                if user_id in self._user_stats:
                    del self._user_stats[user_id]

        self._last_cleanup = now
        self._global_stats['cleanups'] += 1

        if removed:
            logger.info(f"Cleaned up {removed} expired cache entries")
        
        return removed

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if user_id:
                user_cache = self._cache.get(user_id, {})
                user_stats = self._user_stats.get(user_id, {})
                total_requests = user_stats.get('hits', 0) + user_stats.get('misses', 0)
                
                return {
                    "enabled": FeatureFlags.is_enabled("memory_caching"),
                    "entries": len(user_cache),
                    "hits": user_stats.get('hits', 0),
                    "misses": user_stats.get('misses', 0),
                    "hit_rate": user_stats.get('hits', 0) / total_requests if total_requests > 0 else 0.0,
                    "size_bytes": user_stats.get('size_bytes', 0),
                    "ttl_seconds": self.config.default_ttl,
                    "avg_access_count": self._get_avg_access_count(user_id)
                }
            else:
                total_requests = self._global_stats['hits'] + self._global_stats['misses']
                total_size = sum(stats.get('size_bytes', 0) for stats in self._user_stats.values())
                
                return {
                    "enabled": FeatureFlags.is_enabled("memory_caching"),
                    "total_entries": sum(len(cache) for cache in self._cache.values()),
                    "users": len(self._cache),
                    "hits": self._global_stats['hits'],
                    "misses": self._global_stats['misses'],
                    "evictions": self._global_stats['evictions'],
                    "cleanups": self._global_stats['cleanups'],
                    "hit_rate": self._global_stats['hits'] / total_requests if total_requests > 0 else 0.0,
                    "total_size_bytes": total_size,
                    "ttl_seconds": self.config.default_ttl,
                    "config": {
                        "max_entries_per_user": self.config.max_entries_per_user,
                        "max_total_entries": self.config.max_total_entries,
                        "cleanup_interval": self.config.cleanup_interval
                    }
                }

    def _generate_cache_key(self, query: str, tags: Optional[List[str]] = None) -> str:
        """Generate a cache key from query and tags."""
        key_data = {
            "query": query.strip().lower(),
            "tags": sorted(tags) if tags else None
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]  # Shorter hash

    def _record_hit(self, user_id: str):
        """Record a cache hit."""
        self._global_stats['hits'] += 1
        self._user_stats[user_id]['hits'] += 1

    def _record_miss(self, user_id: str):
        """Record a cache miss."""
        self._global_stats['misses'] += 1
        self._user_stats[user_id]['misses'] += 1

    def _remove_entry(self, user_id: str, cache_key: str):
        """Remove a single cache entry and update stats."""
        user_cache = self._cache.get(user_id)
        if user_cache and cache_key in user_cache:
            entry = user_cache[cache_key]
            del user_cache[cache_key]
            self._user_stats[user_id]['size_bytes'] -= entry.size_bytes
            self._user_stats[user_id]['entries'] -= 1

    def _update_user_stats(self, user_id: str, size_bytes: int):
        """Update user statistics after adding entry."""
        self._user_stats[user_id]['size_bytes'] += size_bytes
        self._user_stats[user_id]['entries'] += 1

    def _evict_lru_user(self, user_id: str):
        """Evict least recently used entry for a user."""
        user_cache = self._cache.get(user_id)
        if not user_cache:
            return

        # Find entry with oldest last_accessed time
        lru_key = min(user_cache.items(), key=lambda x: x[1].last_accessed)[0]
        self._remove_entry(user_id, lru_key)
        self._global_stats['evictions'] += 1
        logger.debug(f"Evicted LRU entry for user {user_id}")

    def _evict_lru_global(self):
        """Evict least recently used entry globally."""
        lru_user = None
        lru_key = None
        lru_time = float('inf')

        for user_id, user_cache in self._cache.items():
            for cache_key, entry in user_cache.items():
                if entry.last_accessed < lru_time:
                    lru_time = entry.last_accessed
                    lru_user = user_id
                    lru_key = cache_key

        if lru_user and lru_key:
            self._remove_entry(lru_user, lru_key)
            self._global_stats['evictions'] += 1
            logger.debug(f"Evicted global LRU entry for user {lru_user}")

    def _get_avg_access_count(self, user_id: str) -> float:
        """Get average access count for user's entries."""
        user_cache = self._cache.get(user_id, {})
        if not user_cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in user_cache.values())
        return total_accesses / len(user_cache)

    def _maybe_cleanup(self):
        """Perform cleanup if interval has passed."""
        if time.time() - self._last_cleanup >= self.config.cleanup_interval:
            self.cleanup()


# Global cache instance
_default_cache = None


def get_memory_cache(config: Optional[CacheConfig] = None) -> MemoryCache:
    """Get or create the default memory cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = MemoryCache(config)
    return _default_cache


def generate_query_hash(query: str, tags: Optional[List[str]] = None, 
                       limit: Optional[int] = None) -> str:
    """
    Generate a hash for query parameters.
    Compatible with the cache key generation but includes limit.
    """
    key_data = {
        "query": query.strip().lower(),
        "tags": sorted(tags) if tags else None,
        "limit": limit
    }
    raw = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# Convenience functions for backward compatibility
def cache_get(user_id: str, query: str, tags: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
    """Convenience function for cache retrieval."""
    return get_memory_cache().get(user_id, query, tags)


def cache_store(user_id: str, query: str, results: List[Dict[str, Any]], 
                tags: Optional[List[str]] = None) -> bool:
    """Convenience function for cache storage."""
    return get_memory_cache().store(user_id, query, results, tags)


def cache_clear(user_id: Optional[str] = None) -> int:
    """Convenience function for cache clearing."""
    return get_memory_cache().clear(user_id)