"""
Embedding Cache - LRU cache for embedding vectors.

PERFORMANCE: Caching embeddings for repeated queries provides 3-5x speedup
by avoiding redundant embedding computations.

Usage:
    >>> from backend.embeddings.embedding_cache import EmbeddingCache
    >>> cache = EmbeddingCache(maxsize=1000)
    >>>
    >>> # Store embedding
    >>> cache.set("my query text", [0.1, 0.2, 0.3, ...])
    >>>
    >>> # Retrieve embedding
    >>> vector = cache.get("my query text")
    >>> if vector:
    >>>     print("Cache hit!")
"""

import hashlib
import logging
import time
from typing import Optional, List
from functools import lru_cache
from threading import RLock

logger = logging.getLogger("lexi_middleware.embedding_cache")


class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding vectors.

    Stores embeddings keyed by text hash to avoid recomputing embeddings
    for repeated queries.
    """

    def __init__(self, maxsize: int = 1000, enable_stats: bool = True):
        """
        Initialize embedding cache.

        Args:
            maxsize: Maximum number of embeddings to cache
            enable_stats: Whether to track cache statistics
        """
        self.maxsize = maxsize
        self.enable_stats = enable_stats
        self._cache = {}  # {text_hash: embedding_vector}
        self._lock = RLock()

        # LRU tracking
        self._access_order = []  # Most recently accessed at end

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _hash_text(self, text: str) -> str:
        """
        Generate stable hash for text.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash of text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache.

        Args:
            text: Query text

        Returns:
            Embedding vector if cached, None otherwise
        """
        text_hash = self._hash_text(text)

        with self._lock:
            if text_hash in self._cache:
                # Update LRU: move to end (most recent)
                if text_hash in self._access_order:
                    self._access_order.remove(text_hash)
                self._access_order.append(text_hash)

                if self.enable_stats:
                    self._hits += 1

                logger.debug(f"Cache HIT for text hash: {text_hash[:16]}...")
                return self._cache[text_hash]

            if self.enable_stats:
                self._misses += 1

            logger.debug(f"Cache MISS for text hash: {text_hash[:16]}...")
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Query text
            embedding: Embedding vector to cache
        """
        if not embedding:
            return

        text_hash = self._hash_text(text)

        with self._lock:
            # Check if we need to evict
            if text_hash not in self._cache and len(self._cache) >= self.maxsize:
                # Evict least recently used (first in access_order)
                if self._access_order:
                    lru_hash = self._access_order.pop(0)
                    if lru_hash in self._cache:
                        del self._cache[lru_hash]
                        if self.enable_stats:
                            self._evictions += 1
                        logger.debug(f"Evicted LRU entry: {lru_hash[:16]}...")

            # Store embedding
            self._cache[text_hash] = embedding

            # Update access order
            if text_hash in self._access_order:
                self._access_order.remove(text_hash)
            self._access_order.append(text_hash)

            logger.debug(f"Cached embedding for text hash: {text_hash[:16]}... (cache size: {len(self._cache)})")

    def clear(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            logger.info(f"Cleared {count} cached embeddings")
            return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": f"{hit_rate:.1f}%",
                "total_requests": total_requests
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache statistics reset")


# Global cache instance
_embedding_cache = None
_cache_lock = RLock()


def get_embedding_cache(maxsize: int = 1000) -> EmbeddingCache:
    """
    Get global embedding cache instance (singleton).

    Args:
        maxsize: Maximum cache size (only used on first call)

    Returns:
        Global EmbeddingCache instance
    """
    global _embedding_cache

    with _cache_lock:
        if _embedding_cache is None:
            _embedding_cache = EmbeddingCache(maxsize=maxsize)
            logger.info(f"Initialized global embedding cache (maxsize={maxsize})")
        return _embedding_cache


def cached_embed_query(embeddings_model, query: str) -> List[float]:
    """
    Wrapper for embeddings.embed_query() with caching.

    PERFORMANCE: 3-5x speedup for repeated queries.

    Args:
        embeddings_model: Embeddings model instance
        query: Query text to embed

    Returns:
        Embedding vector

    Example:
        >>> from backend.embeddings.embedding_cache import cached_embed_query
        >>> vector = cached_embed_query(embeddings, "my search query")
    """
    cache = get_embedding_cache()
    start_time = time.time()

    # Try cache first
    cached_vector = cache.get(query)
    if cached_vector is not None:
        # Cache hit - record metrics
        duration_ms = (time.time() - start_time) * 1000

        try:
            from backend.monitoring.performance_metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            metrics_collector.record_embedding_call(duration_ms=duration_ms, cache_hit=True)
        except Exception as e:
            logger.debug(f"Failed to record embedding metrics (non-critical): {e}")

        return cached_vector

    # Cache miss: compute embedding
    vector = embeddings_model.embed_query(query)

    # Record embedding call time
    duration_ms = (time.time() - start_time) * 1000

    try:
        from backend.monitoring.performance_metrics import get_metrics_collector
        metrics_collector = get_metrics_collector()
        metrics_collector.record_embedding_call(duration_ms=duration_ms, cache_hit=False)
    except Exception as e:
        logger.debug(f"Failed to record embedding metrics (non-critical): {e}")

    # Store in cache
    cache.set(query, vector)

    return vector
