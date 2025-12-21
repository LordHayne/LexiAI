"""
Performance Metrics - System-wide performance tracking and analytics.

Tracks:
- Query performance (latency, throughput)
- Cache effectiveness (hit rates)
- Embedding generation (latency, cache)
- Qdrant operations (latency, errors)
- Slow queries detection
- Error tracking

Usage:
    >>> from backend.monitoring.performance_metrics import get_metrics_collector
    >>>
    >>> metrics = get_metrics_collector()
    >>>
    >>> # Track query
    >>> with metrics.track_query("user_query"):
    >>>     results = perform_search(...)
    >>>
    >>> # Get stats
    >>> stats = metrics.get_stats()
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from threading import RLock
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger("lexi_middleware.metrics")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query_type: str  # "semantic", "hybrid", "keyword"
    duration_ms: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    results_count: int = 0
    cache_hit: bool = False
    user_id: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over time window."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    queries_per_second: float = 0.0

    # By type
    semantic_queries: int = 0
    hybrid_queries: int = 0
    keyword_queries: int = 0

    # Slow queries
    slow_queries_count: int = 0
    slowest_query_ms: float = 0.0


@dataclass
class ComponentMetrics:
    """Metrics for individual components."""
    # Embeddings
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    embedding_avg_latency_ms: float = 0.0

    # Qdrant
    qdrant_queries: int = 0
    qdrant_errors: int = 0
    qdrant_avg_latency_ms: float = 0.0

    # Memory operations
    memory_stores: int = 0
    memory_retrievals: int = 0
    memory_cache_hits: int = 0
    memory_cache_misses: int = 0


class PerformanceMetricsCollector:
    """
    Collects and analyzes performance metrics across the system.

    Thread-safe, efficient, with configurable retention.
    """

    def __init__(
        self,
        max_history: int = 10000,
        slow_query_threshold_ms: float = 1000.0,
        retention_hours: int = 24
    ):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of queries to keep in history
            slow_query_threshold_ms: Threshold for slow query detection
            retention_hours: Hours to retain metrics
        """
        self.max_history = max_history
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.retention_hours = retention_hours

        # Query history (thread-safe deque)
        self._query_history: deque = deque(maxlen=max_history)
        self._slow_queries: deque = deque(maxlen=100)  # Keep last 100 slow queries

        # Component metrics
        self._component_metrics = ComponentMetrics()

        # Error tracking
        self._errors: deque = deque(maxlen=1000)

        # Thread safety
        self._lock = RLock()

        # Start time
        self._start_time = datetime.now()

        logger.info(f"Metrics collector initialized (history={max_history}, slow_threshold={slow_query_threshold_ms}ms)")

    @contextmanager
    def track_query(
        self,
        query_type: str = "semantic",
        user_id: Optional[str] = None,
        query_id: Optional[str] = None
    ):
        """
        Context manager to track query performance.

        Usage:
            >>> with metrics.track_query("hybrid", user_id="thomas") as query_id:
            >>>     results = perform_search(...)
            >>>     yield results  # Optional: pass results count
        """
        start_time = time.time()
        query_id = query_id or f"{query_type}_{int(time.time() * 1000)}"
        success = True
        error = None
        results_count = 0
        cache_hit = False

        try:
            # Yield control to calling code
            tracker = {"results_count": 0, "cache_hit": False}
            yield tracker

            results_count = tracker.get("results_count", 0)
            cache_hit = tracker.get("cache_hit", False)

        except Exception as e:
            success = False
            error = str(e)
            logger.error(f"Query {query_id} failed: {e}")
            raise

        finally:
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000

            metric = QueryMetrics(
                query_id=query_id,
                query_type=query_type,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                success=success,
                error=error,
                results_count=results_count,
                cache_hit=cache_hit,
                user_id=user_id
            )

            self.record_query(metric)

    def record_query(self, metric: QueryMetrics):
        """Record a query metric."""
        with self._lock:
            self._query_history.append(metric)

            # Track slow queries
            if metric.duration_ms > self.slow_query_threshold_ms:
                self._slow_queries.append(metric)
                logger.warning(f"Slow query detected: {metric.query_id} ({metric.duration_ms:.0f}ms)")

            # Track errors
            if not metric.success:
                self._errors.append({
                    "query_id": metric.query_id,
                    "error": metric.error,
                    "timestamp": metric.timestamp
                })

    def record_embedding_call(self, duration_ms: float, cache_hit: bool):
        """Record embedding generation metrics."""
        with self._lock:
            self._component_metrics.embedding_calls += 1

            if cache_hit:
                self._component_metrics.embedding_cache_hits += 1
            else:
                self._component_metrics.embedding_cache_misses += 1

            # Update rolling average
            total_calls = self._component_metrics.embedding_calls
            current_avg = self._component_metrics.embedding_avg_latency_ms
            new_avg = (current_avg * (total_calls - 1) + duration_ms) / total_calls
            self._component_metrics.embedding_avg_latency_ms = new_avg

    def record_qdrant_query(self, duration_ms: float, success: bool = True):
        """Record Qdrant query metrics."""
        with self._lock:
            self._component_metrics.qdrant_queries += 1

            if not success:
                self._component_metrics.qdrant_errors += 1

            # Update rolling average
            total_queries = self._component_metrics.qdrant_queries
            current_avg = self._component_metrics.qdrant_avg_latency_ms
            new_avg = (current_avg * (total_queries - 1) + duration_ms) / total_queries
            self._component_metrics.qdrant_avg_latency_ms = new_avg

    def record_memory_operation(self, operation: str, cache_hit: bool = False):
        """
        Record memory operation.

        Args:
            operation: "store" or "retrieve"
            cache_hit: Whether operation hit cache
        """
        with self._lock:
            if operation == "store":
                self._component_metrics.memory_stores += 1
            elif operation == "retrieve":
                self._component_metrics.memory_retrievals += 1

                if cache_hit:
                    self._component_metrics.memory_cache_hits += 1
                else:
                    self._component_metrics.memory_cache_misses += 1

    def get_stats(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregated statistics.

        Args:
            window_minutes: Optional time window (None = all time)

        Returns:
            Dictionary with aggregated metrics
        """
        with self._lock:
            # Filter by time window
            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                queries = [q for q in self._query_history if q.timestamp >= cutoff]
            else:
                queries = list(self._query_history)

            if not queries:
                return {
                    "message": "No queries recorded",
                    "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
                }

            # Compute aggregated metrics
            aggregated = self._compute_aggregated_metrics(queries)

            return {
                "summary": asdict(aggregated),
                "components": asdict(self._component_metrics),
                "slow_queries": [
                    {
                        "query_id": q.query_id,
                        "duration_ms": q.duration_ms,
                        "timestamp": q.timestamp.isoformat()
                    }
                    for q in list(self._slow_queries)[-10:]  # Last 10 slow queries
                ],
                "recent_errors": [
                    {
                        "query_id": e["query_id"],
                        "error": e["error"],
                        "timestamp": e["timestamp"].isoformat()
                    }
                    for e in list(self._errors)[-10:]  # Last 10 errors
                ],
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "history_size": len(queries)
            }

    def _compute_aggregated_metrics(self, queries: List[QueryMetrics]) -> AggregatedMetrics:
        """Compute aggregated metrics from query list."""
        if not queries:
            return AggregatedMetrics()

        # Basic counts
        total = len(queries)
        successful = sum(1 for q in queries if q.success)
        failed = total - successful

        # Latencies (only successful queries)
        successful_queries = [q for q in queries if q.success]
        latencies = [q.duration_ms for q in successful_queries] if successful_queries else [0]
        latencies.sort()

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Percentiles
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if (f + 1) < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)

        # Cache hit rate
        cache_hits = sum(1 for q in queries if q.cache_hit)
        cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0

        # Query types
        semantic = sum(1 for q in queries if q.query_type == "semantic")
        hybrid = sum(1 for q in queries if q.query_type == "hybrid")
        keyword = sum(1 for q in queries if q.query_type == "keyword")

        # Slow queries
        slow_count = sum(1 for q in queries if q.duration_ms > self.slow_query_threshold_ms)
        slowest = max(latencies) if latencies else 0

        # Queries per second (if time window available)
        if len(queries) >= 2:
            time_span = (queries[-1].timestamp - queries[0].timestamp).total_seconds()
            qps = total / time_span if time_span > 0 else 0
        else:
            qps = 0

        return AggregatedMetrics(
            total_queries=total,
            successful_queries=successful,
            failed_queries=failed,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            cache_hit_rate=cache_hit_rate,
            queries_per_second=qps,
            semantic_queries=semantic,
            hybrid_queries=hybrid,
            keyword_queries=keyword,
            slow_queries_count=slow_count,
            slowest_query_ms=slowest
        )

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._query_history.clear()
            self._slow_queries.clear()
            self._errors.clear()
            self._component_metrics = ComponentMetrics()
            self._start_time = datetime.now()
            logger.info("Metrics reset")


# Global collector instance
_metrics_collector: Optional[PerformanceMetricsCollector] = None
_collector_lock = RLock()


def get_metrics_collector(
    max_history: int = 10000,
    slow_query_threshold_ms: float = 1000.0
) -> PerformanceMetricsCollector:
    """
    Get global metrics collector instance (singleton).

    Args:
        max_history: Maximum query history size
        slow_query_threshold_ms: Slow query threshold

    Returns:
        PerformanceMetricsCollector instance
    """
    global _metrics_collector

    with _collector_lock:
        if _metrics_collector is None:
            _metrics_collector = PerformanceMetricsCollector(
                max_history=max_history,
                slow_query_threshold_ms=slow_query_threshold_ms
            )
            logger.info("Global metrics collector initialized")

        return _metrics_collector
