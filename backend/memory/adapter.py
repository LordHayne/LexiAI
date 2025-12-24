"""
Memory Adapter - Unified interface for memory operations.

This module provides the main interface for storing and retrieving memories,
with proper data model handling, validation, and caching.
"""
import logging
import os
import uuid
import datetime
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import asdict

from backend.config.middleware_config import MiddlewareConfig
from backend.api.middleware.error_handler import MemoryError
from backend.embeddings.embedding_model import OllamaEmbeddingModel
from backend.models.memory_entry import MemoryEntry  # Use dataclass internally
from backend.config.feature_flags import FeatureFlags
from backend.memory.cache import get_memory_cache, generate_query_hash
from backend.memory.memory_bootstrap import get_predictor
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.qdrant.client_wrapper import QdrantClient
from backend.memory.category_predictor import ClusteredCategoryPredictor
from backend.core.component_cache import get_cached_components
from backend.memory.memory_intelligence import track_memory_retrieval, track_memory_usage
from backend.monitoring.performance_metrics import get_metrics_collector

logger = logging.getLogger("lexi_middleware.memory_adapter")

# Configuration constants
class MemoryConfig:
    """Configuration constants for memory operations."""
    STREAMING_MEMORY_LIMIT = 1
    RETRIEVAL_OVERSELECTION_FACTOR = 2
    CORRECTION_BOOST_FACTOR = 1.5
    MAX_STATS_ENTRIES = 10000  # Use scroll, not similarity search
    MAX_STATS_DAYS = 180
    DEFAULT_RETRIEVAL_LIMIT = 10
    MAX_RETRIEVAL_LIMIT = 100
    MAX_CONTENT_LENGTH = 50000  # 50KB
    MAX_USER_ID_LENGTH = 255
    MAX_TAGS_COUNT = 50
    MAX_METADATA_SIZE = 10000  # 10KB JSON


# Lazy initialization for category predictor
_category_predictor = None
_predictor_lock = None


def get_category_predictor():
    """Get or initialize category predictor (thread-safe lazy initialization)."""
    global _category_predictor, _predictor_lock

    if _predictor_lock is None:
        import threading
        _predictor_lock = threading.RLock()

    with _predictor_lock:
        if _category_predictor is None:
            _category_predictor = get_predictor()
        return _category_predictor


def validate_user_id(user_id: str) -> str:
    """
    Validate and sanitize user_id.

    Args:
        user_id: User identifier to validate

    Returns:
        Validated user_id

    Raises:
        ValueError: If user_id is invalid
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Invalid user_id: must be a non-empty string")

    if len(user_id) > MemoryConfig.MAX_USER_ID_LENGTH:
        raise ValueError(f"user_id too long: {len(user_id)} characters (max {MemoryConfig.MAX_USER_ID_LENGTH})")

    # Sanitize: only allow alphanumeric, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise ValueError("user_id contains invalid characters (allowed: a-z, A-Z, 0-9, _, -)")

    return user_id


def validate_content(content: str) -> str:
    """
    Validate and sanitize memory content.

    Args:
        content: Content to validate

    Returns:
        Validated content

    Raises:
        ValueError: If content is invalid
    """
    if not content or not content.strip():
        raise ValueError("Content cannot be empty")

    if len(content) > MemoryConfig.MAX_CONTENT_LENGTH:
        raise ValueError(f"Content too large: {len(content)} characters (max {MemoryConfig.MAX_CONTENT_LENGTH})")

    return content


def validate_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    """
    Validate tags list.

    Args:
        tags: Tags to validate

    Returns:
        Validated tags

    Raises:
        ValueError: If tags are invalid
    """
    if tags is None:
        return None

    if not isinstance(tags, list):
        raise ValueError("Tags must be a list")

    if len(tags) > MemoryConfig.MAX_TAGS_COUNT:
        raise ValueError(f"Too many tags: {len(tags)} (max {MemoryConfig.MAX_TAGS_COUNT})")

    return tags


def validate_metadata(metadata: Optional[dict]) -> Optional[dict]:
    """
    Validate metadata dict.

    Args:
        metadata: Metadata to validate

    Returns:
        Validated metadata

    Raises:
        ValueError: If metadata is invalid
    """
    if metadata is None:
        return None

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    metadata_size = len(json.dumps(metadata))
    if metadata_size > MemoryConfig.MAX_METADATA_SIZE:
        raise ValueError(f"Metadata too large: {metadata_size} bytes (max {MemoryConfig.MAX_METADATA_SIZE})")

    return metadata


def validate_limit(limit: int) -> int:
    """
    Validate and clamp limit parameter.

    Args:
        limit: Limit to validate

    Returns:
        Validated limit (clamped to max)
    """
    if limit <= 0:
        return MemoryConfig.DEFAULT_RETRIEVAL_LIMIT

    if limit > MemoryConfig.MAX_RETRIEVAL_LIMIT:
        logger.warning(f"Limit {limit} exceeds maximum, clamping to {MemoryConfig.MAX_RETRIEVAL_LIMIT}")
        return MemoryConfig.MAX_RETRIEVAL_LIMIT

    return limit


def get_basic_memory_stats() -> int:
    """
    Get basic memory count.

    Returns:
        Total number of memories
    """
    try:
        bundle = get_cached_components()
        client = bundle.vectorstore.client
        collection_info = client.get_collection(MiddlewareConfig.get_memory_collection())
        return collection_info.points_count
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        return 0


def get_detailed_memory_stats(user_id: str) -> Dict[str, Any]:
    """
    Get detailed memory statistics for a user using scroll (not similarity search).

    Args:
        user_id: User identifier

    Returns:
        Dictionary with detailed statistics

    Raises:
        MemoryError: If stats retrieval fails
    """
    try:
        user_id = validate_user_id(user_id)
        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        # Use scroll instead of similarity search for efficiency
        if hasattr(vectorstore, 'client'):
            try:
                cutoff_days = int(os.environ.get("LEXI_STATS_DAYS", str(MemoryConfig.MAX_STATS_DAYS)))
            except (TypeError, ValueError):
                cutoff_days = MemoryConfig.MAX_STATS_DAYS
            must_filters = [{"key": "user_id", "match": {"value": user_id}}]
            if cutoff_days > 0:
                cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=cutoff_days)
                cutoff_ms = int(cutoff_dt.timestamp() * 1000)
                must_filters.append({"key": "timestamp_ms", "range": {"gte": cutoff_ms}})

            try:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter={"must": must_filters},
                    with_payload=True,
                    limit=MemoryConfig.MAX_STATS_ENTRIES
                )
            except Exception:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                    with_payload=True,
                    limit=MemoryConfig.MAX_STATS_ENTRIES
                )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            if not points and cutoff_days > 0:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                    with_payload=True,
                    limit=MemoryConfig.MAX_STATS_ENTRIES
                )
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
        else:
            # Fallback to similarity search if scroll not available
            search_filter = {"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}
            docs = vectorstore.similarity_search("statistics query", k=1000, filter=search_filter)
            points = [{"payload": doc.metadata} for doc in docs]

        categories = {}
        tags = {}
        last_updated = datetime.datetime.now().isoformat()
        total_chars = 0

        for point in points:
            payload = point.payload if hasattr(point, 'payload') else point.get('payload', {})

            cat = payload.get("category", "unkategorisiert")
            categories[cat] = categories.get(cat, 0) + 1

            tag = payload.get("source", "untagged")
            tags[tag] = tags.get(tag, 0) + 1

            ts = payload.get("timestamp")
            if ts and ts > last_updated:
                last_updated = ts

            content = payload.get("content", "")
            total_chars += len(content)

        most_common_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]
        storage_usage = f"{total_chars / 1024:.1f}KB"

        return {
            "total_entries": len(points),
            "categories": categories,
            "last_updated": last_updated,
            "storage_usage": storage_usage,
            "most_common_tags": [tag for tag, _ in most_common_tags]
        }
    except ValueError as e:
        raise MemoryError(f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting detailed memory stats: {str(e)}")
        raise MemoryError(f"Failed to retrieve memory statistics: {str(e)}")


async def store_memory_async(content: str, user_id: str, tags: Optional[List[str]] = None, metadata: Optional[dict] = None) -> Tuple[str, str]:
    """
    Store memory asynchronously with proper validation and cache invalidation.

    IMPORTANT: This is a truly async function that doesn't block the event loop.
    Cache is invalidated BEFORE storing to prevent race conditions.

    Args:
        content: Memory content
        user_id: User identifier
        tags: Optional list of tags
        metadata: Optional metadata dictionary

    Returns:
        Tuple of (memory_id, timestamp)

    Raises:
        ValueError: If validation fails
        MemoryError: If storage fails
    """
    try:
        # Validate all inputs (synchronous, fast)
        content = validate_content(content)
        user_id = validate_user_id(user_id)
        tags = validate_tags(tags)
        metadata = validate_metadata(metadata)

        # Generate ID before storing
        doc_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Prepare metadata
        full_metadata = {
            "id": doc_id,
            "created_at": timestamp
        }
        if metadata:
            full_metadata.update(metadata)

        # Store memory using thread pool for blocking I/O
        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        # FIX: Invalidate cache BEFORE storing to prevent race conditions
        # This ensures concurrent requests don't cache stale data
        try:
            cache = get_memory_cache()
            if cache:
                invalidated = cache.invalidate_user(user_id)
                if invalidated > 0:
                    logger.debug(f"Pre-invalidated {invalidated} cache entries for user {user_id}")
        except Exception as cache_error:
            logger.warning(f"Cache invalidation failed (non-critical): {cache_error}")

        # Run blocking vectorstore operation in executor
        await asyncio.get_event_loop().run_in_executor(
            None,
            vectorstore.add_entry,
            content,
            user_id,
            tags,
            full_metadata
        )

        logger.info(f"Memory stored with ID {doc_id} (metadata keys: {list(full_metadata.keys())})")

        # Track memory storage operation
        metrics_collector = get_metrics_collector()
        metrics_collector.record_memory_operation("store")

        return doc_id, timestamp

    except ValueError as e:
        logger.error(f"Validation error storing memory: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise MemoryError(f"Failed to store memory: {str(e)}")


def store_memory(content: str, user_id: str, tags: Optional[List[str]] = None, metadata: Optional[dict] = None) -> Tuple[str, str]:
    """
    Synchronous wrapper for store_memory_async (for backwards compatibility).

    DEPRECATED: Use store_memory_async() directly in async contexts.

    Args:
        content: Memory content
        user_id: User identifier
        tags: Optional list of tags
        metadata: Optional metadata dictionary

    Returns:
        Tuple of (memory_id, timestamp)

    Raises:
        ValueError: If validation fails
        MemoryError: If storage fails
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context - this is a problem
            logger.warning("store_memory() called from async context - this may cause issues. Use store_memory_async() instead.")
            # Create a new event loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, store_memory_async(content, user_id, tags, metadata))
                return future.result()
        else:
            # Safe to run async function
            return asyncio.run(store_memory_async(content, user_id, tags, metadata))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(store_memory_async(content, user_id, tags, metadata))


def retrieve_memories(user_id: str, query: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 10, score_threshold: Optional[float] = None) -> List[MemoryEntry]:
    """
    Retrieve memories with optional caching and score filtering.

    Args:
        user_id: User identifier
        query: Optional search query
        tags: Optional list of tags to filter by
        limit: Maximum number of memories to retrieve
        score_threshold: Minimum similarity score (0.0-1.0) to filter results

    Returns:
        List of MemoryEntry objects
    """
    return retrieve_memories_with_cache(user_id, query, tags, limit, score_threshold=score_threshold)


def retrieve_memories_with_cache(user_id: str, query: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 10, use_cache: bool = True, score_threshold: Optional[float] = None) -> List[MemoryEntry]:
    """
    Retrieve memories with caching support and score filtering.

    Args:
        user_id: User identifier
        query: Optional search query
        tags: Optional list of tags to filter by
        limit: Maximum number of memories to retrieve
        use_cache: Whether to use cache
        score_threshold: Minimum similarity score (0.0-1.0) to filter results

    Returns:
        List of MemoryEntry objects
    """
    if not use_cache or not FeatureFlags.is_enabled("memory_caching"):
        return retrieve_memories_direct(user_id, query, tags, limit, score_threshold=score_threshold)

    # Track cache performance
    metrics_collector = get_metrics_collector()

    cache = get_memory_cache()
    cached = cache.get(user_id, query or "", tags)

    if cached:
        logger.debug(f"Cache hit for user {user_id}")

        # Record memory cache hit
        metrics_collector.record_memory_operation("retrieve", cache_hit=True)

        # Reconstruct MemoryEntry objects from cached dicts
        return [
            MemoryEntry(
                id=entry.get("id", "unknown"),
                content=entry["content"],
                timestamp=datetime.datetime.fromisoformat(entry["timestamp"]) if isinstance(entry["timestamp"], str) else entry["timestamp"],
                category=entry.get("category"),
                tags=entry.get("tags"),
                source=entry.get("source"),
                relevance=entry.get("relevance")
            )
            for entry in cached
        ]

    # Cache miss - record it
    metrics_collector.record_memory_operation("retrieve", cache_hit=False)

    memories = retrieve_memories_direct(user_id, query, tags, limit, score_threshold=score_threshold)

    # Convert MemoryEntry objects to dicts for caching
    memory_dicts = [
        {
            "id": m.id,
            "content": m.content,
            "timestamp": m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp),
            "category": m.category,
            "tags": m.tags,
            "source": m.source,
            "relevance": m.relevance
        }
        for m in memories
    ]
    cache.store(user_id, query or "", memory_dicts, tags)

    return memories


def retrieve_memories_direct(user_id: str, query: Optional[str], tags: Optional[List[str]], limit: int, score_threshold: Optional[float] = None) -> List[MemoryEntry]:
    """
    Retrieve memories directly from vectorstore (bypassing cache).

    Args:
        user_id: User identifier
        query: Optional search query
        tags: Optional list of tags to filter by
        limit: Maximum number of memories to retrieve
        score_threshold: Minimum similarity score (0.0-1.0) to filter results

    Returns:
        List of MemoryEntry objects

    Raises:
        MemoryError: If retrieval fails
    """
    # Track query performance metrics
    metrics_collector = get_metrics_collector()
    query_type = "semantic" if query else "recent"

    with metrics_collector.track_query(query_type=query_type, user_id=user_id) as tracker:
        try:
            # Validate inputs
            user_id = validate_user_id(user_id)
            limit = validate_limit(limit)

            bundle = get_cached_components()
            vectorstore = bundle.vectorstore
            search_filter = {"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}

            # Over-select to allow for tag filtering
            if query:
                raw_docs = vectorstore.similarity_search(
                    query,
                    k=limit * MemoryConfig.RETRIEVAL_OVERSELECTION_FACTOR,
                    filter=search_filter,
                    score_threshold=score_threshold  # Filter by relevance
                )
            else:
                # For non-query retrievals, get recent entries if vectorstore supports it
                if hasattr(vectorstore, 'get_recent_entries'):
                    raw_docs = vectorstore.get_recent_entries(user_id=user_id, limit=limit * MemoryConfig.RETRIEVAL_OVERSELECTION_FACTOR)
                else:
                    # Fallback: use similarity search with generic query
                    raw_docs = vectorstore.similarity_search("recent memories", k=limit * MemoryConfig.RETRIEVAL_OVERSELECTION_FACTOR, filter=search_filter)

            results = []

            for item in raw_docs:
                # Handle both tuple (doc, score) and plain doc formats
                if isinstance(item, tuple):
                    d = item[0]
                    score = item[1] if len(item) > 1 and isinstance(item[1], (int, float)) else d.metadata.get("score", 1.0)
                else:
                    d = item
                    score = d.metadata.get("score", 1.0)

                # Parse tags (handle JSON string or list)
                entry_tags = d.metadata.get("tags", [])
                if isinstance(entry_tags, str):
                    try:
                        entry_tags = json.loads(entry_tags)
                    except Exception:
                        entry_tags = [entry_tags] if entry_tags else []

                # Filter by tags if specified
                if tags and not any(t in entry_tags for t in tags):
                    continue

                memory_id = d.metadata.get("id", "unknown")

                # BOOST CORRECTION MEMORIES
                is_correction = (
                    "correction" in entry_tags or
                    d.metadata.get("category") == "self_correction" or
                    d.metadata.get("source") == "self_correction"
                )

                if is_correction:
                    score = score * MemoryConfig.CORRECTION_BOOST_FACTOR
                    logger.debug(f"Boosted correction memory {memory_id}: score {score}")

                # Parse timestamp
                ts_raw = d.metadata.get("timestamp", "")
                if isinstance(ts_raw, str):
                    try:
                        ts = datetime.datetime.fromisoformat(ts_raw)
                    except Exception:
                        ts = datetime.datetime.now(datetime.timezone.utc)
                elif isinstance(ts_raw, datetime.datetime):
                    ts = ts_raw
                else:
                    ts = datetime.datetime.now(datetime.timezone.utc)

                # Create MemoryEntry (dataclass)
                results.append(MemoryEntry(
                    id=memory_id,
                    content=d.page_content,
                    timestamp=ts,
                    category=d.metadata.get("category"),
                    tags=entry_tags,
                    source=d.metadata.get("source"),
                    relevance=score
                ))

            # Sort: corrections first, then by relevance
            results.sort(key=lambda m: (
                0 if (m.source == "self_correction" or "correction" in (m.tags or [])) else 1,
                -m.relevance
            ))

            # Limit to requested amount
            results = results[:limit]

            # Track memory retrievals
            if results:
                track_memory_retrieval([m.id for m in results])
                logger.debug(f"Retrieved {len(results)} memories for user {user_id}")

            # Record metrics
            tracker["results_count"] = len(results)
            tracker["cache_hit"] = False  # Direct query, not from cache

            return results

        except ValueError as e:
            logger.error(f"Validation error retrieving memories: {str(e)}")
            raise MemoryError(f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories: {str(e)}")


async def process_chat_message_streaming(message: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Tuple[str, bool, str, List[MemoryEntry]], None]:
    """
    Process chat message with streaming response.

    Args:
        message: User message
        user_id: User identifier
        context: Optional context dictionary

    Yields:
        Tuples of (chunk, memory_used, source, memories)
    """
    try:
        logger.info(f"Processing chat for user_id: {user_id} | context: {context}")
        bundle = get_cached_components()
        embeddings = bundle.embeddings
        vectorstore = bundle.vectorstore
        memory = bundle.memory
        chat_client = bundle.chat_client

        search_filter = {"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}
        relevant_docs = vectorstore.similarity_search(message, k=MemoryConfig.STREAMING_MEMORY_LIMIT, filter=search_filter)

        # Parse timestamp from docs
        used_memories = []
        for doc in relevant_docs:
            ts_raw = doc.metadata.get("timestamp", "")
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.datetime.fromisoformat(ts_raw)
                except Exception:
                    ts = datetime.datetime.now(datetime.timezone.utc)
            elif isinstance(ts_raw, datetime.datetime):
                ts = ts_raw
            else:
                ts = datetime.datetime.now(datetime.timezone.utc)

            used_memories.append(MemoryEntry(
                id=doc.metadata.get("id", "unknown"),
                content=doc.page_content,
                timestamp=ts,
                category=doc.metadata.get("category"),
                tags=doc.metadata.get("tags", []),
                source=doc.metadata.get("source"),
                relevance=0.95
            ))

        # Track memory usage
        memory_ids_used = [m.id for m in used_memories]
        if memory_ids_used:
            track_memory_retrieval(memory_ids_used)

        memory_was_used = bool(used_memories)
        response_source = "memory" if memory_was_used else "llm"

        from backend.core.chat_logic import process_chat_message_streaming as core_streaming
        response_generated = False

        async for chunk in core_streaming(message, chat_client, vectorstore, memory, embeddings):
            response_generated = True
            yield chunk, memory_was_used, response_source, used_memories

        # Track successful memory usage
        if response_generated and memory_ids_used:
            for mem_id in memory_ids_used:
                track_memory_usage(mem_id, was_helpful=True)
            logger.debug(f"Tracked successful usage of {len(memory_ids_used)} memories")

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        yield f"Fehler: {str(e)}", False, "fallback", []


def get_memory_stats(user_id: str = "default") -> Dict[str, int]:
    """
    Get memory statistics by category.

    Args:
        user_id: User identifier

    Returns:
        Dictionary mapping categories to counts
    """
    try:
        memories = retrieve_memories(user_id, limit=100)
        category_counts = {}

        for memory in memories:
            raw_cat = memory.category or memory.source
            try:
                cat = str(raw_cat).strip().lower() if raw_cat else "unknown"
                if cat in ("", "none", "null"):
                    cat = "unknown"
            except Exception:
                cat = "unknown"

            category_counts[cat] = category_counts.get(cat, 0) + 1

        return category_counts
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        return {}


def get_memory_stats_from_qdrant(user_id: str = "default") -> Dict[str, int]:
    """
    Get memory statistics directly from Qdrant using scroll.

    Args:
        user_id: User identifier

    Returns:
        Dictionary mapping categories to counts
    """
    try:
        user_id = validate_user_id(user_id)
        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        if hasattr(vectorstore, 'client'):
            try:
                cutoff_days = int(os.environ.get("LEXI_STATS_DAYS", str(MemoryConfig.MAX_STATS_DAYS)))
            except (TypeError, ValueError):
                cutoff_days = MemoryConfig.MAX_STATS_DAYS
            must_filters = [{"key": "user_id", "match": {"value": user_id}}]
            if cutoff_days > 0:
                cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=cutoff_days)
                cutoff_ms = int(cutoff_dt.timestamp() * 1000)
                must_filters.append({"key": "timestamp_ms", "range": {"gte": cutoff_ms}})

            search_filter = {"must": must_filters}
            try:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter=search_filter,
                    with_payload=True,
                    limit=1000
                )
            except Exception:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                    with_payload=True,
                    limit=1000
                )
            points = scroll_result.points if hasattr(scroll_result, "points") else scroll_result.get("points", [])
            if not points and cutoff_days > 0:
                scroll_result = vectorstore.client.scroll(
                    collection_name=MiddlewareConfig.get_memory_collection(),
                    scroll_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                    with_payload=True,
                    limit=1000
                )
                points = scroll_result.points if hasattr(scroll_result, "points") else scroll_result.get("points", [])

            category_counts = {}
            for point in points:
                raw_cat = point.payload.get("category")
                try:
                    cat = str(raw_cat).strip().lower() if raw_cat else "unknown"
                    if cat in ("", "none", "null"):
                        cat = "unknown"
                except Exception:
                    cat = "unknown"

                category_counts[cat] = category_counts.get(cat, 0) + 1

            return category_counts
        else:
            return get_memory_stats(user_id)

    except Exception as e:
        logger.error(f"Error getting memory stats from Qdrant: {str(e)}")
        return {}


def categorize_memory(content: str, predictor: Optional[ClusteredCategoryPredictor] = None) -> str:
    """
    Categorize memory content (synchronous).

    Args:
        content: Memory content to categorize
        predictor: Optional predictor instance

    Returns:
        Category string
    """
    try:
        pred = predictor or get_category_predictor()
        return pred.predict_category(content)
    except Exception as e:
        logger.error(f"Fehler bei der Kategorisierung: {e}")
        return "unkategorisiert"


def delete_memory(memory_id: str, user_id: str) -> bool:
    """
    Delete a single memory entry by ID.

    Args:
        memory_id: Memory entry ID to delete
        user_id: User identifier (for validation)

    Returns:
        True if successfully deleted, False otherwise

    Raises:
        ValueError: If validation fails
        MemoryError: If deletion fails
    """
    try:
        # Validate inputs
        user_id = validate_user_id(user_id)

        if not memory_id or not isinstance(memory_id, str):
            raise ValueError("Invalid memory_id: must be a non-empty string")

        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        # Import UUID for conversion
        from uuid import UUID

        try:
            # Convert string to UUID for Qdrant interface
            uuid_id = UUID(memory_id)
        except ValueError:
            raise ValueError(f"Invalid memory_id format: must be a valid UUID")

        # Delete from Qdrant
        vectorstore.delete_entry(uuid_id)

        # Invalidate cache for this user
        try:
            cache = get_memory_cache()
            if cache:
                invalidated = cache.invalidate_user(user_id)
                if invalidated > 0:
                    logger.debug(f"Invalidated {invalidated} cache entries for user {user_id}")
        except Exception as cache_error:
            logger.warning(f"Cache invalidation failed (non-critical): {cache_error}")

        logger.info(f"Memory deleted: {memory_id} for user {user_id}")
        return True

    except ValueError as e:
        logger.error(f"Validation error deleting memory: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {str(e)}")
        raise MemoryError(f"Failed to delete memory: {str(e)}")


def delete_memories_by_content(query: str, user_id: str, similarity_threshold: float = 0.75) -> List[str]:
    """
    Delete memories matching a semantic query.

    This function searches for memories semantically similar to the query
    and deletes those above the similarity threshold.

    Args:
        query: Search query to find memories to delete
        user_id: User identifier
        similarity_threshold: Minimum similarity score (0.0-1.0) to delete
                            Default: 0.75 (fairly strict to avoid accidental deletion)

    Returns:
        List of deleted memory IDs

    Raises:
        ValueError: If validation fails
        MemoryError: If deletion fails

    Example:
        >>> deleted_ids = delete_memories_by_content("Python programming", "user123")
        >>> print(f"Deleted {len(deleted_ids)} memories about Python")
    """
    try:
        # Validate inputs
        query = validate_content(query)
        user_id = validate_user_id(user_id)

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        # Retrieve memories matching the query with high similarity
        memories = retrieve_memories_direct(
            user_id=user_id,
            query=query,
            tags=None,
            limit=50,  # Limit to prevent mass deletion
            score_threshold=similarity_threshold
        )

        deleted_ids = []
        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        # Import UUID for conversion
        from uuid import UUID

        # Delete matching memories
        for memory in memories:
            try:
                # Convert string ID to UUID
                uuid_id = UUID(memory.id)
                vectorstore.delete_entry(uuid_id)
                deleted_ids.append(memory.id)
                logger.info(f"Deleted memory {memory.id} (relevance: {memory.relevance:.2f})")
            except Exception as e:
                logger.warning(f"Failed to delete memory {memory.id}: {str(e)}")
                # Continue with next memory instead of failing completely

        # Invalidate cache for this user
        try:
            cache = get_memory_cache()
            if cache:
                invalidated = cache.invalidate_user(user_id)
                if invalidated > 0:
                    logger.debug(f"Invalidated {invalidated} cache entries for user {user_id}")
        except Exception as cache_error:
            logger.warning(f"Cache invalidation failed (non-critical): {cache_error}")

        logger.info(f"Deleted {len(deleted_ids)} memories matching query '{query}' for user {user_id}")
        return deleted_ids

    except ValueError as e:
        logger.error(f"Validation error deleting memories by content: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error deleting memories by content: {str(e)}")
        raise MemoryError(f"Failed to delete memories: {str(e)}")
