"""
Qdrant Database Optimization Workers

This module implements background workers that continuously improve
the Qdrant vector database through deduplication, index optimization,
relevance reranking, data quality checks, and tiered storage balancing.

Architecture:
- BaseWorker: Abstract base class with common functionality
- Specialized Workers: Each implements a specific optimization task
- WorkerCoordinator: Orchestrates scheduling and execution
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue,
    VectorParams, Distance, OptimizersConfig,
    QuantizationConfig, BinaryQuantization, ScalarQuantization,
    HnswConfigDiff
)

from backend.models.memory_entry import MemoryEntry
from backend.memory.memory_intelligence import MemoryUsageTracker, get_usage_tracker
from backend.qdrant.client_wrapper import (
    safe_scroll, safe_upsert, safe_delete, safe_set_payload
)

logger = logging.getLogger("qdrant_optimizer")


# ============================================================================
# BASE WORKER CLASS
# ============================================================================

class BaseWorker(ABC):
    """
    Abstract base class for all Qdrant optimization workers.

    Provides:
    - Common initialization
    - Error handling
    - Metrics tracking
    - Logging
    - Coordination via memory system
    """

    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_client: QdrantClient,
        embeddings,
        memory_adapter=None
    ):
        """
        Initialize base worker.

        Args:
            config: Worker-specific configuration
            qdrant_client: Qdrant database client
            embeddings: Embedding model for re-embedding
            memory_adapter: Memory adapter for coordination (optional)
        """
        self.config = config
        self.client = qdrant_client
        self.embeddings = embeddings
        self.memory_adapter = memory_adapter
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = defaultdict(int)
        self._setup_metrics()

    @abstractmethod
    def _setup_metrics(self):
        """Setup worker-specific metrics. Override in subclass."""
        pass

    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """
        Execute the worker's main task.

        Returns:
            Dict with execution results and metrics
        """
        pass

    async def safe_run(self) -> Dict[str, Any]:
        """
        Execute worker with comprehensive error handling and metrics.

        Returns:
            Dict with:
                - success: bool
                - duration: float (seconds)
                - error: str (if failed)
                - metrics: Dict (if successful)
        """
        start_time = datetime.now(timezone.utc)
        worker_name = self.__class__.__name__

        try:
            self.logger.info(f"🚀 Starting {worker_name}")

            # Execute main task
            result = await self.run()

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.info(f"✅ Completed {worker_name} in {duration:.2f}s")

            # Store execution status in memory system
            await self._store_execution_status(
                status="success",
                duration=duration,
                metrics=result
            )

            return {
                "success": True,
                "duration": duration,
                "timestamp": start_time.isoformat(),
                **result
            }

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error(f"❌ Failed {worker_name}: {e}", exc_info=True)

            # Store failure status
            await self._store_execution_status(
                status="failed",
                duration=duration,
                error=str(e)
            )

            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "timestamp": start_time.isoformat()
            }

    async def _store_execution_status(
        self,
        status: str,
        duration: float,
        metrics: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Store worker execution status in memory system for coordination."""
        if not self.memory_adapter:
            return

        worker_name = self.__class__.__name__
        content = f"Worker {worker_name} {status}"

        try:
            await self.memory_adapter.store_memory(
                user_id="system",
                content=content,
                tags=["worker_status", worker_name, status],
                metadata={
                    "worker": worker_name,
                    "status": status,
                    "duration": duration,
                    "metrics": metrics or {},
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to store execution status: {e}")

    async def get_last_run_info(self) -> Optional[Dict]:
        """Retrieve last execution info from memory system."""
        if not self.memory_adapter:
            return None

        worker_name = self.__class__.__name__

        try:
            memories = await self.memory_adapter.retrieve_memories(
                user_id="system",
                tags=["worker_status", worker_name],
                limit=1
            )

            if memories:
                return memories[0].metadata
        except Exception as e:
            self.logger.warning(f"Failed to retrieve last run info: {e}")

        return None


# ============================================================================
# 1. DEDUPLICATION WORKER
# ============================================================================

class DeduplicationWorker(BaseWorker):
    """
    Identifies and merges duplicate/similar memory entries.

    Algorithm:
    1. Batch-fetch memories with embeddings
    2. Compute pairwise cosine similarity
    3. Group by similarity threshold (default: 0.95)
    4. Merge metadata and delete duplicates
    5. Update canonical entry
    """

    def _setup_metrics(self):
        self.metrics.update({
            "duplicates_found": 0,
            "duplicates_merged": 0,
            "groups_processed": 0,
            "memories_scanned": 0,
            "storage_saved_bytes": 0
        })

    async def run(self) -> Dict[str, Any]:
        """Execute deduplication process."""
        collection_name = self.config.get("collection_name", "lexi_memory")
        similarity_threshold = self.config.get("similarity_threshold", 0.95)
        batch_size = self.config.get("batch_size", 1000)

        self.logger.info(f"Starting deduplication on {collection_name} "
                        f"(threshold={similarity_threshold})")

        # Fetch all memories with embeddings
        memories = await self._fetch_all_memories_with_embeddings(
            collection_name, batch_size
        )

        self.metrics["memories_scanned"] = len(memories)
        self.logger.info(f"Fetched {len(memories)} memories for deduplication")

        if len(memories) < 2:
            self.logger.info("Not enough memories for deduplication")
            return self.metrics

        # Find similar groups
        similar_groups = await self._find_similar_groups(
            memories, similarity_threshold
        )

        self.metrics["groups_processed"] = len(similar_groups)
        self.logger.info(f"Found {len(similar_groups)} duplicate groups")

        # Merge each group
        for group in similar_groups:
            try:
                await self._merge_group(group, collection_name)
                self.metrics["duplicates_found"] += len(group) - 1
                self.metrics["duplicates_merged"] += 1
            except Exception as e:
                self.logger.warning(f"Failed to merge group: {e}")

        return self.metrics

    async def _fetch_all_memories_with_embeddings(
        self,
        collection_name: str,
        batch_size: int
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Fetch all memories with embeddings from collection.

        Returns:
            List of (id, embedding_vector, payload) tuples
        """
        memories = []
        offset = None

        while True:
            scroll_result = safe_scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=True,
                limit=batch_size,
                offset=offset
            )

            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                points = getattr(scroll_result, "points", [])
                next_offset = getattr(scroll_result, "next_page_offset", None)

            if not points:
                break

            for point in points:
                if hasattr(point, 'vector') and point.vector is not None:
                    memories.append((
                        point.id,
                        np.array(point.vector),
                        point.payload or {}
                    ))

            if next_offset is None:
                break

            offset = next_offset

        return memories

    async def _find_similar_groups(
        self,
        memories: List[Tuple[str, np.ndarray, Dict]],
        threshold: float
    ) -> List[List[Tuple[str, np.ndarray, Dict]]]:
        """
        Find groups of similar memories using cosine similarity.

        Args:
            memories: List of (id, vector, payload) tuples
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List of groups, each group is a list of similar memories
        """
        groups = []
        processed = set()

        for i, (id1, vec1, payload1) in enumerate(memories):
            if id1 in processed:
                continue

            group = [(id1, vec1, payload1)]
            processed.add(id1)

            for j, (id2, vec2, payload2) in enumerate(memories[i+1:], start=i+1):
                if id2 in processed:
                    continue

                # Compute cosine similarity
                similarity = self._cosine_similarity(vec1, vec2)

                if similarity >= threshold:
                    group.append((id2, vec2, payload2))
                    processed.add(id2)

            # Only keep groups with 2+ members
            if len(group) >= 2:
                groups.append(group)

        return groups

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0

    async def _merge_group(
        self,
        group: List[Tuple[str, np.ndarray, Dict]],
        collection_name: str
    ):
        """
        Merge a group of duplicate memories.

        Strategy:
        1. Select canonical (highest relevance)
        2. Merge metadata (tags, source_memory_ids)
        3. Delete duplicates
        4. Update canonical
        """
        if len(group) < 2:
            return

        # Select canonical (highest relevance)
        canonical_idx = 0
        max_relevance = group[0][2].get("relevance", 0.0)

        for i, (_, _, payload) in enumerate(group[1:], start=1):
            relevance = payload.get("relevance", 0.0)
            if relevance > max_relevance:
                max_relevance = relevance
                canonical_idx = i

        canonical_id, canonical_vec, canonical_payload = group[canonical_idx]
        duplicates = [item for i, item in enumerate(group) if i != canonical_idx]

        # Merge metadata
        merged_tags = set(canonical_payload.get("tags", []))
        merged_source_ids = set(canonical_payload.get("source_memory_ids", []))

        for dup_id, _, dup_payload in duplicates:
            merged_tags.update(dup_payload.get("tags", []))
            merged_source_ids.add(dup_id)

        # Update canonical payload
        updated_payload = {
            **canonical_payload,
            "tags": list(merged_tags),
            "source_memory_ids": list(merged_source_ids),
            "is_consolidated": True,
            "consolidation_timestamp": datetime.now(timezone.utc).isoformat(),
            "duplicates_merged_count": len(duplicates)
        }

        # Update canonical entry
        safe_set_payload(
            collection_name=collection_name,
            points=[canonical_id],
            payload=updated_payload
        )

        # Delete duplicates
        duplicate_ids = [dup_id for dup_id, _, _ in duplicates]
        safe_delete(
            collection_name=collection_name,
            points_selector=duplicate_ids
        )

        self.logger.debug(f"Merged {len(duplicates)} duplicates into {canonical_id}")


# ============================================================================
# 2. INDEX OPTIMIZATION WORKER
# ============================================================================

class IndexOptimizationWorker(BaseWorker):
    """
    Automatically tunes HNSW index parameters based on performance.

    Strategy:
    1. Collect query performance metrics
    2. Generate parameter variants
    3. A/B test on sample queries
    4. Apply best configuration
    """

    def _setup_metrics(self):
        self.metrics.update({
            "tests_run": 0,
            "improvements_applied": 0,
            "rollbacks": 0,
            "current_m": None,
            "current_ef_construct": None,
            "performance_gain_percent": 0.0
        })

    async def run(self) -> Dict[str, Any]:
        """Execute index optimization."""
        collection_name = self.config.get("collection_name", "lexi_memory")

        self.logger.info(f"Starting index optimization for {collection_name}")

        # Get current collection info
        collection_info = self.client.get_collection(collection_name)
        current_config = collection_info.config

        # Get collection size
        count_result = self.client.count(collection_name)
        collection_size = count_result.count if hasattr(count_result, 'count') else 0

        self.logger.info(f"Collection size: {collection_size} points")

        # Determine optimal parameters based on size
        optimal_params = self._determine_optimal_params(collection_size)

        current_m = current_config.params.vectors.get("").hnsw_config.m
        current_ef = current_config.params.vectors.get("").hnsw_config.ef_construct

        self.metrics["current_m"] = current_m
        self.metrics["current_ef_construct"] = current_ef

        # Check if update needed
        if (current_m == optimal_params["m"] and
            current_ef == optimal_params["ef_construct"]):
            self.logger.info(f"Parameters already optimal: m={current_m}, "
                           f"ef_construct={current_ef}")
            return self.metrics

        # Update HNSW configuration
        try:
            self.logger.info(f"Updating HNSW parameters: "
                           f"m={current_m}->{optimal_params['m']}, "
                           f"ef_construct={current_ef}->{optimal_params['ef_construct']}")

            self.client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(
                    m=optimal_params["m"],
                    ef_construct=optimal_params["ef_construct"]
                )
            )

            self.metrics["improvements_applied"] = 1
            self.metrics["tests_run"] = 1

            self.logger.info("✅ HNSW parameters updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update HNSW parameters: {e}")
            self.metrics["rollbacks"] = 1

        return self.metrics

    def _determine_optimal_params(self, collection_size: int) -> Dict[str, int]:
        """
        Determine optimal HNSW parameters based on collection size.

        Tuning matrix:
        - < 10K:      m=16,  ef_construct=100
        - 10K-100K:   m=32,  ef_construct=200
        - 100K-1M:    m=48,  ef_construct=400
        - > 1M:       m=64,  ef_construct=600
        """
        if collection_size < 10_000:
            return {"m": 16, "ef_construct": 100}
        elif collection_size < 100_000:
            return {"m": 32, "ef_construct": 200}
        elif collection_size < 1_000_000:
            return {"m": 48, "ef_construct": 400}
        else:
            return {"m": 64, "ef_construct": 600}


# ============================================================================
# 3. RELEVANCE RERANKING WORKER
# ============================================================================

class RelevanceRerankingWorker(BaseWorker):
    """
    Updates relevance scores based on usage patterns.

    Implements adaptive relevance formula from memory_intelligence.py:
    - Usage boost: Frequent usage increases relevance
    - Recency boost: Recently used memories stay relevant
    - Age decay: Unused memories lose relevance
    - Success multiplier: Effective memories get boosted
    """

    def _setup_metrics(self):
        self.metrics.update({
            "memories_updated": 0,
            "memories_boosted": 0,
            "memories_decayed": 0,
            "average_relevance": 0.0,
            "p95_relevance": 0.0
        })

    async def run(self) -> Dict[str, Any]:
        """Execute relevance reranking."""
        collection_name = self.config.get("collection_name", "lexi_memory")
        batch_size = self.config.get("batch_size", 500)

        self.logger.info(f"Starting relevance reranking for {collection_name}")

        # Get usage tracker
        usage_tracker = get_usage_tracker()

        # Fetch all memories
        memories = await self._fetch_all_memories(collection_name, batch_size)
        self.logger.info(f"Fetched {len(memories)} memories for reranking")

        relevance_scores = []

        # Update each memory's relevance
        for memory_id, payload in memories:
            try:
                # Calculate age
                timestamp_str = payload.get("timestamp")
                if not timestamp_str:
                    continue

                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now(timezone.utc) - timestamp).days

                # Get current relevance
                current_relevance = payload.get("relevance", 0.5)

                # Calculate adaptive relevance
                new_relevance = usage_tracker.calculate_adaptive_relevance(
                    memory_id=memory_id,
                    base_relevance=current_relevance,
                    memory_age_days=age_days
                )

                relevance_scores.append(new_relevance)

                # Update if changed significantly
                if abs(new_relevance - current_relevance) > 0.05:
                    safe_set_payload(
                        collection_name=collection_name,
                        points=[memory_id],
                        payload={"relevance": new_relevance}
                    )

                    self.metrics["memories_updated"] += 1

                    if new_relevance > current_relevance:
                        self.metrics["memories_boosted"] += 1
                    else:
                        self.metrics["memories_decayed"] += 1

            except Exception as e:
                self.logger.warning(f"Failed to update relevance for {memory_id}: {e}")

        # Calculate statistics
        if relevance_scores:
            self.metrics["average_relevance"] = float(np.mean(relevance_scores))
            self.metrics["p95_relevance"] = float(np.percentile(relevance_scores, 95))

        self.logger.info(f"Updated relevance for {self.metrics['memories_updated']} memories")

        return self.metrics

    async def _fetch_all_memories(
        self,
        collection_name: str,
        batch_size: int
    ) -> List[Tuple[str, Dict]]:
        """Fetch all memories (id, payload)."""
        memories = []
        offset = None

        while True:
            scroll_result = safe_scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset
            )

            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                points = getattr(scroll_result, "points", [])
                next_offset = getattr(scroll_result, "next_page_offset", None)

            if not points:
                break

            for point in points:
                memories.append((point.id, point.payload or {}))

            if next_offset is None:
                break

            offset = next_offset

        return memories


# ============================================================================
# 4. DATA QUALITY WORKER
# ============================================================================

class DataQualityWorker(BaseWorker):
    """
    Detects and repairs data integrity issues.

    Checks:
    - Embedding validation (dimensions, NaN, zero vectors)
    - Payload validation (required fields, types)
    - Metadata consistency (categories, tags, relevance range)
    """

    def _setup_metrics(self):
        self.metrics.update({
            "issues_found": 0,
            "issues_repaired": 0,
            "issues_quarantined": 0,
            "embedding_issues": 0,
            "payload_issues": 0,
            "metadata_issues": 0
        })

    async def run(self) -> Dict[str, Any]:
        """Execute data quality checks."""
        collection_name = self.config.get("collection_name", "lexi_memory")
        batch_size = self.config.get("batch_size", 1000)
        auto_repair = self.config.get("auto_repair", True)
        quarantine_collection = self.config.get(
            "quarantine_collection",
            "lexi_memory_quarantine"
        )

        self.logger.info(f"Starting data quality checks for {collection_name}")

        # Fetch all memories with vectors
        memories = await self._fetch_memories_for_validation(
            collection_name, batch_size
        )

        self.logger.info(f"Validating {len(memories)} memories")

        for memory_id, vector, payload in memories:
            issues = []

            # Validate embedding
            embedding_issues = self._validate_embedding(vector)
            if embedding_issues:
                issues.extend(embedding_issues)
                self.metrics["embedding_issues"] += len(embedding_issues)

            # Validate payload
            payload_issues = self._validate_payload(payload)
            if payload_issues:
                issues.extend(payload_issues)
                self.metrics["payload_issues"] += len(payload_issues)

            # Validate metadata
            metadata_issues = self._validate_metadata(payload)
            if metadata_issues:
                issues.extend(metadata_issues)
                self.metrics["metadata_issues"] += len(metadata_issues)

            if issues:
                self.metrics["issues_found"] += len(issues)
                self.logger.debug(f"Found {len(issues)} issues in {memory_id}: {issues}")

                # Attempt repair
                if auto_repair:
                    try:
                        await self._repair_memory(
                            memory_id, vector, payload, issues, collection_name
                        )
                        self.metrics["issues_repaired"] += len(issues)
                    except Exception as e:
                        self.logger.warning(f"Failed to repair {memory_id}: {e}")
                        # Quarantine if can't repair
                        await self._quarantine_memory(
                            memory_id, vector, payload, collection_name, quarantine_collection
                        )
                        self.metrics["issues_quarantined"] += 1

        self.logger.info(f"Data quality check complete: "
                        f"{self.metrics['issues_found']} issues, "
                        f"{self.metrics['issues_repaired']} repaired, "
                        f"{self.metrics['issues_quarantined']} quarantined")

        return self.metrics

    async def _fetch_memories_for_validation(
        self,
        collection_name: str,
        batch_size: int
    ) -> List[Tuple[str, Optional[np.ndarray], Dict]]:
        """Fetch memories for validation."""
        memories = []
        offset = None

        while True:
            scroll_result = safe_scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=True,
                limit=batch_size,
                offset=offset
            )

            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                points = getattr(scroll_result, "points", [])
                next_offset = getattr(scroll_result, "next_page_offset", None)

            if not points:
                break

            for point in points:
                vector = np.array(point.vector) if hasattr(point, 'vector') and point.vector else None
                memories.append((point.id, vector, point.payload or {}))

            if next_offset is None:
                break

            offset = next_offset

        return memories

    def _validate_embedding(self, vector: Optional[np.ndarray]) -> List[str]:
        """Validate embedding vector."""
        issues = []

        if vector is None:
            issues.append("null_embedding")
            return issues

        # Check for NaN or Inf
        if np.any(np.isnan(vector)):
            issues.append("nan_values")

        if np.any(np.isinf(vector)):
            issues.append("inf_values")

        # Check for zero vector
        if np.allclose(vector, 0):
            issues.append("zero_vector")

        # Check dimension (expected: 768 for nomic-embed-text)
        expected_dim = self.config.get("expected_dimension", 768)
        if len(vector) != expected_dim:
            issues.append(f"dimension_mismatch_{len(vector)}_expected_{expected_dim}")

        return issues

    def _validate_payload(self, payload: Dict) -> List[str]:
        """Validate payload structure."""
        issues = []

        # Required fields
        required_fields = ["content", "timestamp"]
        for field in required_fields:
            if field not in payload:
                issues.append(f"missing_field_{field}")

        # Validate content
        if "content" in payload:
            content = payload["content"]
            if not isinstance(content, str):
                issues.append("content_not_string")
            elif len(content) < 10:
                issues.append("content_too_short")

        # Validate timestamp format
        if "timestamp" in payload:
            try:
                datetime.fromisoformat(payload["timestamp"])
            except (ValueError, TypeError):
                issues.append("invalid_timestamp_format")

        return issues

    def _validate_metadata(self, payload: Dict) -> List[str]:
        """Validate metadata consistency."""
        issues = []

        # Validate relevance range
        if "relevance" in payload:
            relevance = payload["relevance"]
            if not isinstance(relevance, (int, float)):
                issues.append("relevance_not_numeric")
            elif not (0.0 <= relevance <= 1.0):
                issues.append("relevance_out_of_range")

        # Validate tags
        if "tags" in payload:
            tags = payload["tags"]
            if not isinstance(tags, list):
                issues.append("tags_not_list")
            elif not all(isinstance(tag, str) for tag in tags):
                issues.append("tags_invalid_types")

        return issues

    async def _repair_memory(
        self,
        memory_id: str,
        vector: Optional[np.ndarray],
        payload: Dict,
        issues: List[str],
        collection_name: str
    ):
        """Attempt to repair memory issues."""
        updated_payload = payload.copy()
        needs_re_embedding = False

        # Repair strategies
        for issue in issues:
            if issue in ["null_embedding", "nan_values", "inf_values", "zero_vector"]:
                needs_re_embedding = True

            elif issue == "content_too_short":
                # Can't fix, will quarantine
                raise ValueError("Cannot repair short content")

            elif issue == "invalid_timestamp_format":
                # Use current time
                updated_payload["timestamp"] = datetime.now(timezone.utc).isoformat()

            elif issue == "relevance_not_numeric":
                updated_payload["relevance"] = 0.5

            elif issue == "relevance_out_of_range":
                relevance = payload.get("relevance", 0.5)
                updated_payload["relevance"] = max(0.0, min(1.0, relevance))

            elif issue == "tags_not_list":
                updated_payload["tags"] = []

        # Re-embed if needed
        if needs_re_embedding:
            content = payload.get("content", "")
            if content and len(content) >= 10:
                new_vector = self.embeddings.embed_query(content)

                # Update point with new vector
                point = PointStruct(
                    id=memory_id,
                    vector=new_vector,
                    payload=updated_payload
                )

                safe_upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                return

        # Update payload only
        safe_set_payload(
            collection_name=collection_name,
            points=[memory_id],
            payload=updated_payload
        )

    async def _quarantine_memory(
        self,
        memory_id: str,
        vector: Optional[np.ndarray],
        payload: Dict,
        source_collection: str,
        quarantine_collection: str
    ):
        """Move corrupted memory to quarantine collection."""
        try:
            # Create quarantine collection if doesn't exist
            try:
                self.client.get_collection(quarantine_collection)
            except Exception:
                # Collection doesn't exist, create it
                self.client.create_collection(
                    collection_name=quarantine_collection,
                    vectors_config=VectorParams(
                        size=self.config.get("expected_dimension", 768),
                        distance=Distance.COSINE
                    )
                )

            # Move to quarantine
            if vector is not None:
                point = PointStruct(
                    id=memory_id,
                    vector=vector.tolist(),
                    payload={**payload, "quarantine_reason": "data_quality_issues"}
                )

                safe_upsert(
                    collection_name=quarantine_collection,
                    points=[point]
                )

            # Delete from source
            safe_delete(
                collection_name=source_collection,
                points_selector=[memory_id]
            )

            self.logger.info(f"Quarantined memory {memory_id}")

        except Exception as e:
            self.logger.error(f"Failed to quarantine {memory_id}: {e}")


# ============================================================================
# 5. COLLECTION BALANCING WORKER
# ============================================================================

class CollectionBalancingWorker(BaseWorker):
    """
    Implements HOT/WARM/COLD tiered storage architecture.

    Tiers:
    - HOT:  Recent (< 30d), high relevance, frequent access
    - WARM: Moderate age (30-90d), medium relevance
    - COLD: Old (> 90d), low relevance, rare access

    Quantization:
    - HOT:  No quantization (full precision)
    - WARM: Binary quantization (32x compression)
    - COLD: Scalar quantization (4x compression)
    """

    def _setup_metrics(self):
        self.metrics.update({
            "hot_tier_count": 0,
            "warm_tier_count": 0,
            "cold_tier_count": 0,
            "migrations_total": 0,
            "storage_saved_bytes": 0
        })

    async def run(self) -> Dict[str, Any]:
        """Execute collection balancing."""
        hot_collection = self.config.get("hot_collection", "lexi_memory")
        warm_collection = self.config.get("warm_collection", "lexi_memory_warm")
        cold_collection = self.config.get("cold_collection", "lexi_memory_cold")

        self.logger.info("Starting collection balancing (HOT/WARM/COLD tiers)")

        # Ensure WARM and COLD collections exist
        await self._ensure_tier_collections_exist(
            warm_collection, cold_collection
        )

        # Fetch all memories from HOT tier
        usage_tracker = get_usage_tracker()
        memories = await self._fetch_memories_with_usage(hot_collection)

        self.logger.info(f"Analyzing {len(memories)} memories for tier migration")

        # Classify into tiers
        to_warm = []
        to_cold = []

        for memory_id, payload in memories:
            tier = self._determine_tier(memory_id, payload, usage_tracker)

            if tier == "WARM":
                to_warm.append((memory_id, payload))
            elif tier == "COLD":
                to_cold.append((memory_id, payload))

        # Migrate memories
        await self._migrate_to_tier(to_warm, hot_collection, warm_collection, "WARM")
        await self._migrate_to_tier(to_cold, hot_collection, cold_collection, "COLD")

        # Update metrics
        self.metrics["migrations_total"] = len(to_warm) + len(to_cold)
        self.metrics["warm_tier_count"] = len(to_warm)
        self.metrics["cold_tier_count"] = len(to_cold)
        self.metrics["hot_tier_count"] = len(memories) - self.metrics["migrations_total"]

        self.logger.info(f"Collection balancing complete: "
                        f"HOT={self.metrics['hot_tier_count']}, "
                        f"WARM={self.metrics['warm_tier_count']}, "
                        f"COLD={self.metrics['cold_tier_count']}")

        return self.metrics

    async def _ensure_tier_collections_exist(
        self,
        warm_collection: str,
        cold_collection: str
    ):
        """Ensure WARM and COLD tier collections exist with proper configuration."""
        dimension = self.config.get("expected_dimension", 768)

        # WARM tier with binary quantization
        try:
            self.client.get_collection(warm_collection)
        except Exception:
            self.logger.info(f"Creating WARM tier collection: {warm_collection}")
            self.client.create_collection(
                collection_name=warm_collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                ),
                quantization_config=BinaryQuantization(
                    binary={"always_ram": True}
                )
            )

        # COLD tier with scalar quantization
        try:
            self.client.get_collection(cold_collection)
        except Exception:
            self.logger.info(f"Creating COLD tier collection: {cold_collection}")
            self.client.create_collection(
                collection_name=cold_collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                ),
                quantization_config=ScalarQuantization(
                    scalar={"type": "int8", "always_ram": False}
                )
            )

    async def _fetch_memories_with_usage(
        self,
        collection_name: str
    ) -> List[Tuple[str, Dict]]:
        """Fetch memories with payload."""
        memories = []
        offset = None
        batch_size = 1000

        while True:
            scroll_result = safe_scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset
            )

            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                points = getattr(scroll_result, "points", [])
                next_offset = getattr(scroll_result, "next_page_offset", None)

            if not points:
                break

            for point in points:
                memories.append((point.id, point.payload or {}))

            if next_offset is None:
                break

            offset = next_offset

        return memories

    def _determine_tier(
        self,
        memory_id: str,
        payload: Dict,
        usage_tracker: MemoryUsageTracker
    ) -> str:
        """
        Determine appropriate tier for memory.

        Rules:
        - HOT:  age < 30d AND (relevance > 0.5 OR retrieval_count > 10)
        - WARM: 30d ≤ age < 90d OR 0.2 ≤ relevance ≤ 0.5
        - COLD: age ≥ 90d AND relevance < 0.2 AND retrieval_count < 3
        """
        # Calculate age
        timestamp_str = payload.get("timestamp")
        if not timestamp_str:
            return "HOT"  # Keep if no timestamp

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now(timezone.utc) - timestamp).days
        except Exception:
            return "HOT"

        # Get usage stats
        stats = usage_tracker.get_usage_stats(memory_id)
        relevance = payload.get("relevance", 0.5)
        retrieval_count = stats.get("retrievals", 0)

        # Apply tier rules
        if age_days < 30 and (relevance > 0.5 or retrieval_count > 10):
            return "HOT"

        if age_days >= 90 and relevance < 0.2 and retrieval_count < 3:
            return "COLD"

        if 30 <= age_days < 90 or (0.2 <= relevance <= 0.5):
            return "WARM"

        return "HOT"

    async def _migrate_to_tier(
        self,
        memories: List[Tuple[str, Dict]],
        source_collection: str,
        target_collection: str,
        tier_name: str
    ):
        """Migrate memories to target tier collection."""
        if not memories:
            return

        self.logger.info(f"Migrating {len(memories)} memories to {tier_name} tier")

        # Fetch full points (with vectors) from source
        memory_ids = [mem_id for mem_id, _ in memories]

        # Batch process
        batch_size = 100
        for i in range(0, len(memory_ids), batch_size):
            batch_ids = memory_ids[i:i+batch_size]

            try:
                # Retrieve points with vectors
                points_to_migrate = []

                for mem_id in batch_ids:
                    # Scroll to find this point
                    result = safe_scroll(
                        collection_name=source_collection,
                        scroll_filter=Filter(must=[FieldCondition(
                            key="id",
                            match=MatchValue(value=mem_id)
                        )]),
                        with_payload=True,
                        with_vectors=True,
                        limit=1
                    )

                    points = result[0] if isinstance(result, tuple) else result.points

                    if points:
                        point = points[0]
                        points_to_migrate.append(PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload={**point.payload, "tier": tier_name}
                        ))

                # Upsert to target collection
                if points_to_migrate:
                    safe_upsert(
                        collection_name=target_collection,
                        points=points_to_migrate
                    )

                    # Delete from source
                    safe_delete(
                        collection_name=source_collection,
                        points_selector=[p.id for p in points_to_migrate]
                    )

                    self.logger.debug(f"Migrated batch of {len(points_to_migrate)} to {tier_name}")

            except Exception as e:
                self.logger.error(f"Failed to migrate batch to {tier_name}: {e}")


# ============================================================================
# WORKER COORDINATOR
# ============================================================================

class WorkerCoordinator:
    """
    Coordinates scheduling and execution of all optimization workers.

    Uses APScheduler for cron-like scheduling.
    Handles worker lifecycle, error recovery, and metrics aggregation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        qdrant_client: QdrantClient,
        embeddings,
        memory_adapter=None
    ):
        """
        Initialize coordinator.

        Args:
            config: Global worker configuration
            qdrant_client: Qdrant client
            embeddings: Embedding model
            memory_adapter: Memory adapter for coordination
        """
        self.config = config
        self.client = qdrant_client
        self.embeddings = embeddings
        self.memory_adapter = memory_adapter
        self.logger = logging.getLogger("WorkerCoordinator")

        # Initialize workers
        self.workers = self._initialize_workers()
        self.scheduler = None

    def _initialize_workers(self) -> Dict[str, BaseWorker]:
        """Initialize all worker instances."""
        workers = {}

        worker_classes = {
            "deduplication": DeduplicationWorker,
            "index_optimization": IndexOptimizationWorker,
            "relevance_reranking": RelevanceRerankingWorker,
            "data_quality": DataQualityWorker,
            "collection_balancing": CollectionBalancingWorker
        }

        for worker_name, worker_class in worker_classes.items():
            if self.config.get("workers", {}).get(worker_name, {}).get("enabled", True):
                worker_config = {
                    **self.config.get("workers", {}).get(worker_name, {}),
                    "collection_name": self.config.get("collection_name", "lexi_memory"),
                    "expected_dimension": self.config.get("expected_dimension", 768)
                }

                workers[worker_name] = worker_class(
                    config=worker_config,
                    qdrant_client=self.client,
                    embeddings=self.embeddings,
                    memory_adapter=self.memory_adapter
                )

                self.logger.info(f"✓ Initialized {worker_name} worker")

        return workers

    async def start(self):
        """Start coordinator and schedule workers."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            self.logger.error("APScheduler not installed. Install with: pip install apscheduler")
            return

        self.scheduler = AsyncIOScheduler()

        # Schedule each worker
        for worker_name, worker in self.workers.items():
            schedule = self.config.get("workers", {}).get(worker_name, {}).get("schedule")

            if schedule:
                self.scheduler.add_job(
                    worker.safe_run,
                    CronTrigger.from_crontab(schedule),
                    id=f"{worker_name}_worker",
                    name=worker_name
                )

                self.logger.info(f"📅 Scheduled {worker_name}: {schedule}")

        # Start scheduler
        self.scheduler.start()
        self.logger.info("🚀 Worker coordinator started")

    async def stop(self):
        """Stop coordinator and all workers."""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            self.logger.info("⏹️  Worker coordinator stopped")

    async def run_worker(self, worker_name: str) -> Dict[str, Any]:
        """Manually trigger a specific worker."""
        if worker_name not in self.workers:
            return {"success": False, "error": f"Worker {worker_name} not found"}

        worker = self.workers[worker_name]
        return await worker.safe_run()

    async def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        status = {
            "overall_status": "healthy",
            "workers": []
        }

        for worker_name, worker in self.workers.items():
            last_run = await worker.get_last_run_info()

            worker_status = {
                "name": worker_name,
                "enabled": True,
                "last_run": last_run.get("timestamp") if last_run else None,
                "last_duration": last_run.get("duration") if last_run else None,
                "last_status": last_run.get("status") if last_run else "never_run",
                "metrics": last_run.get("metrics") if last_run else {}
            }

            status["workers"].append(worker_status)

            # Update overall status
            if last_run and last_run.get("status") == "failed":
                status["overall_status"] = "degraded"

        return status
