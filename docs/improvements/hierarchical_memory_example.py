"""
Hierarchical Memory Architecture - 3-Tier System

Implements hot/warm/cold memory tiers with automatic compression
and promotion/demotion based on access patterns.

Expected Benefits:
- 90% storage reduction for old memories
- 5x faster retrieval (smaller search space)
- Maintained accuracy for recent memories
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

from backend.models.memory_entry import MemoryEntry

logger = logging.getLogger("lexi_middleware.hierarchical_memory")


class MemoryTier(Enum):
    """Memory tier classification"""
    HOT = "hot"      # Last 7 days, full embeddings (768-dim)
    WARM = "warm"    # 7-90 days, compressed (256-dim)
    COLD = "cold"    # 90+ days, ultra-compressed (64-dim)


@dataclass
class TierConfig:
    """Configuration for a memory tier"""
    name: MemoryTier
    max_age_days: int
    embedding_dim: int
    collection_name: str
    priority: int  # Higher = searched first


class HierarchicalMemoryManager:
    """
    Manages memories across multiple tiers with automatic promotion/demotion.

    Architecture:
    - HOT tier: Recent memories (7 days), full quality, fast access
    - WARM tier: Older memories (7-90 days), compressed, moderate speed
    - COLD tier: Ancient memories (90+ days), highly compressed, archive

    Operations:
    - Automatic demotion based on age
    - Smart promotion based on access frequency
    - Compression preserves semantic similarity
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

        # Tier configurations
        self.tiers = {
            MemoryTier.HOT: TierConfig(
                name=MemoryTier.HOT,
                max_age_days=7,
                embedding_dim=768,
                collection_name="lexi_memory_hot",
                priority=1
            ),
            MemoryTier.WARM: TierConfig(
                name=MemoryTier.WARM,
                max_age_days=90,
                embedding_dim=256,
                collection_name="lexi_memory_warm",
                priority=2
            ),
            MemoryTier.COLD: TierConfig(
                name=MemoryTier.COLD,
                max_age_days=999999,  # Infinite
                embedding_dim=64,
                collection_name="lexi_memory_cold",
                priority=3
            )
        }

        # PCA models for compression (trained once, reused)
        self.pca_models = {}
        self._initialize_pca_models()

    def _initialize_pca_models(self):
        """Initialize PCA models for embedding compression"""
        # We'll train these on existing embeddings
        self.pca_models = {
            768 -> 256: None,  # HOT → WARM compression
            768 -> 64: None,   # HOT → COLD compression
            256 -> 64: None    # WARM → COLD compression
        }

    def compress_embedding(self, embedding: List[float],
                          target_dim: int) -> List[float]:
        """
        Compress embedding using PCA.

        Args:
            embedding: Original embedding (768-dim or 256-dim)
            target_dim: Target dimensionality (256 or 64)

        Returns:
            Compressed embedding
        """
        original_dim = len(embedding)
        pca_key = f"{original_dim}->{target_dim}"

        if pca_key not in self.pca_models or self.pca_models[pca_key] is None:
            # Initialize PCA model
            self.pca_models[pca_key] = PCA(n_components=target_dim)
            logger.info(f"Initialized PCA model: {pca_key}")

        pca = self.pca_models[pca_key]

        # Compress
        compressed = pca.transform([embedding])[0]

        return compressed.tolist()

    def determine_tier(self, memory: MemoryEntry) -> MemoryTier:
        """
        Determine which tier a memory belongs to based on age.

        Args:
            memory: Memory to classify

        Returns:
            Appropriate tier
        """
        age_days = (datetime.now(timezone.utc) - memory.timestamp).days

        if age_days <= 7:
            return MemoryTier.HOT
        elif age_days <= 90:
            return MemoryTier.WARM
        else:
            return MemoryTier.COLD

    def store_memory(self, memory: MemoryEntry) -> bool:
        """
        Store memory in appropriate tier.

        Args:
            memory: Memory to store

        Returns:
            Success status
        """
        tier = self.determine_tier(memory)
        tier_config = self.tiers[tier]

        # Compress embedding if needed
        if tier != MemoryTier.HOT:
            original_embedding = memory.embedding
            compressed_embedding = self.compress_embedding(
                original_embedding,
                tier_config.embedding_dim
            )
            memory.embedding = compressed_embedding
            logger.debug(f"Compressed embedding: {len(original_embedding)}d → {len(compressed_embedding)}d")

        # Store in tier collection
        try:
            self.vectorstore.store_entry(
                memory,
                collection_name=tier_config.collection_name
            )
            logger.info(f"Stored memory {memory.id} in {tier.value} tier")
            return True
        except Exception as e:
            logger.error(f"Failed to store memory in {tier.value} tier: {e}")
            return False

    def retrieve_memories(self, query_embedding: List[float],
                         user_id: str,
                         k: int = 10,
                         search_all_tiers: bool = True) -> List[MemoryEntry]:
        """
        Retrieve memories across tiers.

        Strategy:
        1. Search HOT tier first (highest priority, best quality)
        2. If not enough results, search WARM tier
        3. If still not enough, search COLD tier

        Args:
            query_embedding: Query vector (768-dim)
            user_id: User identifier
            k: Number of results desired
            search_all_tiers: Whether to search all tiers or stop after HOT

        Returns:
            List of retrieved memories (ranked by relevance)
        """
        all_results = []

        # Search tiers in priority order
        for tier in sorted(self.tiers.values(), key=lambda t: t.priority):
            if not search_all_tiers and len(all_results) >= k:
                break

            # Compress query embedding to match tier dimension
            tier_query_embedding = query_embedding
            if tier.embedding_dim != len(query_embedding):
                tier_query_embedding = self.compress_embedding(
                    query_embedding,
                    tier.embedding_dim
                )

            # Search this tier
            try:
                tier_results = self.vectorstore.query(
                    query_vector=tier_query_embedding,
                    collection_name=tier.collection_name,
                    filter={"user_id": user_id},
                    limit=k
                )

                # Tag with tier information
                for memory in tier_results:
                    memory.metadata = memory.metadata or {}
                    memory.metadata["tier"] = tier.name.value

                all_results.extend(tier_results)
                logger.debug(f"Retrieved {len(tier_results)} from {tier.name.value} tier")

            except Exception as e:
                logger.error(f"Error searching {tier.name.value} tier: {e}")
                continue

        # Combine and re-rank results
        all_results = self._rerank_cross_tier_results(all_results)

        return all_results[:k]

    def _rerank_cross_tier_results(self, results: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Re-rank results from different tiers.

        Strategy:
        - Boost HOT tier results slightly (more accurate embeddings)
        - Apply recency bonus
        - Consider usage statistics
        """
        tier_boosts = {
            MemoryTier.HOT: 1.1,   # 10% boost
            MemoryTier.WARM: 1.0,  # No boost
            MemoryTier.COLD: 0.9   # Slight penalty
        }

        for memory in results:
            tier = MemoryTier(memory.metadata.get("tier", "hot"))
            boost = tier_boosts.get(tier, 1.0)

            # Apply boost to relevance
            if memory.relevance:
                memory.relevance *= boost

        # Sort by boosted relevance
        results.sort(key=lambda m: m.relevance or 0.0, reverse=True)

        return results

    def promote_memory(self, memory_id: str, from_tier: MemoryTier,
                      to_tier: MemoryTier) -> bool:
        """
        Promote a frequently accessed memory to a higher tier.

        Use case: Cold memory gets accessed often → Move to WARM/HOT

        Args:
            memory_id: Memory to promote
            from_tier: Current tier
            to_tier: Target tier (must be higher priority)

        Returns:
            Success status
        """
        if self.tiers[to_tier].priority >= self.tiers[from_tier].priority:
            logger.warning(f"Cannot promote from {from_tier.value} to {to_tier.value}")
            return False

        try:
            # 1. Retrieve from old tier
            old_collection = self.tiers[from_tier].collection_name
            memory = self.vectorstore.get_entry(memory_id, collection_name=old_collection)

            if not memory:
                logger.warning(f"Memory {memory_id} not found in {from_tier.value}")
                return False

            # 2. Decompress embedding if promoting to HOT
            if to_tier == MemoryTier.HOT and from_tier != MemoryTier.HOT:
                # We can't perfectly reconstruct, so re-embed
                from backend.core.component_cache import get_cached_components
                bundle = get_cached_components()
                memory.embedding = bundle.embeddings.embed_query(memory.content)
                logger.info(f"Re-embedded memory for promotion to HOT tier")

            # 3. Store in new tier
            new_collection = self.tiers[to_tier].collection_name
            self.vectorstore.store_entry(memory, collection_name=new_collection)

            # 4. Delete from old tier
            self.vectorstore.delete_entry(memory_id, collection_name=old_collection)

            logger.info(f"Promoted memory {memory_id}: {from_tier.value} → {to_tier.value}")
            return True

        except Exception as e:
            logger.error(f"Error promoting memory: {e}")
            return False

    def demote_memory(self, memory_id: str, from_tier: MemoryTier,
                     to_tier: MemoryTier) -> bool:
        """
        Demote an old/unused memory to a lower tier.

        Use case: HOT memory becomes old → Move to WARM → COLD

        Args:
            memory_id: Memory to demote
            from_tier: Current tier
            to_tier: Target tier (must be lower priority)

        Returns:
            Success status
        """
        if self.tiers[to_tier].priority <= self.tiers[from_tier].priority:
            logger.warning(f"Cannot demote from {from_tier.value} to {to_tier.value}")
            return False

        try:
            # 1. Retrieve from old tier
            old_collection = self.tiers[from_tier].collection_name
            memory = self.vectorstore.get_entry(memory_id, collection_name=old_collection)

            if not memory:
                logger.warning(f"Memory {memory_id} not found in {from_tier.value}")
                return False

            # 2. Compress embedding
            target_dim = self.tiers[to_tier].embedding_dim
            memory.embedding = self.compress_embedding(memory.embedding, target_dim)

            # 3. Store in new tier
            new_collection = self.tiers[to_tier].collection_name
            self.vectorstore.store_entry(memory, collection_name=new_collection)

            # 4. Delete from old tier
            self.vectorstore.delete_entry(memory_id, collection_name=old_collection)

            logger.info(f"Demoted memory {memory_id}: {from_tier.value} → {to_tier.value}")
            return True

        except Exception as e:
            logger.error(f"Error demoting memory: {e}")
            return False

    def run_tier_maintenance(self, user_id: str) -> Dict[str, int]:
        """
        Automatic tier maintenance: Promote/demote based on age and usage.

        Should run periodically (e.g., nightly).

        Returns:
            Statistics: {"promoted": N, "demoted": N}
        """
        stats = {"promoted": 0, "demoted": 0}

        logger.info(f"Running tier maintenance for user {user_id}")

        # 1. Check HOT tier for old memories → Demote to WARM
        hot_memories = self.vectorstore.get_all_entries(
            collection_name=self.tiers[MemoryTier.HOT].collection_name,
            filter={"user_id": user_id}
        )

        for memory in hot_memories:
            age_days = (datetime.now(timezone.utc) - memory.timestamp).days

            if age_days > 7:
                if self.demote_memory(str(memory.id), MemoryTier.HOT, MemoryTier.WARM):
                    stats["demoted"] += 1

        # 2. Check WARM tier for old memories → Demote to COLD
        warm_memories = self.vectorstore.get_all_entries(
            collection_name=self.tiers[MemoryTier.WARM].collection_name,
            filter={"user_id": user_id}
        )

        for memory in warm_memories:
            age_days = (datetime.now(timezone.utc) - memory.timestamp).days

            if age_days > 90:
                if self.demote_memory(str(memory.id), MemoryTier.WARM, MemoryTier.COLD):
                    stats["demoted"] += 1

        # 3. Check WARM/COLD tiers for frequently accessed → Promote to HOT
        # (Requires usage tracking integration)
        from backend.memory.memory_intelligence import get_usage_tracker
        tracker = get_usage_tracker()

        for tier in [MemoryTier.WARM, MemoryTier.COLD]:
            tier_memories = self.vectorstore.get_all_entries(
                collection_name=self.tiers[tier].collection_name,
                filter={"user_id": user_id}
            )

            for memory in tier_memories:
                usage_stats = tracker.get_usage_stats(str(memory.id))

                # Promote if accessed 5+ times in last 7 days
                if usage_stats["retrievals"] >= 5:
                    last_used = usage_stats.get("last_used")
                    if last_used and (datetime.now(timezone.utc) - last_used).days < 7:
                        target_tier = MemoryTier.HOT if tier == MemoryTier.COLD else MemoryTier.WARM
                        if self.promote_memory(str(memory.id), tier, target_tier):
                            stats["promoted"] += 1

        logger.info(f"Tier maintenance complete: {stats}")
        return stats


# ============================================================================
# Integration Example
# ============================================================================

def example_usage():
    """Example: Using hierarchical memory in chat processing"""

    from backend.core.component_cache import get_cached_components

    # Initialize
    bundle = get_cached_components()
    manager = HierarchicalMemoryManager(bundle.vectorstore)

    # Store a new memory (automatically goes to HOT tier)
    new_memory = MemoryEntry(
        id="mem_123",
        content="User likes Python programming",
        timestamp=datetime.now(timezone.utc),
        category="preference",
        embedding=bundle.embeddings.embed_query("User likes Python programming"),
        user_id="user_456"
    )
    manager.store_memory(new_memory)

    # Retrieve memories (searches across tiers)
    query = "What does the user like?"
    query_embedding = bundle.embeddings.embed_query(query)

    results = manager.retrieve_memories(
        query_embedding=query_embedding,
        user_id="user_456",
        k=5,
        search_all_tiers=True  # Search HOT → WARM → COLD
    )

    # Results are automatically ranked and boosted by tier
    for memory in results:
        tier = memory.metadata.get("tier", "unknown")
        print(f"[{tier}] {memory.content} (relevance: {memory.relevance:.3f})")

    # Run nightly maintenance
    stats = manager.run_tier_maintenance(user_id="user_456")
    print(f"Maintenance: Promoted {stats['promoted']}, Demoted {stats['demoted']}")


if __name__ == "__main__":
    example_usage()
