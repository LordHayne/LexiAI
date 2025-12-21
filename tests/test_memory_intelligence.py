"""
Tests für das intelligente Memory-System.
"""

import pytest
from datetime import datetime, timedelta, timezone
from backend.memory.memory_intelligence import (
    MemoryUsageTracker,
    MemoryConsolidator,
    IntelligentMemoryCleanup,
    update_memory_relevance,
    track_memory_retrieval,
    track_memory_usage
)
from backend.models.memory_entry import MemoryEntry


class TestMemoryUsageTracker:
    """Tests für MemoryUsageTracker."""

    def test_track_retrieval(self):
        tracker = MemoryUsageTracker()
        tracker.track_retrieval("mem_123")

        stats = tracker.get_usage_stats("mem_123")
        assert stats["retrievals"] == 1
        assert stats["last_used"] is not None

    def test_track_usage_in_response(self):
        tracker = MemoryUsageTracker()
        tracker.track_retrieval("mem_123")
        tracker.track_usage_in_response("mem_123", was_helpful=True)

        stats = tracker.get_usage_stats("mem_123")
        assert stats["used_in_response"] == 1
        assert stats["success_rate"] == 1.0

    def test_success_rate_calculation(self):
        tracker = MemoryUsageTracker()

        # 5 retrievals, 3 successful uses
        for _ in range(5):
            tracker.track_retrieval("mem_123")

        for _ in range(3):
            tracker.track_usage_in_response("mem_123", was_helpful=True)

        stats = tracker.get_usage_stats("mem_123")
        assert stats["retrievals"] == 5
        assert stats["used_in_response"] == 3
        assert stats["success_rate"] == 0.6

    def test_adaptive_relevance_usage_boost(self):
        tracker = MemoryUsageTracker()

        # Simulate frequent successful usage
        for _ in range(5):
            tracker.track_retrieval("mem_123")
            tracker.track_usage_in_response("mem_123", was_helpful=True)

        # Base relevance 0.5, should get usage boost
        adaptive_rel = tracker.calculate_adaptive_relevance("mem_123", 0.5, memory_age_days=10)

        # Should be higher than base due to usage boost
        assert adaptive_rel > 0.5

    def test_adaptive_relevance_age_decay(self):
        tracker = MemoryUsageTracker()

        # Old memory, never used
        adaptive_rel = tracker.calculate_adaptive_relevance("mem_old", 0.5, memory_age_days=120)

        # Should have decay
        assert adaptive_rel < 0.5

    def test_adaptive_relevance_recency_boost(self):
        tracker = MemoryUsageTracker()

        # Recently used memory
        tracker.track_retrieval("mem_recent")
        tracker.track_usage_in_response("mem_recent", was_helpful=True)

        adaptive_rel = tracker.calculate_adaptive_relevance("mem_recent", 0.5, memory_age_days=5)

        # Should get recency boost
        assert adaptive_rel > 0.5


class TestMemoryConsolidator:
    """Tests für MemoryConsolidator."""

    def test_cosine_similarity(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = MemoryConsolidator._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01  # Should be nearly 1.0

    def test_cosine_similarity_orthogonal(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = MemoryConsolidator._cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.01  # Should be nearly 0.0

    def test_find_similar_memories_no_matches(self):
        memories = [
            MemoryEntry(
                id="1",
                content="Test 1",
                timestamp=datetime.now(timezone.utc),
                embedding=[1.0, 0.0, 0.0]
            ),
            MemoryEntry(
                id="2",
                content="Test 2",
                timestamp=datetime.now(timezone.utc),
                embedding=[0.0, 1.0, 0.0]
            )
        ]

        consolidator = MemoryConsolidator()
        similar_groups = consolidator.find_similar_memories(memories, similarity_threshold=0.9)

        # No similar groups (vectors are orthogonal)
        assert len(similar_groups) == 0

    def test_find_similar_memories_with_matches(self):
        memories = [
            MemoryEntry(
                id="1",
                content="Test 1",
                timestamp=datetime.now(timezone.utc),
                embedding=[1.0, 0.1, 0.0]
            ),
            MemoryEntry(
                id="2",
                content="Test 2",
                timestamp=datetime.now(timezone.utc),
                embedding=[0.9, 0.2, 0.0]
            ),
            MemoryEntry(
                id="3",
                content="Test 3",
                timestamp=datetime.now(timezone.utc),
                embedding=[0.0, 0.0, 1.0]
            )
        ]

        consolidator = MemoryConsolidator()
        similar_groups = consolidator.find_similar_memories(memories, similarity_threshold=0.8)

        # Should find one group with memories 1 and 2
        assert len(similar_groups) >= 1


class TestIntelligentMemoryCleanup:
    """Tests für IntelligentMemoryCleanup."""

    def test_protect_high_relevance_memories(self):
        tracker = MemoryUsageTracker()
        cleanup = IntelligentMemoryCleanup(tracker)

        # Create old memory with high relevance
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        memory = MemoryEntry(
            id="mem_high",
            content="Important memory",
            timestamp=old_date,
            relevance=0.9  # High relevance
        )

        to_delete = cleanup.identify_memories_for_deletion([memory], max_age_days=90, min_relevance=0.1)

        # Should NOT delete high relevance memory
        assert "mem_high" not in to_delete

    def test_delete_old_low_relevance_memories(self):
        tracker = MemoryUsageTracker()
        cleanup = IntelligentMemoryCleanup(tracker)

        # Create old memory with low relevance, never used
        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        memory = MemoryEntry(
            id="mem_old_low",
            content="Unimportant memory",
            timestamp=old_date,
            relevance=0.05  # Low relevance
        )

        to_delete = cleanup.identify_memories_for_deletion([memory], max_age_days=90, min_relevance=0.1)

        # Should delete old, low relevance memory
        assert "mem_old_low" in to_delete

    def test_protect_frequently_used_memories(self):
        tracker = MemoryUsageTracker()

        # Simulate frequent usage
        for _ in range(5):
            tracker.track_retrieval("mem_frequent")
            tracker.track_usage_in_response("mem_frequent", was_helpful=True)

        cleanup = IntelligentMemoryCleanup(tracker)

        # Create old memory but frequently used
        old_date = datetime.now(timezone.utc) - timedelta(days=150)
        memory = MemoryEntry(
            id="mem_frequent",
            content="Frequently used memory",
            timestamp=old_date,
            relevance=0.3
        )

        to_delete = cleanup.identify_memories_for_deletion([memory], max_age_days=90, min_relevance=0.1)

        # Should NOT delete frequently used memory
        assert "mem_frequent" not in to_delete

    def test_delete_never_used_old_memories(self):
        tracker = MemoryUsageTracker()
        cleanup = IntelligentMemoryCleanup(tracker)

        # Create memory that's 70 days old and never used
        old_date = datetime.now(timezone.utc) - timedelta(days=70)
        memory = MemoryEntry(
            id="mem_unused",
            content="Never used memory",
            timestamp=old_date,
            relevance=0.5
        )

        to_delete = cleanup.identify_memories_for_deletion([memory], max_age_days=90, min_relevance=0.1)

        # Should delete never-used memory older than 60 days
        assert "mem_unused" in to_delete

    def test_should_consolidate_instead_of_delete(self):
        tracker = MemoryUsageTracker()
        cleanup = IntelligentMemoryCleanup(tracker)

        # Memory that was used once, moderate relevance
        tracker.track_retrieval("mem_moderate")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        memory = MemoryEntry(
            id="mem_moderate",
            content="Moderately relevant memory",
            timestamp=old_date,
            relevance=0.3
        )

        should_consolidate = cleanup.should_consolidate_instead_of_delete(memory)

        # Should prefer consolidation over deletion
        assert should_consolidate is True


class TestGlobalFunctions:
    """Tests für globale Helper-Funktionen."""

    def test_track_memory_retrieval_global(self):
        memory_ids = ["mem1", "mem2", "mem3"]
        track_memory_retrieval(memory_ids)

        # Verify tracking was recorded (via global tracker)
        from backend.memory.memory_intelligence import get_usage_tracker
        tracker = get_usage_tracker()

        for mem_id in memory_ids:
            stats = tracker.get_usage_stats(mem_id)
            assert stats["retrievals"] > 0

    def test_track_memory_usage_global(self):
        track_memory_usage("mem_global", was_helpful=True)

        from backend.memory.memory_intelligence import get_usage_tracker
        tracker = get_usage_tracker()

        stats = tracker.get_usage_stats("mem_global")
        assert stats["used_in_response"] > 0

    def test_update_memory_relevance_global(self):
        # Create a memory
        memory = MemoryEntry(
            id="mem_test",
            content="Test memory",
            timestamp=datetime.now(timezone.utc) - timedelta(days=30),
            relevance=0.5
        )

        # Track some usage
        track_memory_retrieval(["mem_test"])
        track_memory_usage("mem_test", was_helpful=True)

        # Update relevance
        new_relevance = update_memory_relevance(memory)

        # Should be different from base
        assert new_relevance != 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
