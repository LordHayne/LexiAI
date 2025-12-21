#!/usr/bin/env python3
"""
Quick Validation Script für das intelligente Memory-System.

Dieser Script überprüft:
1. Imports funktionieren
2. Grundlegende Funktionalität
3. Keine Syntax-Fehler
"""

import sys
from datetime import datetime, timedelta, timezone


def test_imports():
    """Test ob alle neuen Module importiert werden können."""
    print("🔍 Testing Imports...")

    try:
        from backend.memory.memory_intelligence import (
            MemoryUsageTracker,
            MemoryConsolidator,
            IntelligentMemoryCleanup,
            get_usage_tracker,
            track_memory_retrieval,
            track_memory_usage,
            update_memory_relevance
        )
        print("  ✅ memory_intelligence.py imports OK")
    except Exception as e:
        print(f"  ❌ memory_intelligence.py import failed: {e}")
        return False

    try:
        from backend.services.heartbeat_memory import (
            intelligent_memory_maintenance,
            get_heartbeat_status,
            reset_heartbeat_stats
        )
        print("  ✅ heartbeat_memory.py imports OK")
    except Exception as e:
        print(f"  ❌ heartbeat_memory.py import failed: {e}")
        return False

    try:
        from backend.core.component_cache import get_cached_components, clear_component_cache
        print("  ✅ component_cache.py imports OK")
    except Exception as e:
        print(f"  ❌ component_cache.py import failed: {e}")
        return False

    try:
        from backend.memory.adapter import (
            retrieve_memories,
            store_memory,
            get_memory_stats
        )
        print("  ✅ adapter.py imports OK")
    except Exception as e:
        print(f"  ❌ adapter.py import failed: {e}")
        return False

    try:
        from backend.qdrant.qdrant_interface import QdrantMemoryInterface
        print("  ✅ qdrant_interface.py imports OK")
    except Exception as e:
        print(f"  ❌ qdrant_interface.py import failed: {e}")
        return False

    return True


def test_memory_usage_tracker():
    """Test MemoryUsageTracker Funktionalität."""
    print("\n🧪 Testing MemoryUsageTracker...")

    try:
        from backend.memory.memory_intelligence import MemoryUsageTracker

        tracker = MemoryUsageTracker()

        # Test retrieval tracking
        tracker.track_retrieval("test_mem_1")
        stats = tracker.get_usage_stats("test_mem_1")
        assert stats["retrievals"] == 1, "Retrieval count should be 1"
        print("  ✅ Retrieval tracking works")

        # Test usage tracking
        tracker.track_usage_in_response("test_mem_1", was_helpful=True)
        stats = tracker.get_usage_stats("test_mem_1")
        assert stats["used_in_response"] == 1, "Usage count should be 1"
        assert stats["success_rate"] == 1.0, "Success rate should be 1.0"
        print("  ✅ Usage tracking works")

        # Test adaptive relevance
        relevance = tracker.calculate_adaptive_relevance("test_mem_1", 0.5, memory_age_days=10)
        assert relevance > 0.5, "Adaptive relevance should be boosted"
        print(f"  ✅ Adaptive relevance calculation works (0.5 → {relevance:.2f})")

        return True

    except Exception as e:
        print(f"  ❌ MemoryUsageTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_consolidator():
    """Test MemoryConsolidator Funktionalität."""
    print("\n🧪 Testing MemoryConsolidator...")

    try:
        from backend.memory.memory_intelligence import MemoryConsolidator
        from backend.models.memory_entry import MemoryEntry

        consolidator = MemoryConsolidator()

        # Test cosine similarity
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = consolidator._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01, "Identical vectors should have similarity ~1.0"
        print("  ✅ Cosine similarity calculation works")

        # Test finding similar memories
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
            )
        ]

        similar_groups = consolidator.find_similar_memories(memories, similarity_threshold=0.8)
        print(f"  ✅ Find similar memories works (found {len(similar_groups)} groups)")

        return True

    except Exception as e:
        print(f"  ❌ MemoryConsolidator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intelligent_cleanup():
    """Test IntelligentMemoryCleanup Funktionalität."""
    print("\n🧪 Testing IntelligentMemoryCleanup...")

    try:
        from backend.memory.memory_intelligence import (
            MemoryUsageTracker,
            IntelligentMemoryCleanup
        )
        from backend.models.memory_entry import MemoryEntry

        tracker = MemoryUsageTracker()
        cleanup = IntelligentMemoryCleanup(tracker)

        # Test protection of high relevance memories
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        high_rel_memory = MemoryEntry(
            id="high_rel",
            content="Important",
            timestamp=old_date,
            relevance=0.9
        )

        to_delete = cleanup.identify_memories_for_deletion(
            [high_rel_memory],
            max_age_days=90,
            min_relevance=0.1
        )

        assert "high_rel" not in to_delete, "High relevance memory should be protected"
        print("  ✅ High relevance memory protection works")

        # Test deletion of old, low relevance memories
        low_rel_memory = MemoryEntry(
            id="low_rel",
            content="Unimportant",
            timestamp=old_date,
            relevance=0.05
        )

        to_delete = cleanup.identify_memories_for_deletion(
            [low_rel_memory],
            max_age_days=90,
            min_relevance=0.1
        )

        assert "low_rel" in to_delete, "Low relevance old memory should be deleted"
        print("  ✅ Old low relevance memory deletion works")

        return True

    except Exception as e:
        print(f"  ❌ IntelligentMemoryCleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heartbeat_status():
    """Test Heartbeat Status Tracking."""
    print("\n🧪 Testing Heartbeat Status...")

    try:
        from backend.services.heartbeat_memory import (
            get_heartbeat_status,
            reset_heartbeat_stats
        )

        # Reset first
        reset_heartbeat_stats()

        status = get_heartbeat_status()
        assert "last_run" in status, "Status should have last_run"
        assert "deleted_count" in status, "Status should have deleted_count"
        assert "consolidated_count" in status, "Status should have consolidated_count"
        assert "run_count" in status, "Status should have run_count"

        print("  ✅ Heartbeat status tracking works")
        print(f"     Status: {status}")

        return True

    except Exception as e:
        print(f"  ❌ Heartbeat status test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("🚀 LexiAI Intelligent Memory System Validation")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))

    if results[0][1]:  # Only continue if imports work
        results.append(("MemoryUsageTracker", test_memory_usage_tracker()))
        results.append(("MemoryConsolidator", test_memory_consolidator()))
        results.append(("IntelligentCleanup", test_intelligent_cleanup()))
        results.append(("HeartbeatStatus", test_heartbeat_status()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All validation tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
