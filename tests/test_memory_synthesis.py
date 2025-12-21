"""
Tests für Phase 1: Idle-Mode Memory Synthesis

Testet:
- Activity Tracker (Singleton, Idle Detection)
- Memory Synthesizer (Clustering, Synthesis)
- Heartbeat Idle/Active Modi
"""

import pytest
import time
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, MagicMock, patch

# Import components to test
from backend.memory.activity_tracker import ActivityTracker, track_activity, is_system_idle
from backend.memory.memory_synthesizer import MemorySynthesizer, SynthesisResult
from backend.models.memory_entry import MemoryEntry


class TestActivityTracker:
    """Tests für Activity Tracker"""

    def test_singleton_pattern(self):
        """Test: ActivityTracker ist ein Singleton"""
        tracker1 = ActivityTracker()
        tracker2 = ActivityTracker()

        assert tracker1 is tracker2, "Activity Tracker sollte Singleton sein"

    def test_track_activity(self):
        """Test: track_activity() aktualisiert last_activity"""
        tracker = ActivityTracker()
        tracker.reset()  # Reset für Test

        before = datetime.now(UTC)
        time.sleep(0.01)  # Kurze Pause
        tracker.track_activity()
        after = datetime.now(UTC)

        last_activity = tracker.get_last_activity()
        assert before < last_activity <= after, "last_activity sollte aktualisiert werden"

    def test_is_idle_false_when_just_active(self):
        """Test: is_idle() gibt False wenn gerade aktiv"""
        tracker = ActivityTracker()
        tracker.track_activity()

        assert not tracker.is_idle(minutes=1), "Sollte nicht idle sein direkt nach Aktivität"

    def test_is_idle_true_when_old_activity(self):
        """Test: is_idle() gibt True wenn Aktivität lange her"""
        tracker = ActivityTracker()

        # Simuliere alte Aktivität durch direktes Setzen
        # (in echtem System würde man warten, aber das dauert zu lange für Tests)
        old_time = datetime.now(UTC) - timedelta(minutes=35)
        with patch.object(tracker, '_last_activity', old_time):
            assert tracker.is_idle(minutes=30), "Sollte idle sein nach 35 Minuten"

    def test_get_idle_duration(self):
        """Test: get_idle_duration() berechnet korrekt"""
        tracker = ActivityTracker()
        tracker.track_activity()

        time.sleep(0.1)  # 100ms warten
        duration = tracker.get_idle_duration()

        assert duration.total_seconds() >= 0.1, "Idle duration sollte mindestens 100ms sein"
        assert duration.total_seconds() < 1.0, "Idle duration sollte unter 1s sein"

    def test_convenience_function_track_activity(self):
        """Test: track_activity() convenience function"""
        tracker = ActivityTracker()
        before = tracker.get_last_activity()

        time.sleep(0.01)
        track_activity()

        after = tracker.get_last_activity()
        assert after > before, "Convenience function sollte Activity tracken"

    def test_convenience_function_is_system_idle(self):
        """Test: is_system_idle() convenience function"""
        track_activity()  # Track jetzt

        assert not is_system_idle(minutes=1), "System sollte nicht idle sein"


class TestMemorySynthesizer:
    """Tests für Memory Synthesizer"""

    def test_synthesizer_initialization(self):
        """Test: MemorySynthesizer kann initialisiert werden"""
        mock_llm = Mock()
        mock_vectorstore = Mock()

        synthesizer = MemorySynthesizer(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            min_cluster_size=3,
            similarity_threshold=0.85
        )

        assert synthesizer.min_cluster_size == 3
        assert synthesizer.similarity_threshold == 0.85

    def test_find_synthesizable_clusters_no_memories(self):
        """Test: Keine Cluster wenn nicht genug Memories"""
        mock_llm = Mock()
        mock_vectorstore = Mock()

        synthesizer = MemorySynthesizer(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            min_cluster_size=3
        )

        # Nur 2 Memories (zu wenig)
        memories = [
            MemoryEntry(
                id="1",
                content="Test 1",
                timestamp=datetime.now(UTC),
                embedding=[0.1, 0.2]
            ),
            MemoryEntry(
                id="2",
                content="Test 2",
                timestamp=datetime.now(UTC),
                embedding=[0.11, 0.21]
            )
        ]

        clusters = synthesizer._find_synthesizable_clusters(memories)

        assert len(clusters) == 0, "Sollte keine Cluster finden mit zu wenig Memories"

    def test_synthesis_result_structure(self):
        """Test: SynthesisResult hat korrekte Struktur"""
        meta_memory = MemoryEntry(
            id="meta_1",
            content="Meta-Wissen Test",
            timestamp=datetime.now(UTC),
            is_meta_knowledge=True
        )

        source_memories = [
            MemoryEntry(id="1", content="Source 1", timestamp=datetime.now(UTC)),
            MemoryEntry(id="2", content="Source 2", timestamp=datetime.now(UTC)),
        ]

        result = SynthesisResult(
            meta_knowledge=meta_memory,
            source_memories=source_memories,
            cluster_size=2,
            avg_similarity=0.9
        )

        assert result.meta_knowledge.is_meta_knowledge is True
        assert result.cluster_size == 2
        assert result.avg_similarity == 0.9
        assert len(result.source_memories) == 2

    def test_synthesize_from_cluster_calls_llm(self):
        """Test: _synthesize_from_cluster() ruft LLM auf"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Synthetisiertes Meta-Wissen"

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=mock_response)

        mock_vectorstore = Mock()

        synthesizer = MemorySynthesizer(
            llm=mock_llm,
            vectorstore=mock_vectorstore
        )

        cluster = [
            MemoryEntry(id="1", content="Memory 1", timestamp=datetime.now(UTC)),
            MemoryEntry(id="2", content="Memory 2", timestamp=datetime.now(UTC)),
            MemoryEntry(id="3", content="Memory 3", timestamp=datetime.now(UTC)),
        ]

        result = synthesizer._synthesize_from_cluster(cluster, user_id="test")

        # Verify LLM was called
        mock_llm.invoke.assert_called_once()

        # Verify result structure
        assert result is not None
        assert result.meta_knowledge.content == "Synthetisiertes Meta-Wissen"
        assert result.meta_knowledge.is_meta_knowledge is True
        assert len(result.meta_knowledge.source_memory_ids) == 3


class TestMemoryEntryExtensions:
    """Tests für erweiterte MemoryEntry Felder (Phase 1)"""

    def test_memory_entry_with_meta_knowledge_fields(self):
        """Test: MemoryEntry unterstützt Meta-Knowledge Felder"""
        memory = MemoryEntry(
            id="test_1",
            content="Test Memory",
            timestamp=datetime.now(UTC),
            is_meta_knowledge=True,
            source_memory_ids=["mem_1", "mem_2", "mem_3"],
            synthesis_timestamp="2025-11-02T12:00:00"
        )

        assert memory.is_meta_knowledge is True
        assert len(memory.source_memory_ids) == 3
        assert memory.synthesis_timestamp == "2025-11-02T12:00:00"

    def test_memory_entry_default_meta_knowledge_false(self):
        """Test: is_meta_knowledge ist standardmäßig False"""
        memory = MemoryEntry(
            id="test_2",
            content="Normal Memory",
            timestamp=datetime.now(UTC)
        )

        assert memory.is_meta_knowledge is False
        assert memory.source_memory_ids == []
        assert memory.synthesis_timestamp is None


class TestHeartbeatIdleActive:
    """Tests für Heartbeat Idle/Active Modi"""

    @patch('backend.services.heartbeat_memory.is_system_idle')
    @patch('backend.services.heartbeat_memory.run_deep_learning_tasks')
    @patch('backend.services.heartbeat_memory.run_lightweight_maintenance')
    def test_heartbeat_calls_deep_learning_when_idle(
        self,
        mock_lightweight,
        mock_deep_learning,
        mock_is_idle
    ):
        """Test: Heartbeat ruft deep_learning_tasks auf wenn idle"""
        from backend.services.heartbeat_memory import intelligent_memory_maintenance

        # Simuliere IDLE mode
        mock_is_idle.return_value = True
        mock_deep_learning.return_value = {"synthesized": 2, "updated": 5}

        result = intelligent_memory_maintenance()

        # Verify deep learning was called
        mock_deep_learning.assert_called_once()
        mock_lightweight.assert_not_called()

    @patch('backend.services.heartbeat_memory.is_system_idle')
    @patch('backend.services.heartbeat_memory.run_deep_learning_tasks')
    @patch('backend.services.heartbeat_memory.run_lightweight_maintenance')
    def test_heartbeat_calls_lightweight_when_active(
        self,
        mock_lightweight,
        mock_deep_learning,
        mock_is_idle
    ):
        """Test: Heartbeat ruft lightweight_maintenance auf wenn active"""
        from backend.services.heartbeat_memory import intelligent_memory_maintenance

        # Simuliere ACTIVE mode
        mock_is_idle.return_value = False
        mock_lightweight.return_value = {"updated": 3}

        result = intelligent_memory_maintenance()

        # Verify lightweight was called
        mock_lightweight.assert_called_once()
        mock_deep_learning.assert_not_called()

    def test_heartbeat_status_includes_synthesis_count(self):
        """Test: Heartbeat Status enthält synthesis_count"""
        from backend.services.heartbeat_memory import get_heartbeat_status

        status = get_heartbeat_status()

        assert "synthesized_count" in status, "Status sollte synthesized_count enthalten"
        assert "mode" in status, "Status sollte mode enthalten"
        assert "last_synthesis" in status, "Status sollte last_synthesis enthalten"


# Integration Test (erfordert laufende Components - als Optional markiert)
@pytest.mark.skip(reason="Requires running Ollama and Qdrant - run manually")
class TestMemorySynthesisIntegration:
    """Integration Tests für Memory Synthesis (manual)"""

    def test_full_synthesis_flow(self):
        """
        Integration Test: Kompletter Synthesis Flow

        Voraussetzungen:
        - Ollama läuft
        - Qdrant läuft
        - Mindestens 10 Memories vorhanden

        Test:
        1. Erstelle mehrere ähnliche Memories
        2. Führe Synthesis aus
        3. Verifiziere Meta-Knowledge wurde erstellt
        """
        from backend.core.component_cache import get_cached_components
        from backend.memory.memory_synthesizer import synthesize_memories

        # Dieser Test muss manuell ausgeführt werden
        # wenn Ollama & Qdrant laufen

        count = synthesize_memories(user_id="default")

        # Wenn es Cluster gab, sollte Meta-Knowledge erstellt worden sein
        # (kann 0 sein wenn nicht genug ähnliche Memories)
        assert count >= 0, "Synthesis sollte ohne Fehler laufen"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
