"""
Test für Phase 3: Self-Correction Heartbeat Integration

Prüft ob die Integration korrekt ist:
1. Stop-Learning Funktionen vorhanden
2. Phase 3 im Heartbeat registriert
3. Stats korrekt aktualisiert
"""

import pytest
from datetime import datetime, timezone


def test_stop_learning_functions_exist():
    """Teste ob Stop-Learning Funktionen existieren."""
    from backend.services.heartbeat_memory import (
        stop_learning_processes,
        is_learning_in_progress
    )

    # Funktionen sollten aufrufbar sein
    assert callable(stop_learning_processes)
    assert callable(is_learning_in_progress)

    # is_learning_in_progress sollte boolean zurückgeben
    result = is_learning_in_progress()
    assert isinstance(result, bool)
    print("✅ Stop-Learning Funktionen vorhanden")


def test_corrections_in_heartbeat_status():
    """Teste ob corrections_count in Status vorhanden ist."""
    from backend.services.heartbeat_memory import get_heartbeat_status

    status = get_heartbeat_status()

    # Prüfe ob alle relevanten Felder vorhanden sind
    assert "corrections_count" in status
    assert "last_correction" in status
    assert isinstance(status["corrections_count"], int)

    print("✅ Corrections-Tracking in Heartbeat Status vorhanden")


def test_self_correction_import():
    """Teste ob Self-Correction Module importierbar ist."""
    from backend.memory.self_correction import (
        SelfCorrectionAnalyzer,
        analyze_and_correct_failures
    )

    assert SelfCorrectionAnalyzer is not None
    assert callable(analyze_and_correct_failures)

    print("✅ Self-Correction Module importierbar")


def test_activity_tracker_exists():
    """Teste ob Activity Tracker vorhanden ist."""
    from backend.memory.activity_tracker import (
        track_activity,
        is_system_idle,
        ActivityTracker
    )

    # Record Activity
    track_activity()

    # Check Idle (sollte False sein nach Activity)
    idle = is_system_idle(minutes=0)
    assert isinstance(idle, bool)

    # Tracker-Instanz holen
    tracker = ActivityTracker()
    last_activity = tracker.get_last_activity()
    assert last_activity is not None

    print("✅ Activity Tracker funktioniert")


def test_heartbeat_stats_structure():
    """Teste ob Heartbeat Stats alle Felder hat."""
    from backend.services.heartbeat_memory import get_heartbeat_status

    status = get_heartbeat_status()

    required_fields = [
        "last_run",
        "last_consolidation",
        "last_synthesis",
        "last_correction",  # NEU für Phase 3
        "mode",
        "deleted_count",
        "consolidated_count",
        "synthesized_count",  # Phase 1
        "corrections_count",  # Phase 3
        "updated_count",
        "total_memories",
        "run_count",
        "errors"
    ]

    for field in required_fields:
        assert field in status, f"Feld {field} fehlt in Heartbeat Status"

    print("✅ Heartbeat Status hat alle Felder")


def test_phase_numbering():
    """Teste ob Phase-Nummerierung korrekt ist (durch Code-Inspektion)."""
    import inspect
    from backend.services.heartbeat_memory import run_deep_learning_tasks

    # Hole Source Code
    source = inspect.getsource(run_deep_learning_tasks)

    # Prüfe ob Phasen korrekt nummeriert sind
    assert "Phase 1: Memory Synthesis" in source
    assert "Phase 2: Memory Consolidation" in source
    assert "Phase 3: Self-Correction" in source
    assert "Phase 4: Update adaptive Relevance" in source
    assert "Phase 5: Intelligent Cleanup" in source

    print("✅ Phase-Nummerierung ist korrekt")


def test_stop_learning_flag():
    """Teste ob Stop-Learning Flag funktioniert."""
    from backend.services import heartbeat_memory

    # Initial sollte False sein
    assert heartbeat_memory._stop_learning == False
    assert heartbeat_memory._learning_in_progress == False

    # Stop setzen
    heartbeat_memory.stop_learning_processes()
    assert heartbeat_memory._stop_learning == True

    # Reset für andere Tests
    heartbeat_memory._stop_learning = False

    print("✅ Stop-Learning Flag funktioniert")


if __name__ == "__main__":
    print("\n🧪 Testing Phase 3 Integration...\n")

    try:
        test_stop_learning_functions_exist()
        test_corrections_in_heartbeat_status()
        test_self_correction_import()
        test_activity_tracker_exists()
        test_heartbeat_stats_structure()
        test_phase_numbering()
        test_stop_learning_flag()

        print("\n✅ Alle Tests bestanden!")
        print("\n📋 Phase 3 Integration ist vollständig:")
        print("   ✅ Stop-Learning Mechanismus")
        print("   ✅ Corrections-Tracking")
        print("   ✅ Self-Correction im Heartbeat")
        print("   ✅ Activity Tracking Middleware")
        print("   ✅ Phase-Nummerierung korrekt")

    except AssertionError as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
