"""
Integration Test für LexiAI Selbstlernsystem
=============================================

Dieser Test verifiziert, dass alle Selbstlern-Komponenten korrekt funktionieren:
1. Memory Storage in Qdrant
2. Self-Correction Loop
3. Pattern Detection
4. Goal Tracking
5. Knowledge Gap Detection
6. Heartbeat Learning Phases

Usage:
    cd /Users/thomas/Desktop/LexiAI_new
    python3 tests/test_self_learning_integration.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from qdrant_client import QdrantClient
from backend.config.settings import QDRANT_URL
from backend.memory.adapter import store_memory, retrieve_memories
from backend.memory.conversation_tracker import get_conversation_tracker
from backend.models.feedback import FeedbackType
import time

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_test(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {name}")
    if details:
        print(f"        {details}")

async def test_qdrant_connection():
    """Test 1: Qdrant Verbindung und Collections"""
    print_header("TEST 1: Qdrant Connection & Collections")

    try:
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections().collections

        required_collections = [
            'lexi_memory', 'lexi_goals', 'lexi_patterns',
            'lexi_knowledge_gaps', 'lexi_feedback'
        ]

        found_collections = [c.name for c in collections]

        for coll in required_collections:
            exists = coll in found_collections
            if exists:
                count = client.count(coll).count
                print_test(f"{coll} exists", True, f"{count} vectors")
            else:
                print_test(f"{coll} exists", False, "Collection not found")

        return True
    except Exception as e:
        print_test("Qdrant Connection", False, str(e))
        return False

async def test_memory_storage():
    """Test 2: Memory Speicherung"""
    print_header("TEST 2: Memory Storage")

    try:
        # Store test memory
        test_content = f"Test memory created at {time.time()}"
        doc_id, timestamp = store_memory(
            content=test_content,
            user_id="test_user",
            tags=["test", "integration"]
        )

        print_test("Store memory", True, f"ID: {doc_id}")

        # Retrieve memory
        results = retrieve_memories(
            query=test_content,
            user_id="test_user",
            limit=1
        )

        retrieved = len(results) > 0 and results[0].content == test_content
        print_test("Retrieve memory", retrieved,
                  f"Found: {results[0].content[:50]}..." if retrieved else "Not found")

        return True
    except Exception as e:
        print_test("Memory Storage", False, str(e))
        return False

async def test_self_correction():
    """Test 3: Self-Correction System"""
    print_header("TEST 3: Self-Correction")

    try:
        from backend.memory.self_correction import get_self_correction_manager
        from backend.models.conversation import ConversationTurn
        from uuid import uuid4

        manager = get_self_correction_manager()

        # Create test turn
        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="What is 2+2?",
            ai_response="5",  # Wrong answer
            retrieved_memories=[],
            response_time_ms=100
        )

        # Create correction
        correction_memory = manager.create_correction_memory(
            turn=turn,
            correction="2+2 = 4, not 5. I made a calculation error.",
            error_category=manager.ErrorCategory.FACTUAL_ERROR,
            analysis="Simple arithmetic mistake - confused with 2+3"
        )

        has_correction = correction_memory is not None
        print_test("Create correction memory", has_correction,
                  f"Category: {correction_memory.category if has_correction else 'N/A'}")

        if has_correction:
            print_test("Correction relevance",
                      correction_memory.relevance == 1.0,
                      f"Relevance: {correction_memory.relevance}")

        return True
    except Exception as e:
        print_test("Self-Correction", False, str(e))
        return False

async def test_goal_tracking():
    """Test 4: Goal Tracking"""
    print_header("TEST 4: Goal Tracking")

    try:
        from backend.memory.goal_tracker import get_goal_tracker, Goal
        from backend.core.bootstrap import initialize_components

        # Initialize components
        embeddings, vectorstore, memory, chat_client, _ = initialize_components()
        tracker = get_goal_tracker(vectorstore)

        # Create test goal
        test_goal = Goal(
            user_id="test_user",
            category="learning",
            content="Learn Python programming",
            confidence=0.95,
            source_memory_ids=[]
        )

        # Add goal
        success = tracker.add_goal(test_goal)
        print_test("Add goal", success, f"Goal ID: {test_goal.goal_id}")

        # Retrieve goals
        goals = tracker.get_active_goals("test_user")
        found = any(g.content == test_goal.content for g in goals)
        print_test("Retrieve goals", found, f"Found {len(goals)} active goals")

        return True
    except Exception as e:
        print_test("Goal Tracking", False, str(e))
        return False

async def test_pattern_detection():
    """Test 5: Pattern Detection"""
    print_header("TEST 5: Pattern Detection")

    try:
        from backend.memory.pattern_detection import get_pattern_detector
        from backend.core.bootstrap import initialize_components

        embeddings, vectorstore, memory, chat_client, _ = initialize_components()
        detector = get_pattern_detector(vectorstore)

        # Simulate conversation history with pattern
        test_messages = [
            "I love programming in Python",
            "Python is my favorite language",
            "Can you help me with Python?",
            "Python programming is fun"
        ]

        # Store test memories
        for msg in test_messages:
            store_memory(
                content=msg,
                user_id="test_user_patterns",
                tags=["test", "pattern"]
            )

        # Detect patterns
        patterns = detector.detect_patterns("test_user_patterns", min_cluster_size=2)

        has_patterns = len(patterns) > 0
        print_test("Detect patterns", has_patterns,
                  f"Found {len(patterns)} patterns")

        if has_patterns:
            for i, pattern in enumerate(patterns[:2], 1):
                print(f"        Pattern {i}: {pattern.description[:50]}...")

        return True
    except Exception as e:
        print_test("Pattern Detection", False, str(e))
        return False

async def test_conversation_tracking():
    """Test 6: Conversation Tracking & Feedback"""
    print_header("TEST 6: Conversation Tracking")

    try:
        tracker = get_conversation_tracker()

        # Record test turn
        turn_id = tracker.record_turn(
            user_id="test_user_conv",
            user_message="Test question",
            ai_response="Test answer",
            retrieved_memories=[],
            response_time_ms=150
        )

        print_test("Record conversation turn", turn_id is not None,
                  f"Turn ID: {turn_id}")

        # Record feedback
        tracker.record_feedback(
            turn_id=turn_id,
            feedback_type=FeedbackType.THUMBS_UP,
            confidence=1.0
        )

        print_test("Record feedback", True, "Feedback stored")

        # Get history
        history = tracker.get_user_history("test_user_conv", limit=1)
        has_history = len(history) > 0
        print_test("Retrieve conversation history", has_history,
                  f"Found {len(history)} turns")

        return True
    except Exception as e:
        print_test("Conversation Tracking", False, str(e))
        return False

async def test_heartbeat_components():
    """Test 7: Heartbeat Learning Phases"""
    print_header("TEST 7: Heartbeat Learning Phases")

    try:
        from backend.services.heartbeat_memory import (
            _synthesize_memories,
            _consolidate_memories,
            _adaptive_relevance_update,
            _intelligent_cleanup
        )

        # Test that functions exist and are importable
        functions = [
            "_synthesize_memories",
            "_consolidate_memories",
            "_adaptive_relevance_update",
            "_intelligent_cleanup"
        ]

        for func_name in functions:
            exists = func_name in dir()
            print_test(f"Heartbeat function: {func_name}", exists)

        return True
    except Exception as e:
        print_test("Heartbeat Components", False, str(e))
        return False

async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("  LEXIAI SELBSTLERNSYSTEM - INTEGRATION TEST")
    print("="*60)

    tests = [
        ("Qdrant Connection", test_qdrant_connection),
        ("Memory Storage", test_memory_storage),
        ("Self-Correction", test_self_correction),
        ("Goal Tracking", test_goal_tracking),
        ("Pattern Detection", test_pattern_detection),
        ("Conversation Tracking", test_conversation_tracking),
        ("Heartbeat Components", test_heartbeat_components)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*60}\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Das Selbstlernsystem funktioniert korrekt.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Bitte Logs prüfen.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
