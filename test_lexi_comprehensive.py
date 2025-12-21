#!/usr/bin/env python3
"""
Comprehensive LexiAI System Test
=================================

Testet alle Kernfunktionen von LexiAI ohne Server zu starten:
1. Qdrant Storage & Retrieval
2. Chat Processing
3. Self-Learning Komponenten
4. Background Processes

Usage:
    python test_lexi_comprehensive.py
"""

import sys
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("  LEXIAI COMPREHENSIVE SYSTEM TEST")
print("="*60)
print()

# Test Results
results = []

def report_test(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((name, passed))
    print(f"{status} | {name}")
    if details:
        print(f"        {details}")


async def test_1_qdrant_storage():
    """Test 1: Qdrant Speicherung & Abruf"""
    print("\n" + "="*60)
    print("TEST 1: Qdrant Storage & Retrieval")
    print("="*60 + "\n")

    try:
        from qdrant_client import QdrantClient
        from backend.config.settings import QDRANT_URL

        client = QdrantClient(url=QDRANT_URL)

        # Check collections
        collections = client.get_collections().collections
        coll_names = [c.name for c in collections]

        report_test("Qdrant Connection", True, f"{len(collections)} collections found")

        # Test each collection
        for coll in ['lexi_memory', 'lexi_goals', 'lexi_patterns', 'lexi_knowledge_gaps']:
            exists = coll in coll_names
            if exists:
                count = client.count(coll).count
                report_test(f"Collection: {coll}", True, f"{count} vectors")
            else:
                report_test(f"Collection: {coll}", False, "Not found")

        # Test storing a vector
        test_id = f"test_{int(time.time())}"
        from backend.memory.adapter import store_memory

        doc_id, timestamp = store_memory(
            content="Test memory for comprehensive evaluation",
            user_id="test_eval",
            tags=["evaluation", "test"]
        )

        report_test("Store Memory", doc_id is not None, f"ID: {doc_id}")

        # Test retrieving
        from backend.memory.adapter import retrieve_memories

        results_mem = retrieve_memories(
            query="evaluation test",
            user_id="test_eval",
            limit=5
        )

        found = any(r.content == "Test memory for comprehensive evaluation" for r in results_mem)
        report_test("Retrieve Memory", found, f"Found {len(results_mem)} results")

        return True

    except Exception as e:
        report_test("Qdrant Storage Test", False, str(e))
        return False


async def test_2_chat_processing():
    """Test 2: Chat Processing"""
    print("\n" + "="*60)
    print("TEST 2: Chat Processing")
    print("="*60 + "\n")

    try:
        from backend.core.chat_processing import process_chat_message_async
        from backend.core.bootstrap import initialize_components

        # Initialize components
        embeddings, vectorstore, memory, chat_client, _ = initialize_components()
        report_test("Initialize Components", True)

        # Process a test message
        response = await process_chat_message_async(
            message="Was ist 2+2?",
            chat_client=chat_client,
            vectorstore=vectorstore,
            memory=memory,
            embeddings=embeddings,
            user_id="test_eval"
        )

        has_response = 'response' in response and len(response['response']) > 0
        report_test("Chat Processing", has_response, f"Response length: {len(response.get('response', ''))}")

        # Check if memory was saved
        has_memory_id = 'memory_saved_id' in response and response['memory_saved_id'] is not None
        report_test("Memory Auto-Save", has_memory_id, f"Memory ID: {response.get('memory_saved_id', 'None')}")

        return True

    except Exception as e:
        report_test("Chat Processing Test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_3_self_learning_components():
    """Test 3: Self-Learning Komponenten"""
    print("\n" + "="*60)
    print("TEST 3: Self-Learning Components")
    print("="*60 + "\n")

    try:
        # Test Goal Tracker
        from backend.memory.goal_tracker import get_goal_tracker, Goal
        from backend.core.bootstrap import initialize_components

        embeddings, vectorstore, memory, chat_client, _ = initialize_components()
        tracker = get_goal_tracker(vectorstore)

        test_goal = Goal(
            user_id="test_eval",
            category="testing",
            content="Complete comprehensive evaluation",
            confidence=0.95
        )

        success = tracker.add_goal(test_goal)
        report_test("Goal Tracking: Add", success, f"Goal ID: {test_goal.goal_id}")

        goals = tracker.get_active_goals("test_eval")
        found = any(g.content == test_goal.content for g in goals)
        report_test("Goal Tracking: Retrieve", found, f"{len(goals)} active goals")

        # Test Self-Correction
        from backend.memory.self_correction import get_self_correction_manager
        from backend.models.conversation import ConversationTurn
        from uuid import uuid4

        manager = get_self_correction_manager()

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_eval",
            user_message="What is the capital of France?",
            ai_response="Berlin",  # Wrong!
            retrieved_memories=[],
            response_time_ms=100
        )

        correction = manager.create_correction_memory(
            turn=turn,
            correction="The capital of France is Paris, not Berlin.",
            error_category=manager.ErrorCategory.FACTUAL_ERROR,
            analysis="Geographic error - confused Germany and France capitals"
        )

        report_test("Self-Correction: Create", correction is not None)
        report_test("Self-Correction: Priority", correction.relevance == 1.0, f"Relevance: {correction.relevance}")

        # Test Pattern Detector
        from backend.memory.pattern_detector import get_pattern_tracker

        pattern_tracker = get_pattern_tracker(vectorstore)
        report_test("Pattern Detection: Initialize", pattern_tracker is not None)

        # Test Knowledge Gap Detector
        from backend.memory.knowledge_gap_detector import get_knowledge_gap_tracker

        gap_tracker = get_knowledge_gap_tracker(vectorstore)
        report_test("Knowledge Gap Detection: Initialize", gap_tracker is not None)

        # Test Conversation Tracker
        from backend.memory.conversation_tracker import get_conversation_tracker

        conv_tracker = get_conversation_tracker()
        turn_id = conv_tracker.record_turn(
            user_id="test_eval",
            user_message="Test question",
            ai_response="Test answer",
            retrieved_memories=[],
            response_time_ms=150
        )
        report_test("Conversation Tracking", turn_id is not None, f"Turn ID: {turn_id}")

        return True

    except Exception as e:
        report_test("Self-Learning Test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_4_heartbeat_service():
    """Test 4: Heartbeat Service"""
    print("\n" + "="*60)
    print("TEST 4: Heartbeat Service")
    print("="*60 + "\n")

    try:
        from backend.services.heartbeat_memory import (
            _heartbeat_state,
            get_heartbeat_stats
        )

        # Check heartbeat service functions exist
        functions = [
            "_synthesize_memories",
            "_consolidate_memories",
            "_adaptive_relevance_update",
            "_intelligent_cleanup"
        ]

        from backend.services import heartbeat_memory

        for func_name in functions:
            exists = hasattr(heartbeat_memory, func_name)
            report_test(f"Heartbeat: {func_name}", exists)

        # Check heartbeat state
        report_test("Heartbeat State", _heartbeat_state is not None)

        return True

    except Exception as e:
        report_test("Heartbeat Service Test", False, str(e))
        return False


async def run_all_tests():
    """Run all tests"""

    # Test 1: Qdrant
    await test_1_qdrant_storage()

    # Test 2: Chat Processing
    await test_2_chat_processing()

    # Test 3: Self-Learning
    await test_3_self_learning_components()

    # Test 4: Heartbeat
    await test_4_heartbeat_service()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60 + "\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! LexiAI funktioniert korrekt!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
