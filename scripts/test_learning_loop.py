#!/usr/bin/env python3
"""
Test script for LexiAI Self-Learning Loop

This script tests the complete learning cycle:
1. Pattern detection
2. Goal tracking
3. Knowledge gap detection
4. Self-correction
5. Memory usage tracking

Usage:
    python scripts/test_learning_loop.py --verbose
    python scripts/test_learning_loop.py --pattern-only
    python scripts/test_learning_loop.py --correction-only
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.bootstrap import initialize_components
from backend.core.post_chat_learning import post_chat_learning
from backend.memory.pattern_detector import get_pattern_tracker
from backend.memory.goal_tracker import get_goal_tracker
from backend.memory.knowledge_gap_detector import get_knowledge_gap_tracker


async def test_pattern_detection(vectorstore, chat_client):
    """Test pattern detection from multiple related messages."""
    print("\n" + "="*60)
    print("TEST 1: PATTERN DETECTION")
    print("="*60)

    messages = [
        ("User: Ich interessiere mich für Python Programmierung", "Assistant: Python ist eine vielseitige Programmiersprache."),
        ("User: Was ist Flask?", "Assistant: Flask ist ein Web-Framework für Python."),
        ("User: Kannst du mir Python beibringen?", "Assistant: Gerne! Lass uns mit den Grundlagen beginnen."),
    ]

    for user_msg, ai_resp in messages:
        print(f"\n📝 Processing: {user_msg[:50]}...")
        stats = await post_chat_learning(
            user_message=user_msg,
            ai_response=ai_resp,
            user_id="test_user",
            retrieved_memories=[],
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=None
        )
        print(f"✅ Learning stats: {stats}")

    # Check if pattern was detected
    tracker = get_pattern_tracker(vectorstore)
    patterns = tracker.get_all_patterns("test_user")

    print(f"\n📊 RESULT: {len(patterns)} pattern(s) detected")
    for pattern in patterns:
        print(f"   - {pattern.name} (freq={pattern.frequency}, keywords={pattern.keywords[:5]})")

    return len(patterns) > 0


async def test_goal_tracking(vectorstore, chat_client):
    """Test goal extraction and tracking."""
    print("\n" + "="*60)
    print("TEST 2: GOAL TRACKING")
    print("="*60)

    messages = [
        ("User: Ich möchte Python lernen um Web-Apps zu entwickeln", "Assistant: Das ist ein großartiges Ziel!"),
        ("User: Mein Ziel ist es, bis Ende des Jahres eine App zu veröffentlichen", "Assistant: Sehr ambitioniert!"),
    ]

    for user_msg, ai_resp in messages:
        print(f"\n📝 Processing: {user_msg[:50]}...")
        stats = await post_chat_learning(
            user_message=user_msg,
            ai_response=ai_resp,
            user_id="test_user",
            retrieved_memories=[],
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=None
        )
        print(f"✅ Learning stats: {stats}")

    # Check if goals were tracked
    tracker = get_goal_tracker(vectorstore)
    goals = tracker.get_all_goals("test_user", status=None)

    print(f"\n📊 RESULT: {len(goals)} goal(s) tracked")
    for goal in goals:
        print(f"   - {goal.category}: {goal.content[:60]}...")

    return len(goals) > 0


async def test_knowledge_gap_detection(vectorstore, chat_client):
    """Test knowledge gap detection when AI lacks information."""
    print("\n" + "="*60)
    print("TEST 3: KNOWLEDGE GAP DETECTION")
    print("="*60)

    messages = [
        ("User: Was ist Quantum Computing?", "Assistant: Das weiß ich leider nicht genau."),
        ("User: Erkläre mir die neuesten AI Modelle", "Assistant: Ich habe keine aktuellen Informationen dazu."),
    ]

    for user_msg, ai_resp in messages:
        print(f"\n📝 Processing: {user_msg[:50]}...")
        stats = await post_chat_learning(
            user_message=user_msg,
            ai_response=ai_resp,
            user_id="test_user",
            retrieved_memories=[],  # Empty = no context found
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=None
        )
        print(f"✅ Learning stats: {stats}")

    # Check if gaps were detected
    tracker = get_knowledge_gap_tracker(vectorstore)
    gaps = tracker.get_all_gaps("test_user", include_dismissed=False)

    print(f"\n📊 RESULT: {len(gaps)} knowledge gap(s) detected")
    for gap in gaps:
        print(f"   - {gap.title[:60]}... (priority={gap.priority})")

    return len(gaps) > 0


async def test_self_correction(vectorstore, chat_client):
    """Test self-correction recording."""
    print("\n" + "="*60)
    print("TEST 4: SELF-CORRECTION")
    print("="*60)

    messages = [
        ("User: Nein, das ist falsch! Mein Name ist Thomas, nicht Tom!", "Assistant: Du hast recht, ich habe Tom gesagt."),
        ("User: Das stimmt nicht, Python wurde 1991 veröffentlicht, nicht 1989", "Assistant: Entschuldigung, ich korrigiere mich."),
    ]

    for user_msg, ai_resp in messages:
        print(f"\n📝 Processing: {user_msg[:50]}...")
        stats = await post_chat_learning(
            user_message=user_msg,
            ai_response=ai_resp,
            user_id="test_user",
            retrieved_memories=[],
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=None
        )
        print(f"✅ Learning stats: {stats}")

    # Check if corrections were stored
    all_memories = vectorstore.get_all_entries()
    correction_memories = [
        m for m in all_memories
        if m.metadata.get("category") == "self_correction"
    ]

    print(f"\n📊 RESULT: {len(correction_memories)} correction(s) recorded")
    for mem in correction_memories:
        print(f"   - {mem.content[:60]}... (relevance={mem.relevance})")

    return len(correction_memories) > 0


async def test_memory_usage_tracking(vectorstore, chat_client):
    """Test memory usage tracking."""
    print("\n" + "="*60)
    print("TEST 5: MEMORY USAGE TRACKING")
    print("="*60)

    # Create some test memories
    from backend.memory.adapter import store_memory_async

    print("\n📝 Creating test memories...")
    mem1_id, _ = await store_memory_async(
        content="Python is a programming language",
        user_id="test_user",
        tags=["programming"]
    )

    mem2_id, _ = await store_memory_async(
        content="Flask is a web framework for Python",
        user_id="test_user",
        tags=["python", "web"]
    )

    print(f"✅ Created memories: {mem1_id}, {mem2_id}")

    # Simulate retrieval and usage
    print("\n📝 Simulating memory retrieval and usage...")
    stats = await post_chat_learning(
        user_message="User: Tell me about Python",
        ai_response="Assistant: Python is a versatile programming language used for web development with frameworks like Flask.",
        user_id="test_user",
        retrieved_memories=[mem1_id, mem2_id],
        vectorstore=vectorstore,
        chat_client=chat_client,
        doc_id=None
    )

    print(f"✅ Learning stats: {stats}")

    # Check usage tracking
    from backend.memory.memory_intelligence import get_usage_tracker
    tracker = get_usage_tracker()

    print("\n📊 RESULT: Memory usage tracking")
    for mem_id in [mem1_id, mem2_id]:
        usage_stats = tracker.get_usage_stats(mem_id)
        print(f"   - {mem_id}: retrievals={usage_stats['retrievals']}, used={usage_stats['used_in_response']}, success_rate={usage_stats['success_rate']:.2f}")

    return stats["memories_tracked"] > 0


async def run_all_tests(args):
    """Run all tests."""
    print("\n" + "="*60)
    print("LEXIAI SELF-LEARNING LOOP TEST SUITE")
    print("="*60)

    # Initialize components
    print("\n🚀 Initializing components...")
    embeddings, vectorstore, memory, chat_client, _ = initialize_components()
    print("✅ Components initialized")

    results = {}

    # Run tests based on arguments
    if args.pattern_only:
        results["pattern_detection"] = await test_pattern_detection(vectorstore, chat_client)
    elif args.correction_only:
        results["self_correction"] = await test_self_correction(vectorstore, chat_client)
    elif args.goal_only:
        results["goal_tracking"] = await test_goal_tracking(vectorstore, chat_client)
    elif args.gap_only:
        results["knowledge_gap"] = await test_knowledge_gap_detection(vectorstore, chat_client)
    else:
        # Run all tests
        results["pattern_detection"] = await test_pattern_detection(vectorstore, chat_client)
        results["goal_tracking"] = await test_goal_tracking(vectorstore, chat_client)
        results["knowledge_gap"] = await test_knowledge_gap_detection(vectorstore, chat_client)
        results["self_correction"] = await test_self_correction(vectorstore, chat_client)
        results["memory_tracking"] = await test_memory_usage_tracking(vectorstore, chat_client)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name.replace('_', ' ').title()}")

    print("\n" + "="*60)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("="*60)

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Test LexiAI Self-Learning Loop")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pattern-only", action="store_true", help="Test pattern detection only")
    parser.add_argument("--correction-only", action="store_true", help="Test self-correction only")
    parser.add_argument("--goal-only", action="store_true", help="Test goal tracking only")
    parser.add_argument("--gap-only", action="store_true", help="Test knowledge gap detection only")

    args = parser.parse_args()

    # Run tests
    success = asyncio.run(run_all_tests(args))

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
