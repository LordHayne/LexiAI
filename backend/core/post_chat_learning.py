"""
Post-Chat Learning Module for LexiAI Self-Learning System

This module implements immediate learning after every chat interaction:
1. Pattern Detection - Identify user behavioral patterns
2. Goal Tracking - Track user objectives and progress
3. Knowledge Gap Detection - Identify when AI lacks knowledge
4. Self-Correction Recording - Learn from user corrections
5. Memory Usage Tracking - Track which memories are helpful

Executes in parallel to minimize latency (<200ms target).
"""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger("lexi_middleware.post_chat_learning")


async def post_chat_learning(
    user_message: str,
    ai_response: str,
    user_id: str,
    retrieved_memories: List[str],
    vectorstore,
    chat_client,
    doc_id: Optional[str] = None
) -> Dict[str, int]:
    """
    Execute immediate learning after chat interaction.

    Args:
        user_message: User's input message
        ai_response: AI's response
        user_id: User identifier
        retrieved_memories: List of memory IDs retrieved for this chat
        vectorstore: Vector store instance
        chat_client: LLM client for advanced analysis
        doc_id: ID of memory created for this chat (if any)

    Returns:
        Dict with counts of learning activities:
        {
            "patterns_detected": int,
            "goals_tracked": int,
            "knowledge_gaps_found": int,
            "corrections_recorded": int,
            "memories_tracked": int
        }
    """
    stats = {
        "patterns_detected": 0,
        "goals_tracked": 0,
        "knowledge_gaps_found": 0,
        "corrections_recorded": 0,
        "memories_tracked": 0
    }

    try:
        # Execute all learning tasks in parallel
        results = await asyncio.gather(
            _detect_and_store_patterns(user_message, user_id, vectorstore),
            _track_goals(user_message, user_id, vectorstore, chat_client, doc_id),
            _detect_knowledge_gaps(user_message, ai_response, retrieved_memories, user_id, vectorstore),
            _record_corrections(user_message, ai_response, user_id, vectorstore),
            _track_memory_usage(retrieved_memories, ai_response),
            return_exceptions=True  # Continue even if one task fails
        )

        # Unpack results
        stats["patterns_detected"] = results[0] if not isinstance(results[0], Exception) else 0
        stats["goals_tracked"] = results[1] if not isinstance(results[1], Exception) else 0
        stats["knowledge_gaps_found"] = results[2] if not isinstance(results[2], Exception) else 0
        stats["corrections_recorded"] = results[3] if not isinstance(results[3], Exception) else 0
        stats["memories_tracked"] = results[4] if not isinstance(results[4], Exception) else 0

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_names = ["pattern_detection", "goal_tracking", "knowledge_gap", "correction", "memory_tracking"]
                logger.error(f"Error in {task_names[i]}: {result}")

        logger.info(f"📚 Post-chat learning complete: {stats}")
        return stats

    except Exception as e:
        logger.error(f"❌ Error in post_chat_learning: {e}", exc_info=True)
        return stats


async def _detect_and_store_patterns(
    user_message: str,
    user_id: str,
    vectorstore
) -> int:
    """
    Detect behavioral patterns from user message.

    Returns:
        Number of patterns detected
    """
    try:
        from backend.memory.pattern_detector import get_pattern_tracker, PatternAnalyzer

        tracker = get_pattern_tracker(vectorstore)

        # Get recent memories for context (last 20 messages)
        all_memories = vectorstore.get_all_entries()
        user_memories = [m for m in all_memories if m.metadata.get("user_id") == user_id]
        recent_memories = sorted(user_memories, key=lambda m: m.timestamp, reverse=True)[:20]

        if len(recent_memories) < 3:
            # Not enough data for pattern detection
            return 0

        # Detect topic patterns (lower threshold for real-time detection)
        topic_patterns = PatternAnalyzer.detect_topic_patterns(
            memories=recent_memories,
            min_cluster_size=2,  # Lower than heartbeat (3)
            similarity_threshold=0.70  # Lower than heartbeat (0.75)
        )

        # Detect interest patterns
        interest_patterns = PatternAnalyzer.detect_interest_patterns(
            memories=recent_memories,
            min_frequency=2  # Lower than heartbeat (3)
        )

        all_patterns = topic_patterns + interest_patterns

        if not all_patterns:
            return 0

        # Get existing patterns for deduplication
        existing_patterns = tracker.get_all_patterns(user_id)
        existing_keyword_sets = {
            idx: set(pattern.keywords)
            for idx, pattern in enumerate(existing_patterns)
        }

        saved = 0

        for pattern in all_patterns[:3]:  # Max 3 per chat interaction
            # Check for duplicates
            pattern_keywords = set(pattern.keywords)
            is_duplicate = False

            for idx, existing in enumerate(existing_patterns):
                existing_keywords = existing_keyword_sets[idx]

                # Jaccard similarity
                intersection = len(pattern_keywords & existing_keywords)
                union = len(pattern_keywords | existing_keywords)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > 0.6:
                    is_duplicate = True
                    break

            if not is_duplicate:
                if tracker.save_pattern(pattern):
                    saved += 1
                    logger.info(f"🔍 Real-time pattern detected: {pattern.name}")

        return saved

    except Exception as e:
        logger.error(f"Error in pattern detection: {e}", exc_info=True)
        return 0


async def _track_goals(
    user_message: str,
    user_id: str,
    vectorstore,
    chat_client,
    doc_id: Optional[str]
) -> int:
    """
    Track goals mentioned in user message.

    Returns:
        Number of goals tracked
    """
    try:
        from backend.memory.goal_tracker import get_goal_tracker, GoalDetector

        tracker = get_goal_tracker(vectorstore)

        # Use LLM-based goal detection
        detected_goals = await GoalDetector.detect_goals_with_llm(
            text=user_message,
            user_id=user_id,
            chat_client=chat_client
        )

        if not detected_goals:
            return 0

        tracked = 0

        for goal in detected_goals:
            # Link to source memory if available
            if doc_id:
                goal.source_memory_ids = [doc_id]

            success = tracker.add_goal(goal)
            if success:
                tracked += 1
                logger.info(f"🎯 Goal tracked: {goal.category} - {goal.content[:50]}...")

        return tracked

    except Exception as e:
        logger.error(f"Error in goal tracking: {e}", exc_info=True)
        return 0


async def _detect_knowledge_gaps(
    user_query: str,
    ai_response: str,
    retrieved_memories: List[str],
    user_id: str,
    vectorstore
) -> int:
    """
    Detect if AI encountered a knowledge gap.

    Returns:
        1 if gap detected, 0 otherwise
    """
    try:
        from backend.memory.knowledge_gap_detector import (
            get_knowledge_gap_tracker,
            KnowledgeGap
        )

        # Signals indicating knowledge gap
        gap_signals = [
            "weiß nicht",
            "weiß ich nicht",
            "kann ich nicht",
            "habe keine information",
            "habe keine daten",
            "don't know",
            "cannot answer",
            "i don't have",
            "keine ahnung"
        ]

        has_gap = (
            any(signal in ai_response.lower() for signal in gap_signals) or
            (len(retrieved_memories) == 0 and len(user_query) > 20)  # Long query with no context
        )

        if not has_gap:
            return 0

        gap_tracker = get_knowledge_gap_tracker(vectorstore)

        # Create knowledge gap entry
        gap = KnowledgeGap(
            title=f"Fehlende Information: {user_query[:100]}",
            description=f"User fragte: '{user_query}'. AI konnte nicht vollständig antworten.",
            category="realtime_detection",
            priority=0.7,  # Medium-high priority
            user_id=user_id,
            related_topics=[],
            suggested_research=user_query[:200]
        )

        # Check for duplicates
        existing_gaps = gap_tracker.get_all_gaps(user_id, include_dismissed=False)

        for existing in existing_gaps:
            # Simple title similarity check
            title_words = set(gap.title.lower().split())
            existing_words = set(existing.title.lower().split())

            intersection = len(title_words & existing_words)
            union = len(title_words | existing_words)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > 0.7:
                # Duplicate gap, skip
                return 0

        success = gap_tracker.save_gap(gap)
        if success:
            logger.info(f"🧠 Knowledge gap detected: {gap.title}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error in knowledge gap detection: {e}", exc_info=True)
        return 0


async def _record_corrections(
    user_message: str,
    ai_response: str,
    user_id: str,
    vectorstore
) -> int:
    """
    Detect and record user corrections.

    Returns:
        1 if correction detected and recorded, 0 otherwise
    """
    try:
        # Correction signals
        correction_signals = [
            "nein, das ist falsch",
            "das stimmt nicht",
            "das ist nicht richtig",
            "nein, mein",  # "Nein, mein Name ist..."
            "falsch",
            "incorrect",
            "nicht korrekt",
            "that's wrong",
            "nicht richtig"
        ]

        has_correction = any(signal in user_message.lower() for signal in correction_signals)

        if not has_correction:
            return 0

        from backend.memory.adapter import store_memory_async

        # Create high-priority correction memory
        correction_content = f"""SELBST-KORREKTUR:
User hat AI korrigiert.

User Message: {user_message}
AI Response (vorher): {ai_response[:300]}

Dies ist eine wichtige Korrektur und muss bei zukünftigen Antworten berücksichtigt werden.
"""

        # Store with maximum relevance and special category
        doc_id, timestamp = await store_memory_async(
            content=correction_content,
            user_id=user_id,
            tags=["self_correction", "user_feedback", "high_priority"],
            metadata={
                "category": "self_correction",
                "relevance": 1.0,  # Maximum priority
                "is_correction": True,
                "corrected_at": datetime.now(timezone.utc).isoformat()
            }
        )

        if doc_id:
            logger.info(f"✅ Correction recorded with ID: {doc_id}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error in correction recording: {e}", exc_info=True)
        return 0


async def _track_memory_usage(
    retrieved_memory_ids: List[str],
    ai_response: str
) -> int:
    """
    Track usage of retrieved memories.

    Returns:
        Number of memories tracked
    """
    try:
        from backend.memory.memory_intelligence import get_usage_tracker

        if not retrieved_memory_ids:
            return 0

        usage_tracker = get_usage_tracker()

        # Simple heuristic: if response is long and detailed, memories were helpful
        # More sophisticated: could use LLM to judge if memories were used
        was_helpful = len(ai_response) > 50  # Basic heuristic

        tracked = 0

        for memory_id in retrieved_memory_ids:
            if memory_id:
                usage_tracker.track_usage_in_response(memory_id, was_helpful)
                tracked += 1

        if tracked > 0:
            logger.debug(f"📊 Tracked {tracked} memory usages (helpful={was_helpful})")

        return tracked

    except Exception as e:
        logger.error(f"Error in memory usage tracking: {e}", exc_info=True)
        return 0


# Helper function for integration
async def integrate_post_chat_learning(
    user_message: str,
    ai_response: str,
    user_id: str,
    retrieved_memory_ids: List[str],
    vectorstore,
    chat_client,
    doc_id: Optional[str] = None
) -> None:
    """
    Wrapper function for easy integration into chat processing.

    Call this after memory storage in chat_processing.py:

    Example:
        await integrate_post_chat_learning(
            user_message=message,
            ai_response=response_content,
            user_id=user_id,
            retrieved_memory_ids=retrieved_memory_ids,
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=doc_id
        )
    """
    try:
        stats = await post_chat_learning(
            user_message=user_message,
            ai_response=ai_response,
            user_id=user_id,
            retrieved_memories=retrieved_memory_ids,
            vectorstore=vectorstore,
            chat_client=chat_client,
            doc_id=doc_id
        )

        # Log summary
        total_learning = sum(stats.values())
        if total_learning > 0:
            logger.info(
                f"📚 Learning summary: {total_learning} activities "
                f"(patterns={stats['patterns_detected']}, "
                f"goals={stats['goals_tracked']}, "
                f"gaps={stats['knowledge_gaps_found']}, "
                f"corrections={stats['corrections_recorded']}, "
                f"tracked={stats['memories_tracked']})"
            )

    except Exception as e:
        logger.error(f"Error in post-chat learning integration: {e}", exc_info=True)
        # Don't fail the chat request if learning fails
