"""
Memory Handler Module for LexiAI Chat Processing

This module centralizes all memory-related operations including:
- Memory retrieval with circular filtering
- Smart home pattern-based storage decisions
- Conversation memory storage
- Turn recording for conversation tracking

Extracted from chat_processing_with_tools.py (Lines 148-776)
"""

import asyncio
import logging
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


async def retrieve_and_filter_memories(
    clean_message: str,
    vectorstore,
    user_id: str = "default"
) -> List:
    """
    Retrieve memories with intelligent circular filtering and prioritization.

    Args:
        clean_message: The cleaned user message to query
        vectorstore: The vector database interface
        user_id: User identifier for memory isolation

    Returns:
        List of filtered and prioritized relevant documents (max 5)

    Original code: chat_processing_with_tools.py lines 148-255
    """
    logger.info(f"🔍 Retrieving memory context for: '{clean_message[:50]}'")
    relevant_docs = []

    if len(clean_message.strip()) < 8:
        return relevant_docs
    meta_style_query = any(token in clean_message.lower() for token in [
        "roboterhaft", "klingt roboterhaft", "antwortgefühl", "antwort klang", "stil",
        "sag mir warum", "warum klingt", "why does it sound"
    ])
    if meta_style_query:
        logger.info("🧾 Skipping memory retrieval for meta-style question")
        return relevant_docs

    try:
        # Retrieve many memories (k=15) to ensure we have enough after filtering
        all_docs = await asyncio.to_thread(
            vectorstore.query_memories,
            clean_message,
            user_id=user_id,
            limit=15
        )

        # Filter out circular/self-referential memories EARLY
        non_circular_docs = []
        circular_count = 0

        # Indicators that a memory contains facts (not just a question)
        fact_indicators = [
            "heißt", "heiße", "name ist", "my name", "I am", "I'm",
            "Besitzer", "owner", "genannt",
            "Geburtstag", "birthday", "wohne in", "live in", "arbeite bei", "work at",
            "ich bin der", "ich bin ein", "das ist mein"
        ]

        # Questions that are typically circular when asking about memory
        circular_patterns = [
            "weißt du", "erinnerst du", "do you remember", "do you know",
            "kannst du dich erinnern", "was weißt du", "what do you know"
        ]
        question_starters = [
            "wie", "was", "wer", "wo", "wann", "warum", "wieso", "weshalb",
            "welche", "welcher", "welches", "kannst", "weißt", "erinnerst",
            "how", "what", "who", "where", "when", "why", "which", "do you"
        ]

        for doc in all_docs:
            content = getattr(doc, 'page_content', str(doc))
            content_lower = content.lower()

            # Check if it's a conversation memory
            is_conversation = "User:" in content and "Assistant:" in content

            if is_conversation:
                # Split into user and assistant parts
                parts = content.split("Assistant:", 1)
                user_part = parts[0].replace("User:", "").strip()

                user_part_lower = user_part.lower()
                is_question = user_part.strip().endswith("?") or any(
                    user_part_lower.startswith(starter) for starter in question_starters
                )
                # Check if user part contains factual information
                has_facts = (not is_question) and any(
                    indicator.lower() in user_part_lower for indicator in fact_indicators
                )

                # Check if it's just a circular question
                is_circular_question = any(pattern in content_lower
                                          for pattern in circular_patterns)

                # Keep if it has facts, even if it's also a question
                if has_facts:
                    non_circular_docs.append(doc)
                    logger.info(f"✓ Keeping factual memory: {user_part[:50]}...")
                elif is_circular_question and not has_facts:
                    circular_count += 1
                    logger.info(f"✗ Filtering circular question: {user_part[:50]}...")
                else:
                    non_circular_docs.append(doc)
                    logger.info(f"~ Keeping other conversation: {user_part[:50]}...")
            elif content_lower.startswith("q:") and "a:" in content_lower:
                # Handle Q/A formatted memories
                qa_parts = content.split("A:", 1)
                user_part = qa_parts[0].replace("Q:", "").strip()

                user_part_lower = user_part.lower()
                is_question = user_part.strip().endswith("?") or any(
                    user_part_lower.startswith(starter) for starter in question_starters
                )
                has_facts = (not is_question) and any(
                    indicator.lower() in user_part_lower for indicator in fact_indicators
                )
                is_circular_question = any(pattern in user_part_lower
                                          for pattern in circular_patterns)

                if has_facts:
                    non_circular_docs.append(doc)
                    logger.info(f"✓ Keeping factual QA memory: {user_part[:50]}...")
                elif is_circular_question and not has_facts:
                    circular_count += 1
                    logger.info(f"✗ Filtering circular QA question: {user_part[:50]}...")
                else:
                    non_circular_docs.append(doc)
                    logger.info(f"~ Keeping other QA memory: {user_part[:50]}...")
            else:
                # Not a conversation format, keep it
                non_circular_docs.append(doc)

        if circular_count > 0:
            logger.info(f"⚡ Filtered {circular_count} circular memories, "
                       f"{len(non_circular_docs)} remain")

        # Now prioritize from the non-circular memories
        correction_docs = []
        factual_docs = []
        other_docs = []

        for doc in non_circular_docs:
            content = getattr(doc, 'page_content', str(doc))
            content_lower = content.lower()

            # Check category first
            if doc.metadata.get("category") == "self_correction":
                correction_docs.append(doc)
            # Then check if it's factual
            elif any(indicator.lower() in content_lower
                    for indicator in fact_indicators):
                factual_docs.append(doc)
            else:
                other_docs.append(doc)

        # Priority: corrections > factual > other
        identity_query = any(token in clean_message.lower() for token in [
            "wie ich heiße", "mein name", "wer bin ich", "beruf", "arbeite", "job", "profession",
            "über mich", "ueber mich", "information", "informationen"
        ])

        if identity_query:
            personal_indicators = [
                "ich heiße", "mein name", "ich bin", "arbeite", "beruf", "job", "profession"
            ]
            smart_home_indicators = [
                "licht", "lampe", "heizung", "thermostat", "schalte", "wohnzimmer", "badezimmer",
                "sensor", "home assistant"
            ]
            personal_docs = []
            for doc in factual_docs:
                content = getattr(doc, 'page_content', str(doc))
                content_lower = content.lower()
                has_personal = any(ind in content_lower for ind in personal_indicators)
                has_smart_home = any(ind in content_lower for ind in smart_home_indicators)
                if has_personal and not has_smart_home:
                    personal_docs.append(doc)

            # Strongly prefer personal-only memories for identity queries
            relevant_docs = (correction_docs[:3] +
                            (personal_docs[:5] if personal_docs else []))
            if not relevant_docs:
                relevant_docs = (correction_docs[:3] + factual_docs[:5])
        else:
            relevant_docs = (correction_docs[:3] +
                            factual_docs[:3] +
                            other_docs[:2])
        relevant_docs = relevant_docs[:5]

        logger.info(f"📊 After prioritization: {len(relevant_docs)} docs in relevant_docs")
        if relevant_docs:
            for i, doc in enumerate(relevant_docs[:2]):
                content = getattr(doc, 'page_content', str(doc))
                logger.info(f"   Doc {i+1}: {content[:80]}...")

        if correction_docs:
            logger.info(f"✨ Found {len(correction_docs)} correction memories - prioritizing")

    except Exception as e:
        logger.error(f"Error in memory search: {e}")

    return relevant_docs


async def store_conversation_memory(
    clean_message: str,
    response_content: str,
    tool_results: List,
    user_id: str,
    is_german: bool,
    memory: any  # langchain ConversationBufferMemory
) -> Optional[Tuple[str, str]]:
    """
    Store conversation in memory with intelligent smart home pattern handling.

    This function unifies the duplicated storage logic from two locations in the
    original code (lines 590-644 and 704-765).

    Args:
        clean_message: The cleaned user message
        response_content: The AI response
        tool_results: List of tool execution results
        user_id: User identifier
        is_german: Whether the conversation is in German
        memory: langchain ConversationBufferMemory instance

    Returns:
        Tuple of (doc_id, timestamp) if stored, None otherwise

    Original code: chat_processing_with_tools.py lines 590-644 & 704-765 (UNIFIED)
    """
    from backend.memory.adapter import store_memory
    from backend.core.chat_processing_with_tools import classify_smart_home_storage_strategy

    # Save to langchain memory buffer
    memory.save_context({"input": clean_message}, {"output": response_content})

    # SMART HOME: Intelligent storage decision
    should_store, reason, entity_id = classify_smart_home_storage_strategy(
        clean_message, tool_results
    )

    if not should_store:
        # Only pattern tracking, no full text storage
        from backend.services.smart_home_pattern_aggregator import get_pattern_aggregator

        aggregator = get_pattern_aggregator()

        # Extract action from message
        message_lower = clean_message.lower()
        if "ein" in message_lower or "an" in message_lower:
            action = "turn_on"
        elif "aus" in message_lower or "ab" in message_lower:
            action = "turn_off"
        else:
            action = "toggle"

        aggregator.track_simple_action(
            entity_id=entity_id or "unknown",
            action=action,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )

        logger.info(f"📊 Smart Home Pattern tracked (reason: {reason}), "
                   "nicht in Qdrant gespeichert")
        return None
    else:
        # Full text storage
        async def memory_store_task():
            try:
                # Better structured content format
                memory_content = f"Q: {clean_message}\nA: {response_content}"

                # Add metadata for better categorization
                metadata = {
                    "interaction_type": "smart_home" if entity_id else "conversation",
                    "storage_reason": reason,
                    "language": "de" if is_german else "en",
                    "message_length": len(clean_message),
                    "response_length": len(response_content)
                }

                if entity_id:
                    metadata["entity_id"] = entity_id

                doc_id, ts = await asyncio.to_thread(
                    store_memory,
                    content=memory_content,
                    user_id=user_id,
                    tags=["smart_home", "conversation", reason] if entity_id
                         else ["chat", "conversation"],
                    metadata=metadata
                )
                logger.info(f"💾 Memory stored (reason: {reason}): {doc_id}")
                return doc_id, ts
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                return None, None

        # Create task and return immediately (async storage)
        asyncio.create_task(memory_store_task())
        return None  # Don't wait for storage to complete


def record_conversation_turn(
    user_id: str,
    user_message: str,
    ai_response: str,
    relevant_docs: List
) -> str:
    """
    Record conversation turn for tracking and analytics.

    This function unifies the duplicated turn recording logic from two locations
    in the original code (lines 646-655 and 767-776).

    Args:
        user_id: User identifier
        user_message: The user's message
        ai_response: The AI's response
        relevant_docs: List of relevant memory documents used

    Returns:
        Turn ID string

    Original code: chat_processing_with_tools.py lines 646-655 & 767-776 (UNIFIED)
    """
    from backend.memory.conversation_tracker import get_conversation_tracker

    conversation_tracker = get_conversation_tracker()
    turn_id = conversation_tracker.record_turn(
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        retrieved_memories=[doc.metadata.get("id", "")
                           for doc in relevant_docs
                           if doc.metadata.get("id")],
        response_time_ms=None
    )
    logger.info(f"Recorded conversation turn: {turn_id}")
    return turn_id
