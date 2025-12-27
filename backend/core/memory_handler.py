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
import os
import re
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _get_memory_threshold() -> float:
    value = os.environ.get("LEXI_MEMORY_THRESHOLD", "0.8")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 0.8
    return min(max(threshold, 0.0), 1.0)


def _get_fact_confidence() -> float:
    value = os.environ.get("LEXI_FACT_CONFIDENCE", "0.85")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.85
    return min(max(confidence, 0.0), 1.0)


def _get_fact_min_confidence() -> float:
    value = os.environ.get("LEXI_FACT_MIN_CONFIDENCE", "0.6")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.6
    return min(max(confidence, 0.0), 1.0)


def _get_fact_ttl_days() -> int:
    value = os.environ.get("LEXI_FACT_TTL_DAYS", "365")
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = 365
    return max(days, 0)


async def _supersede_existing_facts(
    user_id: str,
    fact_kind: str,
    fact_category: Optional[str],
    new_fact_id: str
) -> None:
    try:
        from backend.core.component_cache import get_cached_components
        from backend.qdrant.client_wrapper import safe_scroll
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        bundle = get_cached_components()
        vectorstore = bundle.vectorstore

        must_filters = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="category", match=MatchValue(value="user_fact")),
            FieldCondition(key="fact_kind", match=MatchValue(value=fact_kind)),
        ]
        if fact_category:
            must_filters.append(FieldCondition(key="fact_category", match=MatchValue(value=fact_category)))

        points, _next_offset = safe_scroll(
            collection_name=vectorstore.collection,
            scroll_filter=Filter(must=must_filters),
            limit=50,
            with_payload=True,
            with_vectors=False
        )

        now = datetime.now(timezone.utc).isoformat()
        for point in points:
            if str(point.id) == str(new_fact_id):
                continue
            vectorstore.update_entry_metadata(
                point.id,
                {
                    "superseded": True,
                    "superseded_at": now,
                    "superseded_by": str(new_fact_id),
                    "fact_active": False
                }
            )
    except Exception as e:
        logger.warning(f"Failed to supersede existing facts: {e}")


def _tokenize(text: str) -> List[str]:
    tokens = []
    for raw in (text or "").lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3:
            tokens.append(token)
    return tokens


def _compute_overlap_score(query_tokens: List[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(doc_text))
    if not doc_tokens:
        return 0.0
    overlap = sum(1 for t in query_tokens if t in doc_tokens)
    return overlap / max(len(query_tokens), 1)


def _bm25_like_score(query_tokens: List[str], doc_text: str, k1: float = 1.2, b: float = 0.75) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    avg_doc_len = 100.0  # rough constant to avoid extra corpus stats
    term_counts = {}
    for token in doc_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    score = 0.0
    for token in query_tokens:
        tf = term_counts.get(token, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        score += norm

    return score


def _recency_boost(metadata: dict) -> float:
    ts = metadata.get("timestamp")
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return 0.0

    if age_days <= 7:
        return 0.05
    if age_days <= 30:
        return 0.03
    if age_days <= 90:
        return 0.015
    return 0.0


def _rerank_memories(query: str, docs: List) -> List:
    query_tokens = _tokenize(query)
    if not docs or not query_tokens:
        return docs

    scored = []
    for doc in docs:
        content = getattr(doc, 'page_content', str(doc))
        overlap = _compute_overlap_score(query_tokens, content)
        bm25 = _bm25_like_score(query_tokens, content)
        relevance = getattr(doc, "relevance", None)
        if relevance is None:
            relevance = 0.0
        metadata = getattr(doc, "metadata", {}) or {}
        category = metadata.get("category")
        confidence = metadata.get("confidence")
        boost = 0.0
        recency = _recency_boost(metadata)
        if category == "self_correction":
            boost += 0.1
        if category == "user_fact":
            try:
                confidence_value = float(confidence) if confidence is not None else 0.85
            except (TypeError, ValueError):
                confidence_value = 0.85
            boost += 0.08 * min(max(confidence_value, 0.0), 1.0)
        combined = (0.6 * relevance) + (0.2 * overlap) + (0.2 * bm25) + boost + recency
        scored.append((combined, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


def _extract_user_facts(message: str, language: str = "de") -> List[dict]:
    text = (message or "").strip()
    if not text:
        return []

    stopwords = set()
    stop_starts = ()
    patterns = []
    if language == "de":
        stopwords = {
            "der", "die", "das", "ein", "eine", "einer", "eines", "einem",
            "mein", "meine", "meiner", "meines", "mir", "mich",
            "du", "dir", "dich", "ihr", "wir", "uns", "euch", "sie", "er", "es",
            "ihm", "ihnen",
        }
        stop_starts = ("dir ", "dich ", "euch ", "ihnen ", "ihm ")
        patterns = [
            (r"\bich hei[ßs]e\s+([^.,;!?]+)", "name"),
            (r"\bmein name ist\s+([^.,;!?]+)", "name"),
            (r"\bich\s+([a-zäöüß-]+)\s+hei[ßs]e\b", "name"),
            (r"\bich wohne in\s+([^.,;!?]+)", "location"),
            (r"\bich lebe in\s+([^.,;!?]+)", "location"),
            (r"\bich arbeite (?:bei|in)\s+([^.,;!?]+)", "workplace"),
            (r"\bich arbeite als\s+([^.,;!?]+)", "occupation"),
            (r"\bmein beruf ist\s+([^.,;!?]+)", "occupation"),
            (r"\bich bin\s+([^.,;!?]+)\s+von beruf\b", "occupation"),
            (r"\bmein geburtstag ist\s+([^.,;!?]+)", "birthday"),
            (r"\bmein hobby ist\s+([^.,;!?]+)", "interest"),
            (r"\bmeine hobbys sind\s+([^.,;!?]+)", "interest"),
            (r"\bich interessiere mich für\s+([^.,;!?]+)", "interest"),
            (r"\bich begeistere mich für\s+([^.,;!?]+)", "interest"),
            (r"\bich kann\s+([^.,;!?]+)", "skill"),
            (r"\bich beherrsche\s+([^.,;!?]+)", "skill"),
            (r"\bich kenne mich mit\s+([^.,;!?]+)\s+aus\b", "skill"),
            (r"\bich spreche\s+([^.,;!?]+)", "language"),
            (r"\bmeine sprachen sind\s+([^.,;!?]+)", "language"),
            (r"\bich bevorzuge\s+([^.,;!?]+)", "preference"),
            (r"\bich mag es\s+([^.,;!?]+)", "preference"),
            (r"\bmeine lieblings([a-zäöüß]+)\s+ist\s+([^.,;!?]+)", "favorite"),
        ]
    else:
        patterns = [
            (r"\bmy name is\s+([^.,;!?]+)", "name"),
            (r"\bi live in\s+([^.,;!?]+)", "location"),
            (r"\bi work at\s+([^.,;!?]+)", "workplace"),
            (r"\bi work as\s+([^.,;!?]+)", "occupation"),
            (r"\bmy job is\s+([^.,;!?]+)", "occupation"),
            (r"\bmy birthday is\s+([^.,;!?]+)", "birthday"),
            (r"\bmy hobby is\s+([^.,;!?]+)", "interest"),
            (r"\bmy hobbies are\s+([^.,;!?]+)", "interest"),
            (r"\bi am interested in\s+([^.,;!?]+)", "interest"),
            (r"\bi like\s+([^.,;!?]+)", "interest"),
            (r"\bi can\s+([^.,;!?]+)", "skill"),
            (r"\bi am good at\s+([^.,;!?]+)", "skill"),
            (r"\bi speak\s+([^.,;!?]+)", "language"),
            (r"\bmy languages are\s+([^.,;!?]+)", "language"),
            (r"\bi prefer\s+([^.,;!?]+)", "preference"),
            (r"\bmy favorite ([a-z]+)\s+is\s+([^.,;!?]+)", "favorite"),
        ]

    facts = []
    lowered = text.lower()
    list_kinds = {"interest", "skill", "language", "preference"}

    def split_values(raw_value: str) -> List[str]:
        parts = re.split(r",|\bund\b|\band\b|&|\bsowie\b", raw_value)
        cleaned = []
        for part in parts:
            value = part.strip()
            if value:
                cleaned.append(value)
        return cleaned

    for pattern, kind in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        if kind == "favorite":
            category = match.group(1)
            value = match.group(2)
            facts.append({"kind": "favorite", "category": category.strip(), "value": value.strip()})
        elif kind in list_kinds:
            values = split_values(match.group(1))
            for value in values:
                if stopwords and value.strip() in stopwords:
                    continue
                if stop_starts and value.strip().startswith(stop_starts):
                    continue
                normalized = value.strip()
                facts.append({"kind": kind, "category": normalized, "value": normalized})
        else:
            value = match.group(1)
            if stopwords and value.strip() in stopwords:
                continue
            if stop_starts and value.strip().startswith(stop_starts):
                continue
            facts.append({"kind": kind, "value": value.strip()})

    return facts


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
        # Retrieve many memories (k=20) to ensure we have enough after filtering
        all_docs = await asyncio.to_thread(
            vectorstore.query_memories,
            clean_message,
            user_id=user_id,
            limit=20
        )

        # Stage 2: filter by similarity threshold, fallback to top-K if too strict
        threshold = _get_memory_threshold()
        low_confidence_fallback = False
        if all_docs:
            filtered_by_score = []
            for doc in all_docs:
                score = getattr(doc, "relevance", None)
                if score is None:
                    score = 0.0
                if score >= threshold:
                    filtered_by_score.append(doc)

            if filtered_by_score:
                all_docs = filtered_by_score
            else:
                fallback_count = min(5, len(all_docs))
                logger.info(
                    f"⚠️ No memories above threshold {threshold:.2f}; using top {fallback_count} low-confidence results."
                )
                all_docs = all_docs[:fallback_count]
                low_confidence_fallback = True

        # Filter out circular/self-referential memories EARLY
        non_circular_docs = []
        circular_count = 0
        expired_count = 0

        # Indicators that a memory contains facts (not just a question)
        fact_indicators = [
            "heißt", "heiße", "name ist", "my name", "I am", "I'm",
            "Besitzer", "owner", "genannt",
            "Geburtstag", "birthday", "wohne in", "live in", "arbeite bei", "work at",
            "ich bin der", "ich bin ein", "das ist mein",
            "hobby", "hobbys", "hobbies", "interess", "begeistere", "spreche",
            "skills", "fähigkeit", "fähigkeiten", "präferenz", "bevorzug", "mag es"
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
            metadata = getattr(doc, "metadata", {}) or {}
            if metadata.get("superseded"):
                continue
            expires_at = metadata.get("expires_at")
            if expires_at:
                try:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) > expires_dt:
                        expired_count += 1
                        continue
                except Exception:
                    pass

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

        if expired_count > 0:
            logger.info(f"🕒 Filtered {expired_count} expired memories")

        if circular_count > 0:
            logger.info(f"⚡ Filtered {circular_count} circular memories, "
                       f"{len(non_circular_docs)} remain")

        # Rerank before prioritization (semantic + lexical overlap)
        non_circular_docs = _rerank_memories(clean_message, non_circular_docs)

        identity_query = any(token in clean_message.lower() for token in [
            "wie ich heiße", "mein name", "wer bin ich", "beruf", "arbeite", "job", "profession",
            "hobby", "hobbys", "hobbies", "interesse", "interessen", "skills", "fähigkeit",
            "fähigkeiten", "sprache", "sprachen", "präferenz", "praeferenz", "bevorzug",
            "über mich", "ueber mich", "information", "informationen", "profil"
        ])

        # Now prioritize from the non-circular memories
        correction_docs = []
        factual_docs = []
        other_docs = []
        profile_docs = []

        for doc in non_circular_docs:
            content = getattr(doc, 'page_content', str(doc))
            content_lower = content.lower()

            # Check category first
            category = doc.metadata.get("category")
            if category == "self_correction":
                correction_docs.append(doc)
            elif category == "user_profile":
                if identity_query:
                    profile_docs.append(doc)
                continue
            elif category == "user_fact":
                confidence = doc.metadata.get("confidence", 0.0)
                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    confidence_value = 0.0
                if confidence_value >= _get_fact_min_confidence():
                    factual_docs.append(doc)
            # Then check if it's factual
            elif any(indicator.lower() in content_lower
                    for indicator in fact_indicators):
                factual_docs.append(doc)
            else:
                other_docs.append(doc)

        # Priority: corrections > factual > other
        if identity_query:
            personal_indicators = [
                "ich heiße", "mein name", "arbeite", "beruf", "job", "profession",
                "hobby", "hobbys", "hobbies", "interesse", "interessen", "skills", "fähigkeit",
                "fähigkeiten", "sprache", "sprachen", "präferenz", "praeferenz", "bevorzug"
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
                            (profile_docs[:3] if profile_docs else []) +
                            (personal_docs[:5] if personal_docs else []))
            if not relevant_docs:
                relevant_docs = (correction_docs[:3] + factual_docs[:5])
        else:
            relevant_docs = (correction_docs[:3] +
                            factual_docs[:3] +
                            other_docs[:2])
        relevant_docs = relevant_docs[:5]
        if low_confidence_fallback and relevant_docs:
            for doc in relevant_docs:
                try:
                    doc.metadata["low_confidence"] = True
                except Exception:
                    pass

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

    language = "de" if is_german else "en"
    user_facts = _extract_user_facts(clean_message, language)
    fact_confidence = _get_fact_confidence()
    fact_ttl_days = _get_fact_ttl_days()

    # Save to langchain memory buffer
    memory.save_context({"input": clean_message}, {"output": response_content})

    # SMART HOME: Intelligent storage decision
    should_store, reason, entity_id = classify_smart_home_storage_strategy(
        clean_message, tool_results
    )

    async def store_user_facts_task():
        try:
            if not user_facts:
                return
            for fact in user_facts[:5]:
                kind = fact.get("kind", "fact")
                value = fact.get("value", "")
                category = fact.get("category")
                if not value:
                    continue
                if kind == "favorite" and category:
                    fact_content = f"User fact: favorite_{category} = {value}"
                else:
                    fact_content = f"User fact: {kind} = {value}"

                fact_metadata = {
                    "category": "user_fact",
                    "source": "user_fact",
                    "language": language,
                    "fact_kind": kind,
                    "confidence": fact_confidence,
                    "superseded": False,
                    "fact_active": True,
                }
                if category:
                    fact_metadata["fact_category"] = category
                if fact_ttl_days > 0:
                    expires_at = datetime.now(timezone.utc) + timedelta(days=fact_ttl_days)
                    fact_metadata["expires_at"] = expires_at.isoformat()

                fact_doc_id, _fact_ts = await asyncio.to_thread(
                    store_memory,
                    content=fact_content,
                    user_id=user_id,
                    tags=["user_fact", "profile", "fact"],
                    metadata=fact_metadata
                )

                await _supersede_existing_facts(
                    user_id=user_id,
                    fact_kind=kind,
                    fact_category=category,
                    new_fact_id=fact_doc_id
                )
        except Exception as e:
            logger.error(f"Error storing user facts: {e}")

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
        if user_facts:
            asyncio.create_task(store_user_facts_task())
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
                    "language": language,
                    "message_length": len(clean_message),
                    "response_length": len(response_content)
                }
                metadata["category"] = "conversation"

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
        if user_facts:
            asyncio.create_task(store_user_facts_task())
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
