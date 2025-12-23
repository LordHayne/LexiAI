import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from backend.core.bootstrap import initialize_components
from backend.utils.model_utils import call_model_async
from backend.memory.adapter import store_memory
from backend.memory.activity_tracker import track_activity
from backend.memory.conversation_tracker import get_conversation_tracker
from backend.api.middleware.response_cache import get_response_cache
from .message_builder import build_messages

# Profile Learning Integration
from backend.services.profile_builder import ProfileBuilder
from backend.services.profile_context import ProfileContextBuilder

logger = logging.getLogger("memory_decisions")

@contextmanager
def timer(step_name: str, logger_instance=None):
    """Context manager for timing code blocks."""
    start = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start) * 1000
        msg = f"⏱️ [{step_name}]: {elapsed_ms:.0f}ms"
        if logger_instance:
            logger_instance.info(msg)
        else:
            logger.info(msg)

class PerformanceTracker:
    """Track cumulative performance metrics for a request."""
    def __init__(self):
        self.start_time = time.time()
        self.steps = {}

    def record(self, step_name: str, duration_ms: float):
        self.steps[step_name] = duration_ms

    def total_time(self) -> float:
        return (time.time() - self.start_time) * 1000

    def summary(self) -> str:
        total = self.total_time()
        accounted_time = sum(self.steps.values())
        unknown_time = total - accounted_time

        breakdown = "\n".join([
            f"  {name}: {dur:.0f}ms ({dur/total*100:.1f}%)"
            for name, dur in sorted(self.steps.items(), key=lambda x: -x[1])
        ])

        unknown_pct = (unknown_time / total * 100) if total > 0 else 0
        breakdown += f"\n  [UNKNOWN/OVERHEAD]: {unknown_time:.0f}ms ({unknown_pct:.1f}%)"

        return f"Performance Summary ({total:.0f}ms total, {accounted_time:.0f}ms accounted):\n{breakdown}"

async def _run_chat_logic(
    message,
    chat_client,
    vectorstore,
    memory,
    embeddings,
    streaming=False,
    user_id="default",
    user_profile: Optional[Dict[str, Any]] = None
):
    """
    Chat Logic mit Profile Learning Integration

    Args:
        message: User message
        chat_client: LLM client
        vectorstore: Vector store
        memory: Conversation memory
        embeddings: Embedding model
        streaming: Streaming mode
        user_id: User ID
        user_profile: User profile dictionary (for personalization)
    """
    perf = PerformanceTracker()
    step_start = time.time()

    # Initialize profile services
    profile_builder = ProfileBuilder(llm_client=chat_client)
    profile_context_builder = ProfileContextBuilder()

    # Parse flags and clean message
    with timer("Parse flags and clean message", logger):
        is_english = "/english" in message.lower() or message.lower().startswith("/en")
        is_german = not is_english  # Deutsch ist Standard
        no_think = "/nothink" in message.lower() or "/no think" in message.lower()
        clean_message = message
        for cmd in ["/nothink", "/no think", "/deutsch", "/de", "/english", "/en"]:
            clean_message = clean_message.replace(cmd, "").strip()

        # Check for forget/delete commands
        forget_patterns = [
            r"vergiss\s+(.+)",
            r"forget\s+(.+)",
            r"lösche\s+erinnerung(?:en)?\s+(?:an|über|zu)\s+(.+)",
            r"delete\s+memor(?:y|ies)\s+(?:about|of)\s+(.+)",
            r"vergiss\s+was\s+du\s+über\s+(.+)\s+weißt",
            r"forget\s+what\s+you\s+know\s+about\s+(.+)"
        ]

        import re
        forget_topic = None
        for pattern in forget_patterns:
            match = re.search(pattern, clean_message.lower())
            if match:
                forget_topic = match.group(1).strip()
                break

        # Handle forget command
        if forget_topic:
            logger.info(f"Forget command detected for topic: {forget_topic}")
            try:
                from backend.memory.adapter import delete_memories_by_content

                # Delete memories about the topic
                deleted_ids = delete_memories_by_content(
                    query=forget_topic,
                    user_id=user_id,
                    similarity_threshold=0.70  # Slightly lower threshold for forget commands
                )

                if deleted_ids:
                    if is_german:
                        response_content = f"Ich habe {len(deleted_ids)} Erinnerung{'en' if len(deleted_ids) > 1 else ''} über '{forget_topic}' gelöscht."
                    else:
                        response_content = f"I deleted {len(deleted_ids)} memor{'ies' if len(deleted_ids) > 1 else 'y'} about '{forget_topic}'."

                    logger.info(f"Successfully deleted {len(deleted_ids)} memories: {deleted_ids}")
                else:
                    if is_german:
                        response_content = f"Ich habe keine Erinnerungen über '{forget_topic}' gefunden."
                    else:
                        response_content = f"I found no memories about '{forget_topic}'."

                # Return immediately without further processing
                if streaming:
                    async for chunk in _stream_response(response_content):
                        yield {"chunk": chunk, "final_chunk": False}
                    yield {
                        "final_chunk": True,
                        "source": "forget_command",
                        "relevant_memory": [],
                        "feedback_possible": False,
                        "memory_saved_id": None,
                        "turn_id": None,
                        "deleted_count": len(deleted_ids)
                    }
                else:
                    yield {
                        "response": response_content,
                        "final": True,
                        "source": "forget_command",
                        "relevant_memory": [],
                        "turn_id": None,
                        "memory_saved_id": None,
                        "feedback_possible": False,
                        "deleted_count": len(deleted_ids)
                    }
                return  # Exit early

            except Exception as e:
                logger.error(f"Error processing forget command: {str(e)}", exc_info=True)
                error_msg = f"Fehler beim Löschen: {str(e)}" if is_german else f"Error deleting memories: {str(e)}"

                if streaming:
                    yield {"chunk": error_msg, "final_chunk": True}
                else:
                    yield {"response": error_msg, "final": True}
                return  # Exit early

    perf.record("Parse flags and clean message", (time.time() - step_start) * 1000)

    # Phase 3.8: Implicit Feedback Detection + Memory Retrieval (PARALLEL OPTIMIZATION)
    step_start = time.time()
    conversation_tracker = get_conversation_tracker()

    # OPTIMIZATION: Define parallel preprocessing tasks
    async def feedback_detection_task():
        """Detect implicit feedback in separate task (thread-safe)"""
        reformulation_turn_id = None

        # Check 1: Reformulation Detection
        reformulation_turn_id = await asyncio.to_thread(
            conversation_tracker.detect_implicit_reformulation,
            user_id, clean_message
        )
        if reformulation_turn_id:
            from backend.models.feedback import FeedbackType
            await asyncio.to_thread(
                conversation_tracker.record_feedback,
                reformulation_turn_id,
                FeedbackType.IMPLICIT_REFORMULATION,
                confidence=0.8
            )
            logger.info(f"🔄 Detected reformulation of turn {reformulation_turn_id}")

        # Check 2: Contradiction Detection
        contradiction_signals = ["falsch", "stimmt nicht", "das ist nicht richtig", "incorrect", "nicht korrekt", "das stimmt nicht"]
        if any(signal in clean_message.lower() for signal in contradiction_signals):
            history = await asyncio.to_thread(conversation_tracker.get_user_history, user_id, 1)
            if history:
                last_turn = history[0]
                from backend.models.feedback import FeedbackType
                await asyncio.to_thread(
                    conversation_tracker.record_feedback,
                    last_turn.turn_id,
                    FeedbackType.IMPLICIT_CONTRADICTION,
                    confidence=0.9
                )
                logger.info(f"⚠️ Detected contradiction for turn {last_turn.turn_id}")

        return reformulation_turn_id

    async def get_context_async():
        """Memory retrieval task"""
        logger.info(f"Suche nach relevanten Informationen für: '{message[:50]}'")
        # Phase 3.8: Bevorzuge Correction Memories
        # FIX: Use query_memories with user_id filter for proper user isolation
        all_docs = await asyncio.to_thread(
            vectorstore.query_memories,
            message,
            user_id=user_id,
            limit=5,
            score_threshold=0.3  # Filter out irrelevant results
        )

        # 🔍 DEBUG MODE: Log raw query results
        logger.info(f"🔍 DEBUG - get_context_async() received from query_memories():")
        logger.info(f"  └─ Total documents: {len(all_docs)}")
        if all_docs:
            for i, doc in enumerate(all_docs, 1):
                # MemoryEntry objects have .category attribute, not metadata dict
                category = getattr(doc, 'category', 'N/A')
                relevance = getattr(doc, 'relevance', 'N/A')
                content_preview = getattr(doc, 'content', '')[:50]
                logger.info(f"  [{i}] Relevance: {relevance:.3f} | Category: {category} | Content: '{content_preview}...'")
        else:
            logger.warning(f"⚠️ DEBUG - query_memories() returned EMPTY LIST!")

        # Separiere Correction Memories und normale Memories
        correction_docs = [doc for doc in all_docs if getattr(doc, 'category', None) == "self_correction"]
        normal_docs = [doc for doc in all_docs if getattr(doc, 'category', None) != "self_correction"]

        # 🔍 DEBUG MODE: Log categorization
        logger.info(f"🔍 DEBUG - After categorization:")
        logger.info(f"  └─ Correction docs: {len(correction_docs)}")
        logger.info(f"  └─ Normal docs: {len(normal_docs)}")

        # Bevorzuge Corrections, dann normale (insgesamt max 3)
        prioritized_docs = correction_docs[:2] + normal_docs[:1]  # Max 2 Corrections + 1 Normal
        prioritized_docs = prioritized_docs[:3]  # Insgesamt max 3

        # 🔍 DEBUG MODE: Log final prioritization
        logger.info(f"🔍 DEBUG - Final prioritized_docs to return: {len(prioritized_docs)}")
        if prioritized_docs:
            for i, doc in enumerate(prioritized_docs, 1):
                category = getattr(doc, 'category', 'N/A')
                content_preview = getattr(doc, 'content', '')[:40]
                logger.info(f"  [{i}] Category: {category} | Content: '{content_preview}...'")

        if correction_docs:
            logger.info(f"✨ Found {len(correction_docs)} correction memories - prioritizing them")

        return prioritized_docs

    # PARALLEL EXECUTION: Run feedback detection and memory retrieval concurrently
    relevant_docs = []
    reformulation_turn_id = None

    if len(message.strip()) >= 8 and message.lower() not in {"ok", "ja", "hm", "versteh ich", "danke"}:
        try:
            logger.info("⚡ Running 2 tasks in parallel: feedback detection + memory retrieval")
            # Run both tasks concurrently
            results = await asyncio.gather(
                feedback_detection_task(),
                get_context_async(),
                return_exceptions=True
            )

            # Extract results with error handling
            if not isinstance(results[0], Exception):
                reformulation_turn_id = results[0]
            else:
                logger.error(f"Feedback detection failed: {results[0]}")

            if not isinstance(results[1], Exception):
                relevant_docs = results[1]
                # 🔍 DEBUG MODE: Log what we got from parallel execution
                logger.info(f"🔍 DEBUG - Parallel execution results[1] (relevant_docs):")
                logger.info(f"  └─ Type: {type(relevant_docs)}")
                logger.info(f"  └─ Length: {len(relevant_docs) if relevant_docs else 0}")
                if relevant_docs:
                    logger.info(f"  └─ First doc preview: {getattr(relevant_docs[0], 'content', 'N/A')[:40]}...")
            else:
                logger.error(f"Memory retrieval failed: {results[1]}")
                relevant_docs = []

        except Exception as e:
            logger.error(f"Fehler bei paralleler Vorverarbeitung: {e}")
    else:
        # For short messages, still run feedback detection
        try:
            reformulation_turn_id = await feedback_detection_task()
        except Exception as e:
            logger.error(f"Fehler bei Feedback Detection: {e}")

    perf.record("Parallel preprocessing (feedback + memory)", (time.time() - step_start) * 1000)

    # Phase 5: Web Search Integration (LLM-based)
    step_start = time.time()
    web_search_result = None
    web_search_feedback = ""
    try:
        from backend.services.web_search import get_web_search_service
        from backend.core.llm_web_search_decision import (
            should_perform_web_search_llm,
            extract_search_query_llm
        )
        from backend.core.web_search_integration import (
            format_web_results_for_context,
            create_web_search_feedback
        )

        web_service = get_web_search_service()

        # Use mock service if real service not available (for testing)
        if not web_service.is_enabled():
            logger.info("⚠️ Real web search disabled - using mock service for testing")
            from backend.services.mock_web_search import get_mock_web_search_service
            web_service = get_mock_web_search_service()

        if web_service.is_enabled():
            language = "de" if is_german else "en"

            # LLM decides if web search is needed
            decision_start = time.time()
            with timer("Web search decision (LLM)", logger):
                should_search, reason = await should_perform_web_search_llm(
                    clean_message, relevant_docs, chat_client, language
                )
            perf.record("Web search decision", (time.time() - decision_start) * 1000)

            if should_search:
                logger.info(f"🤖 LLM decided to perform web search: {reason}")

                # LLM extracts optimal search query
                query_start = time.time()
                with timer("Web search query extraction (LLM)", logger):
                    search_query = await extract_search_query_llm(
                        clean_message, relevant_docs, chat_client, language
                    )
                perf.record("Web search query extraction", (time.time() - query_start) * 1000)

                # Perform async web search
                search_start = time.time()
                with timer("Web search execution", logger):
                    web_search_result = await web_service.search(
                        query=search_query,
                        max_results=5,
                        search_depth="basic"
                    )
                search_time_ms = int((time.time() - search_start) * 1000)
                perf.record("Web search execution", search_time_ms)

                if web_search_result and web_search_result.get("results"):
                    # Phase 1.4: Check relevance of web results
                    from backend.core.llm_result_relevance_check import (
                        check_result_relevance,
                        should_refine_search
                    )

                    # Filter results by relevance
                    relevance_start = time.time()
                    with timer("Web search result relevance check (LLM)", logger):
                        filtered_results, overall_quality = await check_result_relevance(
                            clean_message,
                            web_search_result["results"],
                            chat_client,
                            language
                        )
                    perf.record("Web search relevance check", (time.time() - relevance_start) * 1000)

                    # If quality poor and no good results, try refining search (once)
                    if overall_quality < 0.4 and len(filtered_results) == 0:
                        should_refine, refined_query = await should_refine_search(
                            clean_message,
                            overall_quality,
                            filtered_results,
                            chat_client,
                            language
                        )

                        if should_refine and refined_query:
                            logger.info(f"🔄 Performing refined search: '{refined_query}'")
                            search_start = time_module.time()
                            web_search_result = await web_service.search(
                                query=refined_query,
                                max_results=5,
                                search_depth="basic"
                            )
                            search_time_ms = int((time_module.time() - search_start) * 1000)

                            # Re-check relevance of refined results
                            if web_search_result and web_search_result.get("results"):
                                filtered_results, overall_quality = await check_result_relevance(
                                    clean_message,
                                    web_search_result["results"],
                                    chat_client,
                                    language
                                )

                    # Use filtered results
                    if filtered_results:
                        web_search_result["results"] = filtered_results

                        # Format results for context
                        web_context = format_web_results_for_context(web_search_result, max_results=3)

                        # Create user feedback
                        web_search_feedback = create_web_search_feedback(
                            query=search_query,
                            result_count=len(filtered_results),
                            search_time_ms=search_time_ms,
                            language=language
                        )

                        logger.info(f"✅ Web search completed: {len(filtered_results)} relevant results (quality={overall_quality:.2f})")
                    else:
                        logger.warning("⚠️ No relevant web results found after filtering")
                        web_search_result = None  # Clear results if all filtered out
                else:
                    logger.warning("⚠️ Web search returned no results")
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        # Continue without web search if it fails
    perf.record("Web search (total)", (time.time() - step_start) * 1000)

    # Build messages with profile context
    step_start = time.time()
    with timer("Build LLM messages", logger):
        # Use profile if available and relevant
        profile_to_use = None
        if user_profile and profile_context_builder.should_use_profile(clean_message, user_profile):
            profile_to_use = user_profile
            logger.info("🎯 Using user profile for personalized response")

        messages = build_messages(
            clean_message,
            is_german,
            relevant_docs,
            no_think,
            web_context=web_search_result,
            user_profile=profile_to_use
        )
    perf.record("Build LLM messages", (time.time() - step_start) * 1000)

    # Main LLM call
    step_start = time.time()
    try:
        with timer("Main LLM call", logger):
            chat_response = await call_model_async(chat_client, messages)

        # Handle various response formats safely
        if asyncio.iscoroutine(chat_response):
            chat_response = await chat_response

        # Extract content from response
        if hasattr(chat_response, "content"):
            response_content = chat_response.content
        elif isinstance(chat_response, dict) and "content" in chat_response:
            response_content = chat_response["content"]
        elif isinstance(chat_response, str):
            response_content = chat_response
        else:
            response_content = str(chat_response)

        # Handle if content is still a coroutine
        if asyncio.iscoroutine(response_content):
            response_content = await response_content

        # Ensure response is a string
        if not isinstance(response_content, str):
            response_content = str(response_content)

    except Exception as e:
        logger.error(f"❌ Error calling LLM: {e}", exc_info=True)
        response_content = f"Entschuldigung, es gab einen Fehler beim Verarbeiten deiner Anfrage: {str(e)}"
    perf.record("Main LLM call", (time.time() - step_start) * 1000)

    # Phase 1.3: Self-Reflection - Check for hallucinations
    step_start = time.time()
    # OPTIMIZATION: Only run self-reflection if answer is uncertain or very short
    should_reflect = (
        len(response_content) < 100 or  # Very short answers
        any(marker in response_content.lower() for marker in ["ich bin mir nicht sicher", "vielleicht", "möglicherweise", "I'm not sure", "maybe"]) or
        (not relevant_docs and not web_search_result)  # No sources available
    )

    if should_reflect:
        try:
            with timer("Self-reflection (verify + fallback)", logger):
                from backend.core.llm_self_reflection import verify_answer_quality, generate_honest_fallback

                # Prepare sources for verification
                sources = []
                if relevant_docs:
                    for doc in relevant_docs:
                        content = getattr(doc, 'page_content', str(doc))
                        sources.append(f"Memory: {content[:200]}")

                if web_search_result and web_search_result.get("results"):
                    for result in web_search_result["results"][:3]:
                        sources.append(f"Web: {result.get('title', '')}: {result.get('content', '')[:150]}")

                # Verify answer quality
                is_valid, issue = await verify_answer_quality(
                    clean_message, response_content, sources, chat_client, "de" if is_german else "en"
                )

                if not is_valid:
                    logger.warning(f"🔄 Self-reflection failed: {issue}. Generating honest fallback.")
                    # Generate honest "I don't know" response
                    response_content = await generate_honest_fallback(
                        clean_message, chat_client, "de" if is_german else "en"
                    )

        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            # Continue with original answer if reflection fails
    else:
        logger.info("✓ Skipping self-reflection (answer appears confident and has sources)")
    perf.record("Self-reflection", (time.time() - step_start) * 1000)

    # Add web search feedback to response if available
    if web_search_feedback:
        response_content = web_search_feedback + response_content

    # Save conversation context
    step_start = time.time()
    with timer("Save conversation context to memory buffer", logger):
        memory.save_context({"input": message}, {"output": response_content})
    perf.record("Save conversation context", (time.time() - step_start) * 1000)

    # Record conversation turn for feedback tracking
    step_start = time.time()
    with timer("Record conversation turn", logger):
        retrieved_memory_ids = [doc.metadata.get("id", "") for doc in relevant_docs if doc.metadata.get("id")]
        turn_id = conversation_tracker.record_turn(
            user_id=user_id,
            user_message=message,
            ai_response=response_content,
            retrieved_memories=retrieved_memory_ids if retrieved_memory_ids else None,
            response_time_ms=None  # Could add timing if needed
        )
        logger.info(f"Recorded conversation turn: {turn_id}")
    perf.record("Record conversation turn", (time.time() - step_start) * 1000)

    # Background tasks (memory storage, goal detection, web search storage)
    step_start = time.time()
    doc_id, ts = None, None
    if not no_think:
        async def memory_store_task():
            """
            Store memory in background without blocking event loop.

            QUALITY: Improved content formatting with structured conversation format.
            """
            nonlocal doc_id, ts
            try:
                # Import async version of store_memory
                from backend.memory.adapter import store_memory_async

                # QUALITY IMPROVEMENT: Better structured content format
                # Instead of raw "User:...Assistant:..." format, use clear Q&A structure
                memory_content = f"Q: {clean_message}\nA: {response_content}"

                # QUALITY: Add metadata for better categorization
                metadata = {
                    "interaction_type": "conversation",
                    "language": "de" if is_german else "en",
                    "message_length": len(clean_message),
                    "response_length": len(response_content)
                }

                # FIXED: Use truly async store_memory_async() instead of asyncio.to_thread()
                doc_id, ts = await store_memory_async(
                    content=memory_content,
                    user_id=user_id,
                    tags=["chat", "conversation"],
                    metadata=metadata
                )
                logger.info(f"✅ Memory stored async during chat processing: {doc_id}")
            except Exception as e:
                logger.error(f"❌ Fehler beim Speichern der Erinnerung: {e}", exc_info=True)
                doc_id, ts = None, None  # Explicit None assignment on failure

        async def goal_detection_task():
            """Erkennt und speichert Ziele aus der Nachricht (non-blocking)"""
            try:
                from backend.memory.goal_tracker import get_goal_tracker, GoalDetector

                # OPTIMIZATION: Quick check if message might contain goals
                goal_indicators = ["möchte", "will", "plane", "ziel", "vorhaben", "want to", "plan to", "goal", "intend"]
                if not any(indicator in clean_message.lower() for indicator in goal_indicators):
                    logger.info("✓ No goal indicators found - skipping goal detection")
                    return

                # Nutze LLM-basierte Goal Detection
                detected_goals = await GoalDetector.detect_goals_with_llm(
                    text=message,
                    user_id=user_id,
                    chat_client=chat_client
                )

                if detected_goals:
                    tracker = get_goal_tracker(vectorstore)
                    for goal in detected_goals:
                        # Speichere source memory ID wenn vorhanden
                        if doc_id:
                            goal.source_memory_ids = [doc_id]

                        success = tracker.add_goal(goal)
                        if success:
                            logger.info(f"🎯 New goal tracked: {goal.category} - {goal.content[:50]}...")
            except Exception as e:
                logger.error(f"Error in goal detection: {e}")

        async def web_search_store_task():
            """Speichert Web-Search-Ergebnisse im Memory wenn sinnvoll"""
            try:
                if web_search_result and web_search_result.get("results"):
                    from backend.core.web_search_integration import should_save_web_result_to_memory
                    from backend.memory.adapter import store_memory_async

                    if should_save_web_result_to_memory(web_search_result, clean_message):
                        # QUALITY IMPROVEMENT: Better structured web search format
                        search_summary = web_search_result.get("answer", "")
                        sources = web_search_result.get("sources", [])[:3]

                        # Format sources nicely
                        sources_formatted = "\n".join([f"- {src}" for src in sources])

                        memory_content = f"""Web Search Results
Query: {clean_message}

Summary:
{search_summary}

Sources:
{sources_formatted}"""

                        # QUALITY: Add metadata for web search
                        metadata = {
                            "search_query": clean_message,
                            "source_count": len(sources),
                            "search_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }

                        # FIXED: Use truly async store_memory_async()
                        web_doc_id, web_ts = await store_memory_async(
                            content=memory_content,
                            user_id=user_id,
                            tags=["web_search", "factual", "external"],
                            metadata=metadata
                        )
                        logger.info(f"💾 Web search results saved to memory: {web_doc_id}")
            except Exception as e:
                logger.error(f"❌ Error storing web search results: {e}", exc_info=True)

        async def profile_learning_task():
            """
            Profile Learning: Extrahiert User-Profil-Informationen aus Konversation
            Background Task - blockiert Response nicht
            """
            try:
                # Nur wenn Profil-Extraktion sinnvoll ist
                if not profile_builder.should_update_profile(clean_message):
                    logger.debug("⏭️ Skipping profile update (message too short/generic)")
                    return

                # Profile info extrahieren
                logger.info("🧠 Starting profile learning from conversation...")
                updated_profile = await profile_builder.extract_profile_info(
                    user_message=clean_message,
                    assistant_response=response_content,
                    current_profile=user_profile or {}
                )

                # TODO: Profil in DB speichern (aktuell nur in-memory)
                # In Produktion: Update user profile in database
                if updated_profile != user_profile:
                    logger.info(f"📝 Profile updated with new information: {list(updated_profile.keys())}")

                    # Log Profile Summary
                    summary = profile_builder.get_profile_summary(updated_profile)
                    logger.debug(f"Profile Summary:\n{summary}")

            except Exception as e:
                logger.error(f"❌ Error in profile learning: {e}", exc_info=True)

        # PARALLEL EXECUTION: Run 4 background tasks concurrently (including profile learning)
        logger.info("⚡ Running 4 background tasks in parallel: memory + goals + web search + profile learning")
        background_step_start = time.time()
        await asyncio.gather(
            memory_store_task(),
            goal_detection_task(),
            web_search_store_task(),
            profile_learning_task(),  # Profile Learning als Background Task
            return_exceptions=True  # Don't fail if one task errors
        )
        perf.record("Background tasks (parallel)", (time.time() - background_step_start) * 1000)

    # Log performance summary
    logger.info(f"\n{perf.summary()}")

    if streaming:
        async for chunk in _stream_response(response_content):
            yield {"chunk": chunk, "final_chunk": False}
        yield {
            "final_chunk": True,
            "source": "llm",
            "relevant_memory": [doc.metadata for doc in relevant_docs],
            "feedback_possible": not no_think,
            "memory_saved_id": doc_id,
            "turn_id": turn_id  # NEW: Include turn_id for feedback
        }
    else:
        # FIXED: Return proper dict structure instead of tuple for consistency
        yield {
            "response": response_content,
            "final": True,
            "source": "llm",
            "relevant_memory": [doc.metadata for doc in relevant_docs],
            "turn_id": turn_id,
            "memory_saved_id": doc_id,
            "feedback_possible": not no_think
        }

async def _stream_response(response):
    if asyncio.iscoroutine(response):
        response = await response
    words = response.split()
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= 10:
            yield " ".join(current_chunk)
            current_chunk = []
    if current_chunk:
        yield " ".join(current_chunk)

async def process_chat_message_streaming(
    message,
    chat_client=None,
    vectorstore=None,
    memory=None,
    embeddings=None,
    collect_feedback=False,
    user_id="default",
    user_profile=None
):
    """
    Process chat message with streaming response (with profile learning)

    Args:
        message: User message
        chat_client: LLM client
        vectorstore: Vector store
        memory: Conversation memory
        embeddings: Embedding model
        collect_feedback: Enable feedback collection
        user_id: User ID
        user_profile: User profile dictionary for personalization
    """
    # Track user activity for idle detection
    track_activity()

    if any(x is None for x in [chat_client, vectorstore, memory, embeddings]):
        embeddings, vectorstore, memory, chat_client, _ = initialize_components()

    async for result in _run_chat_logic(
        message,
        chat_client,
        vectorstore,
        memory,
        embeddings,
        streaming=True,
        user_id=user_id,
        user_profile=user_profile
    ):
        yield result

async def process_chat_message_async(
    message,
    chat_client=None,
    vectorstore=None,
    memory=None,
    embeddings=None,
    user_id="default",
    user_profile=None
):
    """
    Process chat message asynchronously with response caching and profile learning.

    OPTIMIZATION: Checks cache before expensive LLM processing.
    Cache hit saves ~24s (entire LLM pipeline).

    Args:
        message: User message to process
        chat_client: LLM client
        vectorstore: Vector store for memories
        memory: Conversation memory
        embeddings: Embedding model
        user_id: User ID
        user_profile: User profile dictionary for personalization
        memory: Conversation memory
        embeddings: Embedding model
        user_id: User identifier

    Returns:
        Dict with response, metadata, and context
    """
    # Track user activity for idle detection
    track_activity()

    # OPTIMIZATION: Check response cache first
    # Skip cache if /nothink flag is present (user wants fresh thinking)
    no_think_flag = "/nothink" in message.lower() or "/no think" in message.lower()

    if not no_think_flag:
        cache = get_response_cache()

        # Detect language for cache key
        is_english = "/english" in message.lower() or message.lower().startswith("/en")
        language = "en" if is_english else "de"

        # Try to get cached response
        cached_response = cache.get(user_id, message, language)

        if cached_response:
            logger.info(f"✓ Cache hit for user={user_id} - returning cached response (saved ~24s)")
            # Add cache hit indicator to response
            cached_response["from_cache"] = True
            return cached_response

    # Cache miss or /nothink flag - process normally
    logger.debug(f"Cache miss for user={user_id} - processing with LLM")

    # Check if any component is None (using list instead of set to avoid unhashable type error)
    if None in [chat_client, vectorstore, memory, embeddings]:
        embeddings, vectorstore, memory, chat_client, _ = initialize_components()

    gen = _run_chat_logic(
        message,
        chat_client,
        vectorstore,
        memory,
        embeddings,
        streaming=False,
        user_id=user_id,
        user_profile=user_profile
    )
    result = await anext(gen)

    # OPTIMIZATION: Store result in cache (unless /nothink flag)
    if not no_think_flag:
        cache = get_response_cache()

        # Detect language for cache key
        is_english = "/english" in message.lower() or message.lower().startswith("/en")
        language = "en" if is_english else "de"

        # Cache the response (TTL: 1 hour)
        cache.set(user_id, message, result, language, ttl=3600)
        logger.debug(f"✓ Cached response for user={user_id} (ttl=3600s)")

    # Add cache indicator
    result["from_cache"] = False

    return result

def process_chat_message(message, chat_client, vectorstore, memory, embeddings, user_id="default"):
    try:
        return asyncio.run(process_chat_message_async(
            message, chat_client, vectorstore, memory, embeddings, user_id=user_id
        ))
    except Exception as e:
        logger.error(f"Fehler in synchroner Verarbeitung: {str(e)}")
        return f"Fehler bei der Verarbeitung deiner Nachricht: {e}"
