"""
Chat Processing with LLM Tool-Calling System
"""
import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional
from backend.core.bootstrap import initialize_components
from backend.utils.model_utils import call_model_async
from backend.memory.activity_tracker import track_activity
from backend.memory.conversation_tracker import get_conversation_tracker
from backend.services.session_manager import get_session_manager
from backend.services.profile_builder import ProfileBuilder
from backend.services.user_store import build_user_profile_context
from backend.core.llm_tool_calling import select_tools, execute_tool, ToolResult
from backend.core.llm_self_reflection import verify_answer_quality, generate_honest_fallback
from backend.core.query_classifier import (
    classify_query, needs_tools, needs_multi_step_check, needs_self_reflection, QueryType
)
# Import QueryType explicitly for direct comparison
from backend.core.query_classifier import QueryType
from backend.core.prompt_builder import (
    format_context_summary,
    build_user_context,
    build_system_prompt
)
from backend.core.memory_handler import (
    retrieve_and_filter_memories,
    store_conversation_memory,
    record_conversation_turn
)
from backend.core.smart_home_handler import classify_smart_home_storage_strategy
from backend.core.forget_handler import (
    detect_forget_topic,
    detect_profile_forget_keys,
    apply_forget_by_topic,
)

logger = logging.getLogger("tool_based_chat")


async def process_chat_with_tools(
    message: str,
    chat_client,
    vectorstore,
    memory,
    embeddings,
    streaming: bool = False,
    user_id: str = "default"
) -> str:
    """
    Process chat message using LLM tool-calling system.

    Flow:
    1. Get initial context from memory
    2. LLM selects tools to use
    3. Execute selected tools
    4. Generate answer with tool results
    5. Self-reflection check
    6. Return final answer
    """
    # Parse flags
    is_english = "/english" in message.lower() or message.lower().startswith("/en")
    is_german = not is_english
    no_think = "/nothink" in message.lower() or "/no think" in message.lower()

    clean_message = message
    for cmd in ["/nothink", "/no think", "/deutsch", "/de", "/english", "/en"]:
        clean_message = clean_message.replace(cmd, "").strip()

    language = "de" if is_german else "en"

    forget_topic = detect_forget_topic(clean_message)

    if forget_topic:
        logger.info(f"Forget command detected for topic: {forget_topic}")
        try:
            profile_keys = detect_profile_forget_keys(clean_message)
            result = await apply_forget_by_topic(
                user_id=user_id,
                topic=forget_topic,
                is_german=is_german,
                similarity_threshold=0.70,
                profile_keys=profile_keys or None
            )
            response_content = result.response
            deleted_ids = result.deleted_ids
        except Exception as e:
            logger.error(f"Error processing forget command: {str(e)}", exc_info=True)
            error_msg = f"Fehler beim Löschen: {str(e)}" if is_german else f"Error deleting memories: {str(e)}"
            session_manager = get_session_manager()
            try:
                session_manager.add_message(user_id, "user", clean_message)
                session_manager.add_message(user_id, "assistant", error_msg)
            except Exception as session_error:
                logger.warning(f"Failed to add forget error to session history: {session_error}")
            return {
                "response": error_msg,
                "relevant_memory": [],
                "final": True,
                "source": "forget_command",
                "turn_id": None,
                "deleted_count": 0
            }

        session_manager = get_session_manager()
        try:
            session_manager.add_message(user_id, "user", clean_message)
            session_manager.add_message(user_id, "assistant", response_content)
        except Exception as e:
            logger.warning(f"Failed to add forget response to session history: {e}")

        return {
            "response": response_content,
            "relevant_memory": [],
            "final": True,
            "source": "forget_command",
            "turn_id": None,
            "deleted_count": len(deleted_ids)
        }

    # FIX: Initialize memory_used flag (used in return statement)
    memory_used = not no_think

    # Phase 1: Get initial memory context using memory_handler
    relevant_docs = await retrieve_and_filter_memories(clean_message, vectorstore, user_id)

    # Format context summary using prompt_builder
    context_summary = format_context_summary(relevant_docs)

    # Phase 1.5: Classify query for performance optimization
    query_type = classify_query(clean_message)
    logger.info(f"🎯 Query classified as: {query_type}")

    def is_time_query(text: str, lang: str) -> bool:
        if lang == "de":
            patterns = [
                r"\bwie spät( ist es)?\b",
                r"\buhrzeit\b",
                r"\bwelches datum\b",
                r"\bwelchen tag haben wir\b"
            ]
        else:
            patterns = [
                r"\bwhat time is it\b",
                r"\bcurrent time\b",
                r"\bwhat(?:'s| is) the date\b",
                r"\bwhat day is it\b"
            ]
        lowered = text.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    def is_confirmation(text: str) -> bool:
        text_lower = text.lower().strip()
        confirm_tokens = [
            "ja", "ok", "okay", "passt", "speichern", "speichere",
            "apply", "ja bitte", "bitte speichern", "mach es", "mach das"
        ]
        return any(token == text_lower or text_lower.startswith(token + " ") for token in confirm_tokens)

    # Check if user has existing memories and if this is first message in session
    session_manager = get_session_manager()
    conversation_history = session_manager.get_conversation_history(user_id, max_turns=5)
    is_first_message_in_session = len(conversation_history) == 0

    pending_automation = session_manager.get_pending_automation(user_id)
    pending_script = session_manager.get_pending_script(user_id)
    if (pending_automation or pending_script) and is_confirmation(clean_message):
        query_type = QueryType.SMART_HOME_AUTOMATION

    # Check if user has ANY memories in Qdrant (not just relevant to current query)
    # This is important for greetings where relevant_docs might be empty
    try:
        # Quick check: count memories for this user_id
        user_memory_count = await asyncio.to_thread(
            vectorstore.count_user_memories,
            user_id=user_id
        )
        has_existing_memories = user_memory_count > 0
        logger.info(f"📊 User {user_id} has {user_memory_count} total memories in Qdrant")
    except Exception as e:
        # Fallback to relevant_docs check if count fails
        logger.warning(f"Failed to count user memories: {e}, using relevant_docs fallback")
        has_existing_memories = len(relevant_docs) > 0

    user_profile = {}
    try:
        user_profile = build_user_profile_context(user_id)
    except Exception as e:
        logger.warning(f"Failed to load user profile for {user_id}: {e}")

    profile_builder = ProfileBuilder(llm_client=chat_client)

    user_display_name = None
    if user_profile:
        display_name = str(user_profile.get("user_profile_name", "")).strip()
        if display_name:
            user_display_name = display_name

    try:
        from backend.services.user_store import get_user_store
        user = get_user_store().get_user(user_id)
        if user and user.display_name:
            display_name = user.display_name.strip()
            if display_name and display_name.lower() not in ["anonymous user", "anonymer benutzer"]:
                user_display_name = display_name
    except Exception as e:
        logger.warning(f"Failed to load display name for user {user_id}: {e}")

    # Build personalized greeting instruction using prompt_builder
    user_context, greeting_instruction = build_user_context(
        user_id=user_id,
        has_existing_memories=has_existing_memories,
        is_first_message_in_session=is_first_message_in_session,
        language=language,
        user_display_name=user_display_name
    )

    # Phase 2: Check if multi-step reasoning is needed (CONDITIONAL)
    needs_multi_step = False
    reasoning = ""
    execution_steps = []

    if needs_multi_step_check(clean_message, query_type):
        logger.info(f"🔀 Phase 2: Checking if multi-step reasoning needed...")
        from backend.core.llm_multi_step_reasoning import (
            detect_if_multi_step_needed,
            create_execution_plan,
            execute_plan
        )

        needs_multi_step, reasoning = await detect_if_multi_step_needed(
            message=clean_message,
            context_docs=relevant_docs,
            chat_client=chat_client,
            language=language
        )
    else:
        logger.info(f"⚡ Phase 2: Skipping multi-step check (simple query)")

    if needs_multi_step:
        logger.info(f"📋 Multi-step reasoning activated: {reasoning}")

        # Create execution plan
        execution_steps = await create_execution_plan(
            message=clean_message,
            context_docs=relevant_docs,
            chat_client=chat_client,
            language=language
        )

        if not execution_steps:
            logger.warning("Failed to create execution plan, falling back to single-step")
            needs_multi_step = False

    # Bundle components for tool execution
    from dataclasses import dataclass
    @dataclass
    class Components:
        vectorstore: Any
        embeddings: Any
        memory: Any
        chat_client: Any

    components = Components(
        vectorstore=vectorstore,
        embeddings=embeddings,
        memory=memory,
        chat_client=chat_client
    )

    def _record_session(user_message: str, assistant_message: str) -> None:
        try:
            session_manager.add_message(user_id, "user", user_message)
            session_manager.add_message(user_id, "assistant", assistant_message)
        except Exception as e:
            logger.warning(f"Failed to add messages to session history: {e}")

    # Phase 3: Execute plan or tools
    sources = []
    tool_results: List[ToolResult] = []  # Initialize for both paths

    if needs_multi_step and execution_steps:
        logger.info(f"⚙️ Phase 3: Executing multi-step plan ({len(execution_steps)} steps)...")

        # Execute the plan
        step_results = await execute_plan(
            steps=execution_steps,
            user_id=user_id,
            components=components,
            chat_client=chat_client
        )

        # Get final answer from synthesis step
        final_step = step_results[-1] if step_results else None
        if final_step and final_step.success:
            response_content = final_step.summary
        else:
            if language == "de":
                response_content = "Es tut mir leid, ich konnte die Aufgabe nicht vollständig ausführen."
            else:
                response_content = "I'm sorry, I couldn't complete the task fully."

        # Prepare sources from all steps for self-reflection
        for result in step_results:
            if result.success and result.summary:
                sources.append(f"Step {result.step_number}: {result.summary[:200]}")

        logger.info(f"✅ Multi-step execution completed with {len(step_results)} steps")

    else:
        # Shortcut: local time/date queries should not rely on web search
        if is_time_query(clean_message, language):
            logger.info("⏰ Time query detected - using system_time tool")
            tool_calls = [{"tool": "system_time", "params": {}}]
            logger.info(f"⚙️ Executing {len(tool_calls)} tools...")
        # SMART HOME: Try Fast-Path first, fallback to LLM if needed
        elif query_type == QueryType.SMART_HOME_CONTROL or query_type == QueryType.SMART_HOME_QUERY:
            logger.info(f"🏠 Phase 3: Smart Home - Trying Fast-Path...")

            # Extract entity and action from message
            message_lower = clean_message.lower()
            use_llm_fallback = False  # Track if we need LLM help

            # Determine action for CONTROL queries
            if query_type == QueryType.SMART_HOME_CONTROL:
                if any(word in message_lower for word in ["ein", "an", "on", "aktiviere"]):
                    action = "turn_on"
                elif any(word in message_lower for word in ["aus", "ab", "off", "deaktiviere"]):
                    action = "turn_off"
                else:
                    action = "toggle"

                # Extract room/device name (simple pattern matching)
                rooms = ["wohnzimmer", "badezimmer", "küche", "schlafzimmer", "büro", "flur", "keller"]
                device_hint = None
                if any(token in message_lower for token in ["licht", "lampe"]):
                    device_hint = "licht"
                elif any(token in message_lower for token in ["heizung", "thermostat"]):
                    device_hint = "heizung"
                elif any(token in message_lower for token in ["steckdose", "stecker", "schalter", "switch"]):
                    device_hint = "schalter"
                elif any(token in message_lower for token in ["rollo", "jalousie", "rolladen", "vorhang", "abdeckung", "cover"]):
                    device_hint = "rollo"
                elif any(token in message_lower for token in ["tv", "fernseher", "radio", "musik", "media", "player"]):
                    device_hint = "media"
                elif any(token in message_lower for token in ["luefter", "lüfter", "ventilator"]):
                    device_hint = "luefter"
                elif any(token in message_lower for token in ["schloss", "tuer", "tür", "lock"]):
                    device_hint = "schloss"
                elif any(token in message_lower for token in ["szene", "scene"]):
                    device_hint = "szene"
                entity_id = None
                confidence = 0  # Track confidence level

                for room in rooms:
                    if room in message_lower:
                        if device_hint:
                            entity_id = f"{room} {device_hint}"
                            confidence = 95
                        else:
                            entity_id = room
                            confidence = 100  # High confidence - exact match
                        break

                if not entity_id:
                    # Try to extract any word after "das" or "im" or "in der"
                    match = re.search(r'(?:das|im|in der|in) (\w+)', message_lower)
                    if match:
                        entity_id = match.group(1)
                        confidence = 50  # Medium confidence - pattern match
                    else:
                        # No entity found - need LLM help!
                        logger.info(f"❓ Fast-Path uncertain - no entity found, falling back to LLM")
                        use_llm_fallback = True

                if not use_llm_fallback:
                    tool_calls = [{
                        "tool": "home_assistant_control",
                        "params": {"entity_id": entity_id, "action": action}
                    }]
                    logger.info(f"🏠 Fast-Path: {entity_id} -> {action} (confidence: {confidence}%)")

            else:  # SMART_HOME_QUERY
                # Extract entity for query
                rooms = ["wohnzimmer", "badezimmer", "küche", "schlafzimmer", "büro", "flur", "keller"]
                device_hint = None
                if any(token in message_lower for token in ["licht", "lampe"]):
                    device_hint = "licht"
                elif any(token in message_lower for token in ["heizung", "thermostat"]):
                    device_hint = "heizung"
                elif any(token in message_lower for token in ["steckdose", "stecker", "schalter", "switch"]):
                    device_hint = "schalter"
                elif any(token in message_lower for token in ["rollo", "jalousie", "rolladen", "vorhang", "abdeckung", "cover"]):
                    device_hint = "rollo"
                elif any(token in message_lower for token in ["tv", "fernseher", "radio", "musik", "media", "player"]):
                    device_hint = "media"
                elif any(token in message_lower for token in ["luefter", "lüfter", "ventilator"]):
                    device_hint = "luefter"
                elif any(token in message_lower for token in ["schloss", "tuer", "tür", "lock"]):
                    device_hint = "schloss"
                elif any(token in message_lower for token in ["szene", "scene"]):
                    device_hint = "szene"
                entity_id = None
                confidence = 0

                for room in rooms:
                    if room in message_lower:
                        if device_hint:
                            entity_id = f"{room} {device_hint}"
                            confidence = 95
                        else:
                            entity_id = room
                            confidence = 100
                        break

                if not entity_id:
                    # No entity found - need LLM help!
                    logger.info(f"❓ Fast-Path uncertain - no entity found, falling back to LLM")
                    use_llm_fallback = True

                if not use_llm_fallback:
                    tool_calls = [{
                        "tool": "home_assistant_query",
                        "params": {"entity_id": entity_id}
                    }]
                    logger.info(f"🏠 Fast-Path: Query {entity_id} (confidence: {confidence}%)")

            # If Fast-Path failed, use LLM
            if use_llm_fallback:
                logger.info(f"🤖 Fallback: Using LLM for Smart Home tool selection...")
                tool_calls = await select_tools(
                    message=clean_message,
                    context_docs=relevant_docs,
                    chat_client=chat_client,
                    language=language
                )

            logger.info(f"⚙️ Executing {len(tool_calls)} Smart Home tools...")

        # Check if tools are needed for this query type
        elif needs_tools(query_type):
            pending_automation = session_manager.get_pending_automation(user_id)
            pending_script = session_manager.get_pending_script(user_id)

            if is_confirmation(clean_message) and pending_automation:
                logger.info("🧾 Confirmation detected - applying pending automation")
                tool_calls = [{
                    "tool": "home_assistant_create_automation",
                    "params": {"automation": pending_automation, "apply": True}
                }]
            elif is_confirmation(clean_message) and pending_script:
                logger.info("🧾 Confirmation detected - applying pending script")
                tool_calls = [{
                    "tool": "home_assistant_create_script",
                    "params": {"script": pending_script, "apply": True}
                }]
            else:
                from backend.core.web_search_integration import should_perform_web_search

                should_search, reason = should_perform_web_search(
                    clean_message,
                    relevant_docs,
                    language=language
                )

                if should_search:
                    tool_calls = [{
                        "tool": "web_search",
                        "params": {
                            "query": clean_message,
                            "reason": reason or "Heuristic web search trigger"
                        }
                    }]
                    logger.info(f"⚙️ Executing {len(tool_calls)} tools (heuristic)")
                else:
                    tool_calls = []
                    logger.info("⚡ No tools selected (heuristic) - answering directly")
        else:
            logger.info(f"⚡ Phase 3: Fast-path - Direct LLM response (no tools needed)")
            tool_calls = []

        for tool_call in tool_calls:
            result = await execute_tool(tool_call, user_id, components)
            tool_results.append(result)

            if result.success:
                logger.info(f"✅ Tool {result.tool_name} succeeded")
            else:
                logger.warning(f"❌ Tool {result.tool_name} failed: {result.error}")

            if result.tool_name == "home_assistant_create_automation" and result.data:
                if result.data.get("preview") and result.data.get("automation") and result.data.get("valid"):
                    session_manager.set_pending_automation(user_id, result.data["automation"])
                elif result.success:
                    session_manager.clear_pending_automation(user_id)

            if result.tool_name == "home_assistant_create_script" and result.data:
                if result.data.get("preview") and result.data.get("script") and result.data.get("valid"):
                    session_manager.set_pending_script(user_id, result.data["script"])
                elif result.success:
                    session_manager.clear_pending_script(user_id)

        # Check if clarification was requested
        for result in tool_results:
            if result.tool_name == "ask_clarification" and result.success:
                meta_style_query = any(token in clean_message.lower() for token in [
                    "roboterhaft", "antwortgefühl", "antwort klang", "stil", "tonalität", "tonfall"
                ])
                if meta_style_query:
                    logger.info("🧾 Skipping clarification for meta-style query")
                    break
                clarification_data = result.data
                question = clarification_data.get('question', '')
                options = clarification_data.get('options', [])

                # Return clarification as proper dict structure
                if options:
                    clarification_response = f"{question}\n\nOptionen:\n"
                    for i, opt in enumerate(options, 1):
                        clarification_response += f"{i}. {opt}\n"
                else:
                    clarification_response = question

                # FIX: Return dict instead of string for consistency
                _record_session(clean_message, clarification_response)
                return {
                    "response": clarification_response,
                    "relevant_memory": relevant_docs,
                    "final": memory_used,
                    "source": "clarification",
                    "turn_id": None
                }

    # Phase 4: Build answer with tool results (ONLY for single-step)
    logger.info(f"📝 Phase 4: Building answer with tool results...")

    # If memory retrieval fell back to low-confidence only, ask a short clarification
    if relevant_docs:
        try:
            low_confidence_only = all(
                getattr(doc, "metadata", {}).get("low_confidence") for doc in relevant_docs
            )
        except Exception:
            low_confidence_only = False

        if low_confidence_only:
            if language == "de":
                clarification = "Ich bin mir nicht sicher, ob ich dich richtig verstanden habe. Was genau meinst du?"
            else:
                clarification = "I'm not fully sure I understood you. What exactly do you mean?"
            _record_session(clean_message, clarification)
            return {
                "response": clarification,
                "relevant_memory": relevant_docs,
                "final": memory_used,
                "source": "low_confidence_clarification",
                "turn_id": None
            }

        # ⚠️ CRITICAL: Check for Home Assistant tool failures BEFORE asking LLM
        ha_tools_used = any(r.tool_name in ["home_assistant_control", "home_assistant_query"] for r in tool_results)
        if ha_tools_used:
            ha_failed = [r for r in tool_results if r.tool_name in ["home_assistant_control", "home_assistant_query"] and not r.success]
            if ha_failed:
                # Return error immediately - DON'T let LLM generate false positive
                error_msg = ha_failed[0].error
                logger.error(f"🚨 Home Assistant tool failed - returning error directly: {error_msg}")
                error_response = f"❌ {error_msg}\n\nBitte prüfe den Entity-Namen in Home Assistant."
                _record_session(clean_message, error_response)
                return {
                    "response": error_response,
                    "relevant_memory": relevant_docs,
                    "final": memory_used,
                    "source": "ha_tool_error",
                    "turn_id": None
                }

        # If all real tools failed, return an honest fallback to avoid hallucination
        real_tool_results = [r for r in tool_results if r.tool_name != "no_tool"]
        if real_tool_results and all(not r.success for r in real_tool_results):
            error_msg = real_tool_results[0].error or "Tool failed"
            logger.error(f"🚨 All tools failed - returning error directly: {error_msg}")
            error_response = f"❌ {error_msg}. Ich kann das gerade nicht zuverlässig beantworten."
            _record_session(clean_message, error_response)
            return {
                "response": error_response,
                "relevant_memory": relevant_docs,
                "final": memory_used,
                "source": "tool_error",
                "turn_id": None
            }

        tool_context = _format_tool_results(tool_results)

        # Check if actual tools (not no_tool) were used
        real_tools_used = any(r.tool_name != "no_tool" for r in tool_results)

        # Build final prompt - adapt based on whether tools were used
        # Determine prompt type based on tool usage
        if real_tools_used:
            # Check if home assistant tools were used
            ha_tools_used = any(r.tool_name in ["home_assistant_control", "home_assistant_query"] for r in tool_results)
            ha_automation_used = any(
                r.tool_name in ["home_assistant_create_automation", "home_assistant_create_script"]
                for r in tool_results
            )
            if ha_automation_used:
                prompt_type = "ha_automation"
            else:
                prompt_type = "ha_control" if ha_tools_used else "tools_used"
        else:
            prompt_type = "no_tools"

        # Build system prompt using prompt_builder
        system_prompt = build_system_prompt(
            prompt_type=prompt_type,
            language=language,
            user_context=user_context,
            greeting_instruction=greeting_instruction,
            has_existing_memories=has_existing_memories,
            context_summary=context_summary,
            tool_context=tool_context,
            user_profile=user_profile,
            user_message=clean_message
        )

        # Conversation history already retrieved earlier (after query classification)
        # session_manager and conversation_history are already available

        # Build messages with history
        messages = [{'role': 'system', 'content': system_prompt}]

        # Add conversation history (previous messages)
        if conversation_history:
            messages.extend(conversation_history)
            logger.info(f"📜 Added {len(conversation_history)} previous messages to context")

        if no_think:
            think_msg = "Antworte direkt ohne zu denken." if is_german else "Respond directly without thinking."
            messages.append({'role': 'system', 'content': think_msg})

        # Add current user message
        messages.append({'role': 'user', 'content': clean_message})

        response = await chat_client.ainvoke(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)

    # Phase 5: Self-Reflection (CONDITIONAL)
    tools_were_used = bool(tool_results) or needs_multi_step
    has_web_results = any(
        result.success and result.tool_name == "web_search" and result.data and result.data.get("results")
        for result in tool_results
    )
    has_sources = bool(sources) or bool(relevant_docs) or has_web_results
    should_reflect, reflection_reason = needs_self_reflection(
        query_type=query_type,
        tools_used=tools_were_used,
        response_content=response_content,
        has_sources=has_sources
    )

    if should_reflect:
        logger.info(f"🔍 Phase 5: Self-reflection check ({reflection_reason})...")
    else:
        logger.info(f"⚡ Phase 5: Skipping self-reflection ({reflection_reason})")
        # Skip directly to Phase 6: Store in memory (INTELLIGENT)
        if not no_think:
            logger.info(f"💾 Phase 6: Storing conversation in memory (using memory_handler)...")
            await store_conversation_memory(
                clean_message=clean_message,
                response_content=response_content,
                tool_results=tool_results,
                user_id=user_id,
                is_german=is_german,
                memory=memory
            )

        await _update_user_profile_from_conversation(
            profile_builder=profile_builder,
            user_id=user_id,
            user_message=clean_message,
            assistant_response=response_content,
            current_profile=user_profile or {}
        )

        # Record conversation turn using memory_handler
        turn_id = record_conversation_turn(user_id, message, response_content, relevant_docs)

        _record_session(clean_message, response_content)

        # FIX: Return dict with response AND memory entries (fast-path without self-reflection)
        return {
            "response": response_content,
            "relevant_memory": relevant_docs,
            "final": memory_used,
            "source": "llm_tool_calling",
            "turn_id": turn_id
        }

    # Add memory as sources (keep existing sources from multi-step if any)
    if relevant_docs:
        for doc in relevant_docs:
            content = getattr(doc, 'page_content', str(doc))
            sources.append(f"Memory: {content[:200]}")

    # Add tool results as sources (only for single-step path)
    if not needs_multi_step:
        for result in tool_results:
            if result.success and result.tool_name == "web_search" and result.data:
                web_results = result.data.get('results', [])
                for web_result in web_results[:3]:
                    title = web_result.get('title', '')
                    content = web_result.get('content', '')[:150]
                    sources.append(f"Web: {title}: {content}")

    # Verify answer quality
    is_valid, issue = await verify_answer_quality(
        clean_message,
        response_content,
        sources,
        chat_client,
        language
    )

    if not is_valid:
        logger.warning(f"🔄 Self-reflection failed: {issue}. Generating honest fallback.")
        response_content = await generate_honest_fallback(
            clean_message,
            chat_client,
            language
        )

    # Phase 6: Store in memory (INTELLIGENT - with self-reflection) using memory_handler
    if not no_think:
        logger.info(f"💾 Phase 6: Storing conversation in memory (using memory_handler)...")
        await store_conversation_memory(
            clean_message=clean_message,
            response_content=response_content,
            tool_results=tool_results,
            user_id=user_id,
            is_german=is_german,
            memory=memory
        )

    await _update_user_profile_from_conversation(
        profile_builder=profile_builder,
        user_id=user_id,
        user_message=clean_message,
        assistant_response=response_content,
        current_profile=user_profile or {}
    )

    # Record conversation turn using memory_handler
    turn_id = record_conversation_turn(user_id, message, response_content, relevant_docs)

    _record_session(clean_message, response_content)

    # FIX: Return dict with response AND memory entries so they appear in API response
    return {
        "response": response_content,
        "relevant_memory": relevant_docs,
        "final": memory_used,
        "source": "llm_tool_calling",
        "turn_id": turn_id
    }


def _format_tool_results(results: List[ToolResult]) -> str:
    """Format tool results for LLM context"""
    if not results:
        return "No tools were executed."

    formatted = ""
    for result in results:
        formatted += f"\n**Tool: {result.tool_name}**\n"

        if not result.success:
            formatted += f"  Status: Failed - {result.error}\n"
            continue

        formatted += "  Status: Success\n"

        if result.tool_name == "web_search" and result.data:
            web_results = result.data.get('results', [])
            formatted += f"  Found {len(web_results)} results:\n"
            for i, web_result in enumerate(web_results[:3], 1):
                title = web_result.get('title', 'Untitled')
                content = web_result.get('content', '')[:200]
                url = web_result.get('url', '')
                formatted += f"    {i}. {title}\n"
                formatted += f"       {content}...\n"
                formatted += f"       Source: {url}\n"

        elif result.tool_name == "memory_search" and result.data:
            memories = result.data.get('memories', [])
            formatted += f"  Found {len(memories)} memory entries:\n"
            for i, mem in enumerate(memories[:3], 1):
                content = getattr(mem, 'page_content', str(mem))
                formatted += f"    {i}. {content[:150]}...\n"

        elif result.tool_name == "no_tool":
            formatted += "  No external tools needed\n"

        elif result.tool_name == "home_assistant_control" and result.data:
            # Format Home Assistant control result
            entity_id = result.data.get('entity_id', '')
            action = result.data.get('action', '')
            formatted += f"  Device: {entity_id}\n"
            formatted += f"  Action: {action}\n"
            if result.data.get('value'):
                formatted += f"  Value: {result.data['value']}\n"
            if result.data.get('state'):
                formatted += f"  Status: {result.data['state']}\n"

        elif result.tool_name == "home_assistant_query" and result.data:
            # Format Home Assistant sensor query result with formatted values
            entity_id = result.data.get('entity_id', '')
            formatted_value = result.data.get('formatted_value', '')
            sensor_type = result.data.get('sensor_type', '')
            friendly_name = result.data.get('friendly_name', entity_id)

            formatted += f"  Gerät: {friendly_name}\n"
            formatted += f"  Wert: {formatted_value}\n"
            if sensor_type:
                formatted += f"  Typ: {sensor_type}\n"

        elif result.tool_name == "home_assistant_create_automation" and result.data:
            preview = result.data.get("preview")
            valid = result.data.get("valid")
            errors = result.data.get("errors", [])
            automation = result.data.get("automation", {})
            alias = automation.get("alias", "Automation")
            summary = result.data.get("summary")

            formatted += f"  Automation: {alias}\n"
            formatted += f"  Preview: {preview}\n"
            if valid is not None:
                formatted += f"  Valid: {valid}\n"
            if summary:
                formatted += f"  Summary: {summary}\n"
            if errors:
                formatted += "  Errors:\n"
                for err in errors[:3]:
                    formatted += f"    - {err}\n"

        elif result.tool_name == "home_assistant_create_script" and result.data:
            preview = result.data.get("preview")
            valid = result.data.get("valid")
            errors = result.data.get("errors", [])
            script = result.data.get("script", {})
            alias = script.get("alias", "Script")
            summary = result.data.get("summary")

            formatted += f"  Script: {alias}\n"
            formatted += f"  Preview: {preview}\n"
            if valid is not None:
                formatted += f"  Valid: {valid}\n"
            if summary:
                formatted += f"  Summary: {summary}\n"
            if errors:
                formatted += "  Errors:\n"
                for err in errors[:3]:
                    formatted += f"    - {err}\n"

    return formatted


async def _update_user_profile_from_conversation(
    profile_builder: ProfileBuilder,
    user_id: str,
    user_message: str,
    assistant_response: str,
    current_profile: Dict[str, Any]
) -> None:
    """
    Extract and persist user profile signals from the latest conversation.
    """
    try:
        if not profile_builder.should_update_profile(user_message):
            logger.debug("⏭️ Skipping profile update (message too short/generic)")
            return

        logger.info("🧠 Starting profile learning from conversation...")
        updated_profile = await profile_builder.extract_profile_info(
            user_message=user_message,
            assistant_response=assistant_response,
            current_profile=current_profile
        )

        if updated_profile != current_profile:
            logger.info(f"📝 Profile updated with new information: {list(updated_profile.keys())}")
            summary = profile_builder.get_profile_summary(updated_profile)
            logger.debug(f"Profile Summary:\n{summary}")

            try:
                from backend.services.user_store import get_user_store
                get_user_store().update_user(user_id, {"profile": updated_profile})
                logger.info(f"💾 Profile persisted for user {user_id}")
                try:
                    from backend.services.user_profile_qdrant import upsert_user_profile_in_qdrant
                    await asyncio.to_thread(upsert_user_profile_in_qdrant, user_id, updated_profile)
                except Exception as e:
                    logger.warning(f"Failed to upsert profile in Qdrant for {user_id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to persist profile for {user_id}: {e}")

    except Exception as e:
        logger.error(f"❌ Error in profile learning: {e}", exc_info=True)


# Convenience wrapper for compatibility
async def _run_chat_logic_with_tools(
    message, chat_client, vectorstore, memory, embeddings, streaming=False, user_id="default"
):
    """Wrapper for compatibility with existing chat API"""
    return await process_chat_with_tools(
        message=message,
        chat_client=chat_client,
        vectorstore=vectorstore,
        memory=memory,
        embeddings=embeddings,
        streaming=streaming,
        user_id=user_id
    )
