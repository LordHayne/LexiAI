"""
Chat Processing with LLM Tool-Calling System
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from backend.core.bootstrap import initialize_components
from backend.utils.model_utils import call_model_async
from backend.memory.activity_tracker import track_activity
from backend.memory.conversation_tracker import get_conversation_tracker
from backend.services.session_manager import get_session_manager
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

    # FIX: Initialize memory_used flag (used in return statement)
    memory_used = not no_think

    # Phase 1: Get initial memory context using memory_handler
    relevant_docs = await retrieve_and_filter_memories(clean_message, vectorstore, user_id)

    # Format context summary using prompt_builder
    context_summary = format_context_summary(relevant_docs)

    # Phase 1.5: Classify query for performance optimization
    query_type = classify_query(clean_message)
    logger.info(f"🎯 Query classified as: {query_type}")

    # Check if user has existing memories and if this is first message in session
    session_manager = get_session_manager()
    conversation_history = session_manager.get_conversation_history(user_id, max_turns=5)
    is_first_message_in_session = len(conversation_history) == 0

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

    user_display_name = None
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
        # SMART HOME: Try Fast-Path first, fallback to LLM if needed
        if query_type == QueryType.SMART_HOME_CONTROL or query_type == QueryType.SMART_HOME_QUERY:
            logger.info(f"🏠 Phase 3: Smart Home - Trying Fast-Path...")

            import re

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
                entity_id = None
                confidence = 0  # Track confidence level

                for room in rooms:
                    if room in message_lower:
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
                entity_id = None
                confidence = 0

                for room in rooms:
                    if room in message_lower:
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
            logger.info(f"🤖 Phase 3: Single-step execution - LLM selecting tools...")
            tool_calls = await select_tools(
                message=clean_message,
                context_docs=relevant_docs,
                chat_client=chat_client,
                language=language
            )

            logger.info(f"⚙️ Executing {len(tool_calls)} tools...")
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
                return {
                    "response": clarification_response,
                    "relevant_memory": relevant_docs,
                    "final": memory_used,
                    "source": "clarification",
                    "turn_id": None
                }

        # Phase 4: Build answer with tool results (ONLY for single-step)
        logger.info(f"📝 Phase 4: Building answer with tool results...")

        # ⚠️ CRITICAL: Check for Home Assistant tool failures BEFORE asking LLM
        ha_tools_used = any(r.tool_name in ["home_assistant_control", "home_assistant_query"] for r in tool_results)
        if ha_tools_used:
            ha_failed = [r for r in tool_results if r.tool_name in ["home_assistant_control", "home_assistant_query"] and not r.success]
            if ha_failed:
                # Return error immediately - DON'T let LLM generate false positive
                error_msg = ha_failed[0].error
                logger.error(f"🚨 Home Assistant tool failed - returning error directly: {error_msg}")
                return {
                    "response": f"❌ {error_msg}\n\nBitte prüfe den Entity-Namen in Home Assistant.",
                    "relevant_memory": relevant_docs,
                    "final": memory_used,
                    "source": "ha_tool_error",
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
            tool_context=tool_context
        )

        # Conversation history already retrieved earlier (after query classification)
        # session_manager and conversation_history are already available

        # Build messages with history
        messages = [{'role': 'system', 'content': system_prompt}]

        # Add conversation history (previous messages)
        if conversation_history:
            messages.extend(conversation_history)
            logger.info(f"📜 Added {len(conversation_history)} previous messages to context")

        # Add current user message
        messages.append({'role': 'user', 'content': clean_message})

        response = await chat_client.ainvoke(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)

    # Phase 5: Self-Reflection (CONDITIONAL)
    tools_were_used = bool(tool_results) or needs_multi_step

    if needs_self_reflection(query_type, tools_were_used):
        logger.info(f"🔍 Phase 5: Self-reflection check...")
    else:
        logger.info(f"⚡ Phase 5: Skipping self-reflection (simple query or no tools)")
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

        # Record conversation turn using memory_handler
        turn_id = record_conversation_turn(user_id, message, response_content, relevant_docs)

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

    # Record conversation turn using memory_handler
    turn_id = record_conversation_turn(user_id, message, response_content, relevant_docs)

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

    return formatted


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
