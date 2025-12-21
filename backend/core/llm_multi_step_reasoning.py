"""
LLM Multi-Step Reasoning: Plan and execute complex tasks in multiple steps
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStep:
    """A single step in the execution plan"""
    step_number: int
    description: str
    tool: str
    params: Dict[str, Any]
    depends_on: List[int]  # Which previous steps this depends on


@dataclass
class StepResult:
    """Result from executing a step"""
    step_number: int
    success: bool
    data: Any
    summary: str  # Brief summary of what was found


async def detect_if_multi_step_needed(
    message: str,
    context_docs: List[Any],
    chat_client,
    language: str = "de"
) -> Tuple[bool, str]:
    """
    Detect if a query requires multi-step reasoning.

    Args:
        message: User's message
        context_docs: Available context
        chat_client: LLM client
        language: Language

    Returns:
        Tuple of (needs_multi_step: bool, reasoning: str)
    """
    # Format context
    context_summary = ""
    if context_docs:
        context_summary = f"\n\nVerfügbarer Kontext ({len(context_docs)} Einträge):\n"
        for i, doc in enumerate(context_docs[:2], 1):
            content = getattr(doc, 'page_content', str(doc))
            context_summary += f"{i}. {content[:100]}...\n"

    if language == "de":
        system_prompt = """Du bist ein Analyzer der erkennt ob eine Aufgabe mehrere Schritte benötigt.

MULTI-STEP ist nötig bei:
- ✅ Vergleichen (z.B. "Vergleiche X und Y")
- ✅ Mehrere Entitäten (z.B. "Info über X und Y")
- ✅ Zeitreihen (z.B. "Was ist in den letzten 3 Monaten passiert")
- ✅ Bedingungen (z.B. "Wenn X dann Y")
- ✅ Komplexe Analysen (z.B. "Analysiere Vor- und Nachteile")

KEIN MULTI-STEP nötig bei:
- ❌ Einfache Fragen (z.B. "Was ist X?")
- ❌ Single-Entity Queries (z.B. "Aktuelle News über Tesla")
- ❌ Konversation (z.B. "Hallo")
- ❌ Direkte Fakten (z.B. "Hauptstadt von Deutschland")

Antworte NUR mit JSON:
{
    "multi_step_needed": true/false,
    "reasoning": "kurze Begründung",
    "confidence": 0.0-1.0
}"""
    else:
        system_prompt = """You are an analyzer that detects if a task requires multiple steps.

MULTI-STEP is needed for:
- ✅ Comparisons (e.g., "Compare X and Y")
- ✅ Multiple entities (e.g., "Info about X and Y")
- ✅ Time series (e.g., "What happened in the last 3 months")
- ✅ Conditions (e.g., "If X then Y")
- ✅ Complex analysis (e.g., "Analyze pros and cons")

NO MULTI-STEP needed for:
- ❌ Simple questions (e.g., "What is X?")
- ❌ Single-entity queries (e.g., "Current news about Tesla")
- ❌ Conversation (e.g., "Hello")
- ❌ Direct facts (e.g., "Capital of Germany")

Answer ONLY with JSON:
{
    "multi_step_needed": true/false,
    "reasoning": "brief justification",
    "confidence": 0.0-1.0
}"""

    user_prompt = f"""Frage: "{message}"{context_summary}

Benötigt diese Frage Multi-Step Reasoning?"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON - handle markdown
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```'):
            lines = cleaned_text.split('\n')
            cleaned_text = '\n'.join(lines[1:])
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3].strip()

        # Extract JSON
        start_idx = cleaned_text.find('{')
        if start_idx == -1:
            return False, "Could not parse response"

        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(cleaned_text)):
            if cleaned_text[i] == '{':
                brace_count += 1
            elif cleaned_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        json_str = cleaned_text[start_idx:end_idx]
        detection = json.loads(json_str)

        needs_multi_step = detection.get('multi_step_needed', False)
        reasoning = detection.get('reasoning', '')
        confidence = detection.get('confidence', 0.5)

        # Only use multi-step if confidence is high
        if needs_multi_step and confidence >= 0.7:
            logger.info(f"🔀 Multi-step reasoning needed: {reasoning} (confidence={confidence:.2f})")
            return True, reasoning
        else:
            logger.info(f"➡️ Single-step execution sufficient (confidence={confidence:.2f})")
            return False, reasoning

    except Exception as e:
        logger.error(f"Error in multi-step detection: {e}")
        return False, str(e)


async def create_execution_plan(
    message: str,
    context_docs: List[Any],
    chat_client,
    language: str = "de"
) -> List[ExecutionStep]:
    """
    Create a step-by-step execution plan for a complex task.

    Args:
        message: User's message
        context_docs: Available context
        chat_client: LLM client
        language: Language

    Returns:
        List of ExecutionStep objects
    """
    # Format context
    context_summary = ""
    if context_docs:
        context_summary = f"\n\nVerfügbarer Kontext:\n"
        for i, doc in enumerate(context_docs[:2], 1):
            content = getattr(doc, 'page_content', str(doc))
            context_summary += f"{i}. {content[:100]}...\n"

    if language == "de":
        system_prompt = """Du bist ein Task Planner der komplexe Aufgaben in Schritte zerlegt.

VERFÜGBARE TOOLS:
- web_search: Suche im Internet (Parameter: query)
- memory_search: Suche im Langzeitgedächtnis (Parameter: query)
- synthesize: Kombiniere Zwischenergebnisse (keine Parameter)

PLANUNGS-REGELN:
1. Jeder Schritt hat GENAU EIN Tool
2. Schritte sind sequenziell (1, 2, 3, ...)
3. Letzter Schritt ist immer "synthesize"
4. Max 3 Schritte
5. WICHTIG: Halte "description" und "reason" KURZ (max 3-4 Wörter)!

BEISPIEL für "Vergleiche Tesla und SpaceX News":
{
  "steps": [
    {"step": 1, "tool": "web_search", "params": {"query": "Tesla News 2025"}},
    {"step": 2, "tool": "web_search", "params": {"query": "SpaceX News 2025"}},
    {"step": 3, "tool": "synthesize", "params": {}}
  ]
}

WICHTIG: Generiere das KOMPLETTE JSON bis zum schließenden }! Nicht abbrechen!"""
    else:
        system_prompt = """You are a task planner that breaks down complex tasks into steps.

AVAILABLE TOOLS:
- web_search: Search internet (Parameter: query)
- memory_search: Search memory (Parameter: query)
- synthesize: Combine results (no parameters)

PLANNING RULES:
1. Each step has EXACTLY ONE tool
2. Steps are sequential (1, 2, 3, ...)
3. Last step is always "synthesize"
4. Max 3 steps

EXAMPLE for "Compare Tesla and SpaceX News":
{
  "steps": [
    {"step": 1, "tool": "web_search", "params": {"query": "Tesla News 2025"}},
    {"step": 2, "tool": "web_search", "params": {"query": "SpaceX News 2025"}},
    {"step": 3, "tool": "synthesize", "params": {}}
  ]
}

IMPORTANT: Generate COMPLETE JSON to closing }! Don't stop early!"""

    user_prompt = f"""Aufgabe: "{message}"{context_summary}

Erstelle einen Schritt-für-Schritt Plan."""

    # Retry logic for incomplete responses
    max_retries = 2
    last_response = ""
    plan_data = {}

    for attempt in range(max_retries):
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]

            # Add continuation prompt for retries
            if attempt > 0:
                messages.append({'role': 'assistant', 'content': last_response})
                messages.append({'role': 'user', 'content': 'Das JSON ist unvollständig. Bitte generiere das KOMPLETTE JSON bis zum schließenden }.'})

            response = await chat_client.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"📋 Plan creation response attempt {attempt+1} (length={len(response_text)}): {response_text[:200]}...")
            if len(response_text) < 50:
                logger.warning(f"Response seems too short! Full response: {response_text}")

            # Parse JSON - handle markdown
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                lines = cleaned_text.split('\n')
                cleaned_text = '\n'.join(lines[1:])
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3].strip()

            # Extract JSON
            start_idx = cleaned_text.find('{')
            if start_idx == -1:
                logger.warning(f"Attempt {attempt+1}: No JSON found in plan response")
                last_response = response_text
                continue

            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(cleaned_text)):
                if cleaned_text[i] == '{':
                    brace_count += 1
                elif cleaned_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            # Check if we found matching braces
            if end_idx == start_idx:
                logger.warning(f"Attempt {attempt+1}: No matching closing brace found")
                if attempt < max_retries - 1:
                    logger.info("Retrying with continuation prompt...")
                    last_response = response_text
                    continue
                else:
                    logger.error(f"Final attempt failed. Cleaned text: {cleaned_text}")
                    return []

            json_str = cleaned_text[start_idx:end_idx]

            if not json_str or json_str == "":
                logger.warning(f"Attempt {attempt+1}: Extracted JSON string is empty")
                last_response = response_text
                continue

            # Try to parse JSON
            plan_data = json.loads(json_str)

            # Success! Break out of retry loop
            logger.info(f"✅ Successfully parsed execution plan with {len(plan_data.get('steps', []))} steps")
            break

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON parsing error: {e}")
            if attempt < max_retries - 1:
                last_response = response_text
                continue
            else:
                logger.error("All retry attempts failed")
                return []
        except Exception as e:
            logger.error(f"Unexpected error in plan creation attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return []

    # Process the parsed plan
    try:
        steps_data = plan_data.get('steps', [])
        execution_steps = []

        for i, step_data in enumerate(steps_data, 1):
            # Generate description from tool and query if not provided
            tool_name = step_data.get('tool', '')
            params = step_data.get('params', {})
            description = step_data.get('description', '')
            if not description:
                if tool_name == 'web_search':
                    description = f"Search: {params.get('query', '')[:30]}"
                elif tool_name == 'memory_search':
                    description = f"Memory: {params.get('query', '')[:30]}"
                elif tool_name == 'synthesize':
                    description = "Synthesize results"
                else:
                    description = f"Step {i}"

            step = ExecutionStep(
                step_number=step_data.get('step', i),
                description=description,
                tool=tool_name,
                params=params,
                depends_on=step_data.get('depends_on', list(range(1, i)) if tool_name == 'synthesize' else [])
            )
            execution_steps.append(step)
            logger.info(f"  📌 Step {step.step_number}: {step.description} (tool: {step.tool})")

        return execution_steps

    except Exception as e:
        logger.error(f"Error creating execution plan: {e}")
        return []


async def execute_plan(
    steps: List[ExecutionStep],
    user_id: str,
    components: Any,
    chat_client
) -> List[StepResult]:
    """
    Execute an execution plan step by step.

    Args:
        steps: List of ExecutionStep objects
        user_id: User ID
        components: Component bundle
        chat_client: LLM client

    Returns:
        List of StepResult objects
    """
    results: List[StepResult] = []

    for step in steps:
        logger.info(f"⚙️ Executing step {step.step_number}: {step.description}")

        # Check dependencies
        if step.depends_on:
            missing_deps = [dep for dep in step.depends_on if dep > len(results)]
            if missing_deps:
                logger.error(f"Missing dependencies for step {step.step_number}: {missing_deps}")
                results.append(StepResult(
                    step_number=step.step_number,
                    success=False,
                    data=None,
                    summary=f"Failed: missing dependencies {missing_deps}"
                ))
                continue

        # Execute the tool
        if step.tool == "synthesize":
            # Special case: synthesize combines previous results
            result = await _synthesize_results(results, step.description, chat_client)
            results.append(result)

        else:
            # Execute tool using the existing tool execution system
            from backend.core.llm_tool_calling import execute_tool

            # Ensure params have required fields
            params = step.params.copy()
            if step.tool == "web_search" and "reason" not in params:
                params["reason"] = f"Step {step.step_number}"

            tool_call = {
                "tool": step.tool,
                "params": params
            }

            tool_result = await execute_tool(tool_call, user_id, components)

            # Convert to StepResult
            summary = _create_summary(tool_result, step.tool)
            step_result = StepResult(
                step_number=step.step_number,
                success=tool_result.success,
                data=tool_result.data,
                summary=summary
            )
            results.append(step_result)

            if step_result.success:
                logger.info(f"✅ Step {step.step_number} completed: {step_result.summary[:100]}")
            else:
                logger.warning(f"❌ Step {step.step_number} failed: {tool_result.error}")

    return results


async def _synthesize_results(
    results: List[StepResult],
    task_description: str,
    chat_client
) -> StepResult:
    """
    Synthesize multiple step results into a final answer.
    """
    # Prepare detailed results with full content for web searches
    results_detail = ""
    for i, result in enumerate(results, 1):
        if result.success:
            # For web search results, include full content
            if result.data and isinstance(result.data, dict) and 'results' in result.data:
                web_results = result.data['results'][:3]  # Top 3 results
                results_detail += f"\n=== Schritt {i} ===\n"
                for j, web_result in enumerate(web_results, 1):
                    title = web_result.get('title', 'Untitled')
                    content = web_result.get('content', web_result.get('snippet', ''))[:500]
                    results_detail += f"{j}. {title}\n{content}\n\n"
            elif result.data and isinstance(result.data, dict) and 'memories' in result.data:
                memories = result.data['memories'][:5]
                results_detail += f"\n=== Schritt {i} (Memory) ===\n"
                for j, mem in enumerate(memories, 1):
                    mem_content = getattr(mem, 'content', getattr(mem, 'page_content', str(mem)))
                    results_detail += f"{j}. {mem_content[:400]}\n"
            else:
                # For other results, use summary
                results_detail += f"\nSchritt {i}: {result.summary[:300]}\n"

    prompt = f"""Du hast mehrere Informations-Schritte ausgeführt. Kombiniere sie zu einer präzisen Antwort.

Aufgabe: {task_description}

Gefundene Informationen:
{results_detail}

Erstelle eine direkte, hilfreiche Antwort basierend auf diesen Informationen."""

    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = await chat_client.ainvoke(messages)
        final_answer = response.content if hasattr(response, 'content') else str(response)

        return StepResult(
            step_number=len(results) + 1,
            success=True,
            data={"final_answer": final_answer},
            summary=final_answer
        )

    except Exception as e:
        logger.error(f"Error synthesizing results: {e}")
        return StepResult(
            step_number=len(results) + 1,
            success=False,
            data=None,
            summary=f"Synthesis failed: {e}"
        )


def _create_summary(tool_result, tool_name: str) -> str:
    """Create a brief summary of a tool result"""
    if not tool_result.success:
        return f"Failed: {tool_result.error}"

    if tool_name == "web_search":
        results = tool_result.data.get('results', []) if tool_result.data else []
        if results:
            titles = [r.get('title', '')[:50] for r in results[:2]]
            return f"Found {len(results)} results: {', '.join(titles)}"
        return "No results found"

    elif tool_name == "memory_search":
        memories = tool_result.data.get('memories', []) if tool_result.data else []
        return f"Found {len(memories)} memory entries"

    else:
        return f"{tool_name} completed successfully"
