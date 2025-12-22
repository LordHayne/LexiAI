"""
LLM Tool-Calling System: LLM chooses tools autonomously
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None


# Tool definitions (JSON-Schema style)
AVAILABLE_TOOLS = {
    "web_search": {
        "name": "web_search",
        "description": "Suche im Internet nach aktuellen Informationen. Verwende dies für: aktuelle News, Fakten über Firmen/Personen/Orte, Informationen die sich ändern (Preise, Termine, etc.)",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Die Suchanfrage (3-8 Wörter, spezifisch und kontextreich)"
            },
            "reason": {
                "type": "string",
                "description": "Warum diese Suche notwendig ist"
            }
        },
        "required": ["query", "reason"]
    },
    "system_time": {
        "name": "system_time",
        "description": "Gibt die aktuelle Uhrzeit und das Datum des Servers zurück. Verwende dies für Fragen wie 'Wie spät ist es?' oder 'Welches Datum haben wir?'.",
        "parameters": {},
        "required": []
    },
    "memory_search": {
        "name": "memory_search",
        "description": "Durchsuche das Langzeitgedächtnis nach spezifischen Informationen. Verwende dies für: frühere Gespräche, persönliche Präferenzen des Users, bereits gelernte Fakten.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Wonach im Memory gesucht werden soll"
            },
            "limit": {
                "type": "integer",
                "description": "Maximale Anzahl an Ergebnissen (default: 5)"
            }
        },
        "required": ["query"]
    },
    "ask_clarification": {
        "name": "ask_clarification",
        "description": "Stelle eine Rückfrage an den User wenn die Anfrage mehrdeutig oder unklar ist. Verwende dies wenn: mehrere Interpretationen möglich sind, wichtige Details fehlen, oder du dir unsicher bist was gemeint ist.",
        "parameters": {
            "question": {
                "type": "string",
                "description": "Die konkrete Frage an den User"
            },
            "options": {
                "type": "array",
                "description": "Optionale Antwortmöglichkeiten (2-4 Optionen)"
            }
        },
        "required": ["question"]
    },
    "no_tool": {
        "name": "no_tool",
        "description": "Kein Tool benötigt - die Frage kann direkt beantwortet werden. Verwende dies für: allgemeines Wissen, Konversation, einfache Fragen die keine externe Info benötigen.",
        "parameters": {},
        "required": []
    },
    "home_assistant_control": {
        "name": "home_assistant_control",
        "description": "**WICHTIG**: Verwende dieses Tool für ALLE Smart Home Steuerungen! Licht ein/aus, Schalter, Thermostat, etc. Funktioniert mit natürlichen Namen wie 'Wohnzimmer', 'Badezimmer', 'Küche' - du musst KEINE Entity-ID kennen!",
        "parameters": {
            "entity_id": {
                "type": "string",
                "description": "Name oder Entity-ID. Beispiele: 'Wohnzimmer' (wird automatisch zu light.wohnzimmer), 'light.bad', 'Küche', 'switch.kaffeemaschine'. Nutze den natürlichen Namen aus der User-Anfrage!"
            },
            "action": {
                "type": "string",
                "description": "Aktion: turn_on (einschalten), turn_off (ausschalten), toggle (umschalten)",
                "enum": ["turn_on", "turn_off", "toggle", "set_brightness", "set_temperature"]
            },
            "value": {
                "type": "number",
                "description": "Optionaler Wert für Helligkeit (0-255) oder Temperatur (in °C)"
            }
        },
        "required": ["entity_id", "action"]
    },
    "home_assistant_query": {
        "name": "home_assistant_query",
        "description": "**WICHTIG**: Verwende dies um Sensor-Daten und Status von Smart Home Geräten abzufragen. Unterstützt: Temperatur, Luftfeuchtigkeit, Helligkeit, Lichtstatus, Schalter-Status, etc. Funktioniert mit natürlichen Namen!",
        "parameters": {
            "entity_id": {
                "type": "string",
                "description": "Name oder Entity-ID zum Abfragen. Beispiele: 'Wohnzimmer' (Temperatur/Licht), 'light.bad', 'Küche', 'climate.heizung'"
            }
        },
        "required": ["entity_id"]
    }
}


async def select_tools(
    message: str,
    context_docs: List[Any],
    chat_client,
    language: str = "de"
) -> List[Dict[str, Any]]:
    """
    LLM selects which tools to use for answering the question.

    Args:
        message: User's message
        context_docs: Available context from memory
        chat_client: LLM client
        language: Language

    Returns:
        List of tool calls: [{"tool": "web_search", "params": {"query": "..."}, ...}]
    """
    # Format context
    context_summary = ""
    if context_docs:
        context_summary = f"\n\nVerfügbarer Kontext aus Memory ({len(context_docs)} Einträge):\n"
        for i, doc in enumerate(context_docs[:3], 1):
            content = getattr(doc, 'page_content', str(doc))
            context_summary += f"{i}. {content[:150]}...\n"
    else:
        context_summary = "\n\nKein relevanter Kontext im Memory gefunden."

    # Format tools
    tools_text = ""
    for tool_name, tool_def in AVAILABLE_TOOLS.items():
        tools_text += f"\n**{tool_name}**\n"
        tools_text += f"  Beschreibung: {tool_def['description']}\n"
        tools_text += f"  Parameter: {json.dumps(tool_def['parameters'], ensure_ascii=False)}\n"

    if language == "de":
        system_prompt = """Du bist ein intelligenter Agent der entscheidet, welche Tools er zur Beantwortung einer Frage benötigt.

VERFÜGBARE TOOLS:
{tools}

DEINE AUFGABE:
Analysiere die Frage und entscheide, welche Tools du brauchst. Du kannst MEHRERE Tools wählen (z.B. memory_search DANN web_search).

ENTSCHEIDUNGSLOGIK (PRIORITÄTEN):

🏠 **HÖCHSTE PRIORITÄT - SMART HOME STEUERUNG**:
   - Erkenne Befehle wie: "schalte X ein/aus", "mach X an/aus", "stelle X auf Y", "dimme X"
   - Erkenne Abfragen wie: "ist X an/aus", "wie hell ist X", "welche Temperatur hat X"
   - **IMMER** home_assistant_control oder home_assistant_query verwenden!
   - Funktioniert mit natürlichen Namen: "Wohnzimmer", "Küche", "Badezimmer"
   - **KEINE no_tool bei Smart Home Anfragen!**

📊 **NORMALE PRIORITÄT**:
1. **web_search**: Wenn aktuelle/externe Infos nötig sind (News, Firmendaten, Fakten)
2. **memory_search**: Nur wenn mehr Memory-Details benötigt werden als im Context vorhanden
3. **ask_clarification**: Nur wenn die Frage wirklich unklar/mehrdeutig ist
4. **no_tool**: Wenn Context ausreicht ODER allgemeines Wissen/Konversation

WICHTIGE REGELN:
- **CRITICAL**: Bei Smart Home Anfragen → **IMMER home_assistant_control/query** wählen!
- Bei Memory-Context relevanter Info → **no_tool** (außer Smart Home!)
- **ask_clarification** NICHT wählen wenn Context die Antwort hat!
- **memory_search** NICHT nötig wenn Context bereits Memories enthält
- **web_search** nur für aktuelle externe Fakten

❗❗❗ JSON SCHEMA - PFLICHTFELDER ❗❗❗
Du MUSST EXAKT diese zwei Felder verwenden:
1. "reasoning" (string) - Deine Begründung
2. "tools" (array) - Liste der Tools

VERBOTENE Felder: "response", "answer", "message", etc.
NUR "reasoning" und "tools" sind erlaubt!

EXAKTES FORMAT (kopiere dieses Schema):
{{
    "reasoning": "kurze Begründung deiner Entscheidung",
    "tools": [
        {{"tool": "tool_name", "params": {{"key": "value"}}}}
    ]
}}

BEISPIELE:

User: "Schalte das Licht im Wohnzimmer ein"
{{
    "reasoning": "Smart Home Steuerungsbefehl - home_assistant_control verwenden",
    "tools": [{{"tool": "home_assistant_control", "params": {{"entity_id": "Wohnzimmer", "action": "turn_on"}}}}]
}}

User: "Mach das Badezimmerlicht aus"
{{
    "reasoning": "Smart Home Steuerung - Licht ausschalten",
    "tools": [{{"tool": "home_assistant_control", "params": {{"entity_id": "Badezimmer", "action": "turn_off"}}}}]
}}

User: "Ist das Küchenlicht an?"
{{
    "reasoning": "Smart Home Status-Abfrage",
    "tools": [{{"tool": "home_assistant_query", "params": {{"entity_id": "Küche"}}}}]
}}

User: "Wie warm ist es im Wohnzimmer?"
{{
    "reasoning": "Temperatur-Sensor-Abfrage im Wohnzimmer",
    "tools": [{{"tool": "home_assistant_query", "params": {{"entity_id": "Wohnzimmer"}}}}]
}}

User: "Wie geht's dir?"
{{
    "reasoning": "Einfache Konversation, keine Tools nötig",
    "tools": [{{"tool": "no_tool", "params": {{}}}}]
}}

User: "Was sind die aktuellen News über Tesla?"
{{
    "reasoning": "Aktuelle Informationen benötigt, Memory hat keine aktuellen News",
    "tools": [{{"tool": "web_search", "params": {{"query": "aktuelle Tesla News 2025", "reason": "User fragt nach aktuellen Tesla News"}}}}]
}}

User: "Weißt du noch, wer ich bin?" (Context: "User: Ich bin Thomas Sigmund, der Besitzer...")
{{
    "reasoning": "Context enthält die Antwort - Thomas Sigmund ist der User. Keine Tools nötig!",
    "tools": [{{"tool": "no_tool", "params": {{}}}}]
}}

========================================
❗❗❗ KRITISCH - OUTPUT FORMAT ❗❗❗
========================================
Du MUSST mit VALIDEM JSON antworten!
KEINE Erklärungen! KEINE natürliche Sprache!
NUR das JSON-Objekt (genau wie in den Beispielen)!

Wenn du NICHT mit JSON antwortest = FEHLER!
========================================
""".format(tools=tools_text)
    else:
        system_prompt = """You are an intelligent agent that decides which tools to use for answering a question.

AVAILABLE TOOLS:
{tools}

YOUR TASK:
Analyze the question and decide which tools you need. You can choose MULTIPLE tools (e.g., memory_search THEN web_search).

DECISION LOGIC:
1. **web_search**: When current/external info is needed (news, company data, facts)
2. **memory_search**: When you want to specifically search memory (in addition to automatic search)
3. **ask_clarification**: When the question is unclear/ambiguous
4. **no_tool**: When the question can be answered directly (general knowledge, conversation)

IMPORTANT RULES:
- Prefer **no_tool** when no external info is needed
- Use **memory_search** only if automatic memory search isn't enough
- Use **web_search** for current/specific external facts
- When unsure: **ask_clarification** instead of guessing!

Answer ONLY with JSON (no explanation):
{{
    "reasoning": "brief justification of your decision",
    "tools": [
        {{"tool": "tool_name", "params": {{"key": "value"}}}},
        ...
    ]
}}
""".format(tools=tools_text)

    user_prompt = f"""User-Frage: "{message}"{context_summary}

DEINE ANTWORT MUSS EXAKT SO AUSSEHEN:
{{
    "reasoning": "deine Begründung hier",
    "tools": [{{"tool": "tool_name", "params": {{...}}}}]
}}

STARTE JETZT MIT DEM JSON:"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        logger.info(f"🔧 Tool selection response: {response_text[:300]}")

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
            logger.warning("No JSON found in tool selection response")
            return [{"tool": "no_tool", "params": {}}]

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
        selection = json.loads(json_str)

        reasoning = selection.get('reasoning', '')
        tools = selection.get('tools', [])

        logger.info(f"🤖 Tool selection: {len(tools)} tools chosen - {reasoning}")
        for tool_call in tools:
            tool_name = tool_call.get('tool', 'unknown')
            logger.info(f"   🔧 {tool_name}: {tool_call.get('params', {})}")

        return tools

    except Exception as e:
        logger.error(f"Error in tool selection: {e}")
        # Default fallback: no tool
        return [{"tool": "no_tool", "params": {}}]


async def execute_tool(
    tool_call: Dict[str, Any],
    user_id: str,
    components: Any
) -> ToolResult:
    """
    Execute a selected tool.

    Args:
        tool_call: {"tool": "tool_name", "params": {...}}
        user_id: User ID for context
        components: ComponentBundle with vectorstore, embeddings, etc.

    Returns:
        ToolResult with execution outcome
    """
    tool_name = tool_call.get('tool')
    params = tool_call.get('params', {})

    logger.info(f"⚙️ Executing tool: {tool_name} with params: {params}")

    try:
        if tool_name == "web_search":
            # Execute web search
            from backend.services.web_search import get_web_search_service
            web_service = get_web_search_service()

            if not web_service.is_enabled():
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error="Web search service not enabled"
                )

        elif tool_name == "system_time":
            from datetime import datetime

            now = datetime.now().astimezone()
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "iso": now.isoformat(timespec="seconds"),
                    "time": now.strftime("%H:%M"),
                    "date": now.strftime("%Y-%m-%d"),
                    "timezone": now.tzname()
                }
            )

            query = params.get('query', '')
            search_result = await web_service.search(
                query=query,
                max_results=5,
                search_depth="basic"
            )

            if search_result and search_result.get('results'):
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=search_result
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error="No results found"
                )

        elif tool_name == "memory_search":
            # Execute specific memory search
            query = params.get('query', '')
            limit = params.get('limit', 5)

            # Use same filtering logic as chat retrieval
            from backend.core.memory_handler import retrieve_and_filter_memories
            memories = await retrieve_and_filter_memories(
                query,
                components.vectorstore,
                user_id=user_id
            )
            # Filter out tool-summary artifacts from memory results
            filtered = []
            for mem in memories:
                mem_content = getattr(mem, 'content', getattr(mem, 'page_content', str(mem)))
                content_lower = mem_content.lower()
                if "es wurden" in content_lower and ("speicher" in content_lower or "memory-einträge" in content_lower):
                    continue
                filtered.append(mem)

            # If query is about the user's info, keep only personal-info memories
            query_lower = query.lower()
            if any(token in query_lower for token in ["über mich", "ueber mich", "meine information", "informationen", "mein name", "wer bin ich", "beruf", "job"]):
                personal_indicators = [
                    "ich heiße", "mein name", "ich bin", "arbeite", "beruf", "job", "profession"
                ]
                personal_filtered = []
                for mem in filtered:
                    mem_content = getattr(mem, 'content', getattr(mem, 'page_content', str(mem)))
                    if any(ind in mem_content.lower() for ind in personal_indicators):
                        personal_filtered.append(mem)
                filtered = personal_filtered or filtered

            memories = filtered[:limit]

            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={"memories": memories, "count": len(memories)}
            )

        elif tool_name == "ask_clarification":
            # Return clarification request
            question = params.get('question', '')
            options = params.get('options', [])

            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "clarification_needed": True,
                    "question": question,
                    "options": options
                }
            )

        elif tool_name == "no_tool":
            # No tool execution needed
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={"message": "No tool needed"}
            )

        elif tool_name == "home_assistant_control":
            # Execute Home Assistant device control
            from backend.services.home_assistant import get_ha_service
            ha_service = get_ha_service()

            if not ha_service.is_enabled():
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error="Home Assistant nicht konfiguriert. Setze LEXI_HA_URL und LEXI_HA_TOKEN."
                )

            entity_id_raw = params.get('entity_id', '')
            action = params.get('action', 'turn_on')
            value = params.get('value')

            # Entity resolution: if entity_id doesn't contain a dot, try to resolve it
            if '.' not in entity_id_raw:
                # Extract domain hint from action
                domain_hint = None
                if 'brightness' in action or action in ['turn_on', 'turn_off', 'toggle']:
                    domain_hint = 'light'
                elif 'temperature' in action:
                    domain_hint = 'climate'

                logger.info(f"🔍 Resolving natural query: '{entity_id_raw}' (domain hint: {domain_hint})")
                entity_id = await ha_service.resolve_entity(entity_id_raw, domain=domain_hint)

                if not entity_id:
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        data=None,
                        error=f"Konnte '{entity_id_raw}' nicht auflösen. Bitte verwende die vollständige Entity-ID (z.B. 'light.wohnzimmer')."
                    )
                logger.info(f"✅ Resolved '{entity_id_raw}' -> '{entity_id}'")
            else:
                entity_id = entity_id_raw

            result = await ha_service.control_device(entity_id, action, value)

            return ToolResult(
                tool_name=tool_name,
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )

        elif tool_name == "home_assistant_query":
            # Query Home Assistant device state
            from backend.services.home_assistant import get_ha_service
            ha_service = get_ha_service()

            if not ha_service.is_enabled():
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error="Home Assistant nicht konfiguriert"
                )

            entity_id_raw = params.get('entity_id', '')

            # ✅ Use new query_sensor() method for formatted sensor data
            logger.info(f"📊 Querying sensor: '{entity_id_raw}'")
            result = await ha_service.query_sensor(entity_id_raw)

            return ToolResult(
                tool_name=tool_name,
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )

        else:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=None,
            error=str(e)
        )
