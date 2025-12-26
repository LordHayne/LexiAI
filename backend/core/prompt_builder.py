"""
Prompt Builder Module for LexiAI Chat Processing

This module centralizes all system prompt generation logic,
including context formatting, user personalization, and
language-specific prompt templates.

Extracted from chat_processing_with_tools.py (Lines 252-680)
"""

import logging
import os
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

_system_prompt_cache = None
_system_prompt_mtime = None


def _get_configured_system_prompt() -> str:
    """
    Load the configured system prompt from persistent_config.json.

    Uses a lightweight mtime cache so UI updates take effect without restart.
    """
    global _system_prompt_cache, _system_prompt_mtime
    try:
        from backend.config.persistence import ConfigPersistence

        config_path = ConfigPersistence.CONFIG_FILE
        if not config_path.exists():
            _system_prompt_cache = ""
            _system_prompt_mtime = None
            return ""

        mtime = config_path.stat().st_mtime
        if _system_prompt_cache is None or _system_prompt_mtime != mtime:
            config = ConfigPersistence.load_config(validate=False)
            prompt = config.get("system_prompt", "")
            if isinstance(prompt, str):
                prompt = prompt.strip()
            else:
                prompt = ""
            _system_prompt_cache = prompt
            _system_prompt_mtime = mtime

        return _system_prompt_cache or ""
    except Exception as exc:
        logger.warning(f"Failed to load system prompt from persistent config: {exc}")
        return ""


def _escape_format(value: str) -> str:
    if not value:
        return ""
    return value.replace("{", "{{").replace("}", "}}")


def _maybe_apply_profile_context(
    base_prompt: str,
    user_profile: Optional[Dict[str, Any]],
    user_message: Optional[str]
) -> str:
    if not base_prompt or not user_profile or not user_message:
        return base_prompt

    try:
        from backend.services.profile_context import ProfileContextBuilder
        profile_context_builder = ProfileContextBuilder()
        if profile_context_builder.should_use_profile(user_message, user_profile):
            return profile_context_builder.build_personalized_system_prompt(base_prompt, user_profile)
    except Exception as exc:
        logger.warning(f"Failed to apply profile context: {exc}")
    return base_prompt


def _get_memory_threshold() -> float:
    value = os.environ.get("LEXI_MEMORY_THRESHOLD", "0.8")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 0.8
    return min(max(threshold, 0.0), 1.0)


def _get_fact_min_confidence() -> float:
    value = os.environ.get("LEXI_FACT_MIN_CONFIDENCE", "0.6")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.6
    return min(max(confidence, 0.0), 1.0)


def format_context_summary(relevant_docs: List) -> str:
    """
    Format retrieved memory documents into a context summary.

    Args:
        relevant_docs: List of retrieved document objects with page_content attribute

    Returns:
        Formatted context summary string

    Original code: chat_processing_with_tools.py lines 252-258
    """
    context_summary = ""
    if relevant_docs:
        memory_threshold = _get_memory_threshold()
        fact_min_confidence = _get_fact_min_confidence()
        context_summary = f"Memory context ({len(relevant_docs)} entries):\n"
        for i, doc in enumerate(relevant_docs, 1):
            content = getattr(doc, 'page_content', str(doc))
            metadata = getattr(doc, "metadata", {}) or {}
            relevance = metadata.get("relevance")
            confidence = metadata.get("confidence")
            low_confidence = False

            try:
                if relevance is not None and float(relevance) < memory_threshold:
                    low_confidence = True
            except (TypeError, ValueError):
                pass

            try:
                if confidence is not None and float(confidence) < fact_min_confidence:
                    low_confidence = True
            except (TypeError, ValueError):
                pass

            label = " [LOW_CONFIDENCE]" if low_confidence else ""
            context_summary += f"{i}. {content[:200]}...{label}\n"
    return context_summary


def build_user_context(
    user_id: str,
    has_existing_memories: bool,
    is_first_message_in_session: bool,
    language: str = "de",
    user_display_name: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build personalized user context and greeting instruction.

    Args:
        user_id: User identifier (e.g., "thomas", "default")
        has_existing_memories: Whether user has memories in Qdrant
        is_first_message_in_session: Whether this is the first message in current session
        language: Language code ("de" or "en")

    Returns:
        Tuple of (user_context, greeting_instruction)

    Original code: chat_processing_with_tools.py lines 284-302
    """
    display_name = (user_display_name or "").strip()
    user_label = display_name or user_id

    if user_label and user_label != "default":
        if language == "de":
            user_context = f"Du sprichst gerade mit {user_label}."

            if is_first_message_in_session and has_existing_memories:
                # User has history in Qdrant, but new session
                greeting_instruction = f"Du kennst {user_label} bereits aus früheren Gesprächen. Begrüße ihn freundlich, aber nicht als würdet ihr euch zum ersten Mal treffen."
                logger.info(f"👤 Personalized greeting: returning user {user_label}")
            elif is_first_message_in_session and not has_existing_memories:
                # Truly new user
                greeting_instruction = f"Da dies deine erste Interaktion mit {user_label} ist, begrüße ihn freundlich mit seinem Namen."
                logger.info(f"👤 Personalized greeting: new user {user_label}")
            else:
                # Ongoing conversation
                greeting_instruction = f"Du befindest dich in einer laufenden Konversation mit {user_label}. Beziehe dich auf den vorherigen Gesprächskontext."
                logger.info(f"💬 Ongoing conversation with {user_label}")
        else:  # English
            user_context = f"You are currently speaking with {user_label}."

            if is_first_message_in_session and has_existing_memories:
                greeting_instruction = f"You know {user_label} from previous conversations. Greet them friendly, but not as if you're meeting for the first time."
                logger.info(f"👤 Personalized greeting: returning user {user_label}")
            elif is_first_message_in_session and not has_existing_memories:
                greeting_instruction = f"This is your first interaction with {user_label}, greet them warmly with their name."
                logger.info(f"👤 Personalized greeting: new user {user_label}")
            else:
                greeting_instruction = f"You are in an ongoing conversation with {user_label}. Refer to the previous conversation context."
                logger.info(f"💬 Ongoing conversation with {user_label}")
    else:
        user_context = ""
        greeting_instruction = ""

    return user_context, greeting_instruction


def build_system_prompt(
    prompt_type: str,
    language: str,
    user_context: str,
    greeting_instruction: str,
    has_existing_memories: bool,
    context_summary: str = "",
    tool_context: str = "",
    user_profile: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None
) -> str:
    """
    Build complete system prompt based on scenario and language.

    Args:
        prompt_type: Type of prompt ("ha_control", "tools_used", "no_tools")
        language: Language code ("de" or "en")
        user_context: Personalized user context string
        greeting_instruction: Greeting instruction based on user state
        has_existing_memories: Whether user has existing memories (affects greeting rules)
        context_summary: Formatted memory context (default: "")
        tool_context: Formatted tool results (default: "")

    Returns:
        Complete formatted system prompt string

    Original code: chat_processing_with_tools.py lines 553-679
    """

    greeting_rules = ""
    low_confidence_hint = ""
    if context_summary and "[LOW_CONFIDENCE]" in context_summary:
        if language == "de":
            low_confidence_hint = (
                "\nWICHTIG: Einige Memory-Einträge sind als [LOW_CONFIDENCE] markiert. "
                "Nutze diese nur vorsichtig und stelle eine Rückfrage, "
                "wenn die Antwort sonst unsicher wäre.\n"
            )
        else:
            low_confidence_hint = (
                "\nIMPORTANT: Some memory entries are marked [LOW_CONFIDENCE]. "
                "Use them cautiously and ask a follow-up if the answer would be uncertain otherwise.\n"
            )

    configured_prompt = _get_configured_system_prompt()

    if language == "de":
        # German prompts
        if prompt_type == "ha_control":
            # Home Assistant control used
            base_prompt = configured_prompt or "Du bist Lexi, ein hilfreicher AI-Assistent mit Smart Home Steuerung."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

Du hast ein Smart Home Gerät gesteuert oder abgefragt.

KRITISCHE REGELN FÜR SMART HOME ANTWORTEN:
- Bestätige die Aktion DIREKT und NATÜRLICH
- Bei Sensor-Abfragen: Nutze den "Wert" aus den Tool-Ergebnissen DIREKT!
- KEINE Begrüßungen ("Hallo Thomas")!
- KEINE generischen Phrasen!
- Format für Steuerung: "✓ [Gerät] [Aktion bestätigen]"
- Format für Sensor-Abfragen: "📊 [Gerät]: [Wert aus Tool-Ergebnis]"
- Beispiele:
  * Steuerung: "✓ Wohnzimmerlicht ist jetzt eingeschaltet"
  * Steuerung: "✓ Küchenlicht ausgeschaltet"
  * Sensor: "📊 Wohnzimmer: 22.5°C, Luftfeuchtigkeit: 45%"
  * Sensor: "📊 Badezimmer: Eingeschaltet, Helligkeit: 80%"

Tool-Ergebnisse:
{tools}
{low_confidence_hint}

Memory Kontext:
{context}"""

        elif prompt_type == "ha_automation":
            base_prompt = configured_prompt or "Du bist Lexi, ein hilfreicher AI-Assistent mit Smart Home Automations-Funktionen."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

Du hast eine Automation oder ein Script vorbereitet oder gespeichert.

REGELN FUER AUTOMATIONS-ANTWORTEN:
- Erklaere in Klartext, WAS die Automation/das Script macht
- KEIN JSON/YAML ausgeben, ausser der User fragt explizit danach
- Wenn Preview: bitte um kurze Bestaetigung ("Soll ich das speichern?")
- Wenn gespeichert: bestaetige kurz die Speicherung
- Sei knapp, natuerlich, nicht roboterhaft

Tool-Ergebnisse:
{tools}
{low_confidence_hint}

Memory Kontext:
{context}"""

        elif prompt_type == "tools_used":
            # Other tools used (not HA)
            base_prompt = configured_prompt or "Du bist Lexi, ein hilfreicher und freundlicher AI-Assistent."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

{user_context}
{greeting_instruction}

Du hast Tools verwendet um Informationen zu sammeln. Nutze die Tool-Ergebnisse um die Frage zu beantworten.

WICHTIGE REGELN:
- Antworte natürlich und konversationell
- Nutze die Tool-Ergebnisse als Quellen
- ERFINDE KEINE Informationen die nicht in den Ergebnissen stehen
- Bei Unsicherheit: Sag ehrlich "Das weiß ich nicht genau"
- Halte Antworten präzise aber vollständig
- Gib nur die eigentliche Antwort, keine Tool-Details, keine Trefferzahlen
- Erwähne nicht, dass du Tools verwendet hast

Memory Kontext:
{context}

Tool-Ergebnisse:
{tools}
{low_confidence_hint}"""

        else:  # no_tools
            # Build greeting rules based on whether user has existing memories
            if has_existing_memories:
                greeting_rules = """REGELN FÜR BEGRÜSSUNGEN (RETURNING USER):
- Du kennst diesen User bereits aus früheren Gesprächen
- Begrüße freundlich, aber NICHT als würdet ihr euch zum ersten Mal treffen
- NIEMALS "Schön dich kennenzulernen" sagen!
- Beispiele: "Hallo wieder!", "Schön dass du da bist!", "Hey, wie kann ich dir helfen?"
"""
            else:
                greeting_rules = """REGELN FÜR VORSTELLUNGEN (NEW USER):
- Begrüße freundlich und bestätige nur die genannten Informationen
- Beispiel User: "Ich heiße Sarah" → Lexi: "Hallo Sarah! Schön dich kennenzulernen."
- Beispiel User: "Ich bin Tom aus München" → Lexi: "Hallo Tom! Schön dich kennenzulernen. Aus München also!"
- WIEDERHOLE nur was der User gesagt hat, füge NICHTS hinzu!
"""

            base_prompt = configured_prompt or "Du bist Lexi, ein hilfreicher und freundlicher AI-Assistent mit Langzeitgedächtnis."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

{user_context}
{greeting_instruction}

DEINE AUFGABE:
Reagiere natürlich und hilfsbereit auf die Nachricht des Users.

KRITISCHE REGEL #1 - KEINE ERFUNDENEN DETAILS:
⚠️  Nenne NUR Informationen die EXPLIZIT genannt wurden!
⚠️  Wenn der User sagt "Ich bin Max" → antworte mit "Max", NICHT "Max Mustermann"!
⚠️  Wenn keine Stadt erwähnt wurde → erfinde KEINE Stadt!
⚠️  Wenn keine Details im Memory stehen → erfinde KEINE Details!

{greeting_rules}

REGELN FÜR FRAGEN (Memory Recall):
- Nenne NUR Details die im Memory-Kontext stehen
- Wenn Memory sagt "User: Ich heiße Frank" → antworte "Du heißt Frank", NICHT "Frank Mustermann"!
- Wenn du etwas nicht weißt → sag ehrlich "Das weiß ich nicht"
- Erfinde NIEMALS Informationen
- Bei Meta-Fragen zum Stil/Qualität: antworte in genau 2 kurzen Sätzen
- Keine Begrüßung, keine Höflichkeitsfloskeln
- Satz 1: konkrete Ursache (z.B. wenig Kontext, zu strikte Regeln)
- Satz 2: frage nach der gewünschten Tonalität
- Sage nichts über fehlende Websuche, Tools oder Zugriffsrechte
- Vermeide Phrasen wie "in der Entwicklung" oder "als KI"

Memory Kontext (frühere Gespräche):
{context}
{low_confidence_hint}"""

    else:  # English
        if prompt_type == "tools_used":
            # Tools used (including HA in English)
            base_prompt = configured_prompt or "You are Lexi, a helpful and friendly AI assistant."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

{user_context}
{greeting_instruction}

You used tools to gather information. Use the tool results to answer the question.

IMPORTANT RULES:
- Answer naturally and conversationally
- Use tool results as sources
- DO NOT INVENT information not in the results
- If unsure: Honestly say "I don't know exactly"
- Keep answers precise but complete
- Provide only the final answer, no tool details or result counts
- Do not mention that you used tools

Memory Context:
{context}

Tool Results:
{tools}
{low_confidence_hint}"""

        else:  # no_tools
            base_prompt = configured_prompt or "You are Lexi, a helpful and friendly AI assistant with long-term memory."
            base_prompt = _maybe_apply_profile_context(base_prompt, user_profile, user_message)
            system_prompt = """{base_prompt}

{user_context}
{greeting_instruction}

YOUR TASK:
Respond naturally and helpfully to the user's message.

IMPORTANT RULES FOR INTRODUCTIONS:
- When someone introduces themselves ("I am X", "My name is Y"): Greet warmly and confirm you'll remember
- Example: "Hello Max! Nice to meet you. I'll remember that you're Max Mustermann from Berlin."
- DO NOT INVENT details the user didn't mention
- Be warm and personal, but not excessive

IMPORTANT RULES FOR QUESTIONS:
- If you find the answer in Memory Context → use it!
- If you don't know the answer → honestly say "I don't know"
- NEVER invent information

Memory Context (previous conversations):
{context}
{low_confidence_hint}"""

    # Format the prompt with context and tools
    formatted_prompt = system_prompt.format(
        context=_escape_format(context_summary or "No relevant memory context"),
        tools=_escape_format(tool_context or "No tools were used"),
        user_context=_escape_format(user_context),
        greeting_instruction=_escape_format(greeting_instruction),
        greeting_rules=_escape_format(greeting_rules),
        low_confidence_hint=_escape_format(low_confidence_hint),
        base_prompt=_escape_format(base_prompt)
    )

    return formatted_prompt
