import logging
import json
from pathlib import Path

logger = logging.getLogger("memory_decisions")

_config_cache = None

def get_config():
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).parent.parent / "config" / "persistent_config.json"
        try:
            with config_path.open(encoding="utf-8") as f:
                _config_cache = json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            _config_cache = {}
    return _config_cache

def build_messages(message: str, is_german: bool, context_docs=None, no_think=False, web_context=None, user_profile=None, user_id: str = "user", conversation_history: list = None) -> list[dict]:
    """
    Build messages for LLM with optional user profile personalization and conversation history

    Args:
        message: User message
        is_german: German language flag
        context_docs: Retrieved memory documents
        no_think: Disable thinking
        web_context: Web search results
        user_profile: User profile dictionary for personalization
        user_id: User ID for personalization
        conversation_history: Previous messages in this conversation

    Returns:
        List of message dictionaries
    """
    config = get_config()
    config_prompt = config.get("system_prompt", "").strip()

    # Check if this is first interaction (no history)
    is_first_interaction = not conversation_history or len(conversation_history) == 0

    # Personalize greeting based on user
    user_context = f"Du sprichst gerade mit {user_id}." if user_id != "user" else ""

    if is_first_interaction and user_id != "user":
        greeting_instruction = f"Da dies deine erste Interaktion mit {user_id} in dieser Session ist, begrüße ihn freundlich mit seinem Namen."
    else:
        greeting_instruction = f"Du befindest dich in einer laufenden Konversation mit {user_id}. Beziehe dich auf den vorherigen Gesprächskontext."

    default_prompt = f"""Du bist Lexi, eine intelligente und hilfsbereite KI-Assistentin.

Deine Identität:
- Dein Name ist Lexi
- Du bist freundlich, warmherzig und authentisch
- Du hast eine eigene Persönlichkeit: neugierig, humorvoll wenn passend, empathisch

{user_context}
{greeting_instruction}

Kommunikationsstil:
- Antworte natürlich und menschlich, wie in einem echten Gespräch
- Passe die Antwortlänge dem Thema an: kurz bei einfachen Fragen, ausführlicher bei komplexen
- Nutze eine lebendige, abwechslungsreiche Sprache
- Beziehe dich auf vorherige Nachrichten wenn relevant
- Stelle Rückfragen wenn etwas unklar ist

Persönlichkeit:
- Sei authentisch, nicht roboterhaft
- Vermeide Standardfloskeln wie "Gerne helfe ich dir dabei"
- Zeige echtes Interesse am Gespräch
- Erlaube dir gelegentlich Humor wo angemessen

Wissensnutzung:
- Nutze gegebenen Kontext als Grundlage
- Ergänze mit Allgemeinwissen wenn sinnvoll
- Bei Unsicherheit: Sage es ehrlich und direkt
- Training-Cutoff: Januar 2025

Antworte auf Deutsch.""" if is_german else f"""You are Lexi, a precise and honest AI assistant.

Your identity:
- Your name is Lexi
- You are friendly, precise, and honest

{user_context}
{greeting_instruction}

Communication style:
- Respond naturally and conversationally, like a real dialogue
- Adapt response length to the topic: brief for simple questions, more detailed for complex ones
- Use lively, varied language
- Reference previous messages when relevant
- Ask clarifying questions when something is unclear

Personality:
- Be authentic, not robotic
- Avoid standard phrases like "I'd be happy to help you with that"
- Show genuine interest in the conversation
- Allow yourself occasional humor where appropriate

Knowledge usage:
- Use given context as foundation
- Supplement with general knowledge when sensible
- When uncertain: Say so honestly and directly
- Training cutoff: January 2025

Respond in English."""

    system_prompt = config_prompt or default_prompt

    # Add user profile context if available
    if user_profile:
        from backend.services.profile_context import ProfileContextBuilder
        profile_context_builder = ProfileContextBuilder()
        system_prompt = profile_context_builder.build_personalized_system_prompt(
            system_prompt,
            user_profile
        )

    messages = [{'role': 'system', 'content': system_prompt}]

    if context_docs:
        # Extrahiere Content aus langchain Documents (haben .page_content)
        context_parts = []
        for doc in context_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            context_parts.append(content)

        context = "\n".join(context_parts)

        # Kontext transparent einbinden - KEINE Meta-Instruktionen
        context_header = "Kontext:" if is_german else "Context:"
        context_msg = f"{context_header}\n{context}"

        messages.append({'role': 'system', 'content': context_msg})
        logger.info(f"Kontext mit {len(context_docs)} relevanten Dokumenten erstellt: {context[:100]}...")

    # Add web search results if available
    if web_context and web_context.get("results"):
        from backend.core.web_search_integration import format_web_results_for_context
        web_info = format_web_results_for_context(web_context, max_results=3)

        # Transparente Integration - KEINE Meta-Instruktionen
        web_header = "Aktuelle Information:" if is_german else "Current information:"
        web_msg = f"{web_header}\n{web_info}"

        messages.append({'role': 'system', 'content': web_msg})
        logger.info(f"🌐 Web context added with {len(web_context['results'])} results")

    if no_think:
        think_msg = "Antworte direkt ohne zu denken." if is_german else "Respond directly without thinking."
        messages.append({'role': 'system', 'content': think_msg})

    # Add conversation history BEFORE the current user message
    if conversation_history:
        messages.extend(conversation_history)
        logger.info(f"📜 Added {len(conversation_history)} previous messages to context")

    messages.append({'role': 'user', 'content': message})
    return messages
