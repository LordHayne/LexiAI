"""
LLM-based intelligent decision making for web search.

This module replaces regex-based pattern matching with LLM reasoning.
"""
import logging
from typing import Tuple, Optional, List, Any
import json

logger = logging.getLogger(__name__)


async def should_perform_web_search_llm(
    message: str,
    context_docs: List[Any],
    chat_client,
    language: str = "de"
) -> Tuple[bool, Optional[str]]:
    """
    Use LLM to intelligently decide if web search is needed.

    OPTIMIZATION: Quick heuristic check BEFORE expensive LLM call.
    Performance: Saves 1-2s per query by avoiding LLM when clearly not needed.

    Args:
        message: User's message
        context_docs: Retrieved memory documents
        chat_client: LLM client for reasoning
        language: Language for reasoning

    Returns:
        Tuple of (should_search: bool, reason: str or None)
    """
    # OPTIMIZATION PHASE 2: Reordered checks with temporal PRIORITY
    message_lower = message.lower()

    # 1. IMMEDIATE TRIGGERS (bypass all other checks)
    explicit_web_search = [
        "such im internet", "search online", "search the web",
        "google", "suche im web", "search for", "suche nach",
        "find online", "finde im internet", "look up online"
    ]

    if any(keyword in message_lower for keyword in explicit_web_search):
        logger.info("🔍 Explicit search request detected - triggering web search")
        return True, "Explicit web search request"

    # 2. PERSONAL QUERIES (never use web search)
    personal_indicators = [
        "mein name", "my name", "ich heiße", "ich heisse", "wer bin ich",
        "kennst du mich", "remember me", "merkst du dir", "du solltest das wissen",
        "hast du dir gemerkt", "weißt du noch", "weißt du wie ich heiße"
    ]

    if any(indicator in message_lower for indicator in personal_indicators):
        logger.info("✓ No web search - personal query")
        return False, "Personal query - no search needed"

    # 3. TEMPORAL QUERIES (high priority - should trigger search)
    temporal_indicators = [
        "neueste", "aktuelle", "heute", "jetzt", "gerade",
        "latest", "current", "today", "now", "recent",
        "2024", "2025", "2026",  # Years
        "gestern", "yesterday", "morgen", "tomorrow",
        "diese woche", "this week", "diesen monat", "this month",
        "this year", "dieses jahr", "aktuell"
    ]

    has_temporal = any(indicator in message_lower for indicator in temporal_indicators)

    if has_temporal:
        # Temporal queries SHOULD trigger search unless explicitly memory-reference
        memory_references = [
            "wie du sagtest", "as you said", "wie bereits erwähnt", "as mentioned",
            "du hast gesagt", "you said", "vorhin", "earlier",
            "letztens", "neulich", "the other day"
        ]

        if any(ref in message_lower for ref in memory_references):
            logger.info("✓ No web search - temporal query but references past conversation")
            return False, "Temporal query referencing conversation history"

        logger.info("🔍 Temporal query detected - triggering web search")
        return True, "Temporal information requires up-to-date data"

    # 3. CONVERSATIONAL QUERIES (quick reject)
    conversational = [
        # Conversational
        "wie geht", "danke", "hallo", "hi", "hey", "moin",
        "tschüss", "bye", "ciao", "servus",
        # Instructional (can answer from training)
        "erkläre mir", "explain", "kannst du", "can you",
        "hilf mir", "help me", "zeig mir", "show me",
        # Greetings
        "guten", "good morning", "good evening", "guten tag",
        # Opinion/subjective
        "meinung", "denke", "finde", "opinion", "think",
        "glaubst du", "do you think", "was hältst", "what do you think"
    ]

    if any(indicator in message_lower for indicator in conversational):
        logger.info("✓ No web search - conversational query")
        return False, "Conversational query - no search needed"

    # 4. SIMPLE FACTUAL with context (reject if 1+ docs)
    simple_factual_patterns = [
        "was ist", "what is", "wer ist", "who is",
        "wie funktioniert", "how does", "erkläre", "explain",
        "definiere", "define", "beschreibe", "describe",
        "was bedeutet", "what does", "what means"
    ]

    if any(pattern in message_lower for pattern in simple_factual_patterns):
        if context_docs and len(context_docs) >= 1:
            logger.info("✓ No web search - simple factual query with context")
            return False, "Simple factual query with sufficient context - no search needed"

    # 5. TECHNICAL QUERIES with ANY context
    # IMPORTANT: Only skip if NOT temporal (temporal queries have priority!)
    tech_patterns = [
        "python", "javascript", "typescript", "java", "c++", "rust",
        "code", "programmier", "algorithm", "funktion", "function",
        "class", "klasse", "methode", "method", "variable",
        "syntax", "error", "fehler", "debug", "compile"
    ]

    if any(pattern in message_lower for pattern in tech_patterns):
        # Check again if temporal (to avoid overriding temporal decision)
        if not has_temporal and context_docs:
            logger.info("✓ No web search - technical query with context (non-temporal)")
            return False, "Technical query with context - no search needed"

    # 6. GENERAL KNOWLEDGE (reject)
    general_knowledge_indicators = [
        "allgemein", "generally", "im allgemeinen", "in general",
        "grundsätzlich", "basically", "prinzipiell", "in principle",
        "typischerweise", "typically", "normalerweise", "usually"
    ]

    if any(indicator in message_lower for indicator in general_knowledge_indicators):
        logger.info("✓ No web search - general knowledge query")
        return False, "General knowledge query - no search needed"

    # 7. DEFAULT: If we have ANY context and no temporal/explicit trigger -> skip search
    if context_docs:
        logger.info("✓ No web search - context available and no explicit search needed")
        return False, "Context sufficient - no search needed"

    # If we get here, use LLM to decide (original logic)
    logger.info("🤔 Using LLM to decide on web search (heuristics inconclusive)")

    # Extract context from memory docs
    memory_context = ""
    if context_docs:
        memory_snippets = []
        for doc in context_docs[:3]:
            content = getattr(doc, 'page_content', str(doc))
            memory_snippets.append(content[:200])  # First 200 chars
        memory_context = "\n".join(memory_snippets)

    # Build prompt for LLM
    if language == "de":
        system_prompt = """Du bist ein intelligenter Assistent der entscheidet, ob eine Web-Suche nötig ist.

WANN IST WEB-SUCHE NÖTIG?
✅ Ja bei:
- Aktuellen Informationen (Nachrichten, Wetter, Preise, Börsenkurse)
- Expliziten Web-Anfragen ("suche im Internet", "google das")
- Faktenfragen zu Firmen/Organisationen ohne ausreichenden Memory-Kontext
- Unbekannten Entitäten die nicht im Memory sind
- Fragen die eindeutig externe/aktuelle Daten benötigen

❌ Nein bei:
- Persönlichen Fragen die im Memory beantwortet werden können
- Allgemeinen Wissensfragen die du mit Training-Wissen beantworten kannst
- Konversations-Phrasen ("Danke", "OK", "Verstehe")
- Fragen über bereits bekannte Informationen im Memory

WICHTIG: Wenn der User explizit "suche im Internet/Web" sagt, IMMER web search!

Antworte NUR mit einem JSON-Objekt:
{
    "search": true/false,
    "reason": "kurze Begründung",
    "confidence": 0.0-1.0
}"""
    else:
        system_prompt = """You are an intelligent assistant deciding if web search is needed.

WHEN IS WEB SEARCH NEEDED?
✅ Yes for:
- Current information (news, weather, prices, stock quotes)
- Explicit web requests ("search online", "google that")
- Factual questions about companies/organizations without sufficient memory context
- Unknown entities not in memory
- Questions clearly requiring external/current data

❌ No for:
- Personal questions that can be answered from memory
- General knowledge questions you can answer with training knowledge
- Conversational phrases ("Thanks", "OK", "I see")
- Questions about already known information in memory

IMPORTANT: If user explicitly says "search online/web", ALWAYS web search!

Answer ONLY with a JSON object:
{
    "search": true/false,
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}"""

    user_prompt = f"""User-Nachricht: "{message}"

Memory-Kontext (was ich bereits weiß):
{memory_context if memory_context else "Kein relevanter Kontext im Memory"}

Brauche ich eine Web-Suche?"""

    try:
        # Call LLM for decision
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON response
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            decision = json.loads(json_match.group())

            should_search = decision.get('search', False)
            reason = decision.get('reason', 'LLM decision')
            confidence = decision.get('confidence', 0.5)

            logger.info(f"🤖 LLM Web-Search Decision: search={should_search}, confidence={confidence:.2f}, reason={reason}")

            # Only search if confidence is high enough
            if should_search and confidence >= 0.6:
                return True, f"LLM decision (conf={confidence:.2f}): {reason}"
            else:
                return False, None
        else:
            logger.warning(f"Could not parse LLM response as JSON: {response_text}")
            # Fallback: check for explicit keywords
            response_lower = response_text.lower()
            if '"search": true' in response_lower or '"search":true' in response_lower:
                return True, "LLM suggested search (JSON parse fallback)"
            return False, None

    except Exception as e:
        logger.error(f"Error in LLM web search decision: {e}")
        # Fallback to conservative approach: search if explicitly requested
        if any(keyword in message.lower() for keyword in ['such', 'search', 'google', 'finde', 'find']):
            return True, "Fallback: explicit search keyword detected"
        return False, None


async def extract_search_query_llm(
    message: str,
    context_docs: List[Any],
    chat_client,
    language: str = "de"
) -> str:
    """
    Use LLM to extract optimal search query with context awareness.

    Args:
        message: User's message
        context_docs: Retrieved memory documents for context
        chat_client: LLM client
        language: Language

    Returns:
        Optimized search query string
    """
    # Extract context entities
    memory_context = ""
    if context_docs:
        memory_snippets = []
        for doc in context_docs[:2]:
            content = getattr(doc, 'page_content', str(doc))
            memory_snippets.append(content[:150])
        memory_context = "\n".join(memory_snippets)

    if language == "de":
        system_prompt = """Du extrahierst die beste Web-Such-Query aus einer User-Nachricht.

REGELN:
- Nutze Kontext aus dem Memory (z.B. Firmennamen, Personen)
- Entferne Frage-Wörter die für Suche unnötig sind ("was", "wie", "kannst du")
- Füge relevante Entitäten aus dem Kontext hinzu
- Halte die Query fokussiert und präzise (3-8 Wörter ideal)
- Bei vagen Fragen: Nutze Memory-Kontext um Spezifität zu erhöhen

BEISPIELE:
User: "was macht die Firma?"
Kontext: "Carwell OG in Klagenfurt..."
→ Query: "Carwell OG Klagenfurt Geschäftstätigkeit"

User: "Suche nach aktuellen KI Nachrichten"
→ Query: "aktuelle KI Nachrichten 2025"

Antworte NUR mit der optimierten Query, keine Erklärung."""
    else:
        system_prompt = """You extract the best web search query from a user message.

RULES:
- Use context from memory (e.g., company names, people)
- Remove question words unnecessary for search ("what", "how", "can you")
- Add relevant entities from context
- Keep query focused and precise (3-8 words ideal)
- For vague questions: Use memory context to increase specificity

EXAMPLES:
User: "what does the company do?"
Context: "Carwell OG in Klagenfurt..."
→ Query: "Carwell OG Klagenfurt business activities"

User: "search for current AI news"
→ Query: "current AI news 2025"

Answer ONLY with the optimized query, no explanation."""

    user_prompt = f"""Nachricht: "{message}"

Memory-Kontext:
{memory_context if memory_context else "Kein Kontext"}

Optimierte Such-Query:"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        query = response.content if hasattr(response, 'content') else str(response)
        query = query.strip().strip('"').strip("'")

        # Limit length
        if len(query) > 100:
            query = query[:100]

        logger.info(f"🤖 LLM extracted search query: '{query}'")
        return query

    except Exception as e:
        logger.error(f"Error in LLM query extraction: {e}")
        # Fallback: use original message
        return message[:100]
