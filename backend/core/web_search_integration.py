"""
Web Search Integration for Chat Processing

Intelligently determines when web search is needed and integrates results into context.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Trigger phrases that indicate current/real-time information is needed
WEB_SEARCH_TRIGGERS = {
    # German triggers
    "de": [
        r"\b(aktuelle|neueste|letzte|jüngste|heutige)\b",
        r"\b(was (ist|sind|gibt es|passiert)|wie (ist|sind|geht es))\b.*\b(heute|jetzt|gerade|aktuell|momentan|derzeit)\b",
        r"\b(suche|finde|recherchiere)\b.*\b(im internet|online|im web)\b",
        r"\b(nachrichten|news|ereignisse|entwicklungen)\b.*\b(heute|diese woche|aktuell)\b",
        r"\b(wetter|temperatur)\b.*\b(heute|morgen|aktuell)\b",
        r"\b(preis|kosten)\b.*\b(aktuell|heute|momentan)\b",
        r"\b(börse|aktien|kurs)\b",
        r"\b(verfügbar|erhältlich)\b",
        r"\bwas ist neu\b",
        r"\bgibt es (neue|neuigkeiten)\b",
        r"\b(website|webseite|url)\b",
        r"\b(informationen über|infos zu|details zu)\b.*\b(aktuelle|neueste)\b",
        # Company/organization queries
        r"\bwas macht (die|das|der) (firma|unternehmen|betrieb|geschäft)\b",
        r"\b(tätigkeitsbereich|geschäftsfeld|branche)\b",
        r"\b(produkte|dienstleistungen|angebot)\b.*\b(firma|unternehmen)\b",
        r"\b(firma|unternehmen|betrieb|geschäft)\b.*\b(spezialisiert|fokussiert|bietet an)\b",
    ],
    # English triggers
    "en": [
        r"\b(current|latest|recent|newest|today's)\b",
        r"\b(what (is|are)|how (is|are))\b.*\b(today|now|currently|right now|at the moment)\b",
        r"\b(search|find|look up|research)\b.*\b(online|on the internet|on the web)\b",
        r"\b(news|events|developments)\b.*\b(today|this week|current|latest)\b",
        r"\b(weather|temperature)\b.*\b(today|tomorrow|current)\b",
        r"\b(price|cost)\b.*\b(current|today|now)\b",
        r"\b(stock|market|exchange)\b",
        r"\b(available|in stock)\b",
        r"\bwhat's new\b",
        r"\b(website|webpage|url)\b",
        r"\b(information about|info on|details about)\b.*\b(current|latest)\b",
    ]
}

# Keywords that indicate factual/encyclopedic queries (don't need web search if in memory)
FACTUAL_KEYWORDS = [
    r"\b(was ist|what is|was sind|what are|define|definition|erkläre|explain)\b",
    r"\b(wie funktioniert|how does|how do)\b",
    r"\b(warum|why|wieso|weshalb)\b",
]


def should_perform_web_search(message: str, relevant_docs: List[Any], language: str = "de") -> Tuple[bool, Optional[str]]:
    """
    Determine if a web search should be performed based on the message content.

    Args:
        message: The user's message
        relevant_docs: Retrieved memory documents
        language: Language of the message ("de" or "en")

    Returns:
        Tuple of (should_search: bool, reason: str or None)
    """
    message_lower = message.lower()

    # Check for explicit web search requests
    explicit_search_patterns = [
        r"\bsuch[e]? (im |nach |für )?(internet|web|online)\b",
        r"\bsearch (the |for |on )?(internet|web|online)\b",
        r"\b(google|bing|duck ?duck ?go)\b",
        r"\bonline (suche|such|search|nachschauen|look up)\b",
        r"\b(finde|find|recherchiere|research) (im |im |on )?(internet|web|online)\b",
    ]

    for pattern in explicit_search_patterns:
        if re.search(pattern, message_lower):
            return True, "Explicit web search request"

    # Check for current/real-time information triggers
    triggers = WEB_SEARCH_TRIGGERS.get(language, WEB_SEARCH_TRIGGERS["de"])
    for trigger_pattern in triggers:
        if re.search(trigger_pattern, message_lower):
            logger.info(f"🌐 Web search trigger detected: {trigger_pattern[:50]}...")
            return True, f"Trigger matched: {trigger_pattern[:50]}"

    # Check if relevant docs are sufficient
    if relevant_docs and len(relevant_docs) >= 2:
        # We have good context from memory, might not need web search
        # But still search if asking for very recent info
        recent_patterns = [
            r"\b(heute|today|jetzt|now|gerade|just|momentan|currently)\b",
            r"\b(neueste|latest|aktuell|current)\b",
        ]
        for pattern in recent_patterns:
            if re.search(pattern, message_lower):
                return True, "Recent information requested despite having memory context"

    # Check for factual questions without good memory context
    if not relevant_docs or len(relevant_docs) < 2:
        for factual_pattern in FACTUAL_KEYWORDS:
            if re.search(factual_pattern, message_lower):
                # Factual question without sufficient memory - could benefit from web search
                if len(message.split()) > 5:  # Only for substantive questions
                    return True, "Factual question without sufficient memory context"

    # Check message length and complexity
    if len(message.split()) > 10 and not relevant_docs:
        # Long, complex question without memory context might benefit from web search
        return True, "Complex question without memory context"

    return False, None


def extract_search_query(message: str, language: str = "de", context_docs: List[Any] = None) -> str:
    """
    Extract or formulate a good search query from the user's message.

    Args:
        message: The user's message
        language: Language of the message
        context_docs: Recent memory documents for context

    Returns:
        Optimized search query string
    """
    # Remove command flags
    clean_message = message
    for cmd in ["/nothink", "/no think", "/deutsch", "/de", "/english", "/en", "/search"]:
        clean_message = clean_message.replace(cmd, "").strip()

    # Extract entities from context (company names, person names, locations)
    context_entities = []
    if context_docs:
        for doc in context_docs[:2]:  # Check last 2 docs for entities
            content = getattr(doc, 'page_content', str(doc))
            # Extract capitalized words (likely entities)
            import re
            entities = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\b', content)
            # Extract company names (with OG, GmbH, etc.)
            companies = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\s+(?:OG|GmbH|AG|KG)\b', content)
            context_entities.extend(companies[:2])  # Prioritize companies
            context_entities.extend(entities[:3])

    # If message is vague (like "was macht die Firma?"), add context
    if context_entities and len(clean_message.split()) < 6:
        # Add most relevant entity to query
        main_entity = context_entities[0] if context_entities else ""
        if main_entity and main_entity.lower() not in clean_message.lower():
            clean_message = f"{main_entity} {clean_message}"
            logger.info(f"🔗 Added context entity '{main_entity}' to query")

    # Remove question words that don't add value to search
    if language == "de":
        remove_words = ["suche", "finde", "kannst du", "könntest du", "bitte", "mir", "mal"]
    else:
        remove_words = ["search", "find", "can you", "could you", "please", "for me"]

    words = clean_message.split()
    filtered_words = [w for w in words if w.lower() not in remove_words]
    search_query = " ".join(filtered_words)

    # Limit query length
    if len(search_query) > 100:
        search_query = search_query[:100]

    logger.info(f"📝 Extracted search query: '{search_query}'")
    return search_query


def format_web_results_for_context(search_result: Dict[str, Any], max_results: int = 3) -> str:
    """
    Format web search results for inclusion in LLM context.

    Args:
        search_result: Result from WebSearchService.search()
        max_results: Maximum number of results to include

    Returns:
        Formatted string for LLM context
    """
    if not search_result or not search_result.get("results"):
        return ""

    formatted = "🌐 **Aktuelle Web-Informationen:**\n\n"

    # Add AI-generated summary if available
    if search_result.get("answer"):
        formatted += f"**Zusammenfassung:**\n{search_result['answer']}\n\n"

    # Add top results
    formatted += "**Quellen:**\n"
    for i, result in enumerate(search_result["results"][:max_results], 1):
        formatted += f"\n{i}. **{result['title']}**\n"
        formatted += f"   {result['content'][:200]}...\n"
        formatted += f"   🔗 {result['url']}\n"

    # Add timestamp
    timestamp = search_result.get("timestamp", datetime.utcnow().isoformat())
    formatted += f"\n_Stand: {timestamp}_\n"

    return formatted


def create_web_search_feedback(
    query: str,
    result_count: int,
    search_time_ms: Optional[int] = None,
    language: str = "de"
) -> str:
    """
    Create user-friendly feedback message about web search.

    Args:
        query: The search query used
        result_count: Number of results found
        search_time_ms: Time taken for search in milliseconds
        language: Language for feedback message

    Returns:
        Formatted feedback message
    """
    # Don't show feedback - let the LLM integrate results naturally into conversation
    # The web results are in the system context already
    return ""


def should_save_web_result_to_memory(search_result: Dict[str, Any], query: str) -> bool:
    """
    Determine if web search results should be saved to memory.

    Args:
        search_result: Result from WebSearchService.search()
        query: The search query

    Returns:
        True if results should be saved to memory
    """
    # Don't save if no results
    if not search_result.get("results"):
        return False

    # Don't save very time-sensitive queries (they'll be outdated quickly)
    time_sensitive_keywords = [
        "heute", "today", "jetzt", "now", "gerade", "just",
        "aktuell", "current", "momentan", "currently",
        "wetter", "weather", "börse", "stock", "preis", "price"
    ]

    query_lower = query.lower()
    for keyword in time_sensitive_keywords:
        if keyword in query_lower:
            logger.info("⏰ Not saving time-sensitive web results to memory")
            return False

    # Save factual/encyclopedic information
    factual_keywords = [
        "was ist", "what is", "definition", "erkläre", "explain",
        "wie funktioniert", "how does", "warum", "why"
    ]

    for keyword in factual_keywords:
        if keyword in query_lower:
            logger.info("💾 Saving factual web results to memory")
            return True

    # Default: save if we have a good answer
    return bool(search_result.get("answer"))
