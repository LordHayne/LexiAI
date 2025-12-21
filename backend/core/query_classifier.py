"""
Query Classifier: Fast detection of simple queries for performance optimization
"""
import re
import logging

logger = logging.getLogger(__name__)


class QueryType:
    """Query type classifications"""
    SIMPLE_GREETING = "simple_greeting"  # Hallo, Hi, Guten Tag
    META_QUESTION = "meta_question"  # Wer bist du?, Was kannst du?
    SIMPLE_THANKS = "simple_thanks"  # Danke, Vielen Dank
    SIMPLE_CONFIRMATION = "simple_confirmation"  # Ja, Nein, Ok
    SMART_HOME_CONTROL = "smart_home_control"  # Schalte X ein, Mach X aus
    SMART_HOME_QUERY = "smart_home_query"  # Ist X an?, Wie hell ist X?
    COMPLEX_QUERY = "complex_query"  # Needs full tool system


# Fast-Path patterns (no tools needed)
SIMPLE_PATTERNS = {
    QueryType.SIMPLE_GREETING: [
        r"^(hallo|hi|hey|guten (tag|morgen|abend)|servus|grüß (dich|gott))[\s!.?]*$",
        r"^(hello|hi|hey|good (morning|afternoon|evening))[\s!.?]*$",
    ],
    QueryType.META_QUESTION: [
        r"^(wer|was) bist du[\s?!.]*$",
        r"^was kannst du( alles)?[\s?!.]*$",
        r"^wie (heißt|heisst) du[\s?!.]*$",
        r"^(stell dich (vor|mal vor)|introduce yourself)[\s?!.]*$",
        r"^who are you[\s?!.]*$",
        r"^what (can you do|are you)[\s?!.]*$",
    ],
    QueryType.SIMPLE_THANKS: [
        r"^(danke|vielen dank|dankeschön|thx|thanks|thank you)[\s!.]*$",
        r"^(super|toll|prima|perfekt|genau|klasse)[\s!.]*$",
    ],
    QueryType.SIMPLE_CONFIRMATION: [
        r"^(ja|nein|ok|okay|alles klar|verstanden|genau)[\s!.]*$",
        r"^(yes|no|ok|okay|got it|sure|fine)[\s!.]*$",
    ],
}


def classify_query(message: str) -> str:
    """
    Classify query type for performance optimization.

    Args:
        message: User message

    Returns:
        QueryType constant
    """
    message_lower = message.lower().strip()

    # Check Smart Home patterns FIRST (highest priority!)
    # Query patterns FIRST (to avoid false positives with control patterns)
    query_patterns = [
        # Status queries
        r"^ist\s+(das|die|der)\s+.+(an|aus|eingeschaltet|ausgeschaltet)",
        r"^is\s+(the)?\s*.+(on|off|turned on|turned off)",
        r"^status.+(licht|lampe|heizung|thermostat|klima|schalter)",
        r"^läuft\s+(das|die|der)",

        # Temperature queries
        r"^wie (warm|kalt|heiß|kühl)",
        r"^welche\s+temperatur",
        r"^temperatur\s+(im|in der|vom|von)",
        r"^how (warm|cold|hot|cool)",
        r"^what.+(temperature|temp)",
        r"^(zeig|zeige|gib).+(temperatur|temp)",

        # Humidity queries
        r"^wie\s+(feucht|trocken)",
        r"^welche\s+(luftfeuchtigkeit|feuchtigkeit)",
        r"^(luftfeuchtigkeit|feuchtigkeit)\s+(im|in der)",
        r"^how\s+(humid|dry)",
        r"^what.+(humidity|moisture)",

        # Brightness/dimming queries
        r"^wie hell",
        r"^welche\s+helligkeit",
        r"^helligkeit\s+(von|vom|im)",
        r"^how bright",
        r"^what.+(brightness|light level)",

        # General sensor queries
        r"^(was|wie)\s+(ist|sind)\s+(die|der|das)\s+(werte|daten|status)",
        r"^(zeig|zeige|gib).+(status|werte|daten)",
    ]

    for pattern in query_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"🏠 Smart Home Query detected")
            return QueryType.SMART_HOME_QUERY

    # Control patterns (check AFTER query patterns)
    control_patterns = [
        r"(schalte|mach|stelle|dimme|aktiviere|deaktiviere).+(ein|aus|an|ab|auf|zu)",
        r"(turn|switch|set|dim|activate|deactivate).+(on|off|up|down)",
        r"(licht|lampe|beleuchtung|light).+(ein|aus|an|ab)",
        r"(heizung|thermostat|heating).+(auf|zu|ein|aus|an|ab)",
        # Specific verb patterns (dimme X without ein/aus)
        r"^dimme\s+",
        r"^stelle\s+.+(auf|ein)",
    ]

    for pattern in control_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"🏠 Smart Home Control detected")
            return QueryType.SMART_HOME_CONTROL

    # Check simple patterns (Fast-Path)
    for query_type, patterns in SIMPLE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, message_lower, re.IGNORECASE):
                logger.info(f"⚡ Fast-Path detected: {query_type}")
                return query_type

    # Check complexity heuristics
    word_count = len(message.split())

    # Very short queries without question marks - likely simple
    if word_count <= 3 and '?' not in message:
        logger.info(f"⚡ Fast-Path: Very short query ({word_count} words)")
        return QueryType.SIMPLE_GREETING

    # Default to complex query (needs full processing)
    logger.info(f"🔄 Complex query detected ({word_count} words) - using full system")
    return QueryType.COMPLEX_QUERY


def needs_tools(query_type: str) -> bool:
    """
    Check if query type needs tool calling.

    Args:
        query_type: QueryType constant

    Returns:
        True if tools are needed
    """
    # Smart Home queries ALWAYS need tools!
    if query_type in [QueryType.SMART_HOME_CONTROL, QueryType.SMART_HOME_QUERY]:
        return True

    return query_type == QueryType.COMPLEX_QUERY


def needs_multi_step_check(message: str, query_type: str) -> bool:
    """
    Check if multi-step reasoning check is needed.

    Args:
        message: User message
        query_type: QueryType constant

    Returns:
        True if multi-step check needed
    """
    # Fast-path queries never need multi-step
    if query_type != QueryType.COMPLEX_QUERY:
        return False

    # Heuristics for multi-step candidates
    message_lower = message.lower()

    # Keywords indicating multi-step need
    multi_step_keywords = [
        "vergleich", "compare", "unterschied", "difference",
        "vor- und nachteile", "pros and cons",
        "analysiere", "analyze",
        "sowohl", "both",
    ]

    has_multi_step_keyword = any(kw in message_lower for kw in multi_step_keywords)

    # Word count threshold
    word_count = len(message.split())
    is_long_query = word_count > 15

    # Multiple questions/entities
    has_multiple_questions = message.count('?') > 1

    needs_check = has_multi_step_keyword or is_long_query or has_multiple_questions

    if not needs_check:
        logger.info(f"⚡ Skipping multi-step check (simple query with {word_count} words)")

    return needs_check


def needs_self_reflection(query_type: str, tools_used: bool) -> bool:
    """
    Check if self-reflection is needed.

    Args:
        query_type: QueryType constant
        tools_used: Whether tools were used

    Returns:
        True if self-reflection needed
    """
    # NEVER use self-reflection for Smart Home (instant confirmation needed!)
    if query_type in [QueryType.SMART_HOME_CONTROL, QueryType.SMART_HOME_QUERY]:
        logger.info("⚡ Skipping self-reflection (Smart Home - needs instant response)")
        return False

    # Only use self-reflection for complex queries with tools
    if query_type != QueryType.COMPLEX_QUERY:
        logger.info("⚡ Skipping self-reflection (simple query)")
        return False

    if not tools_used:
        logger.info("⚡ Skipping self-reflection (no tools used)")
        return False

    return True


def get_fast_path_response(query_type: str, message: str, language: str = "de") -> str:
    """
    Get a quick response for fast-path queries without LLM call.

    Args:
        query_type: QueryType constant
        message: User message
        language: Language code

    Returns:
        Response string or None if LLM needed
    """
    # For now, only handle simple greetings with hardcoded responses
    # Let LLM handle everything else for quality
    return None  # Let LLM handle all responses for quality
