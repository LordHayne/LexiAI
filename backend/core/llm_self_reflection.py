"""
LLM Self-Reflection: The AI checks its own answers for hallucinations.
"""
import logging
from typing import Tuple, Optional, List, Any
import json
import re

logger = logging.getLogger(__name__)


async def verify_answer_quality(
    question: str,
    answer: str,
    sources: List[str],
    chat_client,
    language: str = "de"
) -> Tuple[bool, Optional[str]]:
    """
    LLM verifies if its own answer is plausible and based on sources.

    Args:
        question: User's original question
        answer: Generated answer to verify
        sources: List of source information used (memory, web results)
        chat_client: LLM client for reflection
        language: Language

    Returns:
        Tuple of (is_valid: bool, issue: str or None)
    """
    # Format sources
    sources_text = "\n".join(sources) if sources else "Keine Quellen verfügbar"

    logger.info(f"🔍 Self-Reflection - Sources count: {len(sources)}, Answer length: {len(answer)}")
    if sources:
        for i, src in enumerate(sources[:2]):
            logger.info(f"   Source {i+1}: {src[:100]}...")

    # Check for circular/self-referential sources - Use smart filtering
    # Factual conversation memories should NOT be filtered as circular
    if sources:
        # Fact indicators - same as in chat_processing_with_tools.py
        fact_indicators = [
            "heißt", "heiße", "name ist", "my name", "I am", "I'm",
            "Besitzer", "owner", "genannt",
            "Geburtstag", "birthday", "wohne in", "live in", "arbeite bei", "work at",
            "ich bin der", "ich bin ein", "das ist mein"
        ]

        # Circular patterns
        circular_patterns = [
            "weißt du", "erinnerst du", "do you remember", "do you know",
            "kannst du dich erinnern", "was weißt du", "what do you know"
        ]

        real_sources = []
        circular_count = 0

        for src in sources:
            src_lower = src.lower()
            is_conversation = "User:" in src and "Assistant:" in src

            if is_conversation:
                # Check if it contains facts
                has_facts = any(indicator.lower() in src_lower for indicator in fact_indicators)
                # Check if it's a circular question
                is_circular_question = any(pattern in src_lower for pattern in circular_patterns)

                # Keep if it has facts, filter if purely circular
                if has_facts:
                    real_sources.append(src)
                    logger.info(f"✓ Keeping factual source: {src[:80]}...")
                elif is_circular_question and not has_facts:
                    circular_count += 1
                    logger.info(f"✗ Filtering circular source: {src[:80]}...")
                else:
                    real_sources.append(src)
            else:
                # Not a conversation format, keep it
                real_sources.append(src)

        # Update sources
        sources = real_sources
        sources_text = "\n".join(sources) if sources else "Keine Quellen verfügbar"
        logger.info(f"📊 Filtered sources: {circular_count} circular removed, {len(sources)} real sources remain")

    # Safety check: If no sources and answer is HIGHLY specific, likely hallucination
    # WICHTIG: Nur bei wirklich spezifischen Behauptungen (Firmen, Produkte, etc.)
    if not sources or len(sources) == 0:
        # Nur sehr spezifische Keywords die auf Firmen/Organisationen hindeuten
        # Entfernt: "bietet", "entwickelt", "bereich" - zu allgemein!
        highly_specific_keywords = [
            "spezialisiert auf", "fokussiert auf", "produziert", "herstellt",
            "specialized in", "focuses on", "produces", "manufactures",
            "gegründet", "founded", "hauptsitz", "headquarters"
        ]
        answer_lower = answer.lower()
        # Nur ablehnen wenn MEHRERE spezifische Keywords UND keine allgemeine Aussage
        specific_count = sum(1 for kw in highly_specific_keywords if kw in answer_lower)
        if specific_count >= 2:
            # High chance of hallucination if making multiple specific claims without sources
            logger.warning(f"⚠️ No sources but {specific_count} specific claims detected - likely hallucination")
            return False, "Spezifische Behauptungen ohne Quellen | Ehrlich zugeben dass Info fehlt"

    if language == "de":
        system_prompt = """Du bist ein kritischer Reviewer der AI-Antworten auf Plausibilität prüft.

DEINE AUFGABE:
Prüfe ob die Antwort plausibel ist oder ob sie halluziniert wurde.

WICHTIG - MEMORY-QUELLEN SIND VALIDE:
✅ Memory-Quellen (Format: "Memory: User: ... Assistant: ...") sind VALIDE Informationsquellen!
✅ Wenn Memory eine Info enthält, ist die Antwort darauf KEINE Halluzination!
✅ "User: Ich bin Thomas" → Antwort: "Du bist Thomas" ist VALID (Info aus Memory)!

HALLUZINATION ERKENNEN:
❌ Spezifische Details OHNE JEGLICHE Quelle (kein Memory, kein Web)
❌ Widersprüche zu den Quellen
❌ Erfundene Fakten die zu spezifisch sind
❌ Bei Firmenfragen: Details ohne externe Web-Quelle (Memory reicht NICHT für Firmeninfos!)

✅ VALID wenn:
- Info steht in Memory-Quelle → VALID!
- Info steht in Web-Quelle → VALID!
- Antwort sagt ehrlich "Das weiß ich nicht"
- Rein allgemeines Wissen
- Details KLAR aus Quellen (Memory ODER Web) ableitbar

SPEZIALFALL - PERSÖNLICHE INFO:
- Frage: "Wer bin ich?" + Memory: "User: Ich bin Thomas" → Antwort: "Du bist Thomas" = ✅ VALID!
- Memory-basierte persönliche Info = KEINE Halluzination!

KRITISCHE REGEL FÜR FIRMEN/ORGANISATIONEN:
- Firmenfragen brauchen externe Web-Quellen (Memory reicht NICHT!)
- Persönliche Fragen können aus Memory beantwortet werden!

Antworte NUR mit JSON:
{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "issue": "kurze Erklärung wenn invalid",
    "suggestion": "was stattdessen tun" (nur wenn invalid)
}"""
    else:
        system_prompt = """You are a critical reviewer checking AI answers for plausibility.

YOUR TASK:
Check if the answer is plausible or hallucinated.

DETECT HALLUCINATION:
❌ Specific company details without source (business areas, products, locations)
❌ Concrete numbers, names, addresses not in sources
❌ Contradictions to sources
❌ Invented facts that are too specific
❌ Answer about wrong entity (different company with similar name)

✅ OK if:
- General knowledge (definition of "company")
- Logical conclusions from sources
- Polite responses ("I don't know")
- Answer matches sources

IMPORTANT: Be strict with company/organization information!

Answer ONLY with JSON:
{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "issue": "brief explanation if invalid",
    "suggestion": "what to do instead" (only if invalid)
}"""

    user_prompt = f"""Frage: "{question}"

Antwort: "{answer}"

Verfügbare Quellen:
{sources_text}

Ist diese Antwort plausibel oder halluziniert?"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON - handle markdown code blocks
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```'):
            # Remove opening ```json or ```
            lines = cleaned_text.split('\n')
            cleaned_text = '\n'.join(lines[1:])  # Skip first line
            # Remove closing ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3].strip()

        # Try to extract JSON by finding matching braces
        try:
            start_idx = cleaned_text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found in response")

            # Find matching closing brace
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
            verification = json.loads(json_str)

            is_valid = verification.get('valid', True)
            confidence = verification.get('confidence', 0.5)
            issue = verification.get('issue', '')
            suggestion = verification.get('suggestion', '')

            # Erhöhter Threshold: Nur bei hoher Confidence ablehnen (war: 0.7)
            if not is_valid and confidence >= 0.85:
                logger.warning(f"⚠️ Self-Reflection detected issue (conf={confidence:.2f}): {issue}")
                return False, f"{issue} | {suggestion}"
            elif not is_valid and confidence >= 0.7:
                # Graubereich: Loggen aber nicht ablehnen
                logger.info(f"ℹ️ Self-Reflection uncertain (conf={confidence:.2f}): {issue} - allowing anyway")
                return True, None
            else:
                logger.info(f"✅ Self-Reflection: Answer valid (conf={confidence:.2f})")
                return True, None

            # Fallback: allow if not caught by above conditions
            return True, None

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse self-reflection response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
            return True, None  # Default to valid if can't parse

    except Exception as e:
        logger.error(f"Error in self-reflection: {e}")
        return True, None  # Default to valid on error


async def generate_honest_fallback(
    question: str,
    chat_client,
    language: str = "de"
) -> str:
    """
    Generate honest "I don't know" response when reflection fails.

    Args:
        question: User's question
        chat_client: LLM client
        language: Language

    Returns:
        Honest fallback answer
    """
    if language == "de":
        prompt = f"""Die Frage war: "{question}"

Ich kann diese Frage nicht zuverlässig beantworten, da mir konkrete Informationen fehlen.

Formuliere eine ehrliche, hilfreiche Antwort die:
- Zugibt dass du es nicht weißt
- Kurz ist (1-2 Sätze)
- Vorschlägt wie der User die Info finden kann (z.B. "Schau auf der Website nach")
- Freundlich aber direkt ist"""
    else:
        prompt = f"""The question was: "{question}"

I cannot reliably answer this question as I lack concrete information.

Formulate an honest, helpful answer that:
- Admits you don't know
- Is brief (1-2 sentences)
- Suggests how the user can find the info (e.g., "Check the website")
- Is friendly but direct"""

    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = await chat_client.ainvoke(messages)
        fallback = response.content if hasattr(response, 'content') else str(response)

        return fallback.strip()

    except Exception as e:
        logger.error(f"Error generating fallback: {e}")
        if language == "de":
            return "Das weiß ich leider nicht genau. Schau am besten direkt auf der Website oder kontaktiere die Firma."
        else:
            return "I don't know that exactly. Best check the website directly or contact the company."
