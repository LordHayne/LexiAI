"""
LLM-based Relevance Check for Web Search Results
"""
import logging
from typing import List, Dict, Any, Tuple
import json
import re

logger = logging.getLogger(__name__)


async def check_result_relevance(
    question: str,
    results: List[Dict[str, Any]],
    chat_client,
    language: str = "de"
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Use LLM to check if web search results are relevant to the question.

    Args:
        question: User's original question
        results: List of web search results (each with 'title', 'content', 'url')
        chat_client: LLM client
        language: Language

    Returns:
        Tuple of (filtered_results: List, overall_relevance_score: float)
    """
    if not results:
        return [], 0.0

    # Format results for LLM evaluation
    results_text = ""
    for i, result in enumerate(results[:5], 1):  # Check top 5 results
        title = result.get('title', '')
        content = result.get('content', '')[:300]  # Limit content length
        results_text += f"{i}. {title}\n   {content}...\n\n"

    if language == "de":
        system_prompt = """Du bist ein kritischer Evaluator der prüft ob Web-Suchergebnisse zur Frage passen.

DEINE AUFGABE:
Bewerte jeden Suchresultat-Eintrag einzeln: Beantwortet er die Frage des Users?

BEWERTUNG:
- ✅ RELEVANT (Score 0.8-1.0): Ergebnis beantwortet die Frage direkt und enthält konkrete Infos
- 🔶 TEILWEISE RELEVANT (Score 0.4-0.7): Ergebnis erwähnt das Thema, aber nicht alle Details
- ❌ IRRELEVANT (Score 0.0-0.3): Ergebnis handelt von anderem Thema/anderer Entität

WICHTIGE REGELN:
- Bei Firmenfragen: Prüfe ob es die RICHTIGE Firma ist (gleicher Name ≠ gleiche Firma!)
- Bei Standortfragen: Prüfe ob RICHTIGER Standort (z.B. "München" vs "Berlin")
- Vergleiche Frage mit Inhalt: Beantwortet das Ergebnis was gefragt wurde?
- Vorsicht bei generischen Inhalten (z.B. Wikipedia-Kategorien, allgemeine Definitionen)

Antworte NUR mit JSON:
{
    "results": [
        {"index": 1, "relevant": true/false, "score": 0.0-1.0, "reason": "warum relevant/irrelevant"},
        {"index": 2, "relevant": true/false, "score": 0.0-1.0, "reason": "..."}
    ],
    "overall_quality": 0.0-1.0,
    "suggestion": "Optional: Verbesserungsvorschlag für neue Suche"
}"""
    else:
        system_prompt = """You are a critical evaluator checking if web search results match the question.

YOUR TASK:
Evaluate each search result individually: Does it answer the user's question?

EVALUATION:
- ✅ RELEVANT (Score 0.8-1.0): Result directly answers the question with concrete info
- 🔶 PARTIALLY RELEVANT (Score 0.4-0.7): Result mentions the topic but lacks details
- ❌ IRRELEVANT (Score 0.0-0.3): Result is about different topic/entity

IMPORTANT RULES:
- For company questions: Check if it's the RIGHT company (same name ≠ same company!)
- For location questions: Check if CORRECT location (e.g., "Munich" vs "Berlin")
- Compare question with content: Does the result answer what was asked?
- Watch out for generic content (e.g., Wikipedia categories, general definitions)

Answer ONLY with JSON:
{
    "results": [
        {"index": 1, "relevant": true/false, "score": 0.0-1.0, "reason": "why relevant/irrelevant"},
        {"index": 2, "relevant": true/false, "score": 0.0-1.0, "reason": "..."}
    ],
    "overall_quality": 0.0-1.0,
    "suggestion": "Optional: Improvement suggestion for new search"
}"""

    user_prompt = f"""Frage: "{question}"

Suchergebnisse:
{results_text}

Bewerte die Relevanz jedes Ergebnisses zur Frage."""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        logger.info(f"📄 Raw relevance response (first 600 chars): {response_text[:600]}")

        # Parse JSON - handle markdown code blocks
        try:
            # Remove markdown code block markers if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                # Remove opening ```json or ```
                lines = cleaned_text.split('\n')
                cleaned_text = '\n'.join(lines[1:])  # Skip first line
                # Remove closing ```
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3].strip()

            # Try to extract JSON by finding matching braces
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
            evaluation = json.loads(json_str)

            result_scores = evaluation.get('results', [])
            overall_quality = evaluation.get('overall_quality', 0.5)
            suggestion = evaluation.get('suggestion', '')

            # Filter results based on relevance
            filtered_results = []
            for i, result in enumerate(results[:5], 1):
                # Find score for this result
                score_info = next((r for r in result_scores if r.get('index') == i), None)

                if score_info:
                    is_relevant = score_info.get('relevant', False)
                    score = score_info.get('score', 0.0)
                    reason = score_info.get('reason', '')

                    logger.info(f"   Result {i}: {'✅' if is_relevant else '❌'} (score={score:.2f}) - {reason}")

                    # Only include if score >= 0.4 (at least partially relevant)
                    if score >= 0.4:
                        filtered_results.append(result)
                else:
                    # No score info - include by default
                    filtered_results.append(result)

            logger.info(f"🔍 Relevance Check: {len(filtered_results)}/{len(results)} results relevant (quality={overall_quality:.2f})")
            if suggestion:
                logger.info(f"   💡 Suggestion: {suggestion}")

            return filtered_results, overall_quality

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse relevance check response: {e}")
            logger.info(f"Response was: {response_text[:600]}")
            return results, 0.5  # Return all results if can't parse

    except Exception as e:
        logger.error(f"Error in relevance check: {e}")
        return results, 0.5  # Return all results on error


async def should_refine_search(
    question: str,
    overall_quality: float,
    filtered_results: List[Dict[str, Any]],
    chat_client,
    language: str = "de"
) -> Tuple[bool, str]:
    """
    Decide if search should be refined with a better query.

    Args:
        question: Original question
        overall_quality: Quality score from relevance check
        filtered_results: Results that passed relevance check
        chat_client: LLM client
        language: Language

    Returns:
        Tuple of (should_refine: bool, refined_query: str)
    """
    # Only refine if quality is poor (<0.4) or no results passed filter
    if overall_quality >= 0.4 and len(filtered_results) > 0:
        return False, ""

    logger.info(f"🔄 Search quality poor (quality={overall_quality:.2f}, results={len(filtered_results)}), considering refinement...")

    if language == "de":
        system_prompt = """Du bist ein Such-Experte der schlechte Suchergebnisse analysiert und bessere Queries formuliert.

DEINE AUFGABE:
Die erste Suche hat keine guten Ergebnisse gebracht. Formuliere eine BESSERE Suchanfrage.

STRATEGIEN:
- Füge spezifischere Keywords hinzu (Standort, Branche, etc.)
- Entferne mehrdeutige Begriffe
- Nutze Synonyme oder alternative Formulierungen
- Füge Kontext hinzu der die Suche eingrenzt

BEISPIELE:
Schlecht: "was macht die Firma" → Besser: "Firma XY München Geschäftstätigkeit"
Schlecht: "Carwell" → Besser: "Carwell OG Klagenfurt Autowerkstatt"

Antworte NUR mit JSON:
{
    "refine": true/false,
    "query": "verbesserte Suchanfrage" (nur wenn refine=true),
    "reason": "warum diese Änderung"
}"""
    else:
        system_prompt = """You are a search expert who analyzes poor search results and formulates better queries.

YOUR TASK:
The first search didn't yield good results. Formulate a BETTER search query.

STRATEGIES:
- Add more specific keywords (location, industry, etc.)
- Remove ambiguous terms
- Use synonyms or alternative formulations
- Add context to narrow the search

EXAMPLES:
Bad: "what does company do" → Better: "Company XY Munich business activities"
Bad: "Carwell" → Better: "Carwell OG Klagenfurt auto repair"

Answer ONLY with JSON:
{
    "refine": true/false,
    "query": "improved search query" (only if refine=true),
    "reason": "why this change"
}"""

    user_prompt = f"""Ursprüngliche Frage: "{question}"

Qualität der Ergebnisse: {overall_quality:.2f}
Anzahl brauchbarer Ergebnisse: {len(filtered_results)}

Sollte die Suche mit besserer Query wiederholt werden?"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await chat_client.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())

            should_refine = decision.get('refine', False)
            refined_query = decision.get('query', '')
            reason = decision.get('reason', '')

            if should_refine and refined_query:
                logger.info(f"✨ Refining search: '{refined_query}' - {reason}")
                return True, refined_query
            else:
                return False, ""

        else:
            logger.warning(f"Could not parse refinement response: {response_text[:200]}")
            return False, ""

    except Exception as e:
        logger.error(f"Error in refinement decision: {e}")
        return False, ""
