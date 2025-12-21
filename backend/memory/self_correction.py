"""
Self-Correction System - Analysiert Fehler und generiert Korrekturen.
"""

import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
from uuid import uuid4

from backend.models.feedback import (
    ConversationTurn,
    FeedbackEntry,
    ErrorCategory,
    FeedbackType
)
from backend.models.memory_entry import MemoryEntry

logger = logging.getLogger("lexi_middleware.self_correction")


class SelfCorrectionAnalyzer:
    """
    Analysiert fehlerhafte Antworten und generiert Korrekturen.
    """

    def __init__(self, chat_client, embeddings):
        """
        Args:
            chat_client: ChatOllama für LLM-Calls
            embeddings: Für Embedding-Generierung
        """
        self.chat_client = chat_client
        self.embeddings = embeddings

    def analyze_failure(self, turn: ConversationTurn,
                       feedbacks: list[FeedbackEntry]) -> Tuple[ErrorCategory, str]:
        """
        Analysiert warum die Antwort schlecht war.

        Args:
            turn: Der fehlerhafte Turn
            feedbacks: Liste von Feedbacks

        Returns:
            (ErrorCategory, detailed_analysis)
        """
        logger.info(f"Analyzing failure for turn {turn.turn_id}")

        # Baue Analyse-Prompt
        prompt = self._build_analysis_prompt(turn, feedbacks)

        try:
            response = self.chat_client.invoke([
                {"role": "system", "content": self._get_analysis_system_prompt()},
                {"role": "user", "content": prompt}
            ])

            analysis = response.content.strip()

            # Parse Error Category aus Antwort
            error_category = self._extract_error_category(analysis)

            logger.info(f"Analysis complete: {error_category.value}")

            return error_category, analysis

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return ErrorCategory.INCOMPLETE, f"Analysis error: {str(e)}"

    def generate_correction(self, turn: ConversationTurn,
                          error_category: ErrorCategory,
                          analysis: str) -> str:
        """
        Generiert eine bessere Alternative zur fehlerhaften Antwort.

        Args:
            turn: Der fehlerhafte Turn
            error_category: Kategorie des Fehlers
            analysis: Detaillierte Analyse

        Returns:
            Korrigierte Antwort
        """
        logger.info(f"Generating correction for turn {turn.turn_id}")

        prompt = self._build_correction_prompt(turn, error_category, analysis)

        try:
            response = self.chat_client.invoke([
                {"role": "system", "content": self._get_correction_system_prompt()},
                {"role": "user", "content": prompt}
            ])

            correction = response.content.strip()

            logger.info(f"Correction generated: {len(correction)} chars")

            return correction

        except Exception as e:
            logger.error(f"Correction generation failed: {e}")
            return f"[Correction failed: {str(e)}]"

    def create_correction_memory(self, turn: ConversationTurn,
                                correction: str,
                                error_category: ErrorCategory,
                                analysis: str) -> MemoryEntry:
        """
        Erstellt eine Correction-Memory.

        Diese Memory wird mit hoher Relevanz gespeichert und
        referenziert den fehlerhaften Turn.

        Args:
            turn: Der fehlerhafte Turn
            correction: Die bessere Antwort
            error_category: Kategorie
            analysis: Analyse

        Returns:
            MemoryEntry für Correction
        """
        # Erstelle Content der Correction-Memory
        content = f"""SELBST-KORREKTUR:

Ursprüngliche Frage: {turn.user_message}

Fehlerhafte Antwort: {turn.ai_response[:200]}...

Fehler-Typ: {error_category.value}
Analyse: {analysis[:300]}...

KORRIGIERTE ANTWORT:
{correction}

Gelernter Punkt: Bei ähnlichen Fragen in Zukunft diese verbesserte Antwort als Referenz nutzen."""

        # Generiere Embedding
        embedding = self.embeddings.embed_query(content)

        # Erstelle Memory
        correction_memory = MemoryEntry(
            id=f"correction_{uuid4()}",
            content=content,
            timestamp=datetime.now(timezone.utc),
            category="self_correction",
            tags=["correction", "learning", error_category.value],
            source="self_correction",
            relevance=1.0,  # Hohe Relevanz - wichtig für Lernen!
            embedding=embedding
        )

        logger.info(f"Created correction memory: {correction_memory.id}")

        return correction_memory

    def _get_analysis_system_prompt(self) -> str:
        """System-Prompt für Fehler-Analyse."""
        return """Du bist ein KI-Qualitätsprüfer der eigene Fehler analysiert.

Aufgabe:
1. Analysiere warum eine KI-Antwort schlecht war
2. Klassifiziere den Fehler
3. Erkläre präzise was falsch lief

Fehler-Kategorien:
- FACTUALLY_WRONG: Falsche Fakten
- INCOMPLETE: Wichtige Infos fehlen
- IRRELEVANT: Antwort passt nicht zur Frage
- TOO_TECHNICAL: Zu komplex für Kontext
- TOO_SIMPLE: Zu oberflächlich
- MISSING_CONTEXT: Vorhandener Kontext nicht genutzt
- HALLUCINATION: Info erfunden

Antwortformat:
ERROR_CATEGORY: [kategorie]
ANALYSIS:
[Detaillierte Analyse was falsch lief]

Sei kritisch und präzise!"""

    def _get_correction_system_prompt(self) -> str:
        """System-Prompt für Korrektur-Generierung."""
        return """Du bist ein KI-Verbesserer der aus Fehlern lernt.

Aufgabe:
1. Nutze die Fehler-Analyse
2. Generiere eine BESSERE Antwort
3. Vermeide den identifizierten Fehler

Wichtig:
- Sei präzise und korrekt
- Nutze verfügbaren Kontext
- Passe Komplexität an
- Sei vollständig aber konzise
- Keine Erfindungen!

Antworte direkt mit der verbesserten Antwort (keine Meta-Kommentare)."""

    def _build_analysis_prompt(self, turn: ConversationTurn,
                               feedbacks: list[FeedbackEntry]) -> str:
        """Erstellt Analyse-Prompt."""
        # Sammle Feedback-Typen
        feedback_types = [f.feedback_type.value for f in feedbacks]
        feedback_comments = [f.user_comment for f in feedbacks if f.user_comment]

        prompt = f"""Analysiere diese fehlerhafte KI-Antwort:

USER-FRAGE:
{turn.user_message}

KI-ANTWORT:
{turn.ai_response}

FEEDBACK:
- Typen: {", ".join(feedback_types)}
{"- Kommentare: " + "; ".join(feedback_comments) if feedback_comments else ""}

KONTEXT:
- Genutzte Memories: {len(turn.retrieved_memories) if turn.retrieved_memories else 0}
- Response Time: {turn.response_time_ms}ms

Was lief falsch?"""

        return prompt

    def _build_correction_prompt(self, turn: ConversationTurn,
                                 error_category: ErrorCategory,
                                 analysis: str) -> str:
        """Erstellt Korrektur-Prompt."""
        prompt = f"""Generiere eine verbesserte Antwort:

URSPRÜNGLICHE FRAGE:
{turn.user_message}

FEHLERHAFTE ANTWORT:
{turn.ai_response}

FEHLER-ANALYSE:
Kategorie: {error_category.value}
{analysis}

Generiere jetzt die KORRIGIERTE, BESSERE Antwort:"""

        return prompt

    def _extract_error_category(self, analysis: str) -> ErrorCategory:
        """Extrahiert ErrorCategory aus LLM-Response."""
        analysis_upper = analysis.upper()

        # Suche nach ERROR_CATEGORY: Zeile
        for line in analysis.split("\n"):
            if "ERROR_CATEGORY:" in line.upper():
                category_str = line.split(":", 1)[1].strip().upper()

                # Map zu ErrorCategory
                category_map = {
                    "FACTUALLY_WRONG": ErrorCategory.FACTUALLY_WRONG,
                    "INCOMPLETE": ErrorCategory.INCOMPLETE,
                    "IRRELEVANT": ErrorCategory.IRRELEVANT,
                    "TOO_TECHNICAL": ErrorCategory.TOO_TECHNICAL,
                    "TOO_SIMPLE": ErrorCategory.TOO_SIMPLE,
                    "MISSING_CONTEXT": ErrorCategory.MISSING_CONTEXT,
                    "HALLUCINATION": ErrorCategory.HALLUCINATION
                }

                for key, value in category_map.items():
                    if key in category_str:
                        return value

        # Fallback: INCOMPLETE
        return ErrorCategory.INCOMPLETE


def analyze_and_correct_failures(stop_check_fn=None) -> int:
    """
    Hauptfunktion für Heartbeat: Analysiert Fehler und erstellt Korrekturen.

    Args:
        stop_check_fn: Stop-Check Funktion

    Returns:
        Anzahl der erstellten Correction-Memories
    """
    from backend.core.component_cache import get_cached_components
    from backend.memory.conversation_tracker import get_conversation_tracker

    logger.info("🔍 Starting failure analysis and correction")

    bundle = get_cached_components()
    vectorstore = bundle.vectorstore
    chat_client = bundle.chat_client
    embeddings = bundle.embeddings

    tracker = get_conversation_tracker()
    analyzer = SelfCorrectionAnalyzer(chat_client, embeddings)

    # Hole Turns mit negativem Feedback
    negative_turns = tracker.get_negative_turns(limit=10)  # Max 10 pro Run

    if not negative_turns:
        logger.info("No negative turns found")
        return 0

    logger.info(f"Analyzing {len(negative_turns)} negative turns")

    corrections_created = 0

    for turn, feedbacks in negative_turns:
        # Check Stop-Signal
        if stop_check_fn and stop_check_fn():
            logger.warning(f"Analysis interrupted after {corrections_created} corrections")
            break

        try:
            # 1. Analysiere Fehler
            error_category, analysis = analyzer.analyze_failure(turn, feedbacks)

            # 2. Generiere Korrektur
            correction = analyzer.generate_correction(turn, error_category, analysis)

            # 3. Erstelle Correction-Memory
            correction_memory = analyzer.create_correction_memory(
                turn, correction, error_category, analysis
            )

            # 4. Speichere in Qdrant
            vectorstore.store_entry(correction_memory)

            corrections_created += 1

            # 5. Update Feedback-Einträge mit Analyse
            for feedback in feedbacks:
                feedback.error_category = error_category
                feedback.error_analysis = analysis
                feedback.suggested_correction = correction

            logger.info(f"✅ Created correction for turn {turn.turn_id}")

        except Exception as e:
            logger.error(f"Failed to process turn {turn.turn_id}: {e}")
            continue

    logger.info(f"✅ Self-correction complete: {corrections_created} corrections created")
    return corrections_created
