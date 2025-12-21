"""
Trackt Konversationen für Feedback-Analyse.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
from collections import deque
from uuid import uuid4

from backend.models.feedback import ConversationTurn, FeedbackEntry, FeedbackType

logger = logging.getLogger("lexi_middleware.conversation_tracker")


class ConversationTracker:
    """
    Speichert Konversations-Historie pro User.

    Für jede User-Anfrage wird ein ConversationTurn gespeichert.
    Dies ermöglicht Feedback-Zuordnung und Fehleranalyse.
    """

    def __init__(self, max_history_per_user: int = 100):
        """
        Args:
            max_history_per_user: Max Anzahl Turns pro User
        """
        self.max_history = max_history_per_user
        # user_id → deque of ConversationTurn
        self._history: Dict[str, deque] = {}
        # turn_id → ConversationTurn (für schnellen Lookup)
        self._turns: Dict[str, ConversationTurn] = {}
        # turn_id → List[FeedbackEntry]
        self._feedback: Dict[str, List[FeedbackEntry]] = {}

    def record_turn(self, user_id: str, user_message: str,
                    ai_response: str,
                    retrieved_memories: Optional[List[str]] = None,
                    response_time_ms: Optional[float] = None) -> str:
        """
        Zeichnet einen Konversations-Turn auf.

        Args:
            user_id: User ID
            user_message: User-Anfrage
            ai_response: KI-Antwort
            retrieved_memories: Optional - IDs der verwendeten Memories
            response_time_ms: Optional - Antwortzeit

        Returns:
            turn_id
        """
        turn_id = str(uuid4())

        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=datetime.now(timezone.utc),
            retrieved_memories=retrieved_memories,
            response_time_ms=response_time_ms
        )

        # Speichere in User-Historie
        if user_id not in self._history:
            self._history[user_id] = deque(maxlen=self.max_history)

        self._history[user_id].append(turn)

        # Speichere für schnellen Lookup
        self._turns[turn_id] = turn

        logger.debug(f"Recorded turn {turn_id} for user {user_id}")

        return turn_id

    def record_feedback(self, turn_id: str, feedback_type: FeedbackType,
                       user_comment: Optional[str] = None,
                       confidence: float = 1.0):
        """
        Zeichnet Feedback zu einem Turn auf.

        Args:
            turn_id: ID des Turns
            feedback_type: Art des Feedbacks
            user_comment: Optional User-Kommentar
            confidence: Konfidenz (bei implizitem Feedback niedriger)
        """
        if turn_id not in self._turns:
            logger.warning(f"Turn {turn_id} not found")
            return

        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=turn_id,
            feedback_type=feedback_type,
            timestamp=datetime.now(timezone.utc),
            user_comment=user_comment,
            confidence=confidence
        )

        if turn_id not in self._feedback:
            self._feedback[turn_id] = []

        self._feedback[turn_id].append(feedback)

        logger.info(f"Recorded feedback for turn {turn_id}: {feedback_type.value}")

    def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        """Holt Turn anhand ID."""
        return self._turns.get(turn_id)

    def get_user_history(self, user_id: str, limit: int = 10) -> List[ConversationTurn]:
        """
        Holt Historie für User.

        Args:
            user_id: User ID
            limit: Max Anzahl Turns

        Returns:
            Liste der letzten Turns (neueste zuerst)
        """
        if user_id not in self._history:
            return []

        history = list(self._history[user_id])
        history.reverse()  # Neueste zuerst
        return history[:limit]

    def get_feedback_for_turn(self, turn_id: str) -> List[FeedbackEntry]:
        """Holt alle Feedbacks für einen Turn."""
        return self._feedback.get(turn_id, [])

    def get_negative_turns(self, user_id: Optional[str] = None,
                          limit: int = 50) -> List[tuple]:
        """
        Holt Turns mit negativem Feedback.

        Args:
            user_id: Optional - nur für diesen User
            limit: Max Anzahl

        Returns:
            Liste von (ConversationTurn, List[FeedbackEntry])
        """
        negative_turns = []

        # Durchsuche alle Turns mit Feedback
        for turn_id, feedbacks in self._feedback.items():
            # Check ob negatives Feedback vorhanden
            has_negative = any(
                f.feedback_type in [
                    FeedbackType.EXPLICIT_NEGATIVE,
                    FeedbackType.IMPLICIT_REFORMULATION,
                    FeedbackType.IMPLICIT_CONTRADICTION,
                    FeedbackType.SEMANTIC_IRRELEVANT,
                    FeedbackType.SEMANTIC_CONTRADICTION
                ]
                for f in feedbacks
            )

            if not has_negative:
                continue

            turn = self._turns.get(turn_id)
            if not turn:
                continue

            # User-Filter
            if user_id:
                # Check ob Turn zu diesem User gehört
                # (müssten wir user_id im Turn speichern - TODO)
                pass

            negative_turns.append((turn, feedbacks))

            if len(negative_turns) >= limit:
                break

        return negative_turns

    def detect_implicit_reformulation(self, user_id: str,
                                     current_message: str) -> Optional[str]:
        """
        Erkennt ob aktuelle Frage eine Umformulierung ist.

        Args:
            user_id: User ID
            current_message: Aktuelle User-Message

        Returns:
            turn_id des vorherigen Turns falls Umformulierung, sonst None
        """
        history = self.get_user_history(user_id, limit=3)

        if not history:
            return None

        # Prüfe letzte Turn
        last_turn = history[0]

        # Einfache Heuristik: Ähnlichkeit der Fragen
        # (könnte mit Embedding verbessert werden)
        similarity = self._text_similarity(
            current_message.lower(),
            last_turn.user_message.lower()
        )

        # Wenn sehr ähnlich (>0.7) aber nicht identisch
        if 0.5 < similarity < 0.95:
            logger.info(f"Detected reformulation: {similarity:.2f} similarity")
            return last_turn.turn_id

        return None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Einfache Text-Ähnlichkeit (Jaccard).

        TODO: Könnte mit Embeddings verbessert werden.
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def add_feedback(self, turn_id: str, feedback: FeedbackEntry):
        """
        Fügt ein FeedbackEntry zu einem Turn hinzu.

        Args:
            turn_id: ID des Turns
            feedback: FeedbackEntry Objekt
        """
        if turn_id not in self._turns:
            logger.warning(f"Turn {turn_id} not found")
            return

        if turn_id not in self._feedback:
            self._feedback[turn_id] = []

        self._feedback[turn_id].append(feedback)

        logger.info(f"Added feedback {feedback.feedback_id} to turn {turn_id}: {feedback.feedback_type.value}")

    def get_feedback_stats(self) -> Dict:
        """
        Gibt Statistiken über gesammeltes Feedback zurück.

        Returns:
            Dict mit Feedback-Statistiken
        """
        total_turns = len(self._turns)
        total_feedbacks = sum(len(fb_list) for fb_list in self._feedback.values())

        # Zähle Feedback-Typen
        positive_count = 0
        negative_count = 0
        correction_count = 0

        for fb_list in self._feedback.values():
            for fb in fb_list:
                if fb.feedback_type == FeedbackType.EXPLICIT_POSITIVE:
                    positive_count += 1
                elif fb.feedback_type == FeedbackType.EXPLICIT_NEGATIVE:
                    negative_count += 1

                if fb.suggested_correction:
                    correction_count += 1

        feedback_rate = (total_feedbacks / total_turns * 100) if total_turns > 0 else 0.0

        return {
            "total_turns": total_turns,
            "total_feedbacks": total_feedbacks,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "correction_count": correction_count,
            "feedback_rate": feedback_rate
        }


# Globale Instanz
_global_tracker = ConversationTracker()


def get_conversation_tracker() -> ConversationTracker:
    """Hole globale Tracker-Instanz."""
    return _global_tracker


def record_conversation_turn(user_id: str, user_message: str,
                            ai_response: str,
                            retrieved_memories: Optional[List[str]] = None,
                            response_time_ms: Optional[float] = None) -> str:
    """Wrapper für record_turn."""
    return _global_tracker.record_turn(
        user_id, user_message, ai_response,
        retrieved_memories, response_time_ms
    )


def record_user_feedback(turn_id: str, feedback_type: FeedbackType,
                        user_comment: Optional[str] = None):
    """Wrapper für record_feedback."""
    _global_tracker.record_feedback(turn_id, feedback_type, user_comment)
