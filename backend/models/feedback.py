"""
Feedback-Datenmodelle für Self-Correction.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum


class FeedbackType(Enum):
    """Typ des Feedbacks."""
    EXPLICIT_POSITIVE = "explicit_positive"  # 👍
    EXPLICIT_NEGATIVE = "explicit_negative"  # 👎
    IMPLICIT_REFORMULATION = "implicit_reformulation"  # User fragt nochmal anders
    IMPLICIT_CONTRADICTION = "implicit_contradiction"  # "Das ist falsch"
    IMPLICIT_REPETITION = "implicit_repetition"  # Gleiche Frage nochmal
    SEMANTIC_IRRELEVANT = "semantic_irrelevant"  # Antwort passt nicht zur Frage
    SEMANTIC_CONTRADICTION = "semantic_contradiction"  # Widerspricht Memories


class ErrorCategory(Enum):
    """Kategorie des Fehlers."""
    FACTUALLY_WRONG = "factually_wrong"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    TOO_TECHNICAL = "too_technical"
    TOO_SIMPLE = "too_simple"
    MISSING_CONTEXT = "missing_context"
    HALLUCINATION = "hallucination"


@dataclass
class ConversationTurn:
    """Ein Austausch: User-Message + AI-Response."""
    turn_id: str
    user_id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    retrieved_memories: Optional[List[str]] = None  # Memory IDs
    response_time_ms: Optional[float] = None


@dataclass
class FeedbackEntry:
    """Feedback zu einer AI-Response."""
    feedback_id: str
    turn_id: str
    feedback_type: FeedbackType
    timestamp: datetime

    # Optional: Zusätzliche Infos
    user_comment: Optional[str] = None
    confidence: float = 1.0  # Bei implizitem Feedback niedriger
    processed: bool = False
    immediate_pending: bool = False

    # Analyse-Ergebnisse (gefüllt von Analyzer)
    error_category: Optional[ErrorCategory] = None
    error_analysis: Optional[str] = None
    suggested_correction: Optional[str] = None

    def to_dict(self) -> dict:
        """Konvertiert zu Dict für Speicherung."""
        return {
            "feedback_id": self.feedback_id,
            "turn_id": self.turn_id,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_comment": self.user_comment,
            "confidence": self.confidence,
            "has_correction": bool(self.suggested_correction),
            "processed": self.processed,
            "immediate_pending": self.immediate_pending,
            "error_category": self.error_category.value if self.error_category else None,
            "error_analysis": self.error_analysis,
            "suggested_correction": self.suggested_correction
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FeedbackEntry':
        """Erstellt FeedbackEntry aus Dict."""
        return cls(
            feedback_id=data["feedback_id"],
            turn_id=data["turn_id"],
            feedback_type=FeedbackType(data["feedback_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_comment=data.get("user_comment"),
            confidence=data.get("confidence", 1.0),
            processed=data.get("processed", False),
            immediate_pending=data.get("immediate_pending", False),
            error_category=ErrorCategory(data["error_category"]) if data.get("error_category") else None,
            error_analysis=data.get("error_analysis"),
            suggested_correction=data.get("suggested_correction")
        )
