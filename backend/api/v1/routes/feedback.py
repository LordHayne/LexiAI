"""
Feedback API Endpoints für Self-Correction System.

Ermöglicht Users explizites Feedback (👍/👎) zu AI-Responses zu geben.
"""

import logging
from typing import Optional
from uuid import uuid4
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.memory.conversation_tracker import ConversationTracker
from backend.models.feedback import FeedbackEntry, FeedbackType

logger = logging.getLogger("lexi_middleware.feedback")

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])

# Singleton ConversationTracker
_conversation_tracker = ConversationTracker()


# Request Models
class ThumbsFeedbackRequest(BaseModel):
    """Request für Thumbs Up/Down Feedback."""
    turn_id: str = Field(..., description="ID des Conversation Turns")
    user_id: str = Field(default="default", description="User ID")


class CorrectionFeedbackRequest(BaseModel):
    """Request für explizite Korrektur."""
    turn_id: str = Field(..., description="ID des Conversation Turns")
    user_id: str = Field(default="default", description="User ID")
    correction_text: str = Field(..., description="Korrigierter Text vom User")


class FeedbackResponse(BaseModel):
    """Response nach Feedback-Speicherung."""
    status: str = Field(default="ok")
    feedback_id: str
    message: str
    turn_found: bool


# API Endpoints
@router.post("/thumbs-up", response_model=FeedbackResponse)
async def thumbs_up(request: ThumbsFeedbackRequest):
    """
    Registriert positives Feedback (👍) für eine AI-Response.

    Verwendung:
        POST /v1/feedback/thumbs-up
        {
            "turn_id": "turn_abc123",
            "user_id": "default"
        }

    Returns:
        FeedbackResponse mit feedback_id
    """
    logger.info(f"👍 Positives Feedback für Turn: {request.turn_id}")

    try:
        # Check ob Turn existiert
        turn = _conversation_tracker.get_turn(request.turn_id)

        if not turn:
            logger.warning(f"Turn nicht gefunden: {request.turn_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Turn {request.turn_id} nicht gefunden"
            )

        # Erstelle FeedbackEntry
        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=request.turn_id,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            timestamp=datetime.now(UTC),
            confidence=1.0
        )

        # Speichere Feedback
        _conversation_tracker.add_feedback(request.turn_id, feedback)

        logger.info(f"✅ Positives Feedback gespeichert: {feedback.feedback_id}")

        return FeedbackResponse(
            status="ok",
            feedback_id=feedback.feedback_id,
            message="Positives Feedback erfolgreich gespeichert",
            turn_found=True
        )

    except Exception as e:
        logger.error(f"Fehler beim Speichern von positivem Feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")


@router.post("/thumbs-down", response_model=FeedbackResponse)
async def thumbs_down(request: ThumbsFeedbackRequest):
    """
    Registriert negatives Feedback (👎) für eine AI-Response.

    Negatives Feedback triggert später Self-Correction Analyse im Heartbeat.

    Verwendung:
        POST /v1/feedback/thumbs-down
        {
            "turn_id": "turn_abc123",
            "user_id": "default"
        }

    Returns:
        FeedbackResponse mit feedback_id
    """
    logger.info(f"👎 Negatives Feedback für Turn: {request.turn_id}")

    try:
        # Check ob Turn existiert
        turn = _conversation_tracker.get_turn(request.turn_id)

        if not turn:
            logger.warning(f"Turn nicht gefunden: {request.turn_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Turn {request.turn_id} nicht gefunden"
            )

        # Erstelle FeedbackEntry
        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=request.turn_id,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            timestamp=datetime.now(UTC),
            confidence=1.0
        )

        # Speichere Feedback
        _conversation_tracker.add_feedback(request.turn_id, feedback)

        logger.info(f"✅ Negatives Feedback gespeichert: {feedback.feedback_id}")
        logger.info(f"   → Wird im nächsten Heartbeat (IDLE Mode) analysiert")

        return FeedbackResponse(
            status="ok",
            feedback_id=feedback.feedback_id,
            message="Negatives Feedback gespeichert. Wird analysiert im nächsten Heartbeat.",
            turn_found=True
        )

    except Exception as e:
        logger.error(f"Fehler beim Speichern von negativem Feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")


@router.post("/correction", response_model=FeedbackResponse)
async def correction(request: CorrectionFeedbackRequest):
    """
    Registriert explizite Korrektur vom User.

    Der User gibt direkt die korrekte Antwort.

    Verwendung:
        POST /v1/feedback/correction
        {
            "turn_id": "turn_abc123",
            "user_id": "default",
            "correction_text": "Die richtige Antwort ist..."
        }

    Returns:
        FeedbackResponse mit feedback_id
    """
    logger.info(f"📝 Explizite Korrektur für Turn: {request.turn_id}")

    try:
        # Check ob Turn existiert
        turn = _conversation_tracker.get_turn(request.turn_id)

        if not turn:
            logger.warning(f"Turn nicht gefunden: {request.turn_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Turn {request.turn_id} nicht gefunden"
            )

        # Erstelle FeedbackEntry mit Korrektur
        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=request.turn_id,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            timestamp=datetime.now(UTC),
            confidence=1.0,
            user_comment=request.correction_text,
            suggested_correction=request.correction_text  # User gibt direkt die Korrektur
        )

        # Speichere Feedback
        _conversation_tracker.add_feedback(request.turn_id, feedback)

        logger.info(f"✅ Explizite Korrektur gespeichert: {feedback.feedback_id}")
        logger.info(f"   Korrektur: {request.correction_text[:100]}...")

        return FeedbackResponse(
            status="ok",
            feedback_id=feedback.feedback_id,
            message="Korrektur gespeichert. Wird im nächsten Heartbeat verarbeitet.",
            turn_found=True
        )

    except Exception as e:
        logger.error(f"Fehler beim Speichern der Korrektur: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")


@router.get("/turns/{turn_id}")
async def get_turn(turn_id: str):
    """
    Ruft Details eines Conversation Turns ab.

    Useful für Debugging und UI Display.
    """
    turn = _conversation_tracker.get_turn(turn_id)

    if not turn:
        raise HTTPException(status_code=404, detail=f"Turn {turn_id} nicht gefunden")

    # Get all feedback for this turn
    feedbacks = _conversation_tracker.get_feedback_for_turn(turn_id)

    return {
        "turn": {
            "turn_id": turn.turn_id,
            "user_message": turn.user_message,
            "ai_response": turn.ai_response,
            "timestamp": turn.timestamp.isoformat(),
            "retrieved_memories": turn.retrieved_memories,
            "response_time_ms": turn.response_time_ms
        },
        "feedbacks": [
            {
                "feedback_id": fb.feedback_id,
                "feedback_type": fb.feedback_type.value,
                "timestamp": fb.timestamp.isoformat(),
                "user_comment": fb.user_comment,
                "confidence": fb.confidence
            }
            for fb in feedbacks
        ]
    }


@router.get("/stats")
async def get_feedback_stats():
    """
    Gibt Statistiken über gesammeltes Feedback.

    Returns:
        Dict mit Feedback-Statistiken
    """
    stats = _conversation_tracker.get_feedback_stats()

    return {
        "total_turns": stats.get("total_turns", 0),
        "total_feedbacks": stats.get("total_feedbacks", 0),
        "positive_feedbacks": stats.get("positive_count", 0),
        "negative_feedbacks": stats.get("negative_count", 0),
        "correction_feedbacks": stats.get("correction_count", 0),
        "feedback_rate": stats.get("feedback_rate", 0.0)
    }


# Convenience function für anderen Code
def get_conversation_tracker() -> ConversationTracker:
    """
    Gibt den Singleton ConversationTracker zurück.

    Usage in anderen Modulen:
        from backend.api.v1.routes.feedback import get_conversation_tracker
        tracker = get_conversation_tracker()
    """
    return _conversation_tracker
