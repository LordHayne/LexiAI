"""
Trackt Konversationen für Feedback-Analyse.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
from collections import deque
from uuid import uuid4

from backend.models.feedback import ConversationTurn, FeedbackEntry, FeedbackType, ErrorCategory
from backend.config.middleware_config import MiddlewareConfig
from backend.qdrant.client_wrapper import get_qdrant_client, safe_scroll, safe_upsert, safe_set_payload
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger("lexi_middleware.conversation_tracker")

TURN_COLLECTION = "lexi_turns"
FEEDBACK_COLLECTION = "lexi_feedback"
_storage_ready = False
_storage_failed = False
_storage_failed_at: Optional[float] = None
_storage_lock = threading.Lock()
_zero_vector: Optional[List[float]] = None
_last_feedback_hydration_ts: Optional[int] = None
_last_feedback_hydration_at: Optional[float] = None


def _get_zero_vector() -> List[float]:
    global _zero_vector
    if _zero_vector is None:
        size = MiddlewareConfig.get_memory_dimension()
        _zero_vector = [0.0] * size
    return _zero_vector


def _ensure_storage_ready() -> None:
    global _storage_ready, _storage_failed
    global _storage_failed_at
    if _storage_ready or _storage_failed:
        if not _storage_failed:
            return
        retry_after = 60.0
        if _storage_failed_at and (time.time() - _storage_failed_at) < retry_after:
            return

    with _storage_lock:
        if _storage_ready or _storage_failed:
            if not _storage_failed:
                return
            retry_after = 60.0
            if _storage_failed_at and (time.time() - _storage_failed_at) < retry_after:
                return
            _storage_failed = False

        client = get_qdrant_client()
        vector_size = MiddlewareConfig.get_memory_dimension()

        try:
            for name in (TURN_COLLECTION, FEEDBACK_COLLECTION):
                try:
                    info = client.get_collection(name)
                    existing_dim = info.config.params.vectors.size
                    if existing_dim != vector_size:
                        logger.warning(
                            "Collection '%s' has vector size %s (expected %s)",
                            name,
                            existing_dim,
                            vector_size,
                        )
                except UnexpectedResponse:
                    client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )

            # Payload indices for turns
            try:
                client.create_payload_index(
                    collection_name=TURN_COLLECTION,
                    field_name="turn_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=TURN_COLLECTION,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=TURN_COLLECTION,
                    field_name="ts",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            except Exception as e:
                logger.debug("Failed creating payload indices for %s: %s", TURN_COLLECTION, e)

            # Payload indices for feedback
            try:
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="turn_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="feedback_type",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="immediate_pending",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="has_correction",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="processed",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                client.create_payload_index(
                    collection_name=FEEDBACK_COLLECTION,
                    field_name="ts",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            except Exception as e:
                logger.debug("Failed creating payload indices for %s: %s", FEEDBACK_COLLECTION, e)

            _storage_ready = True
        except Exception as e:
            _storage_failed = True
            _storage_failed_at = time.time()
            logger.warning("Feedback storage unavailable: %s", e)


def _feedback_from_payload(payload: dict) -> Optional[FeedbackEntry]:
    if not payload:
        return None
    try:
        return FeedbackEntry(
            feedback_id=payload.get("feedback_id") or payload.get("id"),
            turn_id=payload["turn_id"],
            feedback_type=FeedbackType(payload["feedback_type"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            user_comment=payload.get("user_comment"),
            confidence=payload.get("confidence", 1.0),
            processed=payload.get("processed", False),
            immediate_pending=payload.get("immediate_pending", False),
            error_category=ErrorCategory(payload["error_category"]) if payload.get("error_category") else None,
            error_analysis=payload.get("error_analysis"),
            suggested_correction=payload.get("suggested_correction"),
        )
    except Exception as e:
        logger.debug("Failed to parse feedback payload: %s", e)
        return None


def _turn_from_payload(payload: dict) -> Optional[ConversationTurn]:
    if not payload:
        return None
    try:
        return ConversationTurn(
            turn_id=payload["turn_id"],
            user_id=payload.get("user_id", "default"),
            user_message=payload.get("user_message", ""),
            ai_response=payload.get("ai_response", ""),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            retrieved_memories=payload.get("retrieved_memories"),
            response_time_ms=payload.get("response_time_ms"),
        )
    except Exception as e:
        logger.debug("Failed to parse turn payload: %s", e)
        return None

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
        self._feedback_ids: Set[str] = set()

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
            user_id=user_id,
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

        self._persist_turn(turn)

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
            loaded_turn = self._load_turn_from_storage(turn_id)
            if loaded_turn:
                self._turns[turn_id] = loaded_turn
            else:
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

        if feedback.feedback_id in self._feedback_ids:
            return
        self._feedback[turn_id].append(feedback)
        self._feedback_ids.add(feedback.feedback_id)
        self._persist_feedback(feedback)

        logger.info(f"Recorded feedback for turn {turn_id}: {feedback_type.value}")

    def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        """Holt Turn anhand ID."""
        turn = self._turns.get(turn_id)
        if turn:
            return turn

        loaded = self._load_turn_from_storage(turn_id)
        if loaded:
            self._turns[turn_id] = loaded
            return loaded

        return None

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
        feedbacks = self._feedback.get(turn_id, [])
        if feedbacks:
            return feedbacks

        loaded = self._load_feedback_for_turn(turn_id)
        if loaded:
            self._feedback[turn_id] = loaded
            for fb in loaded:
                self._feedback_ids.add(fb.feedback_id)
            return loaded

        return []

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
        self._hydrate_recent_feedback()

        negative_turns = []

        # Durchsuche alle Turns mit Feedback
        for turn_id, feedbacks in self._feedback.items():
            unprocessed_feedbacks = [fb for fb in feedbacks if not fb.processed]
            if not unprocessed_feedbacks:
                continue
            # Check ob negatives Feedback vorhanden
            has_negative = any(
                f.feedback_type in [
                    FeedbackType.EXPLICIT_NEGATIVE,
                    FeedbackType.IMPLICIT_REFORMULATION,
                    FeedbackType.IMPLICIT_CONTRADICTION,
                    FeedbackType.SEMANTIC_IRRELEVANT,
                    FeedbackType.SEMANTIC_CONTRADICTION
                ]
                for f in unprocessed_feedbacks
            )

            if not has_negative:
                continue

            turn = self._turns.get(turn_id)
            if not turn:
                continue

            # User-Filter
            if user_id:
                if turn.user_id != user_id:
                    continue

            negative_turns.append((turn, unprocessed_feedbacks))

            if len(negative_turns) >= limit:
                break

        return negative_turns

    def _hydrate_recent_feedback(self, limit: int = 200) -> None:
        """Lädt recent Feedback + Turns aus Storage (best effort)."""
        global _last_feedback_hydration_ts, _last_feedback_hydration_at
        _ensure_storage_ready()
        if not _storage_ready:
            return

        now = time.time()
        if _last_feedback_hydration_at and (now - _last_feedback_hydration_at) < 10.0:
            return

        try:
            last_ts = _last_feedback_hydration_ts or 0
            max_ts_seen = last_ts
            offset = None
            total_loaded = 0

            while True:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="ts",
                            range=models.Range(gt=last_ts)
                        )
                    ]
                )
                result = safe_scroll(
                    collection_name=FEEDBACK_COLLECTION,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                    limit=limit,
                    offset=offset,
                )
                points = getattr(result, "points", None)
                next_offset = getattr(result, "next_page_offset", None)
                if points is None:
                    points, next_offset = result if isinstance(result, tuple) else ([], None)

                if not points:
                    break

                for point in points:
                    payload = getattr(point, "payload", None) or {}
                    feedback = _feedback_from_payload(payload)
                    if not feedback:
                        continue
                    if feedback.feedback_id in self._feedback_ids:
                        continue
                    self._feedback.setdefault(feedback.turn_id, []).append(feedback)
                    self._feedback_ids.add(feedback.feedback_id)
                    total_loaded += 1
                    ts_value = payload.get("ts")
                    if isinstance(ts_value, int) and ts_value > max_ts_seen:
                        max_ts_seen = ts_value
                    if feedback.turn_id not in self._turns:
                        turn = self._load_turn_from_storage(feedback.turn_id)
                        if turn:
                            self._turns[feedback.turn_id] = turn

                if not next_offset:
                    break
                offset = next_offset

            if max_ts_seen > last_ts:
                _last_feedback_hydration_ts = max_ts_seen
            _last_feedback_hydration_at = now
            if total_loaded:
                logger.debug("Hydrated %s feedback entries from storage", total_loaded)
        except Exception as e:
            logger.debug("Failed to hydrate feedback from storage: %s", e)

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

    def _persist_turn(self, turn: ConversationTurn) -> None:
        _ensure_storage_ready()
        if not _storage_ready:
            return

        try:
            ts_ms = int(turn.timestamp.timestamp() * 1000)
            payload = {
                "turn_id": turn.turn_id,
                "user_id": turn.user_id,
                "user_message": turn.user_message,
                "ai_response": turn.ai_response,
                "timestamp": turn.timestamp.isoformat(),
                "retrieved_memories": turn.retrieved_memories,
                "response_time_ms": turn.response_time_ms,
                "ts": ts_ms,
            }
            point = models.PointStruct(
                id=turn.turn_id,
                vector=_get_zero_vector(),
                payload=payload,
            )
            safe_upsert(collection_name=TURN_COLLECTION, points=[point])
        except Exception as e:
            logger.debug("Failed to persist turn %s: %s", turn.turn_id, e)

    def _persist_feedback(self, feedback: FeedbackEntry) -> None:
        _ensure_storage_ready()
        if not _storage_ready:
            return

        try:
            ts_ms = int(feedback.timestamp.timestamp() * 1000)
            payload = feedback.to_dict()
            turn = self._turns.get(feedback.turn_id)
            if turn:
                payload["user_id"] = turn.user_id
            payload.update(
                {
                    "feedback_id": feedback.feedback_id,
                    "ts": ts_ms,
                }
            )
            point = models.PointStruct(
                id=feedback.feedback_id,
                vector=_get_zero_vector(),
                payload=payload,
            )
            safe_upsert(collection_name=FEEDBACK_COLLECTION, points=[point])
        except Exception as e:
            logger.debug("Failed to persist feedback %s: %s", feedback.feedback_id, e)

    def update_feedback_entry(self, feedback: FeedbackEntry) -> None:
        """Persist updated feedback fields for existing feedback."""
        _ensure_storage_ready()
        if not _storage_ready:
            return

        try:
            payload = feedback.to_dict()
            safe_set_payload(
                collection_name=FEEDBACK_COLLECTION,
                payload=payload,
                points=[feedback.feedback_id],
            )
        except Exception as e:
            logger.debug("Failed to update feedback %s: %s", feedback.feedback_id, e)

    def _load_turn_from_storage(self, turn_id: str) -> Optional[ConversationTurn]:
        _ensure_storage_ready()
        if not _storage_ready:
            return None

        try:
            client = get_qdrant_client()
            points = client.retrieve(
                collection_name=TURN_COLLECTION,
                ids=[turn_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            payload = points[0].payload if hasattr(points[0], "payload") else None
            return _turn_from_payload(payload)
        except Exception as e:
            logger.debug("Failed to load turn %s from storage: %s", turn_id, e)
            return None

    def _load_feedback_for_turn(self, turn_id: str) -> List[FeedbackEntry]:
        _ensure_storage_ready()
        if not _storage_ready:
            return []

        try:
            scroll_filter = models.Filter(
                must=[models.FieldCondition(key="turn_id", match=models.MatchValue(value=turn_id))]
            )
            result = safe_scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=100,
            )
            points = getattr(result, "points", None)
            if points is None:
                points = result[0] if isinstance(result, tuple) else []
            feedbacks = []
            for point in points:
                payload = getattr(point, "payload", None) or {}
                feedback = _feedback_from_payload(payload)
                if feedback:
                    feedbacks.append(feedback)
            return feedbacks
        except Exception as e:
            logger.debug("Failed to load feedback for turn %s: %s", turn_id, e)
            return []

    def add_feedback(self, turn_id: str, feedback: FeedbackEntry):
        """
        Fügt ein FeedbackEntry zu einem Turn hinzu.

        Args:
            turn_id: ID des Turns
            feedback: FeedbackEntry Objekt
        """
        if turn_id not in self._turns:
            loaded_turn = self._load_turn_from_storage(turn_id)
            if loaded_turn:
                self._turns[turn_id] = loaded_turn
            else:
                logger.warning(f"Turn {turn_id} not found")
                return

        if turn_id not in self._feedback:
            self._feedback[turn_id] = []

        if feedback.feedback_id in self._feedback_ids:
            return
        self._feedback[turn_id].append(feedback)
        self._feedback_ids.add(feedback.feedback_id)
        self._persist_feedback(feedback)

        logger.info(f"Added feedback {feedback.feedback_id} to turn {turn_id}: {feedback.feedback_type.value}")

    def get_feedback_stats(self) -> Dict:
        """
        Gibt Statistiken über gesammeltes Feedback zurück.

        Returns:
            Dict mit Feedback-Statistiken
        """
        total_turns = len(self._turns)
        total_feedbacks = sum(len(fb_list) for fb_list in self._feedback.values())

        storage_stats = self._get_feedback_stats_from_storage()
        if storage_stats:
            return storage_stats

        # Zähle Feedback-Typen
        positive_count = 0
        negative_count = 0
        correction_count = 0
        unprocessed_count = 0

        for fb_list in self._feedback.values():
            for fb in fb_list:
                if fb.feedback_type == FeedbackType.EXPLICIT_POSITIVE:
                    positive_count += 1
                elif fb.feedback_type == FeedbackType.EXPLICIT_NEGATIVE:
                    negative_count += 1

                if fb.suggested_correction:
                    correction_count += 1
                if not fb.processed:
                    unprocessed_count += 1

        feedback_rate = (total_feedbacks / total_turns * 100) if total_turns > 0 else 0.0

        return {
            "total_turns": total_turns,
            "total_feedbacks": total_feedbacks,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "correction_count": correction_count,
            "unprocessed_count": unprocessed_count,
            "feedback_rate": feedback_rate
        }

    def _get_feedback_stats_from_storage(self) -> Optional[Dict]:
        _ensure_storage_ready()
        if not _storage_ready:
            return None

        try:
            client = get_qdrant_client()
            total_turns = client.count(TURN_COLLECTION, exact=True).count
            total_feedbacks = client.count(FEEDBACK_COLLECTION, exact=True).count

            positive_count = client.count(
                FEEDBACK_COLLECTION,
                exact=True,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="feedback_type",
                        match=models.MatchValue(value=FeedbackType.EXPLICIT_POSITIVE.value)
                    )]
                ),
            ).count

            negative_count = client.count(
                FEEDBACK_COLLECTION,
                exact=True,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="feedback_type",
                        match=models.MatchValue(value=FeedbackType.EXPLICIT_NEGATIVE.value)
                    )]
                ),
            ).count

            correction_count = client.count(
                FEEDBACK_COLLECTION,
                exact=True,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="has_correction",
                        match=models.MatchValue(value=True)
                    )]
                ),
            ).count

            unprocessed_count = client.count(
                FEEDBACK_COLLECTION,
                exact=True,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="processed",
                        match=models.MatchValue(value=False)
                    )]
                ),
            ).count

            feedback_rate = (total_feedbacks / total_turns * 100) if total_turns > 0 else 0.0
            return {
                "total_turns": total_turns,
                "total_feedbacks": total_feedbacks,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "correction_count": correction_count,
                "unprocessed_count": unprocessed_count,
                "feedback_rate": feedback_rate
            }
        except Exception as e:
            logger.debug("Failed to get feedback stats from storage: %s", e)
            return None

    def get_feedback_storage_status(self) -> Dict:
        """Expose storage readiness and hydration timestamps."""
        return {
            "storage_ready": _storage_ready,
            "storage_failed": _storage_failed,
            "storage_failed_at": _storage_failed_at,
            "last_feedback_hydration_ts": _last_feedback_hydration_ts,
            "last_feedback_hydration_at": _last_feedback_hydration_at,
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
