"""
Background worker for immediate feedback corrections.

Polls Qdrant for pending negative feedback and processes sequentially.
"""

import logging
import os
import threading
import time
from typing import Optional

from qdrant_client import models

from backend.qdrant.client_wrapper import safe_scroll, safe_set_payload
from backend.memory.conversation_tracker import FEEDBACK_COLLECTION, get_conversation_tracker
from backend.memory.self_correction import analyze_and_correct_turn

logger = logging.getLogger("lexi_middleware.feedback_worker")

_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_worker_running = False


def _run_loop(poll_interval_s: float, min_interval_s: float) -> None:
    global _worker_running
    last_run_ts = 0.0
    _worker_running = True

    while True:
        try:
            now = time.time()
            delta = now - last_run_ts
            if delta < min_interval_s:
                time.sleep(min_interval_s - delta)

            pending = _fetch_pending_feedback(limit=20)
            if not pending:
                time.sleep(poll_interval_s)
                continue

            for feedback in pending:
                turn_id = feedback.get("turn_id")
                feedback_id = feedback.get("feedback_id")
                if not turn_id or not feedback_id:
                    _mark_not_pending(feedback_id)
                    continue

                success = analyze_and_correct_turn(turn_id)
                if success:
                    _mark_not_pending(feedback_id)
                else:
                    # Clear pending if feedback is no longer valid (turn missing/processed)
                    tracker = get_conversation_tracker()
                    existing = tracker.get_feedback_for_turn(turn_id)
                    if not any((fb.feedback_id == feedback_id and not fb.processed) for fb in existing):
                        _mark_not_pending(feedback_id)

                last_run_ts = time.time()

        except Exception as e:
            logger.warning("Feedback worker loop error: %s", e)
            time.sleep(poll_interval_s)


def _fetch_pending_feedback(limit: int = 20):
    try:
        result = safe_scroll(
            collection_name=FEEDBACK_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="immediate_pending",
                        match=models.MatchValue(value=True)
                    ),
                    models.FieldCondition(
                        key="processed",
                        match=models.MatchValue(value=False)
                    ),
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        points = getattr(result, "points", None)
        if points is None:
            points = result[0] if isinstance(result, tuple) else []

        payloads = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            payloads.append(payload)
        return payloads
    except Exception as e:
        logger.debug("Failed to fetch pending feedback: %s", e)
        return []


def _mark_not_pending(feedback_id: Optional[str]) -> None:
    if not feedback_id:
        return
    try:
        safe_set_payload(
            collection_name=FEEDBACK_COLLECTION,
            payload={"immediate_pending": False},
            points=[feedback_id],
        )
    except Exception as e:
        logger.debug("Failed to clear immediate_pending for %s: %s", feedback_id, e)


def start_feedback_worker(force: bool = False) -> bool:
    """Start background feedback worker if enabled by env or forced."""
    enabled = os.environ.get("LEXI_FEEDBACK_WORKER", "0") == "1"
    if not enabled and not force:
        return False

    poll_interval_s = float(os.environ.get("LEXI_FEEDBACK_WORKER_POLL", "5"))
    min_interval_s = float(os.environ.get("LEXI_FEEDBACK_WORKER_MIN_INTERVAL", "5"))

    global _worker_thread
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return False

        _worker_thread = threading.Thread(
            target=_run_loop,
            args=(poll_interval_s, min_interval_s),
            name="lexi-feedback-worker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("🧠 Feedback worker started (poll=%ss, min_interval=%ss)", poll_interval_s, min_interval_s)
        return True


def is_feedback_worker_running() -> bool:
    if _worker_thread and _worker_thread.is_alive():
        return True
    return False


if __name__ == "__main__":
    # Allow running as standalone worker process
    os.environ.setdefault("LEXI_FEEDBACK_WORKER", "1")
    started = start_feedback_worker()
    if not started:
        logger.info("Feedback worker already running or disabled")
    while True:
        time.sleep(60)
