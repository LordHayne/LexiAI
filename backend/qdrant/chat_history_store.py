import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from backend.config.middleware_config import MiddlewareConfig
from backend.qdrant.client_wrapper import get_qdrant_client, safe_scroll, safe_upsert

logger = logging.getLogger("lexi_chat_history")

CHAT_HISTORY_VECTOR_SIZE = 1
CHAT_HISTORY_VECTOR = [1.0]

_collection_ready = False
_collection_lock = threading.Lock()


def _get_collection_name() -> str:
    return MiddlewareConfig.get_chat_history_collection()


def _ensure_collection_exists() -> None:
    global _collection_ready
    if _collection_ready:
        return

    with _collection_lock:
        if _collection_ready:
            return

        client = get_qdrant_client()
        name = _get_collection_name()

        try:
            info = client.get_collection(name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != CHAT_HISTORY_VECTOR_SIZE:
                logger.warning(
                    "Chat history collection '%s' has vector size %s (expected %s)",
                    name,
                    existing_dim,
                    CHAT_HISTORY_VECTOR_SIZE,
                )
        except UnexpectedResponse:
            logger.info("Creating chat history collection '%s'", name)
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=CHAT_HISTORY_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

            # Payload indices for faster filtering and sorting
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=name,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=name,
                    field_name="role",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                client.create_payload_index(
                    collection_name=name,
                    field_name="ts",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            except Exception as e:
                logger.warning("Failed creating payload indices for '%s': %s", name, e)

        _collection_ready = True


def store_chat_message(
    *,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    seq: int = 0,
    turn_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    if not user_id or not session_id:
        return

    _ensure_collection_exists()

    timestamp = timestamp or datetime.now(timezone.utc)
    ts_ms = int(timestamp.timestamp() * 1000)

    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "seq": seq,
        "turn_id": turn_id,
        "ts": ts_ms,
        "ts_iso": timestamp.isoformat(),
    }

    point = PointStruct(
        id=str(uuid4()),
        vector=CHAT_HISTORY_VECTOR,
        payload={k: v for k, v in payload.items() if v is not None},
    )
    safe_upsert(collection_name=_get_collection_name(), points=[point])


def store_chat_turn(
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    turn_id: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc)
    assistant_ts = now + timedelta(milliseconds=1)
    store_chat_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=user_message,
        seq=0,
        timestamp=now,
    )
    store_chat_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        seq=1,
        turn_id=turn_id,
        timestamp=assistant_ts,
    )


def _scroll_points(scroll_filter: Filter, limit: int = 1000) -> List:
    _ensure_collection_exists()
    points = []
    offset = None

    while True:
        result = safe_scroll(
            collection_name=_get_collection_name(),
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        batch = getattr(result, "points", None)
        next_offset = getattr(result, "next_page_offset", None)
        if batch is None:
            batch, next_offset = result

        points.extend(batch)
        if not next_offset:
            break
        offset = next_offset

    return points


def list_sessions(user_id: str, limit: int = 50) -> List[Dict[str, Optional[str]]]:
    if not user_id:
        return []

    points = _scroll_points(
        Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    )

    sessions: Dict[str, Dict[str, Optional[str]]] = {}
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        session_id = payload.get("session_id")
        ts = payload.get("ts")
        seq = payload.get("seq", 0)
        if not session_id or ts is None:
            continue

        existing = sessions.get(session_id)
        if not existing or (ts, seq) > (existing.get("last_ts", 0), existing.get("last_seq", 0)):
            sessions[session_id] = {
                "session_id": session_id,
                "last_ts": ts,
                "last_ts_iso": payload.get("ts_iso"),
                "last_preview": (payload.get("content") or "")[:120],
                "last_role": payload.get("role"),
                "last_seq": seq,
            }

    session_list = sorted(
        sessions.values(),
        key=lambda item: item.get("last_ts", 0),
        reverse=True,
    )
    return session_list[:limit]


def fetch_session_messages(
    user_id: str,
    session_id: str,
    limit: int = 500,
) -> List[Dict[str, Optional[str]]]:
    if not user_id or not session_id:
        return []

    points = _scroll_points(
        Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
            ]
        )
    )

    messages = []
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        ts = payload.get("ts")
        seq = payload.get("seq", 0)
        messages.append(
            {
                "role": payload.get("role"),
                "content": payload.get("content"),
                "turn_id": payload.get("turn_id"),
                "ts": ts,
                "ts_iso": payload.get("ts_iso"),
                "seq": seq,
            }
        )

    messages.sort(key=lambda item: (item.get("ts", 0), item.get("seq", 0)))
    return messages[:limit]
