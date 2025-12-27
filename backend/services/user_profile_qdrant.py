import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from qdrant_client.models import PointStruct

from backend.core.component_cache import get_cached_components
from backend.embeddings.embedding_cache import cached_embed_query
from backend.qdrant.client_wrapper import safe_upsert, safe_delete
from backend.services.profile_context import ProfileContextBuilder

logger = logging.getLogger(__name__)


def _build_profile_summary(profile: Dict[str, Any]) -> str:
    builder = ProfileContextBuilder()
    summary = builder.get_user_context(profile, include_all=True, max_length=1200)
    return summary.strip() if summary else ""


def upsert_user_profile_in_qdrant(user_id: str, profile: Dict[str, Any]) -> Optional[str]:
    if not user_id:
        return None
    if not profile or all(k.startswith("_") for k in profile.keys()):
        try:
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore
            point_id = f"profile:{user_id}"
            safe_delete(collection_name=vectorstore.collection, points_selector=[point_id])
            logger.info(f"🧹 Profile deleted in Qdrant for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to delete profile in Qdrant for {user_id}: {e}")
        return None

    summary = _build_profile_summary(profile)
    if not summary:
        return None

    bundle = get_cached_components()
    vectorstore = bundle.vectorstore

    embedding = cached_embed_query(bundle.embeddings, summary)
    now = datetime.now(timezone.utc)
    payload = {
        "content": summary,
        "user_id": user_id,
        "category": "user_profile",
        "source": "user_profile",
        "tags": ["user_profile", "profile"],
        "timestamp": now.isoformat(),
        "timestamp_ms": int(now.timestamp() * 1000),
        "profile_snapshot": profile,
        "profile_updated_at": profile.get("_last_updated") or now.isoformat(),
    }

    point_id = f"profile:{user_id}"
    point = PointStruct(id=point_id, vector=embedding, payload=payload)
    safe_upsert(collection_name=vectorstore.collection, points=[point])
    logger.info(f"💾 Profile upserted in Qdrant for user {user_id}")
    return point_id
