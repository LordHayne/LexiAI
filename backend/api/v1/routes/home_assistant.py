"""
Home Assistant event ingestion endpoints for Lexi.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.middleware.auth import verify_api_key
from backend.config.feature_flags import FeatureFlags
from backend.services.home_assistant import get_ha_service

logger = logging.getLogger("lexi_middleware.home_assistant_api")
router = APIRouter()


class HAStateChangeEvent(BaseModel):
    """Minimal state change payload from Home Assistant."""

    entity_id: str = Field(..., description="Home Assistant entity_id")
    state: Optional[str] = Field(None, description="New state value")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    last_changed: Optional[str] = None
    last_updated: Optional[str] = None
    time_fired: Optional[str] = None
    domain: Optional[str] = None


class HAEventBatchRequest(BaseModel):
    """Batch request for state change events."""

    events: List[HAStateChangeEvent]
    source: Optional[str] = "home_assistant"
    user_id: Optional[str] = "default"


@router.post("/ha/events", dependencies=[Depends(verify_api_key)])
async def ingest_home_assistant_events(request: HAEventBatchRequest):
    """
    Ingest Home Assistant state change events and refresh Lexi caches.
    """
    if not FeatureFlags.is_enabled("home_assistant"):
        raise HTTPException(status_code=503, detail="Home Assistant feature is disabled")

    if not request.events:
        raise HTTPException(status_code=400, detail="No events provided")

    ha_service = get_ha_service()
    processed = 0
    skipped = 0

    for event in request.events:
        if not event.entity_id:
            skipped += 1
            continue
        if event.state is None:
            skipped += 1
            continue
        if ha_service.ingest_state_event(event.model_dump()):
            processed += 1
        else:
            skipped += 1

    logger.debug("Ingested HA events: processed=%s skipped=%s", processed, skipped)
    return {
        "success": True,
        "processed": processed,
        "skipped": skipped,
        "received": len(request.events),
    }
