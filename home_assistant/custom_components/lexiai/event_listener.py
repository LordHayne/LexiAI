"""Event listener and batching for LexiAI."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Set

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import Event, HomeAssistant

from .api import LexiApiClient
from .const import DEFAULT_BATCH_INTERVAL, DEFAULT_BATCH_SIZE, DEFAULT_DOMAINS

logger = logging.getLogger(__name__)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(v) for v in value]
    return str(value)


def _serialize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _serialize_value(v) for k, v in attributes.items()}


class LexiEventBatcher:
    """Collects state changes and sends them to LexiAI in batches."""

    def __init__(
        self,
        hass: HomeAssistant,
        api: LexiApiClient,
        domains: Optional[Iterable[str]] = None,
        batch_interval: float = DEFAULT_BATCH_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._hass = hass
        self._api = api
        self._domains: Set[str] = set(domains or DEFAULT_DOMAINS)
        self._batch_interval = batch_interval
        self._batch_size = batch_size
        self._queue: list[Dict[str, Any]] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._unsub = None

    def start(self) -> None:
        """Start listening for Home Assistant state changes."""
        if self._unsub is None:
            self._unsub = self._hass.bus.async_listen(EVENT_STATE_CHANGED, self.handle_state_changed)

    async def stop(self) -> None:
        """Stop listening and flush remaining events."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        async with self._lock:
            self._cancel_flush_task_locked()
        await self._flush_now()

    async def handle_state_changed(self, event: Event) -> None:
        """Handle a Home Assistant state_changed event."""
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        entity_id = new_state.entity_id
        if not entity_id or "." not in entity_id:
            return

        domain = entity_id.split(".", 1)[0]
        if domain not in self._domains:
            return

        payload = {
            "entity_id": entity_id,
            "domain": domain,
            "state": new_state.state,
            "attributes": _serialize_attributes(new_state.attributes or {}),
            "last_changed": new_state.last_changed.isoformat() if new_state.last_changed else None,
            "last_updated": new_state.last_updated.isoformat() if new_state.last_updated else None,
            "time_fired": event.time_fired.isoformat() if event.time_fired else None,
        }

        await self.add_event(payload)

    async def add_event(self, payload: Dict[str, Any]) -> None:
        """Add an event to the batch and schedule a flush if needed."""
        immediate = None
        async with self._lock:
            self._queue.append(payload)
            if len(self._queue) >= self._batch_size:
                immediate = self._drain_locked()
                self._cancel_flush_task_locked()
            elif self._flush_task is None:
                self._flush_task = self._hass.async_create_task(self._flush_after_delay())

        if immediate:
            await self._send(immediate)

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._batch_interval)
            await self._flush_now()
        except asyncio.CancelledError:
            return

    async def _flush_now(self) -> None:
        batch = None
        async with self._lock:
            self._flush_task = None
            if self._queue:
                batch = self._drain_locked()

        if batch:
            await self._send(batch)

    def _drain_locked(self) -> list[Dict[str, Any]]:
        batch = self._queue
        self._queue = []
        return batch

    def _cancel_flush_task_locked(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None

    async def _send(self, batch: list[Dict[str, Any]]) -> None:
        ok = await self._api.post_events(batch)
        if not ok:
            logger.debug("LexiAI event batch dropped (%d events)", len(batch))
