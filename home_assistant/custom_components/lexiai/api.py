"""LexiAI API client for Home Assistant."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import async_timeout
from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class LexiApiClient:
    """Client for sending state change batches to the LexiAI backend."""

    def __init__(self, hass: HomeAssistant, base_url: str, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._hass = hass
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._session = async_get_clientsession(hass)

    async def post_events(self, events: List[Dict[str, Any]]) -> bool:
        """Post a batch of state change events to the LexiAI backend."""
        if not events:
            return True

        url = f"{self._base_url}/v1/ha/events"
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        payload = {"source": "home_assistant", "events": events}

        try:
            async with async_timeout.timeout(self._timeout):
                response = await self._session.post(url, json=payload, headers=headers)
            if response.status != 200:
                body = await response.text()
                logger.warning("LexiAI event post failed: %s %s", response.status, body[:200])
                return False
            return True
        except ClientError as exc:
            logger.warning("LexiAI event post network error: %s", exc)
            return False

    async def check_health(self) -> bool:
        """Check if the LexiAI backend is reachable."""
        url = f"{self._base_url}/v1/health"
        headers = {"X-API-Key": self._api_key} if self._api_key else {}

        try:
            async with async_timeout.timeout(self._timeout):
                response = await self._session.get(url, headers=headers)
            return response.status == 200
        except ClientError as exc:
            logger.warning("LexiAI health check network error: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LexiAI health check error: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LexiAI event post error: %s", exc)
            return False
