"""Button entity for LexiAI connectivity checks."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo

from .api import LexiApiClient
from .const import DOMAIN

logger = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up LexiAI button entities."""
    data = hass.data[DOMAIN][entry.entry_id]
    client: LexiApiClient = data["client"]
    base_url = data.get("config", {}).get("base_url", "")
    async_add_entities([LexiAIPingButton(entry.entry_id, client, base_url)])


class LexiAIPingButton(ButtonEntity):
    """Button to manually test connectivity to the LexiAI backend."""

    _attr_has_entity_name = True
    _attr_name = "Connectivity check"
    _attr_icon = "mdi:lan-connect"

    def __init__(self, entry_id: str, client: LexiApiClient, base_url: str) -> None:
        self._entry_id = entry_id
        self._client = client
        self._base_url = base_url
        self._last_result: str | None = None
        self._last_checked: str | None = None
        self._attr_unique_id = f"{entry_id}_connectivity_check"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="LexiAI",
            manufacturer="LexiAI",
            configuration_url=self._base_url or None,
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        return {
            "last_result": self._last_result,
            "last_checked": self._last_checked,
        }

    async def async_press(self) -> None:
        """Handle the button press."""
        ok = await self._client.check_health()
        self._last_result = "ok" if ok else "failed"
        self._last_checked = datetime.now(timezone.utc).isoformat()
        self.async_write_ha_state()
