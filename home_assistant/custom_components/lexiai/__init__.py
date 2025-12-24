"""LexiAI Home Assistant integration."""

from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .api import LexiApiClient
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_BATCH_INTERVAL,
    CONF_BATCH_SIZE,
    CONF_DOMAINS,
    CONF_TIMEOUT,
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_INTERVAL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DOMAINS,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from .event_listener import LexiEventBatcher

logger = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.BUTTON]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LexiAI from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    config: Dict[str, Any] = {
        **entry.data,
        **entry.options,
    }

    base_url = config.get(CONF_BASE_URL, DEFAULT_BASE_URL)
    api_key = config.get(CONF_API_KEY, "")
    domains = config.get(CONF_DOMAINS, DEFAULT_DOMAINS)
    batch_interval = config.get(CONF_BATCH_INTERVAL, DEFAULT_BATCH_INTERVAL)
    batch_size = config.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)
    timeout = config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

    client = LexiApiClient(hass, base_url=base_url, api_key=api_key, timeout=timeout)
    batcher = LexiEventBatcher(
        hass,
        api=client,
        domains=domains,
        batch_interval=batch_interval,
        batch_size=batch_size,
    )
    batcher.start()

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "batcher": batcher,
        "config": {
            "base_url": base_url,
        },
    }

    logger.info("LexiAI integration set up for %s", base_url)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a LexiAI config entry."""
    data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    if data and "batcher" in data:
        await data["batcher"].stop()
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options updates by reloading the integration."""
    await hass.config_entries.async_reload(entry.entry_id)
