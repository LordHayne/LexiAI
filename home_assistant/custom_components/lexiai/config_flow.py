"""Config flow for LexiAI."""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import async_timeout
from aiohttp import ClientError
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

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
    DOMAIN_OPTIONS,
)


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class LexiAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LexiAI."""

    VERSION = 1

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None):
        errors: Dict[str, str] = {}
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            if not _is_valid_url(user_input[CONF_BASE_URL]):
                errors[CONF_BASE_URL] = "invalid_url"
            else:
                can_connect = await self._async_validate_input(user_input)
                if can_connect:
                    return self.async_create_entry(title="LexiAI", data=user_input)
                errors["base"] = "cannot_connect"

        data_schema = vol.Schema(
            {
                vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
                vol.Optional(CONF_API_KEY, default=""): str,
                vol.Optional(CONF_DOMAINS, default=DEFAULT_DOMAINS): cv.multi_select(DOMAIN_OPTIONS),
                vol.Optional(CONF_BATCH_INTERVAL, default=DEFAULT_BATCH_INTERVAL): vol.Coerce(float),
                vol.Optional(CONF_BATCH_SIZE, default=DEFAULT_BATCH_SIZE): vol.Coerce(int),
                vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(float),
            }
        )

        return self.async_show_form(step_id="user", data_schema=data_schema, errors=errors)

    @staticmethod
    def async_get_options_flow(config_entry):
        return LexiAIOptionsFlow(config_entry)

    async def _async_validate_input(self, data: Dict[str, Any]) -> bool:
        base_url = data[CONF_BASE_URL].rstrip("/")
        api_key = data.get(CONF_API_KEY, "")
        timeout = float(data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        url = f"{base_url}/v1/health"
        headers = {"X-API-Key": api_key} if api_key else {}
        session = async_get_clientsession(self.hass)

        try:
            async with async_timeout.timeout(timeout):
                response = await session.get(url, headers=headers)
            return response.status == 200
        except (ClientError, TimeoutError):
            return False


class LexiAIOptionsFlow(config_entries.OptionsFlow):
    """Handle LexiAI options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        data = {**self._config_entry.data, **self._config_entry.options}
        data_schema = vol.Schema(
            {
                vol.Optional(CONF_DOMAINS, default=data.get(CONF_DOMAINS, DEFAULT_DOMAINS)): cv.multi_select(
                    DOMAIN_OPTIONS
                ),
                vol.Optional(
                    CONF_BATCH_INTERVAL, default=data.get(CONF_BATCH_INTERVAL, DEFAULT_BATCH_INTERVAL)
                ): vol.Coerce(float),
                vol.Optional(CONF_BATCH_SIZE, default=data.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)): vol.Coerce(int),
                vol.Optional(CONF_TIMEOUT, default=data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)): vol.Coerce(float),
            }
        )

        return self.async_show_form(step_id="init", data_schema=data_schema)
