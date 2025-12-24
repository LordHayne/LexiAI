"""
Home Assistant Integration Service

Provides smart home device control via Home Assistant REST API.
"""
import logging
import os
from typing import Dict, Any, Optional, List
from functools import lru_cache
import asyncio
import aiohttp
import re
from difflib import get_close_matches
from uuid import uuid4

logger = logging.getLogger(__name__)


class HomeAssistantService:
    """
    Service for controlling Home Assistant devices.

    Features:
    - Device control (lights, switches, climate)
    - State queries
    - Service calls
    - Entity discovery
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Home Assistant service.

        Args:
            url: Home Assistant URL (e.g., 'http://homeassistant.local:8123')
            token: Long-lived access token
        """
        # Load from environment if not provided
        self.url = (url or os.getenv("LEXI_HA_URL", "")).rstrip('/')
        self.token = token or os.getenv("LEXI_HA_TOKEN")
        self.enabled = bool(self.url and self.token)

        # Session pooling for performance (reuse connections)
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=10.0)  # 10 second timeout

        # Cache for entity states (TTL-based)
        self._state_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self._cache_ttl = 30.0  # 30 seconds cache

        # Post-action verification
        self._verify_state_delay = 0.35  # delay before state check
        self._verify_state_retries = 2  # total attempts

        # Entity resolution cache
        self._entities_cache: Optional[List[Dict[str, Any]]] = None
        self._entities_cache_time: float = 0
        self._entities_cache_ttl = 300.0  # 5 minutes cache for entities
        self._friendly_name_map: Dict[str, List[str]] = {}  # friendly_name -> entity_ids
        self._registry_cache_time: float = 0
        self._registry_cache_ttl = 300.0  # 5 minutes cache for registry data
        self._areas_cache: Optional[List[Dict[str, Any]]] = None
        self._devices_cache: Optional[List[Dict[str, Any]]] = None
        self._entity_registry_cache: Optional[List[Dict[str, Any]]] = None
        self._area_name_map: Dict[str, str] = {}  # area name -> area_id
        self._device_name_map: Dict[str, str] = {}  # device name -> device_id
        self._ws_msg_id = 0

        # Allowed service domains for automation/script creation
        self._allowed_service_domains = {
            "light", "switch", "climate", "cover", "media_player", "lock",
            "fan", "scene", "script", "homeassistant", "automation"
        }

        if not self.url:
            logger.warning("Home Assistant URL nicht konfiguriert. Setze LEXI_HA_URL.")
        if not self.token:
            logger.warning("Home Assistant Token nicht konfiguriert. Setze LEXI_HA_TOKEN.")

        if self.enabled:
            logger.info(f"✅ Home Assistant Service initialisiert: {self.url}")
        else:
            logger.info("⚠️ Home Assistant Service nicht konfiguriert (URL oder Token fehlt)")

    def is_enabled(self) -> bool:
        """Check if Home Assistant is configured and enabled."""
        return self.enabled

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session with connection pooling.

        Returns:
            Reusable aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._timeout
            )
        return self._session

    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("🔒 Home Assistant session closed")

    def _invalidate_cache(self, entity_id: Optional[str] = None):
        """
        Invalidate state cache.

        Args:
            entity_id: Specific entity to invalidate, or None for all
        """
        if entity_id:
            self._state_cache.pop(entity_id, None)
        else:
            self._state_cache.clear()

    def ingest_state_event(self, event: Dict[str, Any]) -> bool:
        """
        Update local caches from a Home Assistant state_changed event.

        Args:
            event: Dict with entity_id, state, attributes, last_changed, last_updated
        """
        entity_id = event.get("entity_id")
        if not entity_id:
            return False

        state = event.get("state")
        attributes = event.get("attributes") or {}
        result = {
            "success": True,
            "entity_id": entity_id,
            "state": state,
            "attributes": attributes,
            "last_changed": event.get("last_changed"),
            "last_updated": event.get("last_updated"),
        }

        self._state_cache[entity_id] = (result, asyncio.get_event_loop().time())

        friendly_name = attributes.get("friendly_name")
        if friendly_name:
            for key in {friendly_name, friendly_name.lower()}:
                self._friendly_name_map.setdefault(key, [])
                if entity_id not in self._friendly_name_map[key]:
                    self._friendly_name_map[key].append(entity_id)

        if self._entities_cache:
            for entity in self._entities_cache:
                if entity.get("entity_id") == entity_id:
                    entity["state"] = state
                    entity["attributes"] = attributes
                    if friendly_name:
                        entity["friendly_name"] = friendly_name
                    break

        return True

    async def _verify_state(
        self,
        entity_id: str,
        expected_state: Optional[str]
    ) -> Dict[str, Any]:
        """
        Verify device state after a control action.

        Returns:
            Dict with verification details and latest state if available
        """
        last_state = None
        last_error = None

        for attempt in range(self._verify_state_retries):
            if attempt > 0:
                await asyncio.sleep(self._verify_state_delay)

            state_result = await self.get_state(entity_id, use_cache=False)
            if not state_result.get("success"):
                last_error = state_result.get("error")
                continue

            last_state = state_result
            state_value = state_result.get("state")

            if expected_state is None or state_value == expected_state:
                return {
                    "verified": True,
                    "state": state_value,
                    "attributes": state_result.get("attributes", {}),
                    "expected_state": expected_state
                }

        return {
            "verified": False,
            "state": last_state.get("state") if last_state else None,
            "attributes": last_state.get("attributes", {}) if last_state else {},
            "expected_state": expected_state,
            "error": last_error or "Status nicht erreicht"
        }

    async def control_device(
        self,
        entity_id: str,
        action: str,
        value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Control a Home Assistant device.

        Args:
            entity_id: Entity ID (e.g., 'light.bedroom', 'switch.coffee_maker')
            action: Action (turn_on, turn_off, toggle, set_brightness, set_temperature)
            value: Optional value for brightness (0-255) or temperature

        Returns:
            Dictionary with success status and result

        Examples:
            >>> await control_device("light.wohnzimmer", "turn_on")
            >>> await control_device("light.wohnzimmer", "set_brightness", 128)
            >>> await control_device("climate.heizung", "set_temperature", 22.5)
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Home Assistant nicht konfiguriert. Setze LEXI_HA_URL und LEXI_HA_TOKEN."
            }

        try:
            # Extract domain from entity_id (e.g., "light" from "light.wohnzimmer")
            domain = entity_id.split('.')[0]

            # Map actions to Home Assistant services
            service_map = {
                'turn_on': 'turn_on',
                'turn_off': 'turn_off',
                'toggle': 'toggle',
                'set_brightness': 'turn_on',  # Brightness uses turn_on with brightness param
                'set_temperature': 'set_temperature'
            }

            service = service_map.get(action, action)

            # Build service data payload
            data = {"entity_id": entity_id}

            # Add action-specific parameters
            if action == 'set_brightness' and value is not None:
                # HA expects brightness in range 0-255
                data['brightness'] = int(min(255, max(0, value)))
            elif action == 'set_temperature' and value is not None:
                data['temperature'] = float(value)

            # Make API call to Home Assistant
            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }

            url = f"{self.url}/api/services/{domain}/{service}"

            # Secure logging (don't expose full data)
            logger.info(f"🏠 Home Assistant: Calling {domain}.{service} for {entity_id}")
            logger.debug(f"Request data: {data}")

            try:
                async with session.post(url, json=data, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✅ Gerät gesteuert: {entity_id} -> {action}")

                        # Invalidate cache for this entity
                        self._invalidate_cache(entity_id)

                        expected_state = None
                        if action in ["turn_on", "set_brightness"]:
                            expected_state = "on"
                        elif action == "turn_off":
                            expected_state = "off"

                        verification = await self._verify_state(entity_id, expected_state)

                        if expected_state and not verification.get("verified"):
                            state_value = verification.get("state")
                            error_msg = (
                                f"Status nach Aktion '{action}' ist '{state_value}' "
                                f"(erwartet '{expected_state}')"
                            )
                            return {
                                "success": False,
                                "entity_id": entity_id,
                                "action": action,
                                "service": f"{domain}.{service}",
                                "result": result,
                                "state": state_value,
                                "attributes": verification.get("attributes", {}),
                                "verification": verification,
                                "error": error_msg
                            }

                        return {
                            "success": True,
                            "entity_id": entity_id,
                            "action": action,
                            "service": f"{domain}.{service}",
                            "result": result,
                            "state": verification.get("state"),
                            "attributes": verification.get("attributes", {}),
                            "verification": verification
                        }
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ Home Assistant Fehler {resp.status}: {error_text[:100]}")

                        return {
                            "success": False,
                            "entity_id": entity_id,
                            "error": f"HTTP {resp.status}: {error_text[:200]}"
                        }

            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout bei Gerätesteuerung: {entity_id}")
                return {
                    "success": False,
                    "entity_id": entity_id,
                    "error": "Request timeout - Home Assistant nicht erreichbar"
                }
            except aiohttp.ClientError as e:
                logger.error(f"🌐 Network error bei Gerätesteuerung: {e}")
                return {
                    "success": False,
                    "entity_id": entity_id,
                    "error": f"Network error: {str(e)}"
                }

        except Exception as e:
            logger.exception(f"❌ Unerwarteter Fehler bei Gerätesteuerung: {e}")
            return {
                "success": False,
                "entity_id": entity_id,
                "error": f"Unexpected error: {str(e)}"
            }

    async def get_state(self, entity_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current state of a Home Assistant entity.

        Args:
            entity_id: Entity ID to query
            use_cache: Use cached result if available (default: True)

        Returns:
            Dictionary with state information

        Example:
            >>> await get_state("light.wohnzimmer")
            {
                "success": True,
                "entity_id": "light.wohnzimmer",
                "state": "on",
                "attributes": {"brightness": 255, ...}
            }
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Home Assistant nicht konfiguriert"
            }

        # Check cache first
        if use_cache and entity_id in self._state_cache:
            cached_data, cached_time = self._state_cache[entity_id]
            if asyncio.get_event_loop().time() - cached_time < self._cache_ttl:
                logger.debug(f"💾 Cache hit for {entity_id}")
                return cached_data

        try:
            logger.info(f"🏠 Home Assistant: State query for {entity_id}")
            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }

            url = f"{self.url}/api/states/{entity_id}"

            async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        state_data = await resp.json()

                        result = {
                            "success": True,
                            "entity_id": entity_id,
                            "state": state_data.get("state"),
                            "attributes": state_data.get("attributes", {}),
                            "last_changed": state_data.get("last_changed"),
                            "last_updated": state_data.get("last_updated")
                        }

                        # Cache the result
                        self._state_cache[entity_id] = (result, asyncio.get_event_loop().time())

                        return result
                    elif resp.status == 404:
                        return {
                            "success": False,
                            "entity_id": entity_id,
                            "error": f"Entity '{entity_id}' nicht gefunden"
                        }
                    else:
                        error_text = await resp.text()
                        return {
                            "success": False,
                            "entity_id": entity_id,
                            "error": f"HTTP {resp.status}: {error_text[:200]}"
                        }

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout bei Status-Abfrage: {entity_id}")
            return {
                "success": False,
                "entity_id": entity_id,
                "error": "Request timeout - Home Assistant nicht erreichbar"
            }
        except aiohttp.ClientError as e:
            logger.error(f"🌐 Network error bei Status-Abfrage: {e}")
            return {
                "success": False,
                "entity_id": entity_id,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"❌ Unerwarteter Fehler bei Status-Abfrage: {e}")
            return {
                "success": False,
                "entity_id": entity_id,
                "error": f"Unexpected error: {str(e)}"
            }

    async def query_sensor(self, entity_id: str) -> Dict[str, Any]:
        """
        Query sensor data with formatted output.

        Supports:
        - Temperature sensors (climate.*, sensor.*temperature*)
        - Humidity sensors (sensor.*humidity*)
        - Brightness sensors (light.*, sensor.*brightness*)
        - Any sensor domain entities

        Args:
            entity_id: Entity ID or natural name (e.g., "wohnzimmer", "sensor.temperature")

        Returns:
            Dictionary with formatted sensor data

        Examples:
            >>> await query_sensor("climate.wohnzimmer")
            {
                "success": True,
                "entity_id": "climate.wohnzimmer",
                "sensor_type": "climate",
                "formatted_value": "22.5°C, Luftfeuchtigkeit: 45%",
                "raw_data": {...}
            }
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Home Assistant nicht konfiguriert"
            }

        logger.info(f"🏠 Home Assistant: Sensor query for '{entity_id}'")

        # Resolve entity if it's not a full entity_id
        resolved_entity = entity_id
        if '.' not in entity_id:
            preferred_domains = self._infer_preferred_domains(entity_id, for_query=True)
            resolved_entity = await self.resolve_entity(entity_id, preferred_domains=preferred_domains)
            if not resolved_entity:
                return {
                    "success": False,
                    "error": (
                        f"Konnte Entity '{entity_id}' nicht finden. "
                        "Bitte pruefe den Namen oder verwende die vollstaendige Entity-ID."
                    )
                }

        # Get entity state
        state_result = await self.get_state(resolved_entity, use_cache=False)

        if not state_result.get("success"):
            return state_result

        # Extract domain and format response
        domain = resolved_entity.split('.')[0]
        state = state_result.get("state")
        attributes = state_result.get("attributes", {})

        # Format based on domain and sensor type
        formatted_value = None
        sensor_type = domain

        if domain == "climate":
            # Climate domain: temperature, humidity, hvac_action
            temp = attributes.get("current_temperature")
            humidity = attributes.get("current_humidity")
            hvac_action = attributes.get("hvac_action", state)

            parts = []
            if temp is not None:
                parts.append(f"{temp}°C")
            if humidity is not None:
                parts.append(f"Luftfeuchtigkeit: {humidity}%")
            if hvac_action:
                action_map = {
                    "heating": "Heizt",
                    "cooling": "Kühlt",
                    "idle": "Bereit",
                    "off": "Aus"
                }
                parts.append(action_map.get(hvac_action, hvac_action))

            formatted_value = ", ".join(parts) if parts else state
            sensor_type = "climate"

        elif domain == "sensor":
            # Sensor domain: various types
            unit = attributes.get("unit_of_measurement", "")
            device_class = attributes.get("device_class", "")

            # Format based on device class
            if "temperature" in device_class or "temperature" in resolved_entity.lower():
                formatted_value = f"{state}°C" if unit == "°C" else f"{state}{unit}"
                sensor_type = "temperature"
            elif "humidity" in device_class or "humidity" in resolved_entity.lower():
                formatted_value = f"{state}%" if unit == "%" else f"{state}{unit}"
                sensor_type = "humidity"
            elif "illuminance" in device_class or "brightness" in resolved_entity.lower():
                formatted_value = f"{state} lx" if unit == "lx" else f"{state}{unit}"
                sensor_type = "brightness"
            elif "power" in device_class:
                formatted_value = f"{state} W" if unit == "W" else f"{state}{unit}"
                sensor_type = "power"
            elif "energy" in device_class:
                formatted_value = f"{state} kWh" if unit == "kWh" else f"{state}{unit}"
                sensor_type = "energy"
            else:
                formatted_value = f"{state}{unit}".strip()
                sensor_type = device_class or "sensor"

        elif domain == "light":
            # Light: brightness level
            brightness = attributes.get("brightness")
            if brightness is not None:
                brightness_pct = int((brightness / 255) * 100)
                formatted_value = f"{state.capitalize()}, Helligkeit: {brightness_pct}%"
            else:
                formatted_value = state.capitalize()
            sensor_type = "light"

        elif domain == "switch":
            formatted_value = "Eingeschaltet" if state == "on" else "Ausgeschaltet"
            sensor_type = "switch"

        elif domain == "cover":
            position = attributes.get("current_position")
            if position is not None:
                formatted_value = f"{state.capitalize()}, Position: {position}%"
            else:
                formatted_value = state.capitalize()
            sensor_type = "cover"

        else:
            # Generic formatting
            formatted_value = str(state)

        return {
            "success": True,
            "entity_id": resolved_entity,
            "sensor_type": sensor_type,
            "state": state,
            "formatted_value": formatted_value,
            "attributes": attributes,
            "friendly_name": attributes.get("friendly_name", resolved_entity)
        }

    async def list_entities(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        List available entities from Home Assistant.

        Args:
            domain: Optional domain filter (e.g., 'light', 'switch', 'climate')

        Returns:
            Dictionary with list of entities

        Example:
            >>> await list_entities("light")
            {
                "success": True,
                "entities": [
                    {"entity_id": "light.wohnzimmer", "state": "on", ...},
                    ...
                ]
            }
        """
        if not self.is_enabled():
            return {
                "success": False,
                "error": "Home Assistant nicht konfiguriert"
            }

        try:
            logger.info(
                "🏠 Home Assistant: Entity list requested"
                + (f" (domain={domain})" if domain else "")
            )
            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }

            url = f"{self.url}/api/states"

            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        all_states = await resp.json()

                        # Filter by domain if specified
                        if domain:
                            filtered = [
                                entity for entity in all_states
                                if entity.get("entity_id", "").startswith(f"{domain}.")
                            ]
                        else:
                            filtered = all_states

                        # Simplify entity data
                        entities = [
                            {
                                "entity_id": e.get("entity_id"),
                                "state": e.get("state"),
                                "friendly_name": e.get("attributes", {}).get("friendly_name"),
                                "domain": e.get("entity_id", "").split(".")[0],
                                "attributes": e.get("attributes", {})
                            }
                            for e in filtered
                        ]

                        return {
                            "success": True,
                            "entities": entities,
                            "count": len(entities),
                            "domain_filter": domain
                        }
                    else:
                        error_text = await resp.text()
                        return {
                            "success": False,
                            "error": f"HTTP {resp.status}: {error_text[:200]}"
                        }

            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout bei Entity-Liste")
                return {
                    "success": False,
                    "error": "Request timeout - Home Assistant nicht erreichbar"
                }
            except aiohttp.ClientError as e:
                logger.error(f"🌐 Network error bei Entity-Liste: {e}")
                return {
                    "success": False,
                    "error": f"Network error: {str(e)}"
                }

        except Exception as e:
            logger.exception(f"❌ Unerwarteter Fehler bei Entity-Liste: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    async def _call_service(self, domain: str, service: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a Home Assistant service without requiring an entity_id."""
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}/api/services/{domain}/{service}"
        payload = data or {}

        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    return {"success": True, "result": await resp.json()}
                error_text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout - Home Assistant nicht erreichbar"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Network error: {str(e)}"}

    async def _ws_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single websocket request to Home Assistant."""
        ws_url = self.url.replace("http", "ws", 1) + "/api/websocket"
        self._ws_msg_id += 1
        message["id"] = self._ws_msg_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    greeting = await ws.receive_json()
                    if greeting.get("type") != "auth_required":
                        return {"success": False, "error": "WebSocket auth_required missing"}

                    await ws.send_json({"type": "auth", "access_token": self.token})
                    auth_resp = await ws.receive_json()
                    if auth_resp.get("type") != "auth_ok":
                        return {"success": False, "error": "WebSocket auth failed"}

                    await ws.send_json(message)
                    response = await ws.receive_json()
                    if response.get("type") == "result" and response.get("success"):
                        return {"success": True, "result": response.get("result")}
                    return {
                        "success": False,
                        "error": response.get("error", {}).get("message", "WebSocket request failed")
                    }
        except asyncio.TimeoutError:
            return {"success": False, "error": "WebSocket timeout"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"WebSocket network error: {str(e)}"}

    async def _fetch_registry(self, path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch registry data (areas/devices/entities) from Home Assistant.

        Returns:
            List of registry entries, or None if unavailable.
        """
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}{path}"

        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status in [400, 401, 403, 404]:
                    logger.info(f"🏠 Home Assistant registry endpoint not available: {path} ({resp.status})")
                    return None
                error_text = await resp.text()
                logger.warning(f"🏠 Home Assistant registry error {resp.status}: {error_text[:120]}")
                return None
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout beim Registry-Request: {path}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"🌐 Network error beim Registry-Request {path}: {e}")
            return None

    async def _load_registry_cache(self) -> None:
        """Load area/device/entity registry caches if available."""
        current_time = asyncio.get_event_loop().time()
        if self._entity_registry_cache and (current_time - self._registry_cache_time) < self._registry_cache_ttl:
            return

        areas = await self._fetch_registry("/api/areas")
        devices = await self._fetch_registry("/api/devices")
        entities = await self._fetch_registry("/api/entities")

        if areas is not None:
            self._areas_cache = areas
            self._area_name_map = {
                area.get("name", "").lower(): area.get("area_id")
                for area in areas
                if area.get("name") and area.get("area_id")
            }

        if devices is not None:
            self._devices_cache = devices
            self._device_name_map = {
                (device.get("name_by_user") or device.get("name") or "").lower(): device.get("id") or device.get("device_id")
                for device in devices
                if device.get("name_by_user") or device.get("name")
            }

        if entities is not None:
            self._entity_registry_cache = entities

        if any(x is not None for x in [areas, devices, entities]):
            self._registry_cache_time = current_time
            logger.info(
                "✅ Loaded registry cache (areas=%s, devices=%s, entities=%s)",
                len(areas or []),
                len(devices or []),
                len(entities or [])
            )

    async def create_automation(self, automation: Dict[str, Any], apply: bool = False) -> Dict[str, Any]:
        """
        Create a Home Assistant automation.

        Args:
            automation: Automation config dict
            apply: If False, return preview only
        """
        if not self.is_enabled():
            return {"success": False, "error": "Home Assistant nicht konfiguriert"}

        normalized, errors = self._normalize_automation_payload(automation)
        validation = await self._validate_actions(normalized.get("action", []))
        errors.extend(validation)

        if errors:
            return {
                "success": False,
                "preview": True,
                "valid": False,
                "errors": errors,
                "automation": normalized
            }

        if not apply:
            return {
                "success": True,
                "preview": True,
                "valid": True,
                "automation": normalized
            }

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}/api/config/automation/config"

        async with session.post(url, json=normalized, headers=headers) as resp:
            if resp.status == 401 or resp.status == 403:
                return {
                    "success": False,
                    "error": "Token hat keine Berechtigung fuer Automationen (Admin-Rechte erforderlich)."
                }
            if resp.status == 404:
                logger.warning("REST Config API 404 - trying WebSocket fallback for automation")
                ws_result = await self._ws_request({
                    "type": "config/automation/create",
                    "automation": normalized
                })
                if ws_result.get("success"):
                    created = ws_result.get("result", {})
                    reload_result = await self._ws_request({"type": "config/automation/reload"})
                    if not reload_result.get("success"):
                        return {
                            "success": False,
                            "error": "Automation gespeichert, aber Reload fehlgeschlagen (WebSocket)."
                        }
                    return {"success": True, "preview": False, "automation": created}
                return {
                    "success": False,
                    "error": (
                        "Home Assistant REST Config API fuer Automationen ist nicht verfuegbar (404) "
                        "und WebSocket-Fallback ist fehlgeschlagen."
                    )
                }
            if resp.status not in [200, 201]:
                error_text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}"}

            created = await resp.json()

        reload_result = await self._call_service("automation", "reload")
        if not reload_result.get("success"):
            await self._rollback_automation(created)
            return {
                "success": False,
                "error": f"Automation gespeichert, aber Reload fehlgeschlagen: {reload_result.get('error')}"
            }

        return {"success": True, "preview": False, "automation": created}

    async def create_script(self, script: Dict[str, Any], apply: bool = False) -> Dict[str, Any]:
        """
        Create a Home Assistant script.

        Args:
            script: Script config dict
            apply: If False, return preview only
        """
        if not self.is_enabled():
            return {"success": False, "error": "Home Assistant nicht konfiguriert"}

        normalized, errors = self._normalize_script_payload(script)
        validation = await self._validate_actions(normalized.get("sequence", []))
        errors.extend(validation)

        if errors:
            return {
                "success": False,
                "preview": True,
                "valid": False,
                "errors": errors,
                "script": normalized
            }

        if not apply:
            return {
                "success": True,
                "preview": True,
                "valid": True,
                "script": normalized
            }

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}/api/config/script/config"

        async with session.post(url, json=normalized, headers=headers) as resp:
            if resp.status == 401 or resp.status == 403:
                return {
                    "success": False,
                    "error": "Token hat keine Berechtigung fuer Scripts (Admin-Rechte erforderlich)."
                }
            if resp.status == 404:
                logger.warning("REST Config API 404 - trying WebSocket fallback for script")
                ws_result = await self._ws_request({
                    "type": "config/script/create",
                    "script": normalized
                })
                if ws_result.get("success"):
                    created = ws_result.get("result", {})
                    reload_result = await self._ws_request({"type": "config/script/reload"})
                    if not reload_result.get("success"):
                        return {
                            "success": False,
                            "error": "Script gespeichert, aber Reload fehlgeschlagen (WebSocket)."
                        }
                    return {"success": True, "preview": False, "script": created}
                return {
                    "success": False,
                    "error": (
                        "Home Assistant REST Config API fuer Scripts ist nicht verfuegbar (404) "
                        "und WebSocket-Fallback ist fehlgeschlagen."
                    )
                }
            if resp.status not in [200, 201]:
                error_text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}"}

            created = await resp.json()

        reload_result = await self._call_service("script", "reload")
        if not reload_result.get("success"):
            await self._rollback_script(created)
            return {
                "success": False,
                "error": f"Script gespeichert, aber Reload fehlgeschlagen: {reload_result.get('error')}"
            }

        return {"success": True, "preview": False, "script": created}

    async def _load_entities_cache(self) -> bool:
        """
        Load and cache all entities with friendly names for resolution.

        Returns:
            True if successful, False otherwise
        """
        # Check if cache is still valid
        current_time = asyncio.get_event_loop().time()
        if self._entities_cache and (current_time - self._entities_cache_time) < self._entities_cache_ttl:
            return True  # Cache still valid

        logger.info("🔄 Loading entities for name resolution...")
        result = await self.list_entities()

        if not result.get("success"):
            logger.warning("❌ Failed to load entities for caching")
            return False

        entities = result.get("entities", [])
        self._entities_cache = entities
        self._entities_cache_time = current_time

        # Build friendly name mapping
        self._friendly_name_map = {}
        for entity in entities:
            entity_id = entity.get("entity_id")
            friendly_name = entity.get("friendly_name")

            if friendly_name and entity_id:
                # Store both exact match and lowercase
                for key in {friendly_name, friendly_name.lower()}:
                    if key not in self._friendly_name_map:
                        self._friendly_name_map[key] = []
                    if entity_id not in self._friendly_name_map[key]:
                        self._friendly_name_map[key].append(entity_id)

        logger.info(f"✅ Loaded {len(entities)} entities ({len(self._friendly_name_map)} name mappings)")
        return True

    async def resolve_entity(
        self,
        natural_query: str,
        domain: Optional[str] = None,
        preferred_domains: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Resolve natural language query to entity ID.

        Examples:
            "Wohnzimmer" -> "light.wohnzimmer"
            "Licht im Wohnzimmer" -> "light.wohnzimmer"
            "Kaffeemaschine" -> "switch.kaffeemaschine"

        Args:
            natural_query: Natural language description
            domain: Optional domain hint ('light', 'switch', etc.)

        Returns:
            Entity ID if found, None otherwise
        """
        if not await self._load_entities_cache():
            return None

        query_lower = natural_query.lower()

        # Remove common prefixes
        prefixes_to_remove = [
            "licht im ", "licht in der ", "licht in ", "das licht im ",
            "schalte ", "schalte das ", "schalte den ", "schalte die ",
            "light in ", "the light in ", "turn on ", "turn off "
        ]

        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
                break

        # Remove common suffixes
        suffixes_to_remove = [" an", " aus", " on", " off", " ein", " aus"]
        for suffix in suffixes_to_remove:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)]
                break

        query_lower = query_lower.strip()

        # 1. Exact friendly name match
        if query_lower in self._friendly_name_map:
            entity_id = self._select_entity(self._friendly_name_map[query_lower], domain, preferred_domains)
            if entity_id:
                logger.info(f"✓ Exact match: '{natural_query}' -> {entity_id}")
                return entity_id

        # 2. Fuzzy matching on friendly names
        all_names = list(self._friendly_name_map.keys())
        matches = get_close_matches(query_lower, all_names, n=3, cutoff=0.6)

        if matches:
            best_match = matches[0]
            entity_id = self._select_entity(self._friendly_name_map[best_match], domain, preferred_domains)
            if entity_id:
                logger.info(f"✓ Fuzzy match: '{natural_query}' -> {entity_id} (via '{best_match}')")
                return entity_id

        # 3. Partial match in entity_id itself
        query_clean = re.sub(r'[^a-z0-9_]', '', query_lower)

        candidates = []
        for entity in self._entities_cache:
            entity_id = entity.get("entity_id", "")

            # Check domain filter
            if domain and not entity_id.startswith(f"{domain}."):
                continue

            # Check if query appears in entity_id
            entity_id_clean = entity_id.split(".")[-1]  # Get part after domain
            if query_clean in entity_id_clean or entity_id_clean in query_clean:
                candidates.append(entity_id)

        entity_id = self._select_entity(candidates, domain, preferred_domains)
        if entity_id:
            logger.info(f"✓ Partial match: '{natural_query}' -> {entity_id}")
            return entity_id

        # 4. Area/Device-based match (if registry available)
        await self._load_registry_cache()
        area_match = self._match_area_or_device(query_lower, is_area=True)
        if area_match:
            entity_id = self._select_entity_from_registry(area_match, is_area=True, domain=domain, preferred_domains=preferred_domains)
            if entity_id:
                logger.info(f"✓ Area match: '{natural_query}' -> {entity_id}")
                return entity_id

        device_match = self._match_area_or_device(query_lower, is_area=False)
        if device_match:
            entity_id = self._select_entity_from_registry(device_match, is_area=False, domain=domain, preferred_domains=preferred_domains)
            if entity_id:
                logger.info(f"✓ Device match: '{natural_query}' -> {entity_id}")
                return entity_id

        # 5. No match found
        logger.warning(f"❌ No entity found for: '{natural_query}'" + (f" (domain={domain})" if domain else ""))
        return None

    def _select_entity(
        self,
        candidates: List[str],
        domain: Optional[str],
        preferred_domains: Optional[List[str]]
    ) -> Optional[str]:
        """Pick best entity based on domain filters and preferences."""
        if not candidates:
            return None

        if domain:
            for candidate in candidates:
                if candidate.startswith(f"{domain}."):
                    return candidate
            return None

        if preferred_domains:
            for pref_domain in preferred_domains:
                for candidate in candidates:
                    if candidate.startswith(f"{pref_domain}."):
                        return candidate

        return candidates[0]

    def _infer_preferred_domains(self, text: str, for_query: bool) -> Optional[List[str]]:
        """Infer preferred domains based on keywords."""
        text_lower = text.lower()

        if any(token in text_lower for token in ["temperatur", "heizung", "thermostat", "klima", "warm", "kühl", "kalt"]):
            return ["climate", "sensor"]
        if any(token in text_lower for token in ["feuchtigkeit", "luftfeuchtigkeit"]):
            return ["sensor", "climate"]
        if any(token in text_lower for token in ["helligkeit", "licht", "lampe", "beleuchtung"]):
            return ["light"]
        if any(token in text_lower for token in ["steckdose", "schalter", "kaffeemaschine", "switch"]):
            return ["switch"]
        if any(token in text_lower for token in ["rollladen", "jalousie", "markise", "cover"]):
            return ["cover"]
        if any(token in text_lower for token in ["tv", "fernseher", "musik", "radio", "media"]):
            return ["media_player"]
        if any(token in text_lower for token in ["lüfter", "ventilator", "fan"]):
            return ["fan"]
        if any(token in text_lower for token in ["schloss", "tür", "tueren", "lock"]):
            return ["lock"]

        if for_query:
            return ["light", "switch", "sensor", "climate", "binary_sensor", "cover", "media_player"]

        return None

    def _normalize_automation_payload(self, automation: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Normalize automation payload to list-based triggers/conditions/actions."""
        errors: List[str] = []
        normalized = dict(automation or {})

        alias = normalized.get("alias")
        if not alias:
            errors.append("Automation: 'alias' fehlt.")

        trigger = normalized.get("trigger")
        if trigger is None:
            errors.append("Automation: 'trigger' fehlt.")
            trigger_list = []
        else:
            trigger_list = trigger if isinstance(trigger, list) else [trigger]

        condition = normalized.get("condition", [])
        condition_list = condition if isinstance(condition, list) else [condition] if condition else []

        action = normalized.get("action")
        if action is None:
            errors.append("Automation: 'action' fehlt.")
            action_list = []
        else:
            action_list = action if isinstance(action, list) else [action]

        normalized["trigger"] = trigger_list
        normalized["condition"] = condition_list
        normalized["action"] = action_list
        if "id" not in normalized:
            normalized["id"] = str(uuid4())

        return normalized, errors

    def _normalize_script_payload(self, script: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Normalize script payload to list-based sequence."""
        errors: List[str] = []
        normalized = dict(script or {})

        alias = normalized.get("alias")
        if not alias:
            errors.append("Script: 'alias' fehlt.")

        sequence = normalized.get("sequence")
        if sequence is None:
            errors.append("Script: 'sequence' fehlt.")
            sequence_list = []
        else:
            sequence_list = sequence if isinstance(sequence, list) else [sequence]

        normalized["sequence"] = sequence_list
        if "id" not in normalized:
            normalized["id"] = str(uuid4())

        return normalized, errors

    async def _validate_actions(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Validate actions for allowed services and existing entities."""
        errors: List[str] = []
        if not actions:
            return errors

        entities_result = await self.list_entities()
        entity_ids = set()
        if entities_result.get("success"):
            entity_ids = {e.get("entity_id") for e in entities_result.get("entities", []) if e.get("entity_id")}

        for action in actions:
            if not isinstance(action, dict):
                errors.append("Aktion muss ein Objekt sein.")
                continue
            service = action.get("service")
            if service:
                domain = service.split(".")[0]
                if domain not in self._allowed_service_domains:
                    errors.append(f"Service-Domain nicht erlaubt: {domain}")

            for entity_id in self._extract_entity_ids(action):
                if isinstance(entity_id, str) and ("{{" in entity_id or "{%" in entity_id):
                    continue
                if entity_ids and entity_id not in entity_ids:
                    errors.append(f"Entity nicht gefunden: {entity_id}")

        return errors

    def _extract_entity_ids(self, action: Dict[str, Any]) -> List[str]:
        """Extract entity_id references from an action."""
        entity_ids: List[str] = []

        if "entity_id" in action:
            value = action.get("entity_id")
            if isinstance(value, list):
                entity_ids.extend(value)
            elif isinstance(value, str):
                entity_ids.append(value)

        target = action.get("target", {})
        target_entity = target.get("entity_id") if isinstance(target, dict) else None
        if target_entity:
            if isinstance(target_entity, list):
                entity_ids.extend(target_entity)
            elif isinstance(target_entity, str):
                entity_ids.append(target_entity)

        data = action.get("data", {})
        data_entity = data.get("entity_id") if isinstance(data, dict) else None
        if data_entity:
            if isinstance(data_entity, list):
                entity_ids.extend(data_entity)
            elif isinstance(data_entity, str):
                entity_ids.append(data_entity)

        return entity_ids

    async def _rollback_automation(self, created: Dict[str, Any]) -> None:
        """Best-effort rollback for automation creation."""
        automation_id = created.get("id") or created.get("automation_id")
        if not automation_id:
            return
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}/api/config/automation/config/{automation_id}"
        try:
            async with session.delete(url, headers=headers) as resp:
                if resp.status in [200, 204]:
                    logger.warning(f"↩️ Rolled back automation {automation_id}")
        except Exception:
            logger.warning("↩️ Rollback automation failed", exc_info=True)

    async def _rollback_script(self, created: Dict[str, Any]) -> None:
        """Best-effort rollback for script creation."""
        script_id = created.get("id") or created.get("script_id")
        if not script_id:
            return
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"{self.url}/api/config/script/config/{script_id}"
        try:
            async with session.delete(url, headers=headers) as resp:
                if resp.status in [200, 204]:
                    logger.warning(f"↩️ Rolled back script {script_id}")
        except Exception:
            logger.warning("↩️ Rollback script failed", exc_info=True)

    def _match_area_or_device(self, query_lower: str, is_area: bool) -> Optional[str]:
        """Try to match query against area or device names."""
        name_map = self._area_name_map if is_area else self._device_name_map
        if not name_map:
            return None

        if query_lower in name_map:
            return name_map[query_lower]

        matches = get_close_matches(query_lower, list(name_map.keys()), n=1, cutoff=0.7)
        if matches:
            return name_map[matches[0]]

        return None

    def _select_entity_from_registry(
        self,
        registry_id: str,
        is_area: bool,
        domain: Optional[str],
        preferred_domains: Optional[List[str]]
    ) -> Optional[str]:
        """Pick an entity from registry cache by area/device id."""
        if not self._entity_registry_cache:
            return None

        key = "area_id" if is_area else "device_id"
        candidates = [
            entry.get("entity_id")
            for entry in self._entity_registry_cache
            if entry.get(key) == registry_id and entry.get("entity_id")
        ]

        return self._select_entity(candidates, domain, preferred_domains)


# Singleton instance
_instance: Optional[HomeAssistantService] = None


def get_ha_service(url: Optional[str] = None, token: Optional[str] = None) -> HomeAssistantService:
    """
    Get or create Home Assistant service instance (singleton pattern).

    Args:
        url: Optional Home Assistant URL (uses env var if not provided)
        token: Optional access token (uses env var if not provided)

    Returns:
        HomeAssistantService instance
    """
    global _instance
    if _instance is None:
        _instance = HomeAssistantService(url, token)
    return _instance
