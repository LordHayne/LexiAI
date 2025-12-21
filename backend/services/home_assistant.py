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

        # Entity resolution cache
        self._entities_cache: Optional[List[Dict[str, Any]]] = None
        self._entities_cache_time: float = 0
        self._entities_cache_ttl = 300.0  # 5 minutes cache for entities
        self._friendly_name_map: Dict[str, str] = {}  # friendly_name -> entity_id

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

                        return {
                            "success": True,
                            "entity_id": entity_id,
                            "action": action,
                            "service": f"{domain}.{service}",
                            "result": result
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

        # Resolve entity if it's not a full entity_id
        resolved_entity = entity_id
        if '.' not in entity_id:
            resolved_entity = await self.resolve_entity(entity_id)
            if not resolved_entity:
                return {
                    "success": False,
                    "error": f"Konnte Entity '{entity_id}' nicht finden"
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
                                "domain": e.get("entity_id", "").split(".")[0]
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
                self._friendly_name_map[friendly_name.lower()] = entity_id
                self._friendly_name_map[friendly_name] = entity_id

        logger.info(f"✅ Loaded {len(entities)} entities ({len(self._friendly_name_map)} name mappings)")
        return True

    async def resolve_entity(self, natural_query: str, domain: Optional[str] = None) -> Optional[str]:
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
            entity_id = self._friendly_name_map[query_lower]
            if not domain or entity_id.startswith(f"{domain}."):
                logger.info(f"✓ Exact match: '{natural_query}' -> {entity_id}")
                return entity_id

        # 2. Fuzzy matching on friendly names
        all_names = list(self._friendly_name_map.keys())
        matches = get_close_matches(query_lower, all_names, n=3, cutoff=0.6)

        if matches:
            best_match = matches[0]
            entity_id = self._friendly_name_map[best_match]

            if not domain or entity_id.startswith(f"{domain}."):
                logger.info(f"✓ Fuzzy match: '{natural_query}' -> {entity_id} (via '{best_match}')")
                return entity_id

        # 3. Partial match in entity_id itself
        query_clean = re.sub(r'[^a-z0-9_]', '', query_lower)

        for entity in self._entities_cache:
            entity_id = entity.get("entity_id", "")

            # Check domain filter
            if domain and not entity_id.startswith(f"{domain}."):
                continue

            # Check if query appears in entity_id
            entity_id_clean = entity_id.split(".")[-1]  # Get part after domain
            if query_clean in entity_id_clean or entity_id_clean in query_clean:
                logger.info(f"✓ Partial match: '{natural_query}' -> {entity_id}")
                return entity_id

        # 4. No match found
        logger.warning(f"❌ No entity found for: '{natural_query}'" + (f" (domain={domain})" if domain else ""))
        return None


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
