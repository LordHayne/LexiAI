"""
Home Assistant LLM Tools

Provides LangChain tools for Home Assistant device control via LLM.
"""
import logging
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from backend.services.home_assistant import get_ha_service
from backend.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


# Tool Input Schemas
class ControlDeviceInput(BaseModel):
    """Input schema for controlling Home Assistant devices."""
    entity_id: str = Field(
        description="Home Assistant entity ID (e.g., 'light.wohnzimmer', 'switch.kaffeemaschine')"
    )
    action: str = Field(
        description="Action to perform: turn_on, turn_off, toggle, set_brightness, set_temperature"
    )
    value: Optional[float] = Field(
        default=None,
        description="Optional value for brightness (0-255) or temperature (in Celsius)"
    )


class GetStateInput(BaseModel):
    """Input schema for getting device state."""
    entity_id: str = Field(
        description="Home Assistant entity ID to query (e.g., 'light.wohnzimmer')"
    )


class ListEntitiesInput(BaseModel):
    """Input schema for listing entities."""
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain filter (e.g., 'light', 'switch', 'climate', 'cover')"
    )


class CreateAutomationInput(BaseModel):
    """Input schema for creating Home Assistant automations."""
    automation: Dict[str, Any] = Field(
        description="Automation config dict (alias, trigger, condition, action)"
    )
    apply: bool = Field(
        default=False,
        description="Wenn true, wird die Automation in Home Assistant gespeichert (sonst nur Preview)."
    )


class CreateScriptInput(BaseModel):
    """Input schema for creating Home Assistant scripts."""
    script: Dict[str, Any] = Field(
        description="Script config dict (alias, sequence)"
    )


def _describe_trigger(trigger: Dict[str, Any]) -> str:
    platform = trigger.get("platform")
    if platform == "time":
        at = trigger.get("at")
        return f"um {at}" if at else "zeitbasiert"
    if platform == "state":
        entity_id = trigger.get("entity_id", "unbekannt")
        to_state = trigger.get("to")
        from_state = trigger.get("from")
        parts = [f"{entity_id}"]
        if from_state:
            parts.append(f"von {from_state}")
        if to_state:
            parts.append(f"nach {to_state}")
        return "Status " + " ".join(parts)
    if platform == "sun":
        event = trigger.get("event", "sun")
        offset = trigger.get("offset")
        return f"{event}" + (f" ({offset})" if offset else "")
    return f"Trigger ({platform})" if platform else "Trigger"


def _describe_action(action: Dict[str, Any]) -> str:
    service = action.get("service", "service")
    entity_id = action.get("entity_id")
    if not entity_id:
        target = action.get("target", {})
        entity_id = target.get("entity_id") if isinstance(target, dict) else None
    if not entity_id:
        data = action.get("data", {})
        entity_id = data.get("entity_id") if isinstance(data, dict) else None
    if entity_id:
        if isinstance(entity_id, list):
            entity_str = ", ".join(entity_id[:3])
        else:
            entity_str = str(entity_id)
        return f"{service} ({entity_str})"
    return service


def _describe_automation(automation: Dict[str, Any]) -> str:
    triggers = automation.get("trigger", [])
    actions = automation.get("action", [])
    trigger_list = triggers if isinstance(triggers, list) else [triggers]
    action_list = actions if isinstance(actions, list) else [actions]

    trigger_desc = "; ".join(_describe_trigger(t) for t in trigger_list if isinstance(t, dict))
    action_desc = "; ".join(_describe_action(a) for a in action_list if isinstance(a, dict))

    parts = []
    if trigger_desc:
        parts.append(f"Ausloeser: {trigger_desc}")
    if action_desc:
        parts.append(f"Aktivitaeten: {action_desc}")

    return " | ".join(parts) if parts else "Automation vorbereitet"


def _describe_script(script: Dict[str, Any]) -> str:
    sequence = script.get("sequence", [])
    seq_list = sequence if isinstance(sequence, list) else [sequence]
    action_desc = "; ".join(_describe_action(a) for a in seq_list if isinstance(a, dict))
    return f"Aktivitaeten: {action_desc}" if action_desc else "Script vorbereitet"
    apply: bool = Field(
        default=False,
        description="Wenn true, wird das Script in Home Assistant gespeichert (sonst nur Preview)."
    )


# LangChain Tools
class HomeAssistantControlTool(BaseTool):
    """
    Tool for controlling Home Assistant devices.

    Examples:
    - "Mach das Wohnzimmerlicht an" → control_device("light.wohnzimmer", "turn_on")
    - "Schalte die Kaffeemaschine aus" → control_device("switch.kaffeemaschine", "turn_off")
    - "Stelle das Licht auf 50% Helligkeit" → control_device("light.wohnzimmer", "set_brightness", 128)
    - "Stelle die Heizung auf 22 Grad" → control_device("climate.heizung", "set_temperature", 22)
    """
    name: str = "home_assistant_control"
    description: str = """
    Steuere Home Assistant Smart Home Geräte (Lichter, Schalter, Heizung, etc.).

    Aktionen:
    - turn_on: Gerät einschalten
    - turn_off: Gerät ausschalten
    - toggle: Gerät umschalten (an/aus)
    - set_brightness: Helligkeit setzen (0-255)
    - set_temperature: Temperatur setzen (Grad Celsius)

    Beispiele:
    - Licht einschalten: entity_id="light.wohnzimmer", action="turn_on"
    - Helligkeit setzen: entity_id="light.wohnzimmer", action="set_brightness", value=128
    - Heizung einstellen: entity_id="climate.heizung", action="set_temperature", value=22
    """
    args_schema: Type[BaseModel] = ControlDeviceInput
    return_direct: bool = False  # LLM should interpret result

    async def _arun(self, entity_id: str, action: str, value: Optional[float] = None) -> str:
        """Async implementation."""
        try:
            if not FeatureFlags.is_enabled("home_assistant"):
                return "❌ Home Assistant ist deaktiviert (Feature-Flag)."

            ha_service = get_ha_service()

            if not ha_service.is_enabled():
                return "❌ Home Assistant ist nicht konfiguriert. Bitte setze LEXI_HA_URL und LEXI_HA_TOKEN."

            result = await ha_service.control_device(entity_id, action, value)

            if result.get("success"):
                # Format success message
                action_msg = {
                    "turn_on": "eingeschaltet",
                    "turn_off": "ausgeschaltet",
                    "toggle": "umgeschaltet",
                    "set_brightness": f"Helligkeit auf {value} gesetzt",
                    "set_temperature": f"Temperatur auf {value}°C eingestellt"
                }.get(action, f"Aktion '{action}' ausgeführt")

                return f"✅ {entity_id} wurde {action_msg}"
            else:
                error = result.get("error", "Unbekannter Fehler")
                return f"❌ Fehler bei {entity_id}: {error}"

        except Exception as e:
            logger.error(f"Error in home_assistant_control tool: {e}")
            return f"❌ Fehler bei der Gerätesteuerung: {str(e)}"

    def _run(self, entity_id: str, action: str, value: Optional[float] = None) -> str:
        """Sync implementation (not supported)."""
        return "❌ Home Assistant Tool requires async execution"


class HomeAssistantStateTool(BaseTool):
    """
    Tool for querying Home Assistant device states.

    Examples:
    - "Ist das Wohnzimmerlicht an?" → get_state("light.wohnzimmer")
    - "Wie hell ist das Licht?" → get_state("light.wohnzimmer")
    - "Welche Temperatur hat die Heizung?" → get_state("climate.heizung")
    """
    name: str = "home_assistant_get_state"
    description: str = """
    Frage den aktuellen Status von Home Assistant Geräten ab.

    Gibt zurück:
    - Aktueller Zustand (on/off/etc.)
    - Attribute (brightness, temperature, etc.)
    - Letzte Änderung

    Beispiele:
    - Licht-Status: entity_id="light.wohnzimmer"
    - Heizungs-Temperatur: entity_id="climate.heizung"
    - Schalter-Status: entity_id="switch.kaffeemaschine"
    """
    args_schema: Type[BaseModel] = GetStateInput
    return_direct: bool = False

    async def _arun(self, entity_id: str) -> str:
        """Async implementation."""
        try:
            if not FeatureFlags.is_enabled("home_assistant"):
                return "❌ Home Assistant ist deaktiviert (Feature-Flag)."

            ha_service = get_ha_service()

            if not ha_service.is_enabled():
                return "❌ Home Assistant ist nicht konfiguriert."

            result = await ha_service.get_state(entity_id)

            if result.get("success"):
                state = result.get("state")
                attributes = result.get("attributes", {})

                # Format response with relevant attributes
                response = f"📊 Status von {entity_id}: {state}"

                # Add brightness for lights
                if "brightness" in attributes:
                    brightness_pct = int(attributes["brightness"] / 255 * 100)
                    response += f"\n   Helligkeit: {brightness_pct}%"

                # Add temperature
                if "temperature" in attributes:
                    response += f"\n   Temperatur: {attributes['temperature']}°C"

                # Add current temperature for climate devices
                if "current_temperature" in attributes:
                    response += f"\n   Aktuelle Temperatur: {attributes['current_temperature']}°C"

                # Add friendly name
                if "friendly_name" in attributes:
                    response += f"\n   Name: {attributes['friendly_name']}"

                return response
            else:
                error = result.get("error", "Unbekannter Fehler")
                return f"❌ Fehler bei {entity_id}: {error}"

        except Exception as e:
            logger.error(f"Error in home_assistant_get_state tool: {e}")
            return f"❌ Fehler bei der Status-Abfrage: {str(e)}"

    def _run(self, entity_id: str) -> str:
        """Sync implementation (not supported)."""
        return "❌ Home Assistant Tool requires async execution"


class HomeAssistantListTool(BaseTool):
    """
    Tool for listing available Home Assistant entities.

    Examples:
    - "Welche Lichter gibt es?" → list_entities("light")
    - "Zeige alle Geräte" → list_entities()
    - "Welche Schalter sind verfügbar?" → list_entities("switch")
    """
    name: str = "home_assistant_list_entities"
    description: str = """
    Liste alle verfügbaren Home Assistant Geräte auf.

    Domains (Gerätetypen):
    - light: Lichter
    - switch: Schalter
    - climate: Heizung/Klimaanlage
    - cover: Rollladen/Jalousien
    - lock: Schlösser
    - media_player: Media Player
    - sensor: Sensoren
    - binary_sensor: Binäre Sensoren

    Beispiele:
    - Alle Lichter: domain="light"
    - Alle Schalter: domain="switch"
    - Alle Geräte: domain=None
    """
    args_schema: Type[BaseModel] = ListEntitiesInput
    return_direct: bool = False

    async def _arun(self, domain: Optional[str] = None) -> str:
        """Async implementation."""
        try:
            if not FeatureFlags.is_enabled("home_assistant"):
                return "❌ Home Assistant ist deaktiviert (Feature-Flag)."

            ha_service = get_ha_service()

            if not ha_service.is_enabled():
                return "❌ Home Assistant ist nicht konfiguriert."

            result = await ha_service.list_entities(domain)

            if result.get("success"):
                entities = result.get("entities", [])
                count = result.get("count", 0)

                if count == 0:
                    if domain:
                        return f"📋 Keine Geräte vom Typ '{domain}' gefunden."
                    else:
                        return "📋 Keine Geräte gefunden."

                # Format entity list
                filter_msg = f" (Typ: {domain})" if domain else ""
                response = f"📋 Gefundene Geräte{filter_msg} ({count} gesamt):\n\n"

                for entity in entities[:20]:  # Limit to 20 for readability
                    entity_id = entity.get("entity_id", "unknown")
                    state = entity.get("state", "unknown")
                    friendly_name = entity.get("friendly_name", entity_id)

                    response += f"  • {friendly_name} ({entity_id}): {state}\n"

                if count > 20:
                    response += f"\n... und {count - 20} weitere Geräte"

                return response
            else:
                error = result.get("error", "Unbekannter Fehler")
                return f"❌ Fehler beim Auflisten der Geräte: {error}"

        except Exception as e:
            logger.error(f"Error in home_assistant_list_entities tool: {e}")
            return f"❌ Fehler beim Auflisten: {str(e)}"

    def _run(self, domain: Optional[str] = None) -> str:
        """Sync implementation (not supported)."""
        return "❌ Home Assistant Tool requires async execution"


class HomeAssistantCreateAutomationTool(BaseTool):
    """
    Tool for creating Home Assistant automations (preview by default).
    """
    name: str = "home_assistant_create_automation"
    description: str = """
    Erstelle Home Assistant Automationen. Standard ist Preview (kein Speichern).
    Verwende apply=true nur nach ausdruecklicher Bestaetigung des Users.
    """
    args_schema: Type[BaseModel] = CreateAutomationInput
    return_direct: bool = False

    async def _arun(self, automation: Dict[str, Any], apply: bool = False) -> str:
        try:
            if not FeatureFlags.is_enabled("home_assistant"):
                return "❌ Home Assistant ist deaktiviert (Feature-Flag)."

            ha_service = get_ha_service()
            if not ha_service.is_enabled():
                return "❌ Home Assistant ist nicht konfiguriert."

            result = await ha_service.create_automation(automation, apply=apply)

            if not result.get("success"):
                errors = result.get("errors") or [result.get("error", "Unbekannter Fehler")]
                return "❌ Automation nicht erstellt:\n- " + "\n- ".join(errors)

            if result.get("preview"):
                automation_payload = result.get("automation", {})
                alias = automation_payload.get("alias", "Automation")
                description = _describe_automation(automation_payload)
                return (
                    f"📝 Ich habe eine Automation vorbereitet: {alias}\n"
                    f"{description}\n"
                    "Soll ich sie speichern?"
                )

            created = result.get("automation", {})
            alias = created.get("alias", "Automation")
            return f"✅ Automation gespeichert: {alias}"

        except Exception as e:
            logger.error(f"Error in home_assistant_create_automation tool: {e}")
            return f"❌ Fehler bei Automation: {str(e)}"

    def _run(self, automation: Dict[str, Any], apply: bool = False) -> str:
        return "❌ Home Assistant Tool requires async execution"


class HomeAssistantCreateScriptTool(BaseTool):
    """
    Tool for creating Home Assistant scripts (preview by default).
    """
    name: str = "home_assistant_create_script"
    description: str = """
    Erstelle Home Assistant Scripts. Standard ist Preview (kein Speichern).
    Verwende apply=true nur nach ausdruecklicher Bestaetigung des Users.
    """
    args_schema: Type[BaseModel] = CreateScriptInput
    return_direct: bool = False

    async def _arun(self, script: Dict[str, Any], apply: bool = False) -> str:
        try:
            if not FeatureFlags.is_enabled("home_assistant"):
                return "❌ Home Assistant ist deaktiviert (Feature-Flag)."

            ha_service = get_ha_service()
            if not ha_service.is_enabled():
                return "❌ Home Assistant ist nicht konfiguriert."

            result = await ha_service.create_script(script, apply=apply)

            if not result.get("success"):
                errors = result.get("errors") or [result.get("error", "Unbekannter Fehler")]
                return "❌ Script nicht erstellt:\n- " + "\n- ".join(errors)

            if result.get("preview"):
                script_payload = result.get("script", {})
                alias = script_payload.get("alias", "Script")
                description = _describe_script(script_payload)
                return (
                    f"📝 Ich habe ein Script vorbereitet: {alias}\n"
                    f"{description}\n"
                    "Soll ich es speichern?"
                )

            created = result.get("script", {})
            alias = created.get("alias", "Script")
            return f"✅ Script gespeichert: {alias}"

        except Exception as e:
            logger.error(f"Error in home_assistant_create_script tool: {e}")
            return f"❌ Fehler bei Script: {str(e)}"

    def _run(self, script: Dict[str, Any], apply: bool = False) -> str:
        return "❌ Home Assistant Tool requires async execution"


# Tool factory functions
def get_home_assistant_tools():
    """
    Get all Home Assistant LangChain tools.

    Returns:
        List of HomeAssistant tools for LLM
    """
    return [
        HomeAssistantControlTool(),
        HomeAssistantStateTool(),
        HomeAssistantListTool(),
        HomeAssistantCreateAutomationTool(),
        HomeAssistantCreateScriptTool()
    ]


def is_home_assistant_enabled() -> bool:
    """
    Check if Home Assistant integration is enabled.

    Returns:
        True if HA is configured and enabled
    """
    try:
        ha_service = get_ha_service()
        return ha_service.is_enabled()
    except Exception:
        return False
