"""
Smart Home Handler Module for LexiAI Chat Processing

This module handles Smart Home (Home Assistant) specific logic including:
- Intelligent storage decisions (pattern tracking vs full-text storage)
- Entity ID extraction from tool results
- Pattern classification for memory management

Extracted from chat_processing_with_tools.py (Lines 35-120)
"""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def classify_smart_home_storage_strategy(
    message: str,
    tool_results: List
) -> Tuple[bool, str, Optional[str]]:
    """
    Entscheidet ob Smart Home Interaktion in Qdrant gespeichert werden soll.

    Diese Funktion implementiert eine intelligente Speicherstrategie:
    - Simple Toggles (z.B. "Licht ein") → NUR Pattern-Tracking (kein Volltext)
    - Komplexe Anfragen → Volltext-Speicherung in Qdrant
    - Sensor-Abfragen → Volltext-Speicherung
    - Präferenz-Angaben → Volltext-Speicherung

    Args:
        message: User-Nachricht
        tool_results: Liste von ToolResult-Objekten

    Returns:
        Tuple von (should_store, reason, entity_id):
        - should_store: True = Volltext in Qdrant, False = nur Pattern-Tracking
        - reason: Grund für Entscheidung (für Logging/Debugging)
        - entity_id: Entity ID falls bekannt (für Pattern-Tracking)

    Original code: chat_processing_with_tools.py lines 35-120
    """
    message_lower = message.lower()

    # Prüfe ob Home Assistant Tools verwendet wurden
    ha_tools = [r for r in tool_results if r.tool_name in ["home_assistant_control", "home_assistant_query"]]
    if not ha_tools:
        # Kein Smart Home Tool → normale Speicherung
        return True, "no_ha_tool", None

    # Extrahiere entity_id aus Tool-Result
    entity_id = None
    for result in ha_tools:
        if result.success and result.data:
            entity_id = result.data.get("entity_id")
            if not entity_id and "entity_id" in result.data.get("params", {}):
                entity_id = result.data["params"]["entity_id"]
            if entity_id:
                break

    # IMMER speichern: Explizite Präferenz-Werte
    preference_patterns = [
        r'\d+\s*grad',      # "22 grad"
        r'\d+\s*°',         # "22°"
        r'\d+\s*prozent',   # "60 prozent"
        r'\d+\s*%',         # "60%"
    ]

    for pattern in preference_patterns:
        if re.search(pattern, message_lower):
            return True, "explicit_preference", entity_id

    # IMMER speichern: Preference-Keywords
    if any(word in message_lower for word in ["hell", "dunkel", "warm", "kalt", "kühl", "heiß"]):
        return True, "preference_keyword", entity_id

    # IMMER speichern: Scene/Stimmungs-Keywords
    if any(word in message_lower for word in [
        "szene", "scene", "stimmung", "atmosphäre", "romantisch", "gemütlich", "cozy"
    ]):
        return True, "scene_preference", entity_id

    # IMMER speichern: Sensor-Abfragen (Query-Tool)
    if any(r.tool_name == "home_assistant_query" for r in ha_tools):
        return True, "sensor_query", entity_id

    # IMMER speichern: Komplexe Anfragen (>7 Wörter oder Kontext-Anfragen)
    if len(message.split()) > 7:
        return True, "complex_interaction", entity_id

    # IMMER speichern: Fragen
    if any(word in message_lower for word in ["wie", "was", "wann", "wo", "warum", "welche", "?", "zeig"]):
        return True, "query_interaction", entity_id

    # NIE speichern: Simple Toggles (nur Pattern-Tracking)
    # Patterns für "schalte licht ein/aus", "mach heizung an/ab"
    simple_toggle_patterns = [
        r'^(schalte|mach|stelle)\s+(das\s+)?(\w+\s*)?(licht|lampe|heizung|thermostat|schalter|switch)\s+(ein|aus|an|ab)$',
        r'^(licht|lampe|heizung|schalter)\s+(ein|aus|an|ab)$',
        r'^(ein|aus|an|ab)schalten\s+(\w+)$'
    ]

    for pattern in simple_toggle_patterns:
        if re.match(pattern, message_lower):
            logger.info(f"📊 Simple Toggle erkannt: '{message}' → Pattern-Tracking statt Volltext")
            return False, "simple_toggle", entity_id

    # Default: Bei Unsicherheit speichern (safe choice)
    return True, "default_store", entity_id
