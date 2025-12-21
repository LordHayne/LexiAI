"""
User Profile Models für Smart Home Pattern Learning

Definiert Datenstrukturen für gelernte Präferenzen, Routinen und Automationsvorschläge.

Autor: LexiAI Development Team
Version: 1.0
Datum: 2025-01-23
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class DayOfWeek(str, Enum):
    """Wochentage für Routine-Planung."""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class TemperaturePreference(BaseModel):
    """Temperatur-Präferenzen für verschiedene Tageszeiten."""
    day: float = Field(21.0, ge=16, le=28, description="Tagestemperatur (6-18 Uhr)")
    evening: float = Field(22.0, ge=16, le=28, description="Abendtemperatur (18-22 Uhr)")
    night: float = Field(18.0, ge=16, le=28, description="Nachttemperatur (22-6 Uhr)")

    def get_preferred_temp(self, hour: int) -> float:
        """
        Gibt bevorzugte Temperatur für eine Stunde zurück.

        Args:
            hour: Stunde (0-23)

        Returns:
            Bevorzugte Temperatur in Celsius
        """
        if 6 <= hour < 18:
            return self.day
        elif 18 <= hour < 22:
            return self.evening
        else:
            return self.night


class BrightnessPreference(BaseModel):
    """Helligkeits-Präferenzen für verschiedene Tageszeiten."""
    morning: int = Field(80, ge=0, le=100, description="Morgen-Helligkeit (6-12 Uhr)")
    afternoon: int = Field(100, ge=0, le=100, description="Nachmittag-Helligkeit (12-18 Uhr)")
    evening: int = Field(60, ge=0, le=100, description="Abend-Helligkeit (18-22 Uhr)")
    night: int = Field(20, ge=0, le=100, description="Nacht-Helligkeit (22-6 Uhr)")

    def get_preferred_brightness(self, hour: int) -> int:
        """
        Gibt bevorzugte Helligkeit für eine Stunde zurück.

        Args:
            hour: Stunde (0-23)

        Returns:
            Bevorzugte Helligkeit (0-100%)
        """
        if 6 <= hour < 12:
            return self.morning
        elif 12 <= hour < 18:
            return self.afternoon
        elif 18 <= hour < 22:
            return self.evening
        else:
            return self.night


class RoomPreferences(BaseModel):
    """Raum-spezifische Präferenzen."""
    temperature: Optional[TemperaturePreference] = None
    brightness: Optional[BrightnessPreference] = None
    preferred_scenes: List[str] = Field(default_factory=list, description="Bevorzugte Szenen für diesen Raum")
    favorite_color: Optional[str] = Field(None, description="Bevorzugte Lichtfarbe (z.B. 'warm_white', 'cool_white')")


class RoutineAction(BaseModel):
    """Einzelne Aktion in einer Routine."""
    entity_id: str = Field(..., description="Home Assistant Entity ID")
    action: str = Field(..., description="Aktion (turn_on, turn_off, set_temperature, etc.)")
    frequency: float = Field(..., ge=0, le=1, description="Häufigkeit dieser Aktion (0-1)")
    typical_value: Optional[float] = Field(None, description="Typischer Wert (z.B. Helligkeit, Temperatur)")
    time_offset_minutes: int = Field(0, description="Zeit-Offset in Minuten relativ zu Routine-Start")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_id": "light.wohnzimmer",
                "action": "turn_on",
                "frequency": 0.9,
                "typical_value": 80.0,
                "time_offset_minutes": 0
            }
        }


class Routine(BaseModel):
    """Gelernte Routine (z.B. Morgen-Routine, Abend-Routine)."""
    name: str = Field(..., description="Name der Routine (z.B. 'Morgen-Routine')")
    time_range: str = Field(..., description="Zeitbereich (z.B. '06:30-07:00')")
    typical_start_hour: int = Field(..., ge=0, le=23, description="Typische Start-Stunde")
    typical_start_minute: int = Field(..., ge=0, le=59, description="Typische Start-Minute")
    actions: List[RoutineAction] = Field(default_factory=list, description="Aktionen in dieser Routine")
    confidence: float = Field(..., ge=0, le=1, description="Konfidenz dieser Routine (0-1)")
    days_of_week: List[DayOfWeek] = Field(
        default_factory=lambda: list(DayOfWeek),
        description="Tage an denen diese Routine aktiv ist"
    )
    enabled: bool = Field(True, description="Ist diese Routine aktiviert?")
    last_executed: Optional[datetime] = Field(None, description="Letzte Ausführung")
    execution_count: int = Field(0, description="Wie oft wurde diese Routine ausgeführt")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Morgen-Routine",
                "time_range": "06:30-07:00",
                "typical_start_hour": 6,
                "typical_start_minute": 30,
                "actions": [
                    {
                        "entity_id": "light.kitchen",
                        "action": "turn_on",
                        "frequency": 0.95,
                        "typical_value": 80.0,
                        "time_offset_minutes": 0
                    }
                ],
                "confidence": 0.85,
                "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "enabled": True
            }
        }


class AutomationSuggestion(BaseModel):
    """Vorschlag für eine Automation basierend auf gelernten Patterns."""
    id: str = Field(..., description="Eindeutige ID für diesen Vorschlag")
    title: str = Field(..., description="Titel des Vorschlags")
    description: str = Field(..., description="Beschreibung was automatisiert werden würde")
    routine: Routine = Field(..., description="Die vorgeschlagene Routine")
    confidence: float = Field(..., ge=0, le=1, description="Konfidenz dieses Vorschlags (0-1)")
    based_on_patterns: int = Field(..., description="Anzahl der Patterns auf denen dieser Vorschlag basiert")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    dismissed: bool = Field(False, description="Wurde dieser Vorschlag abgelehnt?")
    accepted: bool = Field(False, description="Wurde dieser Vorschlag akzeptiert?")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "suggestion_morning_lights",
                "title": "Automatisches Küchenlicht morgens",
                "description": "Schalte das Küchenlicht jeden Morgen um 6:30 Uhr automatisch ein (basierend auf 14 Tagen konstanter Nutzung)",
                "confidence": 0.9,
                "based_on_patterns": 14
            }
        }


class UsageStatistics(BaseModel):
    """Nutzungs-Statistiken für ein Entity."""
    entity_id: str
    total_activations: int = Field(0, description="Gesamt-Anzahl der Aktivierungen")
    total_deactivations: int = Field(0, description="Gesamt-Anzahl der Deaktivierungen")
    peak_hours: Dict[int, int] = Field(default_factory=dict, description="Aktivierungen pro Stunde (hour → count)")
    average_duration_minutes: Optional[float] = Field(None, description="Durchschnittliche Nutzungsdauer in Minuten")
    last_used: Optional[datetime] = Field(None, description="Letzte Nutzung")


class UserProfile(BaseModel):
    """
    Haupt-User-Profile mit allen gelernten Präferenzen und Routinen.

    Wird als JSON gespeichert unter: profiles/{user_id}.json
    """
    user_id: str = Field(..., description="Eindeutige User ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Raum-spezifische Präferenzen
    room_preferences: Dict[str, RoomPreferences] = Field(
        default_factory=dict,
        description="Präferenzen pro Raum (entity_name → RoomPreferences)"
    )

    # Gelernte Routinen
    morning_routine: Optional[Routine] = Field(None, description="Morgen-Routine")
    evening_routine: Optional[Routine] = Field(None, description="Abend-Routine")
    night_routine: Optional[Routine] = Field(None, description="Nacht-Routine")
    custom_routines: List[Routine] = Field(default_factory=list, description="Weitere benutzerdefinierte Routinen")

    # Nutzungsmuster
    usage_statistics: Dict[str, UsageStatistics] = Field(
        default_factory=dict,
        description="Nutzungs-Statistiken pro Entity"
    )

    # Automationsvorschläge
    suggested_automations: List[AutomationSuggestion] = Field(
        default_factory=list,
        description="Aktuelle Automationsvorschläge"
    )

    # Einstellungen
    automation_enabled: bool = Field(True, description="Sind Automationen aktiviert?")
    learning_enabled: bool = Field(True, description="Ist Pattern-Learning aktiviert?")
    suggestion_threshold: float = Field(0.7, ge=0, le=1, description="Minimum Confidence für Vorschläge")

    def get_room_preference(self, room_name: str) -> Optional[RoomPreferences]:
        """
        Hole Raum-Präferenzen für einen Raum.

        Args:
            room_name: Name des Raums (z.B. "wohnzimmer")

        Returns:
            RoomPreferences oder None
        """
        return self.room_preferences.get(room_name)

    def add_usage_event(self, entity_id: str, action: str, timestamp: datetime):
        """
        Füge ein Nutzungs-Event hinzu.

        Args:
            entity_id: Entity ID
            action: Aktion (turn_on, turn_off)
            timestamp: Zeitpunkt
        """
        if entity_id not in self.usage_statistics:
            self.usage_statistics[entity_id] = UsageStatistics(entity_id=entity_id)

        stats = self.usage_statistics[entity_id]

        if action == "turn_on":
            stats.total_activations += 1
        elif action == "turn_off":
            stats.total_deactivations += 1

        hour = timestamp.hour
        if hour not in stats.peak_hours:
            stats.peak_hours[hour] = 0
        stats.peak_hours[hour] += 1

        stats.last_used = timestamp
        self.last_updated = datetime.utcnow()

    def get_active_suggestions(self) -> List[AutomationSuggestion]:
        """
        Gibt alle aktiven (nicht dismissed/accepted) Vorschläge zurück.

        Returns:
            Liste von AutomationSuggestion
        """
        return [
            s for s in self.suggested_automations
            if not s.dismissed and not s.accepted and s.confidence >= self.suggestion_threshold
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "thomas",
                "room_preferences": {
                    "wohnzimmer": {
                        "temperature": {"day": 21, "evening": 22, "night": 18},
                        "brightness": {"morning": 80, "afternoon": 100, "evening": 60, "night": 20}
                    }
                },
                "automation_enabled": True,
                "learning_enabled": True
            }
        }
