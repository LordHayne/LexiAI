"""
User Profile Manager Service

Verwaltet User Profiles und extrahiert Präferenzen aus Smart Home Patterns.

Autor: LexiAI Development Team
Version: 1.0
Datum: 2025-01-23
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from backend.models.user_profile import (
    UserProfile, RoomPreferences, Routine, RoutineAction,
    TemperaturePreference, BrightnessPreference,
    AutomationSuggestion, DayOfWeek
)

logger = logging.getLogger(__name__)


class UserProfileManager:
    """
    Manager für User Profiles mit JSON-Persistenz.

    Speichert Profile unter: profiles/{user_id}.json
    """

    def __init__(self, profiles_dir: str = "profiles"):
        """
        Initialisiert den Profile Manager.

        Args:
            profiles_dir: Verzeichnis für Profile-Speicherung
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ UserProfileManager initialized (dir: {self.profiles_dir})")

    def _get_profile_path(self, user_id: str) -> Path:
        """Gibt Pfad zur Profile-Datei zurück."""
        return self.profiles_dir / f"{user_id}.json"

    def load_profile(self, user_id: str) -> UserProfile:
        """
        Lädt User Profile oder erstellt neues.

        Args:
            user_id: User ID

        Returns:
            UserProfile
        """
        profile_path = self._get_profile_path(user_id)

        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    profile = UserProfile(**data)
                    logger.info(f"📂 Profile loaded for {user_id}")
                    return profile
            except Exception as e:
                logger.error(f"❌ Fehler beim Laden von Profile für {user_id}: {e}")
                # Fallback: Neues Profile erstellen
                return UserProfile(user_id=user_id)
        else:
            logger.info(f"📝 Creating new profile for {user_id}")
            profile = UserProfile(user_id=user_id)
            self.save_profile(profile)
            return profile

    def save_profile(self, profile: UserProfile) -> bool:
        """
        Speichert User Profile als JSON.

        Args:
            profile: UserProfile zu speichern

        Returns:
            True bei Erfolg
        """
        profile_path = self._get_profile_path(profile.user_id)

        try:
            # Update timestamp
            profile.last_updated = datetime.utcnow()

            # Serialize to JSON
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.model_dump(), f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"💾 Profile saved for {profile.user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern von Profile für {profile.user_id}: {e}")
            return False

    def extract_preferences_from_patterns(
        self,
        user_id: str,
        memory_patterns: List[Dict]
    ) -> UserProfile:
        """
        Extrahiert Präferenzen aus Memory Patterns und aktualisiert Profile.

        Args:
            user_id: User ID
            memory_patterns: Liste von Pattern-Memories aus Qdrant

        Returns:
            Aktualisiertes UserProfile
        """
        profile = self.load_profile(user_id)

        # Gruppiere Patterns nach Entity
        entity_patterns = defaultdict(list)

        for pattern in memory_patterns:
            # Parse Pattern-Memory
            # Format: "Smart Home Pattern: light.wohnzimmer wurde 5x turn_on (Peak: 7:00 Uhr, 3x)"
            try:
                metadata = pattern.get("metadata", {})
                entity_id = metadata.get("entity_id")
                action = metadata.get("action")
                frequency = metadata.get("frequency", 0)
                peak_hour = metadata.get("peak_hour", 0)

                if entity_id and action:
                    entity_patterns[entity_id].append({
                        "action": action,
                        "frequency": frequency,
                        "peak_hour": peak_hour,
                        "pattern": pattern
                    })

            except Exception as e:
                logger.warning(f"Fehler beim Parsen von Pattern: {e}")
                continue

        # Extrahiere Raum-Namen aus Entity IDs
        for entity_id, patterns in entity_patterns.items():
            # Extrahiere Raum-Name (z.B. "wohnzimmer" aus "light.wohnzimmer")
            if "." in entity_id:
                domain, room_name = entity_id.split(".", 1)
            else:
                room_name = entity_id

            # Erstelle/Update Room Preferences
            if room_name not in profile.room_preferences:
                profile.room_preferences[room_name] = RoomPreferences()

            # Extrahiere Helligkeit-Präferenzen für Lichter
            if domain == "light":
                brightness_by_hour = defaultdict(list)
                for p in patterns:
                    # Wenn Helligkeit-Wert vorhanden
                    brightness = p.get("pattern", {}).get("metadata", {}).get("typical_value")
                    if brightness:
                        brightness_by_hour[p["peak_hour"]].append(brightness)

                # Berechne Durchschnitt pro Tageszeit
                if brightness_by_hour:
                    morning_vals = [v for h, vals in brightness_by_hour.items() if 6 <= h < 12 for v in vals]
                    afternoon_vals = [v for h, vals in brightness_by_hour.items() if 12 <= h < 18 for v in vals]
                    evening_vals = [v for h, vals in brightness_by_hour.items() if 18 <= h < 22 for v in vals]
                    night_vals = [v for h, vals in brightness_by_hour.items() if (h >= 22 or h < 6) for v in vals]

                    profile.room_preferences[room_name].brightness = BrightnessPreference(
                        morning=int(sum(morning_vals) / len(morning_vals)) if morning_vals else 80,
                        afternoon=int(sum(afternoon_vals) / len(afternoon_vals)) if afternoon_vals else 100,
                        evening=int(sum(evening_vals) / len(evening_vals)) if evening_vals else 60,
                        night=int(sum(night_vals) / len(night_vals)) if night_vals else 20
                    )

        self.save_profile(profile)
        logger.info(f"✅ Preferences extracted from {len(memory_patterns)} patterns for {user_id}")
        return profile

    def build_routines_from_patterns(
        self,
        user_id: str,
        memory_patterns: List[Dict]
    ) -> UserProfile:
        """
        Erstellt Routinen aus zeitbasierten Patterns.

        Args:
            user_id: User ID
            memory_patterns: Liste von Pattern-Memories

        Returns:
            Aktualisiertes UserProfile mit Routinen
        """
        profile = self.load_profile(user_id)

        # Gruppiere Patterns nach Peak-Hour
        hour_patterns = defaultdict(list)

        for pattern in memory_patterns:
            try:
                metadata = pattern.get("metadata", {})
                peak_hour = metadata.get("peak_hour")
                if peak_hour is not None:
                    hour_patterns[peak_hour].append(pattern)
            except Exception as e:
                logger.warning(f"Fehler beim Gruppieren von Pattern: {e}")

        # Erkenne Morgen-Routine (6-9 Uhr)
        morning_patterns = []
        for hour in range(6, 10):
            morning_patterns.extend(hour_patterns.get(hour, []))

        if len(morning_patterns) >= 3:  # Mindestens 3 Patterns
            profile.morning_routine = self._create_routine_from_patterns(
                name="Morgen-Routine",
                patterns=morning_patterns,
                time_range="06:00-09:00"
            )
            logger.info(f"🌅 Morgen-Routine erstellt für {user_id} ({len(morning_patterns)} Patterns)")

        # Erkenne Abend-Routine (18-22 Uhr)
        evening_patterns = []
        for hour in range(18, 23):
            evening_patterns.extend(hour_patterns.get(hour, []))

        if len(evening_patterns) >= 3:
            profile.evening_routine = self._create_routine_from_patterns(
                name="Abend-Routine",
                patterns=evening_patterns,
                time_range="18:00-22:00"
            )
            logger.info(f"🌆 Abend-Routine erstellt für {user_id} ({len(evening_patterns)} Patterns)")

        # Erkenne Nacht-Routine (22-24 Uhr)
        night_patterns = []
        for hour in range(22, 24):
            night_patterns.extend(hour_patterns.get(hour, []))

        if len(night_patterns) >= 2:
            profile.night_routine = self._create_routine_from_patterns(
                name="Nacht-Routine",
                patterns=night_patterns,
                time_range="22:00-00:00"
            )
            logger.info(f"🌙 Nacht-Routine erstellt für {user_id} ({len(night_patterns)} Patterns)")

        self.save_profile(profile)
        return profile

    def _create_routine_from_patterns(
        self,
        name: str,
        patterns: List[Dict],
        time_range: str
    ) -> Routine:
        """
        Erstellt eine Routine aus Patterns.

        Args:
            name: Name der Routine
            patterns: Liste von Patterns
            time_range: Zeitbereich (z.B. "06:00-09:00")

        Returns:
            Routine
        """
        # Berechne typische Start-Zeit (Durchschnitt aller Peak-Hours)
        peak_hours = []
        for p in patterns:
            peak_hour = p.get("metadata", {}).get("peak_hour")
            if peak_hour is not None:
                peak_hours.append(peak_hour)

        typical_hour = int(sum(peak_hours) / len(peak_hours)) if peak_hours else 0
        typical_minute = 0  # Vereinfachung

        # Erstelle Actions aus Patterns
        actions = []
        for p in patterns:
            metadata = p.get("metadata", {})
            entity_id = metadata.get("entity_id", "unknown")
            action = metadata.get("action", "turn_on")
            frequency = metadata.get("frequency", 0)
            frequency_score = metadata.get("frequency_score", 0.5)

            actions.append(RoutineAction(
                entity_id=entity_id,
                action=action,
                frequency=frequency_score,
                typical_value=None,  # TODO: Extract from pattern
                time_offset_minutes=0
            ))

        # Berechne Konfidenz (Durchschnitt aller Frequency-Scores)
        freq_scores = [p.get("metadata", {}).get("frequency_score", 0.5) for p in patterns]
        confidence = sum(freq_scores) / len(freq_scores) if freq_scores else 0.5

        return Routine(
            name=name,
            time_range=time_range,
            typical_start_hour=typical_hour,
            typical_start_minute=typical_minute,
            actions=actions,
            confidence=confidence,
            days_of_week=list(DayOfWeek),  # Alle Tage erstmal
            enabled=False  # User muss aktivieren
        )

    def generate_automation_suggestions(
        self,
        user_id: str,
        min_confidence: float = 0.7
    ) -> List[AutomationSuggestion]:
        """
        Generiert Automationsvorschläge basierend auf Routinen.

        Args:
            user_id: User ID
            min_confidence: Minimum Confidence für Vorschläge

        Returns:
            Liste von AutomationSuggestion
        """
        profile = self.load_profile(user_id)
        suggestions = []

        # Prüfe jede Routine
        routines = [
            profile.morning_routine,
            profile.evening_routine,
            profile.night_routine
        ] + profile.custom_routines

        for routine in routines:
            if not routine or routine.enabled:
                continue  # Überspringe None oder bereits aktivierte

            if routine.confidence >= min_confidence:
                # Erstelle Suggestion
                suggestion = AutomationSuggestion(
                    id=f"suggestion_{routine.name.lower().replace(' ', '_').replace('-', '_')}",
                    title=f"Automatische {routine.name}",
                    description=self._generate_suggestion_description(routine),
                    routine=routine,
                    confidence=routine.confidence,
                    based_on_patterns=len(routine.actions)
                )
                suggestions.append(suggestion)

        # Füge Suggestions zu Profile hinzu
        profile.suggested_automations = suggestions
        self.save_profile(profile)

        logger.info(f"💡 {len(suggestions)} Automationsvorschläge generiert für {user_id}")
        return suggestions

    def _generate_suggestion_description(self, routine: Routine) -> str:
        """Generiert Beschreibung für Automation Suggestion."""
        actions_text = ", ".join([
            f"{action.entity_id.split('.')[-1]} {action.action}"
            for action in routine.actions[:3]  # Nur erste 3
        ])

        return (
            f"Basierend auf deinen Nutzungsmustern schlage ich vor, "
            f"täglich um {routine.typical_start_hour:02d}:{routine.typical_start_minute:02d} Uhr "
            f"automatisch folgende Aktionen auszuführen: {actions_text}"
            + (f" (und {len(routine.actions) - 3} weitere)" if len(routine.actions) > 3 else "")
            + f". Confidence: {routine.confidence:.0%}"
        )


# Singleton Instance
_profile_manager: Optional[UserProfileManager] = None


def get_profile_manager() -> UserProfileManager:
    """
    Get or create singleton profile manager.

    Returns:
        UserProfileManager
    """
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = UserProfileManager()
    return _profile_manager
