"""
Profile Context Service für LexiAI
Bereitet User-Profil für personalisierte LLM-Antworten auf
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProfileContextBuilder:
    """
    Erstellt Kontext-String aus User-Profil für LLM-Prompts
    """

    def get_user_context(
        self,
        user_profile: Dict[str, Any],
        include_all: bool = False,
        max_length: int = 500
    ) -> str:
        """
        Erstellt personalisierten Kontext aus User-Profil

        Args:
            user_profile: User-Profil Dictionary
            include_all: Alle Kategorien einschließen (auch leere)
            max_length: Maximale Länge des Kontexts

        Returns:
            Kontext-String für LLM-Prompt
        """
        if not user_profile or all(k.startswith("_") for k in user_profile.keys()):
            return ""

        context_parts = []

        # Kategorie-Mapping für lesbare Namen
        category_display = {
            "user_profile_name": "Name",
            "user_profile_occupation": "Beruf",
            "user_profile_interests": "Interessen",
            "user_profile_preferences": "Präferenzen",
            "user_profile_background": "Hintergrund",
            "user_profile_goals": "Ziele",
            "user_profile_location": "Wohnort",
            "user_profile_languages": "Sprachen",
            "user_profile_technical_level": "Technisches Niveau",
            "user_profile_communication_style": "Kommunikationsstil",
            "user_profile_topics": "Häufige Themen"
        }

        # Priorisierte Kategorien (wichtigste zuerst)
        priority_categories = [
            "user_profile_name",
            "user_profile_occupation",
            "user_profile_technical_level",
            "user_profile_interests",
            "user_profile_communication_style",
            "user_profile_preferences",
            "user_profile_background",
            "user_profile_goals",
            "user_profile_topics"
        ]

        # Wichtigste Kategorien zuerst
        for category in priority_categories:
            if category in user_profile:
                value = user_profile[category]
                display_name = category_display.get(category, category)

                # Wert formatieren
                if isinstance(value, list):
                    if value:  # Nur nicht-leere Listen
                        value_str = ", ".join(str(v) for v in value[:3])  # Max 3 Items
                        if len(value) > 3:
                            value_str += f" (+{len(value) - 3} weitere)"
                        context_parts.append(f"{display_name}: {value_str}")
                elif value:  # Nur nicht-leere Strings/Werte
                    context_parts.append(f"{display_name}: {value}")

        # Restliche Kategorien (wenn include_all)
        if include_all:
            for key, value in user_profile.items():
                if key.startswith("_") or key in priority_categories:
                    continue

                display_name = category_display.get(key, key.replace("user_profile_", "").title())

                if isinstance(value, list) and value:
                    value_str = ", ".join(str(v) for v in value[:2])
                    context_parts.append(f"{display_name}: {value_str}")
                elif value:
                    context_parts.append(f"{display_name}: {value}")

        if not context_parts:
            return ""

        # Zusammenbauen
        context = "User-Profil:\n" + "\n".join(f"• {part}" for part in context_parts)

        # Länge begrenzen
        if len(context) > max_length:
            context = context[:max_length] + "..."

        return context

    def build_personalized_system_prompt(
        self,
        base_prompt: str,
        user_profile: Dict[str, Any]
    ) -> str:
        """
        Erweitert System-Prompt mit User-Profil-Kontext

        Args:
            base_prompt: Basis System-Prompt
            user_profile: User-Profil

        Returns:
            Erweiterter System-Prompt
        """
        user_context = self.get_user_context(user_profile, include_all=False)

        if not user_context:
            return base_prompt

        # Profil-Kontext einfügen
        personalized_prompt = f"""{base_prompt}

{user_context}

Bitte berücksichtige diese Informationen bei deinen Antworten und passe deinen Kommunikationsstil entsprechend an."""

        return personalized_prompt

    def get_relevant_profile_fields(
        self,
        user_message: str,
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extrahiert nur für die aktuelle Nachricht relevante Profil-Felder

        Args:
            user_message: Aktuelle User-Nachricht
            user_profile: Vollständiges Profil

        Returns:
            Relevante Profil-Felder
        """
        if not user_profile:
            return {}

        relevant_fields = {}
        message_lower = user_message.lower()

        # Keyword-Mapping für Relevanz
        relevance_keywords = {
            "user_profile_name": ["name", "heiße", "heisse", "ich bin", "mein name"],
            "user_profile_occupation": ["arbeit", "beruf", "job", "karriere", "firma"],
            "user_profile_interests": ["hobby", "freizeit", "interesse", "mag", "gerne"],
            "user_profile_technical_level": ["technisch", "programmier", "code", "entwickl", "software"],
            "user_profile_languages": ["sprache", "englisch", "deutsch", "sprechen"],
            "user_profile_communication_style": ["kommunikation", "stil", "gespräch"],
            "user_profile_goals": ["ziel", "plan", "möchte", "will", "erreichen"],
            "user_profile_location": ["wohnort", "stadt", "land", "wo wohnst"],
        }

        # Prüfe jede Kategorie auf Relevanz
        for category, keywords in relevance_keywords.items():
            if category in user_profile:
                # Prüfe ob Keywords in Nachricht vorkommen
                if any(keyword in message_lower for keyword in keywords):
                    relevant_fields[category] = user_profile[category]

        # Immer technisches Level einschließen (wichtig für Antwort-Stil)
        if "user_profile_technical_level" in user_profile:
            relevant_fields["user_profile_technical_level"] = user_profile["user_profile_technical_level"]

        return relevant_fields

    def should_use_profile(
        self,
        user_message: str,
        user_profile: Dict[str, Any]
    ) -> bool:
        """
        Entscheidet ob Profil-Kontext verwendet werden soll

        Args:
            user_message: User-Nachricht
            user_profile: User-Profil

        Returns:
            True wenn Profil verwendet werden soll
        """
        # Kein Profil vorhanden
        if not user_profile or all(k.startswith("_") for k in user_profile.keys()):
            return False

        # Sehr kurze Nachrichten (einfache Fragen)
        if len(user_message) < 10:
            return False

        # System-Commands
        if user_message.startswith("/"):
            return False

        # Immer verwenden bei längeren Konversationen
        return True
