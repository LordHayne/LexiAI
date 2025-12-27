"""
Profile Builder Service für LexiAI
Automatische Extraktion von User-Profil-Informationen aus Konversationen
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime, timezone

# LAZY IMPORT: langchain_ollama blockiert wenn Ollama nicht läuft
if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """
    Automatische Profil-Extraktion aus Chat-Nachrichten
    Verwendet LLM zur Identifikation relevanter User-Informationen
    """

    # Profile-Kategorien
    CATEGORIES = [
        "user_profile_name",            # Name
        "user_profile_occupation",      # Beruf/Tätigkeit
        "user_profile_interests",        # Hobbys/Interessen
        "user_profile_skills",           # Fähigkeiten/Skills
        "user_profile_preferences",      # Präferenzen (Kommunikationsstil, etc.)
        "user_profile_background",       # Hintergrund/Ausbildung
        "user_profile_goals",            # Ziele/Absichten
        "user_profile_location",         # Wohnort/Zeitzone
        "user_profile_languages",        # Sprachen
        "user_profile_technical_level",  # Technisches Niveau
        "user_profile_communication_style",  # Kommunikationsstil
        "user_profile_topics"            # Häufige Themen
    ]

    def __init__(self, llm_client: Optional["ChatOllama"] = None, ollama_url: str = "http://localhost:11434"):
        """
        Initialisiert Profile Builder

        Args:
            llm_client: Optionaler LLM Client (wird erstellt wenn None)
            ollama_url: Ollama Server URL
        """
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # LAZY IMPORT: Nur importieren wenn tatsächlich gebraucht
            from langchain_ollama import ChatOllama
            self.llm_client = ChatOllama(
                base_url=ollama_url,
                model="gemma3:4b-it-qat",  # Schnelles Modell für Profile-Extraktion
                temperature=0.3  # Niedrig für konsistente Extraktion
            )

    async def extract_profile_info(
        self,
        user_message: str,
        assistant_response: str,
        current_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extrahiert Profil-Informationen aus Chat-Nachricht

        Args:
            user_message: Nachricht vom User
            assistant_response: Antwort vom Assistant
            current_profile: Aktuelles User-Profil

        Returns:
            Aktualisiertes Profil-Dictionary oder leeres Dict bei Fehler
        """
        try:
            rule_based_info = self._extract_rule_based_profile(user_message)

            # Prompt für LLM zur Profil-Extraktion
            extraction_prompt = self._build_extraction_prompt(
                user_message,
                assistant_response,
                current_profile
            )

            # LLM aufrufen (async)
            response = await self.llm_client.ainvoke(extraction_prompt)

            # Response parsen
            extracted_info = self._parse_llm_response(response.content)

            if extracted_info or rule_based_info:
                # Profil aktualisieren (merge mit current_profile)
                updated_profile = self._merge_profiles(current_profile, extracted_info)
                updated_profile = self._merge_profiles(updated_profile, rule_based_info)

                extracted_keys = set(extracted_info.keys()) | set(rule_based_info.keys())
                logger.info(f"Profil-Informationen extrahiert: {sorted(extracted_keys)}")
                return updated_profile

            return current_profile

        except Exception as e:
            logger.error(f"Fehler bei Profil-Extraktion: {e}", exc_info=True)
            return current_profile

    def _build_extraction_prompt(
        self,
        user_message: str,
        assistant_response: str,
        current_profile: Dict[str, Any]
    ) -> str:
        """
        Erstellt Prompt für LLM zur Profil-Extraktion
        """
        categories_str = ", ".join(self.CATEGORIES)

        prompt = f"""Du bist ein Profil-Analyse-Assistent. Deine Aufgabe ist es, aus einer Chat-Konversation relevante Informationen über den User zu extrahieren.

Analysiere folgende Konversation:

USER: {user_message}

ASSISTANT: {assistant_response}

Aktuelles User-Profil:
{json.dumps(current_profile, indent=2, ensure_ascii=False)}

Extrahiere NEUE oder AKTUALISIERTE Informationen in folgenden Kategorien:
{categories_str}

WICHTIG:
- Nur explizit genannte oder klar ableitbare Informationen extrahieren
- Keine Spekulationen oder Vermutungen
- Wenn keine neuen Infos: Leeres JSON zurückgeben
- Format: JSON mit Kategorie als Key, Wert als String oder Liste

Beispiel Output:
{{
  "user_profile_occupation": "Software-Entwickler",
  "user_profile_skills": ["Python", "FastAPI"],
  "user_profile_interests": ["Künstliche Intelligenz", "Machine Learning"],
  "user_profile_technical_level": "Fortgeschritten"
}}

Output (NUR JSON, keine Erklärung):"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parst LLM Response zu Dictionary

        Args:
            response: LLM Response String

        Returns:
            Extrahiertes Profil-Dictionary
        """
        try:
            # JSON extrahieren (manchmal mit Markdown Code Blocks)
            response = response.strip()

            # Entferne Markdown Code Blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])  # Entferne erste und letzte Zeile

            # Parse JSON
            extracted = json.loads(response)

            # Validiere Kategorien
            valid_extracted = {}
            for key, value in extracted.items():
                if key in self.CATEGORIES:
                    valid_extracted[key] = value
                else:
                    logger.warning(f"Ungültige Profil-Kategorie: {key}")

            return valid_extracted

        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim JSON-Parsing: {e}\nResponse: {response}")
            return {}
        except Exception as e:
            logger.error(f"Fehler beim Response-Parsing: {e}")
            return {}

    def _split_list_values(self, raw_value: str) -> List[str]:
        parts = re.split(r",|\bund\b|\band\b|&|\bsowie\b", raw_value)
        cleaned = []
        for part in parts:
            value = part.strip()
            if value:
                cleaned.append(value)
        return cleaned

    def _extract_rule_based_profile(self, user_message: str) -> Dict[str, Any]:
        text = (user_message or "").strip()
        if not text:
            return {}

        extracted: Dict[str, Any] = {}
        stop_starts = ("dir ", "dich ", "euch ", "ihnen ", "ihm ")
        list_fields = {
            "user_profile_interests",
            "user_profile_skills",
            "user_profile_languages",
        }

        patterns = [
            ("user_profile_name", [
                r"\bich hei[ßs]e\s+([^.,;!?]+)",
                r"\bmein name ist\s+([^.,;!?]+)",
                r"\bich\s+([a-zäöüß-]+)\s+hei[ßs]e\b",
            ]),
            ("user_profile_occupation", [
                r"\bich arbeite als\s+([^.,;!?]+)",
                r"\bmein beruf ist\s+([^.,;!?]+)",
                r"\bich bin\s+([^.,;!?]+)\s+von beruf\b",
            ]),
            ("user_profile_interests", [
                r"\bmein hobby ist\s+([^.,;!?]+)",
                r"\bmeine hobbys sind\s+([^.,;!?]+)",
                r"\bich interessiere mich für\s+([^.,;!?]+)",
                r"\bich begeistere mich für\s+([^.,;!?]+)",
            ]),
            ("user_profile_skills", [
                r"\bich kann\s+([^.,;!?]+)",
                r"\bich beherrsche\s+([^.,;!?]+)",
                r"\bich kenne mich mit\s+([^.,;!?]+)\s+aus\b",
            ]),
            ("user_profile_languages", [
                r"\bich spreche\s+([^.,;!?]+)",
                r"\bmeine sprachen sind\s+([^.,;!?]+)",
            ]),
            ("user_profile_location", [
                r"\bich wohne in\s+([^.,;!?]+)",
                r"\bich lebe in\s+([^.,;!?]+)",
            ]),
            ("user_profile_preferences", [
                r"\bich bevorzuge\s+([^.,;!?]+)",
                r"\bich mag es\s+([^.,;!?]+)",
            ]),
            ("user_profile_goals", [
                r"\bmein ziel ist\s+([^.,;!?]+)",
            ]),
        ]

        for field, field_patterns in patterns:
            for pattern in field_patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue
                raw_value = match.group(1).strip()
                if raw_value.lower().startswith(stop_starts):
                    continue
                if not raw_value:
                    continue
                if field in list_fields:
                    values = self._split_list_values(raw_value)
                    if values:
                        extracted.setdefault(field, []).extend(values)
                else:
                    extracted[field] = raw_value
                break

        for field in list_fields:
            if field in extracted:
                unique = []
                for value in extracted[field]:
                    if value not in unique:
                        unique.append(value)
                extracted[field] = unique

        return extracted

    def _merge_profiles(
        self,
        current_profile: Dict[str, Any],
        new_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merged neues Profil mit bestehendem

        Args:
            current_profile: Aktuelles Profil
            new_info: Neue Informationen

        Returns:
            Gemergtes Profil
        """
        merged = current_profile.copy()

        for key, new_value in new_info.items():
            if key not in merged:
                # Neue Kategorie
                merged[key] = new_value
            else:
                # Existierende Kategorie updaten
                current_value = merged[key]

                # Listen zusammenführen
                if isinstance(current_value, list) and isinstance(new_value, list):
                    # Duplikate vermeiden
                    merged[key] = list(set(current_value + new_value))

                # String updaten (neuer Wert überschreibt)
                elif isinstance(new_value, str):
                    merged[key] = new_value

                # Dictionary mergen
                elif isinstance(current_value, dict) and isinstance(new_value, dict):
                    merged[key] = {**current_value, **new_value}

                else:
                    # Default: Überschreiben
                    merged[key] = new_value

        # Metadaten hinzufügen
        merged["_last_updated"] = datetime.now(timezone.utc).isoformat()

        return merged

    def get_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Erstellt lesbare Zusammenfassung des Profils

        Args:
            profile: User-Profil

        Returns:
            Lesbare Zusammenfassung als String
        """
        if not profile or all(k.startswith("_") for k in profile.keys()):
            return "Noch kein Profil vorhanden."

        summary_parts = []

        category_names = {
            "user_profile_name": "Name",
            "user_profile_occupation": "Beruf/Tätigkeit",
            "user_profile_interests": "Interessen",
            "user_profile_skills": "Fähigkeiten",
            "user_profile_preferences": "Präferenzen",
            "user_profile_background": "Hintergrund",
            "user_profile_goals": "Ziele",
            "user_profile_location": "Wohnort",
            "user_profile_languages": "Sprachen",
            "user_profile_technical_level": "Technisches Niveau",
            "user_profile_communication_style": "Kommunikationsstil",
            "user_profile_topics": "Häufige Themen"
        }

        for key, value in profile.items():
            if key.startswith("_"):
                continue

            category_name = category_names.get(key, key)

            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)

            summary_parts.append(f"• {category_name}: {value_str}")

        return "\n".join(summary_parts)

    def should_update_profile(
        self,
        user_message: str,
        min_message_length: int = 20
    ) -> bool:
        """
        Entscheidet ob Profil-Update sinnvoll ist

        Args:
            user_message: User-Nachricht
            min_message_length: Minimale Nachrichtenlänge

        Returns:
            True wenn Update sinnvoll
        """
        # Zu kurze Nachrichten überspringen, außer klare Profilsignale
        if len(user_message) < min_message_length:
            if self._extract_rule_based_profile(user_message):
                return True
            return False

        # System-Commands überspringen
        if user_message.startswith("/"):
            return False

        # Einfache Grüße überspringen
        greetings = ["hallo", "hi", "hey", "guten tag", "guten morgen", "guten abend"]
        if user_message.lower().strip() in greetings:
            return False

        return True
