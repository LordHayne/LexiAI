"""
User Database Persistence
Speichert User-Daten persistent in JSON-Datei
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import shutil

logger = logging.getLogger(__name__)

# Pfad zur User-Datenbank
USERS_DB_PATH = Path(__file__).parent.parent / "config" / "users_db.json"
BACKUP_DIR = Path(__file__).parent.parent / "config" / "backups"


class UserPersistence:
    """Verwaltet persistente Speicherung von User-Daten"""

    @staticmethod
    def ensure_directories() -> None:
        """Stellt sicher, dass Verzeichnisse existieren"""
        USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_users() -> Dict[str, dict]:
        """
        Lädt User-Datenbank aus JSON-Datei

        Returns:
            Dictionary mit User-Daten (user_id -> user_dict)
        """
        UserPersistence.ensure_directories()

        if not USERS_DB_PATH.exists():
            logger.info("Keine User-Datenbank gefunden, erstelle neue")
            return {}

        try:
            with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Datetime-Strings zurück in datetime konvertieren
            for user_id, user in data.items():
                if user.get("created_at"):
                    user["created_at"] = datetime.fromisoformat(user["created_at"])
                if user.get("last_login"):
                    user["last_login"] = datetime.fromisoformat(user["last_login"])

            logger.info(f"✅ {len(data)} User aus Datenbank geladen")
            return data

        except Exception as e:
            logger.error(f"Fehler beim Laden der User-Datenbank: {e}", exc_info=True)
            # Bei Fehler Backup versuchen
            backup_file = BACKUP_DIR / "users_db.json.backup"
            if backup_file.exists():
                logger.info("Versuche Backup zu laden...")
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as backup_error:
                    logger.error(f"Backup laden fehlgeschlagen: {backup_error}")

            return {}

    @staticmethod
    def save_users(users: Dict[str, dict]) -> bool:
        """
        Speichert User-Datenbank in JSON-Datei

        Args:
            users: Dictionary mit User-Daten

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        UserPersistence.ensure_directories()

        try:
            # Backup der aktuellen Datei erstellen
            if USERS_DB_PATH.exists():
                backup_file = BACKUP_DIR / f"users_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(USERS_DB_PATH, backup_file)

                # Nur die letzten 10 Backups behalten
                backups = sorted(BACKUP_DIR.glob("users_db_*.json"))
                for old_backup in backups[:-10]:
                    old_backup.unlink()

            # Datetime-Objekte in Strings konvertieren für JSON
            users_serializable = {}
            for user_id, user in users.items():
                user_copy = user.copy()
                if user_copy.get("created_at"):
                    user_copy["created_at"] = user_copy["created_at"].isoformat()
                if user_copy.get("last_login"):
                    user_copy["last_login"] = user_copy["last_login"].isoformat()
                users_serializable[user_id] = user_copy

            # Temporäre Datei schreiben (atomic write)
            temp_file = USERS_DB_PATH.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(users_serializable, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(USERS_DB_PATH)

            logger.info(f"✅ User-Datenbank gespeichert ({len(users)} User)")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern der User-Datenbank: {e}", exc_info=True)
            return False

    @staticmethod
    def load_refresh_tokens() -> Dict[str, dict]:
        """
        Lädt Refresh Tokens aus JSON-Datei

        Returns:
            Dictionary mit Refresh Token-Daten
        """
        tokens_path = USERS_DB_PATH.parent / "refresh_tokens.json"

        if not tokens_path.exists():
            return {}

        try:
            with open(tokens_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Datetime-Strings zurück in datetime konvertieren
            for token, info in data.items():
                if info.get("created_at"):
                    info["created_at"] = datetime.fromisoformat(info["created_at"])

            logger.info(f"✅ {len(data)} Refresh Tokens geladen")
            return data

        except Exception as e:
            logger.error(f"Fehler beim Laden der Refresh Tokens: {e}")
            return {}

    @staticmethod
    def save_refresh_tokens(tokens: Dict[str, dict]) -> bool:
        """
        Speichert Refresh Tokens in JSON-Datei

        Args:
            tokens: Dictionary mit Refresh Token-Daten

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        UserPersistence.ensure_directories()
        tokens_path = USERS_DB_PATH.parent / "refresh_tokens.json"

        try:
            # Datetime-Objekte in Strings konvertieren
            tokens_serializable = {}
            for token, info in tokens.items():
                info_copy = info.copy()
                if info_copy.get("created_at"):
                    info_copy["created_at"] = info_copy["created_at"].isoformat()
                tokens_serializable[token] = info_copy

            # Temporäre Datei schreiben
            temp_file = tokens_path.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(tokens_serializable, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(tokens_path)

            logger.debug(f"Refresh Tokens gespeichert ({len(tokens)} Tokens)")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern der Refresh Tokens: {e}")
            return False
