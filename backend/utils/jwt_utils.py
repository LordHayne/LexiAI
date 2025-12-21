"""
JWT Token Utilities für LexiAI Authentication
Implementiert HS256 Token-Erzeugung und -Validierung mit Refresh-Token-Support
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import logging

logger = logging.getLogger(__name__)


class JWTConfig:
    """JWT Configuration from Environment"""

    @staticmethod
    def get_secret_key() -> str:
        """Get JWT secret from environment or generate temporary one"""
        secret = os.getenv("LEXI_JWT_SECRET")
        if not secret:
            logger.warning("LEXI_JWT_SECRET nicht gesetzt! Verwende temporären Schlüssel (NICHT für Produktion!)")
            secret = "TEMPORARY_DEV_SECRET_CHANGE_ME_IN_PRODUCTION"
        return secret

    @staticmethod
    def get_algorithm() -> str:
        """JWT algorithm (HS256)"""
        return "HS256"

    @staticmethod
    def get_access_token_expire_minutes() -> int:
        """Access token expiration in minutes (default: 15)"""
        return int(os.getenv("LEXI_JWT_ACCESS_EXPIRE_MINUTES", "15"))

    @staticmethod
    def get_refresh_token_expire_days() -> int:
        """Refresh token expiration in days (default: 7)"""
        return int(os.getenv("LEXI_JWT_REFRESH_EXPIRE_DAYS", "7"))


def create_access_token(
    user_id: str,
    email: str,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Erstellt JWT Access Token mit 15 Minuten Gültigkeit

    Args:
        user_id: Eindeutige User-ID
        email: User-Email
        additional_claims: Zusätzliche Claims (optional)

    Returns:
        JWT Token String
    """
    config = JWTConfig()

    # Standard Claims
    payload = {
        "sub": user_id,  # Subject (User ID)
        "email": email,
        "type": "access",
        "iat": datetime.now(timezone.utc),  # Issued at
        "exp": datetime.now(timezone.utc) + timedelta(minutes=config.get_access_token_expire_minutes())
    }

    # Zusätzliche Claims hinzufügen
    if additional_claims:
        payload.update(additional_claims)

    # Token erstellen
    token = jwt.encode(
        payload,
        config.get_secret_key(),
        algorithm=config.get_algorithm()
    )

    logger.debug(f"Access Token erstellt für User {user_id} (gültig bis {payload['exp']})")
    return token


def create_refresh_token(user_id: str, email: str) -> str:
    """
    Erstellt JWT Refresh Token mit 7 Tagen Gültigkeit

    Args:
        user_id: Eindeutige User-ID
        email: User-Email

    Returns:
        JWT Refresh Token String
    """
    config = JWTConfig()

    payload = {
        "sub": user_id,
        "email": email,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=config.get_refresh_token_expire_days())
    }

    token = jwt.encode(
        payload,
        config.get_secret_key(),
        algorithm=config.get_algorithm()
    )

    logger.debug(f"Refresh Token erstellt für User {user_id} (gültig bis {payload['exp']})")
    return token


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Validiert JWT Token und gibt Payload zurück

    Args:
        token: JWT Token String
        token_type: Erwarteter Token-Typ ("access" oder "refresh")

    Returns:
        Token Payload als Dictionary

    Raises:
        InvalidTokenError: Token ist ungültig
        ExpiredSignatureError: Token ist abgelaufen
        ValueError: Falscher Token-Typ
    """
    config = JWTConfig()

    try:
        # Token dekodieren und validieren
        payload = jwt.decode(
            token,
            config.get_secret_key(),
            algorithms=[config.get_algorithm()]
        )

        # Token-Typ prüfen
        if payload.get("type") != token_type:
            raise ValueError(f"Falscher Token-Typ: erwartet '{token_type}', erhalten '{payload.get('type')}'")

        logger.debug(f"Token erfolgreich validiert für User {payload.get('sub')}")
        return payload

    except ExpiredSignatureError:
        logger.warning("Token ist abgelaufen")
        raise
    except InvalidTokenError as e:
        logger.error(f"Token-Validierung fehlgeschlagen: {e}")
        raise


def decode_token_unsafe(token: str) -> Optional[Dict[str, Any]]:
    """
    Dekodiert Token OHNE Validierung (für Debugging/Logging)
    NIEMALS für Authentifizierung verwenden!

    Args:
        token: JWT Token String

    Returns:
        Token Payload oder None bei Fehler
    """
    try:
        # Decode ohne Verifikation
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except Exception as e:
        logger.error(f"Fehler beim unsicheren Token-Dekodieren: {e}")
        return None


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Gibt Ablaufdatum des Tokens zurück

    Args:
        token: JWT Token String

    Returns:
        Datetime des Ablaufs oder None bei Fehler
    """
    payload = decode_token_unsafe(token)
    if payload and "exp" in payload:
        return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    return None


def is_token_expired(token: str) -> bool:
    """
    Prüft, ob Token abgelaufen ist (ohne Validierung der Signatur)

    Args:
        token: JWT Token String

    Returns:
        True wenn abgelaufen, False wenn noch gültig
    """
    expiry = get_token_expiry(token)
    if expiry:
        return datetime.now(timezone.utc) > expiry
    return True


def extract_user_id_from_token(token: str) -> Optional[str]:
    """
    Extrahiert User-ID aus Token OHNE Validierung
    Für Logging/Debug-Zwecke

    Args:
        token: JWT Token String

    Returns:
        User-ID oder None
    """
    payload = decode_token_unsafe(token)
    if payload:
        return payload.get("sub")
    return None
