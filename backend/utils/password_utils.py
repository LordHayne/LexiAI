"""
Password Utilities für LexiAI Authentication
Implementiert bcrypt Hashing mit Cost Factor 12 und Password-Validierung
"""
import re
import bcrypt
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PasswordConfig:
    """Password Policy Configuration"""

    # bcrypt cost factor (2^12 = 4096 rounds)
    BCRYPT_ROUNDS = 12

    # Password strength requirements
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = False  # Optional für bessere UX

    # Special characters erlaubt
    ALLOWED_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"


def hash_password(password: str) -> str:
    """
    Erstellt bcrypt Hash mit Salt (Cost Factor 12)

    Args:
        password: Klartext-Passwort

    Returns:
        Bcrypt Hash String (60 chars)

    Raises:
        ValueError: Bei leerem Passwort
    """
    if not password:
        raise ValueError("Passwort darf nicht leer sein")

    # NIEMALS Passwörter loggen!
    logger.debug("Erstelle bcrypt Hash...")

    # Salt generieren und Hash erstellen
    salt = bcrypt.gensalt(rounds=PasswordConfig.BCRYPT_ROUNDS)
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)

    logger.debug("Passwort Hash erfolgreich erstellt")
    return password_hash.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifiziert Passwort gegen bcrypt Hash

    Args:
        plain_password: Klartext-Passwort vom User
        hashed_password: Gespeicherter bcrypt Hash

    Returns:
        True wenn Passwort korrekt, False sonst
    """
    if not plain_password or not hashed_password:
        return False

    try:
        # bcrypt vergleicht automatisch inkl. Salt
        result = bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )

        if result:
            logger.debug("Passwort-Verifikation erfolgreich")
        else:
            logger.debug("Passwort-Verifikation fehlgeschlagen")

        return result
    except Exception as e:
        logger.error(f"Fehler bei Passwort-Verifikation: {e}")
        return False


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validiert Passwort-Stärke gemäß Policy

    Args:
        password: Zu prüfendes Passwort

    Returns:
        Tuple (is_valid, error_message)
        - is_valid: True wenn Passwort Policy erfüllt
        - error_message: Fehlerbeschreibung oder leerer String
    """
    config = PasswordConfig()

    # Längenprüfung
    if len(password) < config.MIN_LENGTH:
        return False, f"Passwort muss mindestens {config.MIN_LENGTH} Zeichen lang sein"

    if len(password) > config.MAX_LENGTH:
        return False, f"Passwort darf maximal {config.MAX_LENGTH} Zeichen lang sein"

    # Großbuchstaben prüfen
    if config.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        return False, "Passwort muss mindestens einen Großbuchstaben enthalten"

    # Kleinbuchstaben prüfen
    if config.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        return False, "Passwort muss mindestens einen Kleinbuchstaben enthalten"

    # Ziffern prüfen
    if config.REQUIRE_DIGIT and not re.search(r'\d', password):
        return False, "Passwort muss mindestens eine Ziffer enthalten"

    # Sonderzeichen prüfen (optional)
    if config.REQUIRE_SPECIAL:
        special_pattern = f"[{re.escape(config.ALLOWED_SPECIAL_CHARS)}]"
        if not re.search(special_pattern, password):
            return False, f"Passwort muss mindestens ein Sonderzeichen enthalten ({config.ALLOWED_SPECIAL_CHARS})"

    # Common password check (Basic)
    common_passwords = [
        "password", "12345678", "qwerty123", "admin123", "welcome1",
        "Password1", "Qwerty123", "Admin123"
    ]
    if password.lower() in [p.lower() for p in common_passwords]:
        return False, "Passwort ist zu unsicher (häufig verwendetes Passwort)"

    return True, ""


def validate_email_format(email: str) -> Tuple[bool, str]:
    """
    Validiert Email-Format (Basic Regex)

    Args:
        email: Email-Adresse

    Returns:
        Tuple (is_valid, error_message)
    """
    if not email:
        return False, "Email darf nicht leer sein"

    # Basic Email Regex (RFC 5322 simplified)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email):
        return False, "Ungültiges Email-Format"

    if len(email) > 254:  # RFC 5321
        return False, "Email-Adresse ist zu lang"

    return True, ""


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitized User-Input gegen SQL Injection und XSS

    Args:
        input_string: Zu säubernder String
        max_length: Maximale Länge

    Returns:
        Gesäuberter String
    """
    if not input_string:
        return ""

    # Maximale Länge begrenzen
    sanitized = input_string[:max_length]

    # SQL Injection Keywords entfernen (Basic Protection - ORM macht mehr)
    sql_keywords = [
        "--", ";--", "/*", "*/", "xp_", "sp_",
        "DROP TABLE", "DELETE FROM", "INSERT INTO",
        "'; DROP", "OR 1=1", "OR '1'='1"
    ]

    for keyword in sql_keywords:
        sanitized = sanitized.replace(keyword, "")

    # XSS-gefährliche Zeichen escapen
    xss_chars = {
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "&": "&amp;"
    }

    for char, replacement in xss_chars.items():
        sanitized = sanitized.replace(char, replacement)

    return sanitized.strip()


def generate_safe_username(email: str) -> str:
    """
    Generiert sicheren Username aus Email

    Args:
        email: Email-Adresse

    Returns:
        Sanitized Username (Teil vor @)
    """
    if not email or "@" not in email:
        return "user"

    username = email.split("@")[0]

    # Nur alphanumerische Zeichen und Unterstrich
    username = re.sub(r'[^a-zA-Z0-9_]', '', username)

    # Mindestlänge 3 Zeichen
    if len(username) < 3:
        username = "user_" + username

    return username[:50]  # Max 50 chars
