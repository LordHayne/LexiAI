"""
Authentication Routes für LexiAI
/register, /login, /logout, /refresh Endpoints mit Rate Limiting
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from collections import defaultdict
import asyncio

from fastapi import APIRouter, HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.models.auth_models import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    LogoutRequest,
    UserProfileResponse,
    AuthErrorResponse,
    PasswordChangeRequest
)
from backend.utils.jwt_utils import (
    create_access_token,
    create_refresh_token,
    verify_token,
    extract_user_id_from_token,
    JWTConfig
)
from backend.utils.password_utils import (
    hash_password,
    verify_password,
    generate_safe_username,
    sanitize_input
)
from backend.utils.user_persistence import UserPersistence

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


def _sync_user_store(user_id: str, display_name: str, email: Optional[str]) -> None:
    from backend.services.user_store import get_user_store
    from backend.models.user import User, UserTier

    user_store = get_user_store()
    now = datetime.now(timezone.utc).isoformat()
    existing = user_store.get_user(user_id)

    if existing:
        updates = {"last_seen": now}
        if display_name:
            updates["display_name"] = display_name
        if email:
            updates["email"] = email
        user_store.update_user(user_id, updates)
    else:
        created_user = User(
            user_id=user_id,
            display_name=display_name or "Anonymous User",
            created_at=now,
            last_seen=now,
            tier=UserTier.REGISTERED,
            email=email
        )
        user_store.create_user(created_user)


# Rate Limiting (In-Memory - für Produktion Redis verwenden!)
class RateLimiter:
    """Simple in-memory rate limiter für Login-Versuche"""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 60):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts: Dict[str, list] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Prüft ob weitere Versuche erlaubt sind"""
        now = datetime.now(timezone.utc).timestamp()

        # Alte Einträge entfernen
        self.attempts[identifier] = [
            timestamp for timestamp in self.attempts[identifier]
            if now - timestamp < self.window_seconds
        ]

        # Prüfen ob Limit überschritten
        if len(self.attempts[identifier]) >= self.max_attempts:
            return False

        # Neuen Versuch hinzufügen
        self.attempts[identifier].append(now)
        return True

    def reset(self, identifier: str):
        """Setzt Rate Limit für Identifier zurück"""
        if identifier in self.attempts:
            del self.attempts[identifier]


# Rate Limiter Instanzen
login_rate_limiter = RateLimiter(max_attempts=5, window_seconds=60)
register_rate_limiter = RateLimiter(max_attempts=3, window_seconds=300)  # 3 pro 5min


# Persistent User Store - lädt aus JSON-Datei
# User-Daten werden automatisch gespeichert
USERS_DB: Dict[str, dict] = UserPersistence.load_users()
REFRESH_TOKENS: Dict[str, dict] = UserPersistence.load_refresh_tokens()

logger.info(f"🔐 User-Datenbank initialisiert: {len(USERS_DB)} User, {len(REFRESH_TOKENS)} Refresh Tokens")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency: Validiert JWT Token und gibt User zurück

    Args:
        credentials: Bearer Token aus Authorization Header

    Returns:
        User Dictionary

    Raises:
        HTTPException: Bei ungültigem/abgelaufenem Token
    """
    try:
        token = credentials.credentials
        payload = verify_token(token, token_type="access")

        user_id = payload.get("sub")
        if not user_id or user_id not in USERS_DB:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User nicht gefunden"
            )

        return USERS_DB[user_id]

    except Exception as e:
        logger.error(f"Token-Validierung fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Ungültiges oder abgelaufenes Token",
            headers={"WWW-Authenticate": "Bearer"}
        )


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description="Registriert neuen User mit Email und Passwort, setzt HttpOnly Cookies"
)
async def register(request: RegisterRequest, http_request: Request, response: Response):
    """
    User Registration mit automatischer Token-Generierung

    - Email-Format wird validiert
    - Passwort muss Policy erfüllen (min. 8 Zeichen, 1 Großbuchstabe, 1 Ziffer)
    - Username wird aus Email generiert wenn nicht angegeben
    - Gibt Access + Refresh Token zurück
    """
    client_ip = http_request.client.host if http_request.client else "unknown"

    # Rate Limiting
    if not register_rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate Limit überschritten für Registration von IP {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Zu viele Registrierungsversuche. Bitte warten Sie 5 Minuten."
        )

    try:
        # Email bereits registriert?
        email_lower = request.email.lower()
        if any(user["email"] == email_lower for user in USERS_DB.values()):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email-Adresse bereits registriert"
            )

        # Username generieren oder validieren
        username = request.username or generate_safe_username(request.email)
        username = sanitize_input(username, max_length=50)

        # User ID generieren
        user_id = f"user_{len(USERS_DB) + 1}_{int(datetime.now(timezone.utc).timestamp())}"

        # Passwort hashen
        password_hash = hash_password(request.password)

        # User erstellen
        user = {
            "user_id": user_id,
            "email": email_lower,
            "username": username,
            "password_hash": password_hash,
            "profile": {},  # Leeres Profil
            "created_at": datetime.now(timezone.utc),
            "last_login": None
        }

        # In DB speichern
        USERS_DB[user_id] = user
        UserPersistence.save_users(USERS_DB)

        _sync_user_store(user_id, username, email_lower)

        # Tokens erstellen
        access_token = create_access_token(user_id, email_lower)
        refresh_token = create_refresh_token(user_id, email_lower)

        # Refresh Token speichern
        REFRESH_TOKENS[refresh_token] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc)
        }
        UserPersistence.save_refresh_tokens(REFRESH_TOKENS)

        logger.info(f"User erfolgreich registriert: {email_lower} (ID: {user_id})")

        # Rate Limiter zurücksetzen bei Erfolg
        register_rate_limiter.reset(client_ip)

        # SECURITY: Set HttpOnly cookies instead of returning tokens in JSON body
        # This prevents XSS attacks from stealing tokens via localStorage

        # Access Token Cookie (15 minutes)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,  # Cannot be accessed via JavaScript (XSS protection)
            secure=True,    # Only sent over HTTPS
            samesite="lax", # CSRF protection
            max_age=900,    # 15 minutes in seconds
            path="/"        # Available for all routes
        )

        # Refresh Token Cookie (7 days)
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,  # Cannot be accessed via JavaScript (XSS protection)
            secure=True,    # Only sent over HTTPS
            samesite="lax", # CSRF protection
            max_age=604800, # 7 days in seconds
            path="/"        # Available for all routes
        )

        # User-Objekt für Frontend vorbereiten (ohne password_hash und OHNE TOKENS)
        user_data = {
            "user_id": user["user_id"],
            "email": user["email"],
            "username": user["username"],
            "display_name": user["username"],
            "created_at": user["created_at"].isoformat(),
            "last_login": user.get("last_login").isoformat() if user.get("last_login") else None,
            "profile": user.get("profile", {})
        }

        # Return user data WITHOUT tokens (tokens are in cookies now!)
        return {
            "success": True,
            "message": "Registrierung erfolgreich",
            "user": user_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler bei Registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interner Server-Fehler bei Registration"
        )


@router.post(
    "/login",
    summary="User Login",
    description="Login mit Email und Passwort, setzt HttpOnly Cookies für JWT Tokens"
)
async def login(request: LoginRequest, http_request: Request, response: Response):
    """
    User Login mit Rate Limiting (5 Versuche pro Minute)

    - Validiert Email + Passwort
    - Gibt Access Token (15min) + Refresh Token (7d) zurück
    - Rate Limiting: 5 Versuche pro Minute pro IP
    """
    client_ip = http_request.client.host if http_request.client else "unknown"

    # Rate Limiting
    if not login_rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate Limit überschritten für Login von IP {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Zu viele Login-Versuche. Bitte warten Sie 1 Minute."
        )

    try:
        email_lower = request.email.lower()

        # User finden
        user = None
        for u in USERS_DB.values():
            if u["email"] == email_lower:
                user = u
                break

        if not user:
            logger.warning(f"Login-Versuch für unbekannte Email: {email_lower}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Ungültige Email oder Passwort"
            )

        # Passwort prüfen
        if not verify_password(request.password, user["password_hash"]):
            logger.warning(f"Falsches Passwort für User: {email_lower}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Ungültige Email oder Passwort"
            )

        # Login erfolgreich - Last Login aktualisieren
        user["last_login"] = datetime.now(timezone.utc)
        UserPersistence.save_users(USERS_DB)

        _sync_user_store(user["user_id"], user.get("username"), user.get("email"))

        # Tokens erstellen
        remember_me = bool(request.remember_me)
        config = JWTConfig()
        refresh_days = (
            config.get_refresh_token_remember_days()
            if remember_me
            else config.get_refresh_token_expire_days()
        )
        access_token = create_access_token(user["user_id"], email_lower)
        refresh_token = create_refresh_token(user["user_id"], email_lower, remember_me=remember_me)

        # Refresh Token speichern
        REFRESH_TOKENS[refresh_token] = {
            "user_id": user["user_id"],
            "created_at": datetime.now(timezone.utc)
        }
        UserPersistence.save_refresh_tokens(REFRESH_TOKENS)

        logger.info(f"Login erfolgreich für User: {email_lower}")

        # Rate Limiter zurücksetzen bei Erfolg
        login_rate_limiter.reset(client_ip)

        # SECURITY: Set HttpOnly cookies instead of returning tokens in JSON body
        # This prevents XSS attacks from stealing tokens via localStorage

        # Access Token Cookie (15 minutes)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,  # Cannot be accessed via JavaScript (XSS protection)
            secure=True,    # Only sent over HTTPS
            samesite="lax", # CSRF protection (can use "strict" for more security)
            max_age=900,    # 15 minutes in seconds
            path="/"        # Available for all routes
        )

        # Refresh Token Cookie (variable days)
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,  # Cannot be accessed via JavaScript (XSS protection)
            secure=True,    # Only sent over HTTPS
            samesite="lax", # CSRF protection
            max_age=refresh_days * 86400, # days in seconds
            path="/"        # Available for all routes
        )

        # User-Objekt für Frontend vorbereiten (ohne password_hash und OHNE TOKENS)
        user_data = {
            "user_id": user["user_id"],
            "email": user["email"],
            "username": user["username"],
            "display_name": user["username"],
            "created_at": user["created_at"].isoformat(),
            "last_login": user["last_login"].isoformat() if user.get("last_login") else None,
            "profile": user.get("profile", {})
        }

        # Return user data WITHOUT tokens (tokens are in cookies now!)
        return {
            "success": True,
            "message": "Login erfolgreich",
            "user": user_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler bei Login: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interner Server-Fehler bei Login"
        )


@router.post(
    "/refresh",
    summary="Refresh Access Token",
    description="Erneuert Access Token mit Refresh Token aus Cookie"
)
async def refresh_token(
    http_request: Request,
    response: Response,
    request: Optional[RefreshTokenRequest] = None
):
    """
    Erneuert Access Token mit gültigem Refresh Token

    - Liest Refresh Token aus Cookie (oder Request Body für Backwards Compatibility)
    - Validiert Refresh Token
    - Setzt neue HttpOnly Cookies
    - Alter Refresh Token wird invalidiert
    """
    try:
        # Read refresh token from cookie (preferred) or request body (backwards compatibility)
        refresh_token_value = http_request.cookies.get("refresh_token")

        # Fallback to request body if cookie not present (backwards compatibility)
        if not refresh_token_value and request and request.refresh_token:
            refresh_token_value = request.refresh_token

        if not refresh_token_value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh Token nicht gefunden (Cookie oder Body)"
            )

        # Refresh Token validieren
        payload = verify_token(refresh_token_value, token_type="refresh")
        user_id = payload.get("sub")

        # Prüfen ob Token nicht auf Blacklist
        if refresh_token_value not in REFRESH_TOKENS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh Token wurde widerrufen"
            )

        # User existiert?
        if user_id not in USERS_DB:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User nicht gefunden"
            )

        user = USERS_DB[user_id]
        email = user["email"]

        # Neue Tokens erstellen
        remember_me = bool(payload.get("remember", False))
        config = JWTConfig()
        refresh_days = (
            config.get_refresh_token_remember_days()
            if remember_me
            else config.get_refresh_token_expire_days()
        )
        new_access_token = create_access_token(user_id, email)
        new_refresh_token = create_refresh_token(user_id, email, remember_me=remember_me)

        # Alten Refresh Token invalidieren
        del REFRESH_TOKENS[refresh_token_value]

        # Neuen Refresh Token speichern
        REFRESH_TOKENS[new_refresh_token] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc)
        }
        UserPersistence.save_refresh_tokens(REFRESH_TOKENS)

        logger.info(f"Token erfolgreich erneuert für User: {email}")

        # SECURITY: Set new HttpOnly cookies
        # Access Token Cookie (15 minutes)
        response.set_cookie(
            key="access_token",
            value=new_access_token,
            httponly=True,  # XSS protection
            secure=True,    # HTTPS only
            samesite="lax", # CSRF protection
            max_age=900,    # 15 minutes
            path="/"
        )

        # Refresh Token Cookie (variable days)
        response.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            httponly=True,  # XSS protection
            secure=True,    # HTTPS only
            samesite="lax", # CSRF protection
            max_age=refresh_days * 86400, # days
            path="/"
        )

        # Return success WITHOUT tokens (tokens are in cookies now!)
        return {
            "success": True,
            "message": "Token erfolgreich erneuert",
            "user_id": user_id,
            "email": email
        }

    except Exception as e:
        logger.error(f"Fehler bei Token-Refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Ungültiger oder abgelaufener Refresh Token"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User Logout",
    description="Löscht HttpOnly Cookies und invalidiert Refresh Token"
)
async def logout(
    http_request: Request,
    response: Response,
    request: Optional[LogoutRequest] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    User Logout - Löscht Cookies und Invalidiert Refresh Token

    - Löscht access_token und refresh_token Cookies
    - Invalidiert Refresh Token in Datenbank
    - Client hat danach keinen Zugriff mehr
    """
    try:
        # Read refresh token from cookie or request body
        refresh_token_value = http_request.cookies.get("refresh_token")
        if not refresh_token_value and request and request.refresh_token:
            refresh_token_value = request.refresh_token

        # Refresh Token invalidieren wenn vorhanden
        if refresh_token_value and refresh_token_value in REFRESH_TOKENS:
            del REFRESH_TOKENS[refresh_token_value]
            UserPersistence.save_refresh_tokens(REFRESH_TOKENS)
            logger.info(f"Refresh Token invalidiert für User: {current_user['email']}")

        # SECURITY: Clear HttpOnly cookies by setting max_age=0
        response.delete_cookie(key="access_token", path="/")
        response.delete_cookie(key="refresh_token", path="/")

        logger.info(f"Logout erfolgreich für User: {current_user['email']}")
        return None  # 204 No Content

    except Exception as e:
        logger.error(f"Fehler bei Logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler bei Logout"
        )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get Current User Profile",
    description="Gibt Profil des aktuell eingeloggten Users zurück"
)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Gibt User-Profil zurück (erfordert gültigen Access Token)
    """
    return UserProfileResponse(
        user_id=current_user["user_id"],
        email=current_user["email"],
        username=current_user["username"],
        created_at=current_user["created_at"],
        profile=current_user.get("profile", {}),
        last_login=current_user.get("last_login")
    )


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change User Password",
    description="Ändert das Passwort des aktuell eingeloggten Users"
)
async def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ändert das Passwort des Users

    Args:
        request: PasswordChangeRequest mit old_password und new_password
        current_user: Aktuell eingeloggter User (aus JWT Token)

    Returns:
        Success message

    Raises:
        HTTPException 401: Wenn altes Passwort falsch ist
        HTTPException 400: Wenn neues Passwort ungültig ist
    """
    try:
        user_id = current_user["user_id"]

        # Verify old password
        if not verify_password(request.old_password, current_user["password_hash"]):
            logger.warning(f"Failed password change attempt for user {user_id}: incorrect old password")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Aktuelles Passwort ist falsch"
            )

        # Hash new password
        new_password_hash = hash_password(request.new_password)

        # Update user in database
        USERS_DB[user_id]["password_hash"] = new_password_hash
        USERS_DB[user_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Save to persistent storage
        UserPersistence.save_users(USERS_DB)

        logger.info(f"Password successfully changed for user {user_id}")

        return {
            "success": True,
            "message": "Passwort erfolgreich geändert",
            "user_id": user_id
        }

    except HTTPException:
        # Re-raise HTTP exceptions (like incorrect password)
        raise
    except Exception as e:
        logger.error(f"Error changing password for user {current_user.get('user_id')}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Ändern des Passworts"
        )
