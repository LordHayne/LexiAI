"""
Pydantic Models für Authentication
Login, Register, Token Request/Response Models
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator
from backend.utils.password_utils import validate_password_strength, validate_email_format


class RegisterRequest(BaseModel):
    """User Registration Request"""

    email: EmailStr = Field(
        ...,
        description="Email-Adresse des Users",
        examples=["user@example.com"]
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Passwort (min. 8 Zeichen, 1 Großbuchstabe, 1 Ziffer)",
        examples=["SecurePass123"]
    )

    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        description="Username (optional, wird aus Email generiert wenn leer)",
        examples=["johndoe"]
    )

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validiert Passwort-Stärke"""
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Zusätzliche Email-Validierung"""
        is_valid, error_msg = validate_email_format(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v.lower()  # Normalisiere zu Kleinbuchstaben


class LoginRequest(BaseModel):
    """User Login Request"""

    email: EmailStr = Field(
        ...,
        description="Email-Adresse",
        examples=["user@example.com"]
    )

    password: str = Field(
        ...,
        description="Passwort",
        examples=["SecurePass123"]
    )

    @field_validator('email')
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalisiere Email zu Kleinbuchstaben"""
        return v.lower()


class TokenResponse(BaseModel):
    """JWT Token Response"""

    access_token: str = Field(
        ...,
        description="JWT Access Token (15 Minuten gültig)"
    )

    refresh_token: str = Field(
        ...,
        description="JWT Refresh Token (7 Tage gültig)"
    )

    token_type: str = Field(
        default="Bearer",
        description="Token-Typ für Authorization Header"
    )

    expires_in: int = Field(
        default=900,  # 15 Minuten in Sekunden
        description="Access Token Gültigkeitsdauer in Sekunden"
    )

    user_id: str = Field(
        ...,
        description="Eindeutige User-ID"
    )

    email: str = Field(
        ...,
        description="User Email"
    )


class RefreshTokenRequest(BaseModel):
    """Token Refresh Request"""

    refresh_token: str = Field(
        ...,
        description="Gültiger Refresh Token"
    )


class LogoutRequest(BaseModel):
    """Logout Request (optional body)"""

    refresh_token: Optional[str] = Field(
        None,
        description="Refresh Token zum Invalidieren (optional)"
    )


class UserProfileResponse(BaseModel):
    """User Profile Response"""

    user_id: str = Field(..., description="Eindeutige User-ID")
    email: str = Field(..., description="User Email")
    username: str = Field(..., description="Username")
    created_at: datetime = Field(..., description="Account-Erstellungsdatum")

    profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-Profil (automatisch gelernt)"
    )

    last_login: Optional[datetime] = Field(
        None,
        description="Letzter Login"
    )


class PasswordChangeRequest(BaseModel):
    """Password Change Request"""

    old_password: str = Field(
        ...,
        description="Aktuelles Passwort"
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Neues Passwort"
    )

    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validiert neues Passwort"""
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v


class PasswordResetRequest(BaseModel):
    """Password Reset Request (ohne altes Passwort)"""

    email: EmailStr = Field(
        ...,
        description="Email-Adresse für Reset"
    )


class PasswordResetConfirm(BaseModel):
    """Password Reset Confirmation mit Token"""

    reset_token: str = Field(
        ...,
        description="Reset Token aus Email"
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Neues Passwort"
    )

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validiert neues Passwort"""
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v


class AuthErrorResponse(BaseModel):
    """Standard Authentication Error Response"""

    error: str = Field(
        ...,
        description="Error-Typ",
        examples=["invalid_credentials", "token_expired", "user_exists"]
    )

    message: str = Field(
        ...,
        description="Detaillierte Fehlerbeschreibung"
    )

    status_code: int = Field(
        ...,
        description="HTTP Status Code"
    )
