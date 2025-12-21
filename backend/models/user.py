"""
User data models for LexiAI.

This module defines the User model and related Pydantic schemas for the user management system.
Supports multi-tier user system (anonymous, registered, premium).
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class UserTier(str, Enum):
    """User tier levels."""
    ANONYMOUS = "anonymous"
    REGISTERED = "registered"
    PREMIUM = "premium"


class User(BaseModel):
    """
    User model representing a LexiAI user.

    Attributes:
        user_id: Unique identifier (UUID v4)
        display_name: User's display name
        created_at: ISO 8601 timestamp of user creation
        last_seen: ISO 8601 timestamp of last activity
        tier: User tier (anonymous/registered/premium)
        preferences: User-specific preferences (e.g., language, theme)
        email: Email address (for registered users only)
        password_hash: Bcrypt password hash (for authentication)
        profile: Auto-learned user profile dictionary
        last_login: ISO 8601 timestamp of last login
    """
    user_id: str = Field(..., description="Unique user identifier (UUID v4)")
    display_name: str = Field(default="Anonymous User", description="User's display name")
    created_at: str = Field(..., description="ISO 8601 timestamp of creation")
    last_seen: str = Field(..., description="ISO 8601 timestamp of last activity")
    tier: UserTier = Field(default=UserTier.ANONYMOUS, description="User tier level")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    email: Optional[str] = Field(default=None, description="Email address (registered users only)")

    # Authentication fields
    password_hash: Optional[str] = Field(default=None, description="Bcrypt password hash (60 chars)")

    # Profile learning fields
    profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auto-learned user profile (occupation, interests, preferences, etc.)"
    )

    last_login: Optional[str] = Field(default=None, description="ISO 8601 timestamp of last login")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "display_name": "Sarah",
                "created_at": "2025-01-22T10:30:00Z",
                "last_seen": "2025-01-22T15:45:30Z",
                "tier": "anonymous",
                "preferences": {
                    "language": "de",
                    "theme": "dark"
                },
                "email": None
            }
        }


class UserCreateRequest(BaseModel):
    """Request model for creating a user (init endpoint)."""
    display_name: Optional[str] = Field(default="Anonymous User", description="Optional display name")


class UserUpdateRequest(BaseModel):
    """Request model for updating user profile."""
    display_name: Optional[str] = Field(default=None, description="New display name")
    preferences: Optional[Dict[str, Any]] = Field(default=None, description="Updated preferences")


class UserResponse(BaseModel):
    """Response model for user endpoints."""
    user: User
    message: Optional[str] = Field(default=None, description="Optional status message")


class UserStatsResponse(BaseModel):
    """Response model for user statistics."""
    user_id: str
    total_memories: int = Field(default=0, description="Total number of memories")
    categories: Dict[str, int] = Field(default_factory=dict, description="Memory count by category")
    last_activity: Optional[str] = Field(default=None, description="Timestamp of last memory activity")
    account_age_days: int = Field(default=0, description="Days since account creation")
