"""
User storage service for LexiAI.

Provides thread-safe user persistence with JSON file storage.
Designed with an interface to support future database migration.
"""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List, Any
from filelock import FileLock

from backend.models.user import User, UserTier

logger = logging.getLogger(__name__)


class UserStoreInterface(ABC):
    """
    Abstract interface for user storage.

    This interface allows easy migration from JSON file storage to a database
    (e.g., SQLite, PostgreSQL) in the future without changing business logic.
    """

    @abstractmethod
    def create_user(self, user: User) -> User:
        """Create a new user."""
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve user by ID."""
        pass

    @abstractmethod
    def update_user(self, user_id: str, updates: Dict) -> Optional[User]:
        """Update user attributes."""
        pass

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        pass

    @abstractmethod
    def list_users(self, tier: Optional[UserTier] = None) -> List[User]:
        """List all users, optionally filtered by tier."""
        pass

    @abstractmethod
    def update_last_seen(self, user_id: str) -> None:
        """Update user's last_seen timestamp."""
        pass


class JSONUserStore(UserStoreInterface):
    """
    Thread-safe JSON file-based user storage.

    Uses filelock to ensure atomic read/write operations.
    Stores users in backend/data/users.json.
    """

    def __init__(self, data_dir: str = "backend/data"):
        """
        Initialize JSON user store.

        Args:
            data_dir: Directory to store users.json file
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.users_file = self.data_dir / "users.json"
        self.lock_file = self.data_dir / "users.json.lock"

        # Initialize file if it doesn't exist
        if not self.users_file.exists():
            self._write_data({"users": {}})
            logger.info(f"Created new users database at {self.users_file}")
        else:
            logger.info(f"Using existing users database at {self.users_file}")

    def _read_data(self) -> Dict:
        """
        Read users data from JSON file with file locking.

        Returns:
            Dictionary with 'users' key containing user_id -> user_data mapping
        """
        with FileLock(str(self.lock_file), timeout=10):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Corrupted JSON file at {self.users_file}, reinitializing")
                return {"users": {}}
            except Exception as e:
                logger.error(f"Error reading users file: {e}")
                return {"users": {}}

    def _write_data(self, data: Dict) -> None:
        """
        Write users data to JSON file with file locking.

        Args:
            data: Dictionary with 'users' key
        """
        with FileLock(str(self.lock_file), timeout=10):
            try:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = self.users_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                temp_file.replace(self.users_file)
            except Exception as e:
                logger.error(f"Error writing users file: {e}")
                raise

    def create_user(self, user: User) -> User:
        """
        Create a new user.

        Args:
            user: User object to create

        Returns:
            Created user object

        Raises:
            ValueError: If user_id already exists
        """
        data = self._read_data()

        if user.user_id in data["users"]:
            raise ValueError(f"User {user.user_id} already exists")

        data["users"][user.user_id] = user.model_dump()
        self._write_data(data)

        logger.info(f"Created user {user.user_id} ({user.tier})")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Retrieve user by ID.

        Args:
            user_id: User identifier

        Returns:
            User object if found, None otherwise
        """
        data = self._read_data()
        user_data = data["users"].get(user_id)

        if user_data is None:
            return None

        return User(**user_data)

    def update_user(self, user_id: str, updates: Dict) -> Optional[User]:
        """
        Update user attributes.

        Args:
            user_id: User identifier
            updates: Dictionary of fields to update

        Returns:
            Updated User object if found, None otherwise
        """
        data = self._read_data()

        if user_id not in data["users"]:
            return None

        # Update fields
        data["users"][user_id].update(updates)
        self._write_data(data)

        logger.info(f"Updated user {user_id}: {list(updates.keys())}")
        return User(**data["users"][user_id])

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        data = self._read_data()

        if user_id not in data["users"]:
            return False

        del data["users"][user_id]
        self._write_data(data)

        logger.info(f"Deleted user {user_id}")
        return True

    def list_users(self, tier: Optional[UserTier] = None) -> List[User]:
        """
        List all users, optionally filtered by tier.

        Args:
            tier: Optional tier filter

        Returns:
            List of User objects
        """
        data = self._read_data()
        users = [User(**user_data) for user_data in data["users"].values()]

        if tier is not None:
            users = [u for u in users if u.tier == tier]

        return users

    def update_last_seen(self, user_id: str) -> None:
        """
        Update user's last_seen timestamp to current UTC time.

        Args:
            user_id: User identifier
        """
        now = datetime.now(timezone.utc).isoformat()
        data = self._read_data()

        if user_id in data["users"]:
            data["users"][user_id]["last_seen"] = now
            self._write_data(data)


def generate_anonymous_user() -> User:
    """
    Generate a new anonymous user with UUID v4.

    Returns:
        New anonymous User object
    """
    now = datetime.now(timezone.utc).isoformat()
    user_id = str(uuid.uuid4())

    return User(
        user_id=user_id,
        display_name="Anonymous User",
        created_at=now,
        last_seen=now,
        tier=UserTier.ANONYMOUS,
        preferences={},
        email=None
    )


# Global user store instance (singleton pattern)
_user_store: Optional[JSONUserStore] = None


def get_user_store() -> JSONUserStore:
    """
    Get global user store instance.

    Returns:
        JSONUserStore singleton instance
    """
    global _user_store
    if _user_store is None:
        _user_store = JSONUserStore()
    return _user_store


def build_user_profile_context(user_id: str) -> Dict[str, Any]:
    """
    Build a user profile context dict for personalization prompts.

    Includes the stored profile plus a display name if available.
    """
    user = get_user_store().get_user(user_id)
    if not user:
        return {}

    profile = dict(user.profile) if user.profile else {}
    display_name = (user.display_name or "").strip()
    if display_name and display_name.lower() not in {"anonymous user", "anonymer benutzer"}:
        profile.setdefault("user_profile_name", display_name)
    return profile
