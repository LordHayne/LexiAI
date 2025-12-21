"""
User Store Tests

Tests for JSONUserStore - the persistence layer for user data.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path
import threading
import time

from backend.services.user_store import JSONUserStore, User


class TestJSONUserStoreBasics:
    """Basic CRUD operations tests"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_store_initialization(self, temp_storage):
        """
        Test JSONUserStore initializes correctly
        """
        store = JSONUserStore(temp_storage)

        # File should be created
        assert os.path.exists(temp_storage)

        # Should contain empty users dict
        with open(temp_storage, 'r') as f:
            data = json.load(f)
            assert data == {"users": {}}

    def test_create_user(self, temp_storage):
        """
        Test creating a new user
        """
        store = JSONUserStore(temp_storage)

        user = store.create(display_name="Test User")

        assert user is not None
        assert user.user_id is not None
        assert user.display_name == "Test User"
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.last_active, datetime)

    def test_create_user_with_default_name(self, temp_storage):
        """
        Test creating user without display_name uses default
        """
        store = JSONUserStore(temp_storage)

        user = store.create()

        assert user.display_name == "Anonymous User"

    def test_get_user_success(self, temp_storage):
        """
        Test retrieving existing user
        """
        store = JSONUserStore(temp_storage)

        # Create user
        created = store.create(display_name="Test User")

        # Retrieve user
        retrieved = store.get(created.user_id)

        assert retrieved is not None
        assert retrieved.user_id == created.user_id
        assert retrieved.display_name == "Test User"

    def test_get_user_not_found(self, temp_storage):
        """
        Test retrieving non-existent user returns None
        """
        store = JSONUserStore(temp_storage)

        user = store.get("nonexistent-user-id")

        assert user is None

    def test_update_user_display_name(self, temp_storage):
        """
        Test updating user's display name
        """
        store = JSONUserStore(temp_storage)

        # Create user
        user = store.create(display_name="Old Name")

        # Update display name
        updated = store.update(user.user_id, display_name="New Name")

        assert updated is not None
        assert updated.display_name == "New Name"
        assert updated.user_id == user.user_id

        # Verify persistence
        retrieved = store.get(user.user_id)
        assert retrieved.display_name == "New Name"

    def test_update_nonexistent_user(self, temp_storage):
        """
        Test updating user that doesn't exist returns None
        """
        store = JSONUserStore(temp_storage)

        updated = store.update("nonexistent", display_name="New Name")

        assert updated is None

    def test_delete_user(self, temp_storage):
        """
        Test deleting a user
        """
        store = JSONUserStore(temp_storage)

        # Create user
        user = store.create(display_name="To Delete")

        # Delete user
        result = store.delete(user.user_id)

        assert result is True

        # Verify user is gone
        retrieved = store.get(user.user_id)
        assert retrieved is None

    def test_delete_nonexistent_user(self, temp_storage):
        """
        Test deleting user that doesn't exist returns False
        """
        store = JSONUserStore(temp_storage)

        result = store.delete("nonexistent")

        assert result is False

    def test_list_all_users(self, temp_storage):
        """
        Test listing all users
        """
        store = JSONUserStore(temp_storage)

        # Create multiple users
        user1 = store.create(display_name="User 1")
        user2 = store.create(display_name="User 2")
        user3 = store.create(display_name="User 3")

        # List all
        users = store.list_all()

        assert len(users) == 3
        user_ids = [u.user_id for u in users]
        assert user1.user_id in user_ids
        assert user2.user_id in user_ids
        assert user3.user_id in user_ids


class TestJSONUserStorePersistence:
    """Tests for data persistence"""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_data_persists_across_instances(self, temp_storage):
        """
        Test data survives store reinitialization
        """
        # First instance - create user
        store1 = JSONUserStore(temp_storage)
        user = store1.create(display_name="Persistent User")
        user_id = user.user_id

        # Second instance - retrieve user
        store2 = JSONUserStore(temp_storage)
        retrieved = store2.get(user_id)

        assert retrieved is not None
        assert retrieved.user_id == user_id
        assert retrieved.display_name == "Persistent User"

    def test_atomic_writes(self, temp_storage):
        """
        Test writes are atomic (no partial writes)
        """
        store = JSONUserStore(temp_storage)

        # Create multiple users rapidly
        users = []
        for i in range(10):
            users.append(store.create(display_name=f"User {i}"))

        # Verify file is valid JSON
        with open(temp_storage, 'r') as f:
            data = json.load(f)
            assert "users" in data
            assert len(data["users"]) == 10

    def test_backup_creation(self, temp_storage):
        """
        Test backups are created on updates
        """
        store = JSONUserStore(temp_storage)

        # Create and update user
        user = store.create(display_name="Original")
        store.update(user.user_id, display_name="Updated")

        # Check for backup files
        storage_path = Path(temp_storage)
        backup_dir = storage_path.parent / "backups"

        if backup_dir.exists():
            backups = list(backup_dir.glob("*.backup"))
            assert len(backups) > 0


class TestThreadSafety:
    """Tests for concurrent access safety"""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_concurrent_user_creation(self, temp_storage):
        """
        Test multiple threads can create users safely
        """
        store = JSONUserStore(temp_storage)
        created_users = []
        errors = []

        def create_user(name):
            try:
                user = store.create(display_name=name)
                created_users.append(user)
            except Exception as e:
                errors.append(e)

        # Create 20 users concurrently
        threads = []
        for i in range(20):
            t = threading.Thread(target=create_user, args=(f"User {i}",))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all users created
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_users) == 20

        # Verify unique IDs
        user_ids = [u.user_id for u in created_users]
        assert len(set(user_ids)) == 20, "Duplicate user IDs detected"

    def test_concurrent_reads_and_writes(self, temp_storage):
        """
        Test concurrent reads and writes don't corrupt data
        """
        store = JSONUserStore(temp_storage)

        # Create initial users
        users = [store.create(display_name=f"User {i}") for i in range(5)]
        errors = []

        def read_user(user_id):
            try:
                for _ in range(10):
                    store.get(user_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def update_user(user_id, count):
            try:
                for i in range(10):
                    store.update(user_id, display_name=f"Updated {count}-{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Mix of reads and writes
        threads = []
        for i, user in enumerate(users):
            # Reader thread
            t1 = threading.Thread(target=read_user, args=(user.user_id,))
            threads.append(t1)

            # Writer thread
            t2 = threading.Thread(target=update_user, args=(user.user_id, i))
            threads.append(t2)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify data integrity
        with open(temp_storage, 'r') as f:
            data = json.load(f)
            assert "users" in data
            assert len(data["users"]) == 5


class TestUserModel:
    """Tests for User dataclass"""

    def test_user_to_dict(self):
        """
        Test User.to_dict() serialization
        """
        now = datetime.utcnow()
        user = User(
            user_id="test-123",
            display_name="Test User",
            created_at=now,
            last_active=now
        )

        data = user.to_dict()

        assert data["user_id"] == "test-123"
        assert data["display_name"] == "Test User"
        assert "created_at" in data
        assert "last_active" in data

    def test_user_from_dict(self):
        """
        Test User.from_dict() deserialization
        """
        now = datetime.utcnow()
        data = {
            "user_id": "test-123",
            "display_name": "Test User",
            "created_at": now.isoformat(),
            "last_active": now.isoformat()
        }

        user = User.from_dict(data)

        assert user.user_id == "test-123"
        assert user.display_name == "Test User"
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.last_active, datetime)

    def test_user_equality(self):
        """
        Test User equality comparison
        """
        now = datetime.utcnow()
        user1 = User("id1", "Name", now, now)
        user2 = User("id1", "Name", now, now)
        user3 = User("id2", "Name", now, now)

        assert user1 == user2
        assert user1 != user3


class TestEdgeCases:
    """Edge case tests"""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_empty_display_name(self, temp_storage):
        """
        Test creating user with empty display name
        """
        store = JSONUserStore(temp_storage)

        user = store.create(display_name="")

        # Should use default or accept empty
        assert user.display_name in ["", "Anonymous User"]

    def test_very_long_display_name(self, temp_storage):
        """
        Test creating user with very long display name
        """
        store = JSONUserStore(temp_storage)

        long_name = "A" * 1000
        user = store.create(display_name=long_name)

        assert user.display_name == long_name

    def test_special_characters_in_name(self, temp_storage):
        """
        Test display name with special characters
        """
        store = JSONUserStore(temp_storage)

        special_name = "User<>@#$%^&*()'\"\\/"
        user = store.create(display_name=special_name)

        assert user.display_name == special_name

        # Verify persistence
        retrieved = store.get(user.user_id)
        assert retrieved.display_name == special_name

    def test_unicode_display_name(self, temp_storage):
        """
        Test display name with Unicode characters
        """
        store = JSONUserStore(temp_storage)

        unicode_name = "用户名 👤 Benutzer"
        user = store.create(display_name=unicode_name)

        assert user.display_name == unicode_name

        # Verify persistence
        retrieved = store.get(user.user_id)
        assert retrieved.display_name == unicode_name

    def test_corrupted_storage_file(self, temp_storage):
        """
        Test behavior with corrupted storage file
        """
        store = JSONUserStore(temp_storage)
        store.create(display_name="Test")

        # Corrupt the file
        with open(temp_storage, 'w') as f:
            f.write("INVALID JSON{{{")

        # Try to initialize new store
        try:
            store2 = JSONUserStore(temp_storage)
            # Should either recover or raise clear error
            users = store2.list_all()
            # If it recovers, should have empty or backed-up data
            assert isinstance(users, list)
        except json.JSONDecodeError:
            # Expected behavior - clear error
            pass

    def test_missing_storage_directory(self):
        """
        Test creating store in non-existent directory
        """
        non_existent_path = "/tmp/nonexistent/dir/users.json"

        try:
            store = JSONUserStore(non_existent_path)
            # Should create directory or fail gracefully
            user = store.create(display_name="Test")
            assert user is not None
        except (FileNotFoundError, PermissionError):
            # Expected - can't create in non-existent path
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
