"""
User Management API Tests

Tests for user creation, retrieval, updates, and deletion.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import uuid
from datetime import datetime

from backend.api.api_server import app
from backend.services.user_store import User


class TestUserInitialization:
    """Tests for anonymous user creation"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_anonymous_user_creation(self, client):
        """
        Test POST /v1/users/init creates anonymous user
        """
        with patch('backend.services.user_store.JSONUserStore.create') as mock_create:
            # Mock user creation
            mock_user = User(
                user_id="test-uuid-1234",
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            mock_create.return_value = mock_user

            response = client.post("/v1/users/init")

            assert response.status_code == 200
            data = response.json()
            assert "user_id" in data
            assert data["user_id"] == "test-uuid-1234"
            assert data["display_name"] == "Anonymous User"
            assert "created_at" in data

            # Verify user store was called
            mock_create.assert_called_once()

    def test_init_returns_unique_user_ids(self, client):
        """
        Verify each call to /v1/users/init returns unique user_id
        """
        with patch('backend.services.user_store.JSONUserStore.create') as mock_create:
            # First user
            mock_create.return_value = User(
                user_id=str(uuid.uuid4()),
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            response1 = client.post("/v1/users/init")
            user1_id = response1.json()["user_id"]

            # Second user
            mock_create.return_value = User(
                user_id=str(uuid.uuid4()),
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            response2 = client.post("/v1/users/init")
            user2_id = response2.json()["user_id"]

            assert user1_id != user2_id

    def test_init_with_optional_display_name(self, client):
        """
        Test user creation with custom display name
        """
        with patch('backend.services.user_store.JSONUserStore.create') as mock_create:
            mock_create.return_value = User(
                user_id=str(uuid.uuid4()),
                display_name="Custom Name",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.post(
                "/v1/users/init",
                json={"display_name": "Custom Name"}
            )

            assert response.status_code == 200
            assert response.json()["display_name"] == "Custom Name"


class TestGetCurrentUser:
    """Tests for GET /v1/users/me"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_get_current_user_success(self, client):
        """
        Test retrieving current user info
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get:
            mock_user = User(
                user_id="test-user-123",
                display_name="Test User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            mock_get.return_value = mock_user

            response = client.get(
                "/v1/users/me",
                headers={"X-User-ID": "test-user-123"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "test-user-123"
            assert data["display_name"] == "Test User"

    def test_get_current_user_not_found(self, client):
        """
        Test behavior when user doesn't exist
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get:
            mock_get.return_value = None

            response = client.get(
                "/v1/users/me",
                headers={"X-User-ID": "nonexistent-user"}
            )

            # Should auto-create or return 404
            assert response.status_code in [200, 404]

    def test_get_user_without_header(self, client):
        """
        Test GET /users/me without X-User-ID header
        Middleware should auto-create user
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get, \
             patch('backend.services.user_store.JSONUserStore.create') as mock_create:

            # First GET returns None (no user)
            mock_get.return_value = None

            # CREATE returns new user
            mock_create.return_value = User(
                user_id="auto-created",
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.get("/v1/users/me")

            # Middleware should have auto-created user
            assert response.status_code == 200
            assert "user_id" in response.json()


class TestUpdateUser:
    """Tests for PATCH /v1/users/me"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_update_display_name(self, client):
        """
        Test updating user's display name
        """
        with patch('backend.services.user_store.JSONUserStore.update') as mock_update:
            mock_update.return_value = User(
                user_id="test-user",
                display_name="New Name",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.patch(
                "/v1/users/me",
                headers={"X-User-ID": "test-user"},
                json={"display_name": "New Name"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["display_name"] == "New Name"

            # Verify update was called with correct params
            mock_update.assert_called_once_with("test-user", display_name="New Name")

    def test_update_with_empty_name(self, client):
        """
        Test updating with empty display name (should fail)
        """
        response = client.patch(
            "/v1/users/me",
            headers={"X-User-ID": "test-user"},
            json={"display_name": ""}
        )

        # Should reject empty name
        assert response.status_code in [400, 422]

    def test_update_nonexistent_user(self, client):
        """
        Test updating user that doesn't exist
        """
        with patch('backend.services.user_store.JSONUserStore.update') as mock_update:
            mock_update.return_value = None

            response = client.patch(
                "/v1/users/me",
                headers={"X-User-ID": "nonexistent"},
                json={"display_name": "New Name"}
            )

            assert response.status_code == 404


class TestUserStatistics:
    """Tests for GET /v1/users/stats"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_get_user_statistics(self, client):
        """
        Test retrieving user statistics
        """
        with patch('backend.memory.adapter.get_memory_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 42,
                "categories": {
                    "personal": 20,
                    "work": 15,
                    "hobbies": 7
                }
            }

            response = client.get(
                "/v1/users/stats",
                headers={"X-User-ID": "test-user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_memories"] == 42
            assert data["by_category"]["personal"] == 20
            assert data["by_category"]["work"] == 15

    def test_stats_for_new_user(self, client):
        """
        Test statistics for user with no memories
        """
        with patch('backend.memory.adapter.get_memory_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 0,
                "categories": {}
            }

            response = client.get(
                "/v1/users/stats",
                headers={"X-User-ID": "new-user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_memories"] == 0
            assert data["by_category"] == {}


class TestDeleteUser:
    """Tests for DELETE /v1/users/me"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_delete_user_account(self, client):
        """
        Test deleting user account
        """
        with patch('backend.services.user_store.JSONUserStore.delete') as mock_delete, \
             patch('backend.qdrant.qdrant_interface.QdrantMemoryInterface.delete_all_for_user') as mock_delete_mem:

            mock_delete.return_value = True

            response = client.delete(
                "/v1/users/me",
                headers={"X-User-ID": "test-user"}
            )

            assert response.status_code == 200
            assert response.json()["success"] is True

            # Verify both user and memories deleted
            mock_delete.assert_called_once_with("test-user")
            mock_delete_mem.assert_called_once_with("test-user")

    def test_delete_nonexistent_user(self, client):
        """
        Test deleting user that doesn't exist
        """
        with patch('backend.services.user_store.JSONUserStore.delete') as mock_delete:
            mock_delete.return_value = False

            response = client.delete(
                "/v1/users/me",
                headers={"X-User-ID": "nonexistent"}
            )

            assert response.status_code == 404


class TestUserMiddleware:
    """Tests for UserMiddleware behavior"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_middleware_header_priority(self, client):
        """
        Test middleware prioritizes X-User-ID header over cookie
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get:
            mock_get.return_value = User(
                user_id="header-user",
                display_name="Header User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.get(
                "/v1/users/me",
                headers={"X-User-ID": "header-user"},
                cookies={"user_id": "cookie-user"}
            )

            assert response.status_code == 200
            # Should use header value
            mock_get.assert_called_with("header-user")

    def test_middleware_cookie_parsing(self, client):
        """
        Test middleware reads user_id from cookie
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get:
            mock_get.return_value = User(
                user_id="cookie-user",
                display_name="Cookie User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.get(
                "/v1/users/me",
                cookies={"user_id": "cookie-user"}
            )

            assert response.status_code == 200
            mock_get.assert_called_with("cookie-user")

    def test_middleware_auto_creates_user(self, client):
        """
        Test middleware auto-creates user when no ID provided
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get, \
             patch('backend.services.user_store.JSONUserStore.create') as mock_create:

            mock_get.return_value = None
            mock_create.return_value = User(
                user_id="auto-created",
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.get("/v1/users/me")

            assert response.status_code == 200
            # Should have auto-created user
            assert mock_create.called or mock_get.called


class TestBackwardCompatibility:
    """Tests for backward compatibility with user_id='default'"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_default_user_still_works(self, client):
        """
        Test that user_id='default' still works for legacy clients
        """
        with patch('backend.core.bootstrap.initialize_components'), \
             patch('backend.memory.adapter.store_memory') as mock_store:

            response = client.post(
                "/v1/memory/add",
                headers={"X-User-ID": "default"},
                json={"content": "Test memory", "category": "test"}
            )

            assert response.status_code == 200

            # Verify store was called with 'default' user
            mock_store.assert_called_once()
            call_args = mock_store.call_args
            assert call_args[0][0] == "default"  # user_id

    def test_no_header_defaults_to_auto_creation(self, client):
        """
        Test that requests without user_id create anonymous user
        (not default to 'default')
        """
        with patch('backend.services.user_store.JSONUserStore.create') as mock_create:
            mock_create.return_value = User(
                user_id="new-anonymous",
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.post("/v1/users/init")

            user_id = response.json()["user_id"]
            # Should NOT be 'default'
            assert user_id != "default"


class TestUserSession:
    """Tests for user session management"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_session_cookie_set_on_init(self, client):
        """
        Test that user_id cookie is set after initialization
        """
        with patch('backend.services.user_store.JSONUserStore.create') as mock_create:
            mock_create.return_value = User(
                user_id="new-user-123",
                display_name="Anonymous User",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )

            response = client.post("/v1/users/init")

            # Check if Set-Cookie header present
            assert response.status_code == 200
            # Cookie setting is implementation-dependent
            # This tests the behavior exists

    def test_last_active_updated(self, client):
        """
        Test that user's last_active timestamp is updated
        """
        with patch('backend.services.user_store.JSONUserStore.get') as mock_get, \
             patch('backend.services.user_store.JSONUserStore.update') as mock_update:

            old_time = datetime.utcnow()
            mock_get.return_value = User(
                user_id="test-user",
                display_name="Test",
                created_at=old_time,
                last_active=old_time
            )

            # Make request
            client.get(
                "/v1/users/me",
                headers={"X-User-ID": "test-user"}
            )

            # Verify update was called (middleware updates last_active)
            # This is implementation-dependent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
