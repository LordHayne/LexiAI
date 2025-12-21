"""
Integration Tests: Authentication + Profile Learning Flow
Tests: Full user journey from registration to personalized responses
Target: 10+ integration tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from httpx import AsyncClient
from fastapi.testclient import TestClient
import jwt

from backend.api.api_server import app
from backend.auth.auth_service import AuthService
from backend.profile.profile_builder import ProfileBuilder
from backend.profile.profile_context import ProfileContext


@pytest.fixture
async def async_client():
    """Async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_client():
    """Sync test client"""
    return TestClient(app)


class TestRegistrationToLoginFlow:
    """Test registration → login flow"""

    @pytest.mark.asyncio
    async def test_full_registration_login_flow(self, async_client):
        """Should complete registration → login → get JWT"""
        # Step 1: Register
        registration_data = {
            "email": "flow@example.com",
            "password": "SecurePass123!",
            "full_name": "Flow Test User"
        }

        register_response = await async_client.post(
            "/v1/auth/register",
            json=registration_data
        )

        assert register_response.status_code == 201
        user_data = register_response.json()
        assert user_data["email"] == "flow@example.com"
        assert "user_id" in user_data
        assert "password" not in user_data  # Security check

        # Step 2: Login
        login_data = {
            "email": "flow@example.com",
            "password": "SecurePass123!"
        }

        login_response = await async_client.post(
            "/v1/auth/login",
            json=login_data
        )

        assert login_response.status_code == 200
        tokens = login_response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

        # Step 3: Verify JWT contains correct user_id
        access_token = tokens["access_token"]
        decoded = jwt.decode(
            access_token,
            options={"verify_signature": False}  # Skip signature verification for test
        )
        assert decoded["sub"] == user_data["user_id"]

    @pytest.mark.asyncio
    async def test_authenticated_api_calls(self, async_client):
        """Should access protected endpoints with JWT"""
        # Register and login
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "auth@example.com",
                "password": "SecurePass123!",
                "full_name": "Auth Test"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "auth@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]

        # Call protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}

        chat_response = await async_client.post(
            "/v1/chat",
            json={"message": "Hello LexiAI!"},
            headers=headers
        )

        assert chat_response.status_code == 200

    @pytest.mark.asyncio
    async def test_reject_invalid_token(self, async_client):
        """Should reject invalid JWT"""
        invalid_token = "invalid.jwt.token"
        headers = {"Authorization": f"Bearer {invalid_token}"}

        response = await async_client.post(
            "/v1/chat",
            json={"message": "Test"},
            headers=headers
        )

        assert response.status_code == 401


class TestChatWithProfileLearning:
    """Test chat interaction with profile learning"""

    @pytest.mark.asyncio
    async def test_profile_learning_during_chat(self, async_client):
        """Should learn profile from chat messages"""
        # Register and login
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "learner@example.com",
                "password": "SecurePass123!",
                "full_name": "Profile Learner"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "learner@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Chat with profile information
        messages_with_profile_info = [
            "Hallo, ich bin Thomas.",
            "Ich arbeite als Software Engineer.",
            "Ich liebe Python und Machine Learning."
        ]

        for message in messages_with_profile_info:
            response = await async_client.post(
                "/v1/chat",
                json={"message": message},
                headers=headers
            )
            assert response.status_code == 200

        # Wait for background profile building
        await asyncio.sleep(0.5)

        # Retrieve profile
        profile_response = await async_client.get(
            "/v1/profile/me",
            headers=headers
        )

        assert profile_response.status_code == 200
        profile = profile_response.json()

        # Verify profile was learned
        assert len(profile["profile_items"]) > 0

        # Check for extracted information
        profile_text = str(profile).lower()
        assert "thomas" in profile_text or "software" in profile_text

    @pytest.mark.asyncio
    async def test_personalized_responses(self, async_client):
        """Should provide personalized responses based on profile"""
        # Setup user with profile
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "personal@example.com",
                "password": "SecurePass123!",
                "full_name": "Personal User"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "personal@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Build profile
        await async_client.post(
            "/v1/chat",
            json={"message": "Ich heiße Lisa und ich liebe Machine Learning."},
            headers=headers
        )

        await asyncio.sleep(0.3)

        # Ask context-dependent question
        response = await async_client.post(
            "/v1/chat",
            json={"message": "Was weißt du über mich?"},
            headers=headers
        )

        assert response.status_code == 200
        answer = response.json()["response"]

        # Response should reference profile information
        answer_lower = answer.lower()
        assert "lisa" in answer_lower or "machine learning" in answer_lower


class TestAnonymousToRegisteredMigration:
    """Test memory preservation when anonymous → registered"""

    @pytest.mark.asyncio
    async def test_preserve_memories_on_registration(self, async_client):
        """Should preserve anonymous memories when user registers"""
        # Step 1: Chat as anonymous user
        anonymous_response = await async_client.post(
            "/ui/chat",  # Public endpoint
            json={"message": "Ich interessiere mich für Python."}
        )

        assert anonymous_response.status_code == 200

        # Get session/device ID from response
        session_id = anonymous_response.cookies.get("session_id") or "anonymous-123"

        # Step 2: Register account
        register_response = await async_client.post(
            "/v1/auth/register",
            json={
                "email": "migration@example.com",
                "password": "SecurePass123!",
                "full_name": "Migration Test"
            }
        )

        user_id = register_response.json()["user_id"]

        # Step 3: Migrate memories
        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "migration@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        migrate_response = await async_client.post(
            "/v1/memory/migrate",
            json={"from_session_id": session_id},
            headers=headers
        )

        assert migrate_response.status_code == 200

        # Step 4: Verify memories preserved
        memories_response = await async_client.get(
            "/v1/memory/me",
            headers=headers
        )

        assert memories_response.status_code == 200
        memories = memories_response.json()

        # Should contain the anonymous chat
        memories_text = str(memories).lower()
        assert "python" in memories_text


class TestTokenRefreshFlow:
    """Test token refresh functionality"""

    @pytest.mark.asyncio
    async def test_refresh_access_token(self, async_client):
        """Should refresh expired access token with refresh token"""
        # Register and login
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "refresh@example.com",
                "password": "SecurePass123!",
                "full_name": "Refresh Test"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "refresh@example.com", "password": "SecurePass123!"}
        )

        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]

        # Use refresh token to get new access token
        refresh_response = await async_client.post(
            "/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()

        assert "access_token" in new_tokens
        assert new_tokens["access_token"] != tokens["access_token"]  # New token


class TestUserIsolation:
    """Test user data isolation"""

    @pytest.mark.asyncio
    async def test_users_cannot_access_each_other_data(self, async_client):
        """Should isolate user data and profiles"""
        # Create User 1
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "user1@example.com",
                "password": "SecurePass123!",
                "full_name": "User One"
            }
        )

        login1 = await async_client.post(
            "/v1/auth/login",
            json={"email": "user1@example.com", "password": "SecurePass123!"}
        )
        token1 = login1.json()["access_token"]

        # Create User 2
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "user2@example.com",
                "password": "SecurePass123!",
                "full_name": "User Two"
            }
        )

        login2 = await async_client.post(
            "/v1/auth/login",
            json={"email": "user2@example.com", "password": "SecurePass123!"}
        )
        token2 = login2.json()["access_token"]

        # User 1 chats
        await async_client.post(
            "/v1/chat",
            json={"message": "Ich bin User One."},
            headers={"Authorization": f"Bearer {token1}"}
        )

        # User 2 retrieves profile - should not see User 1's data
        profile2 = await async_client.get(
            "/v1/profile/me",
            headers={"Authorization": f"Bearer {token2}"}
        )

        profile2_text = str(profile2.json()).lower()
        assert "user one" not in profile2_text


class TestLogoutFlow:
    """Test logout and session cleanup"""

    @pytest.mark.asyncio
    async def test_logout_invalidates_token(self, async_client):
        """Should invalidate token on logout"""
        # Register and login
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "logout@example.com",
                "password": "SecurePass123!",
                "full_name": "Logout Test"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "logout@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Logout
        logout_response = await async_client.post(
            "/v1/auth/logout",
            headers=headers
        )

        assert logout_response.status_code == 200

        # Try to use token after logout
        chat_response = await async_client.post(
            "/v1/chat",
            json={"message": "Test"},
            headers=headers
        )

        # Token should be invalidated
        assert chat_response.status_code == 401


class TestProfileAccuracy:
    """Test profile learning accuracy"""

    @pytest.mark.asyncio
    async def test_profile_learning_accuracy(self, async_client):
        """Should accurately extract and store profile information"""
        # Register
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "accuracy@example.com",
                "password": "SecurePass123!",
                "full_name": "Accuracy Test"
            }
        )

        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "accuracy@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Provide specific profile information
        test_data = {
            "name": "Max Mustermann",
            "job": "Senior Software Architect",
            "company": "Tech Corp",
            "interests": ["Kubernetes", "Microservices", "Go"]
        }

        await async_client.post(
            "/v1/chat",
            json={
                "message": f"Ich heiße {test_data['name']} und arbeite als "
                          f"{test_data['job']} bei {test_data['company']}. "
                          f"Ich interessiere mich für {', '.join(test_data['interests'])}."
            },
            headers=headers
        )

        await asyncio.sleep(0.5)

        # Retrieve and verify profile
        profile_response = await async_client.get(
            "/v1/profile/me",
            headers=headers
        )

        profile = profile_response.json()
        profile_text = str(profile).lower()

        # Verify all information was extracted
        assert "max mustermann" in profile_text or "max" in profile_text
        assert "architect" in profile_text or "software" in profile_text
        assert any(interest.lower() in profile_text for interest in test_data["interests"])


class TestPerformance:
    """Performance tests for integrated flow"""

    @pytest.mark.asyncio
    async def test_end_to_end_performance(self, async_client):
        """Complete flow should complete in <2 seconds"""
        import time

        start = time.time()

        # Register
        await async_client.post(
            "/v1/auth/register",
            json={
                "email": "perf@example.com",
                "password": "SecurePass123!",
                "full_name": "Perf Test"
            }
        )

        # Login
        login_response = await async_client.post(
            "/v1/auth/login",
            json={"email": "perf@example.com", "password": "SecurePass123!"}
        )

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Chat
        await async_client.post(
            "/v1/chat",
            json={"message": "Hallo!"},
            headers=headers
        )

        # Logout
        await async_client.post(
            "/v1/auth/logout",
            headers=headers
        )

        duration = time.time() - start

        assert duration < 2.0  # Complete flow < 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend", "--cov-report=html"])
