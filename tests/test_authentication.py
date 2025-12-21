"""
Comprehensive Authentication Tests for LexiAI
Tests: Registration, Login, JWT, Password Hashing, Rate Limiting
Target: 20+ tests with >95% coverage
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
import jwt
import bcrypt
from fastapi import HTTPException, status

from backend.auth.auth_service import AuthService
from backend.auth.models import (
    UserRegistration,
    UserLogin,
    User,
    TokenPair,
    TokenRefresh
)
from backend.config.auth_config import AuthConfig


@pytest.fixture
def auth_config():
    """Mock auth configuration"""
    config = MagicMock()
    config.jwt_secret = "test_secret_key_change_in_production"
    config.jwt_algorithm = "HS256"
    config.access_token_expire_minutes = 30
    config.refresh_token_expire_days = 7
    config.password_min_length = 8
    config.max_login_attempts = 5
    config.lockout_duration_minutes = 15
    return config


@pytest.fixture
def mock_db():
    """Mock database operations"""
    db = MagicMock()
    db.users = {}
    db.login_attempts = {}
    return db


@pytest.fixture
def auth_service(auth_config, mock_db):
    """Create AuthService instance with mocks"""
    service = AuthService(auth_config)
    service.db = mock_db
    return service


class TestUserRegistration:
    """Test user registration functionality"""

    @pytest.mark.asyncio
    async def test_register_valid_user(self, auth_service, mock_db):
        """Should successfully register a user with valid data"""
        registration = UserRegistration(
            email="test@example.com",
            password="SecurePass123!",
            full_name="Test User"
        )

        user = await auth_service.register_user(registration)

        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.user_id is not None
        assert user.created_at is not None
        assert "password" not in user.dict()  # Password never exposed

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, auth_service, mock_db):
        """Should reject duplicate email registration"""
        registration = UserRegistration(
            email="duplicate@example.com",
            password="SecurePass123!",
            full_name="User One"
        )

        # Register first user
        await auth_service.register_user(registration)

        # Try to register with same email
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(registration)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "email already registered" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_register_weak_password(self, auth_service):
        """Should reject weak passwords"""
        registration = UserRegistration(
            email="test@example.com",
            password="weak",  # Too short
            full_name="Test User"
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(registration)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, auth_service):
        """Should reject invalid email formats"""
        registration = UserRegistration(
            email="not-an-email",
            password="SecurePass123!",
            full_name="Test User"
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(registration)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_password_is_hashed(self, auth_service, mock_db):
        """Should hash password before storing"""
        registration = UserRegistration(
            email="test@example.com",
            password="PlainTextPassword123!",
            full_name="Test User"
        )

        user = await auth_service.register_user(registration)

        # Retrieve stored password hash
        stored_hash = mock_db.users[user.user_id]["password_hash"]

        # Verify it's bcrypt hash (starts with $2b$)
        assert stored_hash.startswith("$2b$")
        assert stored_hash != "PlainTextPassword123!"

        # Verify password can be checked
        assert bcrypt.checkpw(
            "PlainTextPassword123!".encode('utf-8'),
            stored_hash.encode('utf-8')
        )


class TestUserLogin:
    """Test user login and authentication"""

    @pytest.mark.asyncio
    async def test_login_correct_credentials(self, auth_service):
        """Should successfully login with correct credentials"""
        # Register user first
        registration = UserRegistration(
            email="login@example.com",
            password="SecurePass123!",
            full_name="Login Test"
        )
        await auth_service.register_user(registration)

        # Login
        login = UserLogin(email="login@example.com", password="SecurePass123!")
        tokens = await auth_service.login(login)

        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_service):
        """Should reject login with wrong password"""
        # Register user
        registration = UserRegistration(
            email="test@example.com",
            password="CorrectPass123!",
            full_name="Test User"
        )
        await auth_service.register_user(registration)

        # Try wrong password
        login = UserLogin(email="test@example.com", password="WrongPass123!")

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login(login)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid credentials" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_login_non_existent_user(self, auth_service):
        """Should reject login for non-existent user"""
        login = UserLogin(
            email="nonexistent@example.com",
            password="AnyPassword123!"
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login(login)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_password_never_logged(self, auth_service, caplog):
        """Should never log password in plain text"""
        registration = UserRegistration(
            email="secure@example.com",
            password="VerySecretPassword123!",
            full_name="Secure User"
        )

        await auth_service.register_user(registration)

        # Check all log messages
        for record in caplog.records:
            assert "VerySecretPassword123!" not in record.message
            assert "password" not in record.message.lower() or \
                   "hashed" in record.message.lower()


class TestJWTTokens:
    """Test JWT token creation and validation"""

    @pytest.mark.asyncio
    async def test_create_access_token(self, auth_service, auth_config):
        """Should create valid access token"""
        user_id = "test-user-123"
        token = auth_service.create_access_token(user_id)

        # Decode token
        payload = jwt.decode(
            token,
            auth_config.jwt_secret,
            algorithms=[auth_config.jwt_algorithm]
        )

        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_create_refresh_token(self, auth_service, auth_config):
        """Should create valid refresh token with longer expiry"""
        user_id = "test-user-123"
        token = auth_service.create_refresh_token(user_id)

        payload = jwt.decode(
            token,
            auth_config.jwt_secret,
            algorithms=[auth_config.jwt_algorithm]
        )

        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

        # Verify expiry is longer than access token
        exp = datetime.fromtimestamp(payload["exp"])
        now = datetime.utcnow()
        assert exp > now + timedelta(days=1)

    @pytest.mark.asyncio
    async def test_validate_valid_token(self, auth_service):
        """Should validate correct token"""
        user_id = "test-user-123"
        token = auth_service.create_access_token(user_id)

        validated_user_id = await auth_service.validate_token(token)

        assert validated_user_id == user_id

    @pytest.mark.asyncio
    async def test_validate_expired_token(self, auth_service, auth_config):
        """Should reject expired token"""
        user_id = "test-user-123"

        # Create token that expired 1 hour ago
        expired_payload = {
            "sub": user_id,
            "type": "access",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        expired_token = jwt.encode(
            expired_payload,
            auth_config.jwt_secret,
            algorithm=auth_config.jwt_algorithm
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.validate_token(expired_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_validate_invalid_signature(self, auth_service):
        """Should reject token with invalid signature"""
        # Create token with different secret
        malicious_token = jwt.encode(
            {"sub": "hacker", "type": "access", "exp": datetime.utcnow() + timedelta(hours=1)},
            "wrong_secret",
            algorithm="HS256"
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.validate_token(malicious_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_token_contains_correct_user_id(self, auth_service):
        """Should ensure JWT contains correct user_id"""
        # Register and login
        registration = UserRegistration(
            email="jwt@example.com",
            password="SecurePass123!",
            full_name="JWT User"
        )
        user = await auth_service.register_user(registration)

        login = UserLogin(email="jwt@example.com", password="SecurePass123!")
        tokens = await auth_service.login(login)

        # Validate token contains correct user_id
        validated_user_id = await auth_service.validate_token(tokens.access_token)
        assert validated_user_id == user.user_id


class TestTokenRefresh:
    """Test token refresh functionality"""

    @pytest.mark.asyncio
    async def test_refresh_valid_token(self, auth_service):
        """Should refresh access token with valid refresh token"""
        user_id = "test-user-123"
        refresh_token = auth_service.create_refresh_token(user_id)

        new_tokens = await auth_service.refresh_access_token(refresh_token)

        assert new_tokens.access_token is not None
        assert new_tokens.refresh_token is not None

        # Verify new access token is valid
        validated_user_id = await auth_service.validate_token(new_tokens.access_token)
        assert validated_user_id == user_id

    @pytest.mark.asyncio
    async def test_refresh_expired_token(self, auth_service, auth_config):
        """Should reject expired refresh token"""
        user_id = "test-user-123"

        # Create expired refresh token
        expired_payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() - timedelta(days=1)
        }
        expired_token = jwt.encode(
            expired_payload,
            auth_config.jwt_secret,
            algorithm=auth_config.jwt_algorithm
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.refresh_access_token(expired_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestRateLimiting:
    """Test rate limiting and account lockout"""

    @pytest.mark.asyncio
    async def test_rate_limit_too_many_attempts(self, auth_service):
        """Should lock account after too many failed login attempts"""
        # Register user
        registration = UserRegistration(
            email="ratelimit@example.com",
            password="CorrectPass123!",
            full_name="Rate Limit Test"
        )
        await auth_service.register_user(registration)

        # Try wrong password multiple times
        login = UserLogin(email="ratelimit@example.com", password="WrongPass123!")

        # First 5 attempts should fail with invalid credentials
        for i in range(5):
            with pytest.raises(HTTPException) as exc_info:
                await auth_service.login(login)
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

        # 6th attempt should be rate limited
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login(login)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "too many attempts" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_lockout_duration(self, auth_service):
        """Should enforce lockout duration"""
        registration = UserRegistration(
            email="lockout@example.com",
            password="CorrectPass123!",
            full_name="Lockout Test"
        )
        await auth_service.register_user(registration)

        # Trigger rate limit
        login_wrong = UserLogin(email="lockout@example.com", password="WrongPass!")
        for _ in range(6):
            try:
                await auth_service.login(login_wrong)
            except HTTPException:
                pass

        # Even correct password should be rejected during lockout
        login_correct = UserLogin(email="lockout@example.com", password="CorrectPass123!")
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login(login_correct)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestPasswordSecurity:
    """Test password hashing security"""

    def test_bcrypt_salt_uniqueness(self, auth_service):
        """Should use unique salt for each password hash"""
        password = "SamePassword123!"

        hash1 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hash2 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Same password, different hashes (different salts)
        assert hash1 != hash2

        # Both verify correctly
        assert bcrypt.checkpw(password.encode('utf-8'), hash1)
        assert bcrypt.checkpw(password.encode('utf-8'), hash2)

    def test_bcrypt_verification(self, auth_service):
        """Should correctly verify password hashes"""
        password = "TestPassword123!"
        wrong_password = "WrongPassword123!"

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Correct password verifies
        assert bcrypt.checkpw(password.encode('utf-8'), password_hash)

        # Wrong password fails
        assert not bcrypt.checkpw(wrong_password.encode('utf-8'), password_hash)


class TestSecurityBestPractices:
    """Test security best practices"""

    @pytest.mark.asyncio
    async def test_jwt_secret_from_env(self, monkeypatch):
        """Should load JWT secret from environment variable"""
        monkeypatch.setenv("LEXI_JWT_SECRET", "env_secret_key")

        config = AuthConfig()

        assert config.jwt_secret == "env_secret_key"

    @pytest.mark.asyncio
    async def test_no_password_in_response(self, auth_service):
        """Should never include password in API responses"""
        registration = UserRegistration(
            email="nopw@example.com",
            password="SecurePass123!",
            full_name="No PW Test"
        )

        user = await auth_service.register_user(registration)
        user_dict = user.dict()

        assert "password" not in user_dict
        assert "password_hash" not in user_dict


# Performance benchmarks
class TestPerformance:
    """Performance tests for authentication"""

    @pytest.mark.asyncio
    async def test_login_performance(self, auth_service):
        """Should complete login in <200ms"""
        import time

        # Register user
        registration = UserRegistration(
            email="perf@example.com",
            password="SecurePass123!",
            full_name="Performance Test"
        )
        await auth_service.register_user(registration)

        # Measure login time
        login = UserLogin(email="perf@example.com", password="SecurePass123!")

        start = time.time()
        await auth_service.login(login)
        duration = time.time() - start

        assert duration < 0.2  # 200ms

    @pytest.mark.asyncio
    async def test_token_validation_performance(self, auth_service):
        """Should validate token in <50ms"""
        import time

        user_id = "perf-user"
        token = auth_service.create_access_token(user_id)

        start = time.time()
        await auth_service.validate_token(token)
        duration = time.time() - start

        assert duration < 0.05  # 50ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend/auth", "--cov-report=html"])
