"""
Pytest Configuration and Shared Fixtures for LexiAI Tests
This file provides common fixtures used across all test files
"""

import pytest
import asyncio
import os
import sys
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add backend to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config.middleware_config import MiddlewareConfig
# from backend.config.auth_config import AuthConfig  # Not needed for these tests


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    config = MagicMock(spec=MiddlewareConfig)
    config.ollama_url = "http://localhost:11434"
    config.embedding_model = "nomic-embed-text"
    config.llm_model = "gemma3:4b-it-qat"
    config.qdrant_host = "localhost"
    config.qdrant_port = 6333
    config.memory_collection = "test_lexi_memory"
    config.memory_dimension = 768
    config.api_key = "test_api_key"
    return config


@pytest.fixture
def auth_config():
    """Test authentication configuration"""
    config = MagicMock(spec=AuthConfig)
    config.jwt_secret = "test_secret_key_for_testing_only"
    config.jwt_algorithm = "HS256"
    config.access_token_expire_minutes = 30
    config.refresh_token_expire_days = 7
    config.password_min_length = 8
    config.max_login_attempts = 5
    config.lockout_duration_minutes = 15
    return config


# ============================================================================
# Mock External Services
# ============================================================================

@pytest.fixture
def mock_ollama_embeddings():
    """Mock Ollama embeddings model"""
    embeddings = AsyncMock()
    embeddings.embed_query = AsyncMock(return_value=[0.1] * 768)
    embeddings.embed_documents = AsyncMock(return_value=[[0.1] * 768])
    embeddings.dimension = 768
    return embeddings


@pytest.fixture
def mock_ollama_chat():
    """Mock Ollama chat model"""
    chat = AsyncMock()
    chat.invoke = AsyncMock(return_value=MagicMock(content="Test response"))
    chat.ainvoke = AsyncMock(return_value=MagicMock(content="Test async response"))
    return chat


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    client = MagicMock()
    client.get_collections = MagicMock(return_value=MagicMock(collections=[]))
    client.create_collection = MagicMock()
    client.upsert = MagicMock()
    client.search = MagicMock(return_value=[])
    client.delete = MagicMock()
    client.scroll = MagicMock(return_value=([], None))
    return client


@pytest.fixture
def mock_vectorstore():
    """Mock Qdrant vectorstore"""
    vectorstore = AsyncMock()
    vectorstore.add_texts = AsyncMock(return_value=["id1", "id2"])
    vectorstore.similarity_search = AsyncMock(return_value=[])
    vectorstore.similarity_search_with_score = AsyncMock(return_value=[])
    vectorstore.delete = AsyncMock()
    return vectorstore


# ============================================================================
# Mock Database
# ============================================================================

@pytest.fixture
def mock_database():
    """Mock database for user storage"""
    db = MagicMock()
    db.users = {}  # In-memory user storage
    db.login_attempts = {}  # Track login attempts
    db.sessions = {}  # Active sessions
    db.profiles = {}  # User profiles
    return db


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_user_data():
    """Sample user registration data"""
    return {
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_profile_data():
    """Sample profile information"""
    return {
        "personal": {
            "name": "Thomas",
            "age": 28,
            "location": "München"
        },
        "professional": {
            "job": "Software Engineer",
            "company": "Google",
            "experience": "5 years"
        },
        "interests": ["Python", "Machine Learning", "Kubernetes"],
        "preferences": {
            "language": "German",
            "ide": "VSCode"
        }
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing"""
    return [
        "Hallo, ich bin Thomas.",
        "Ich arbeite als Software Engineer bei Google.",
        "Ich interessiere mich für Python und Machine Learning.",
        "Ich wohne in München und bin 28 Jahre alt."
    ]


# ============================================================================
# Time Mocking
# ============================================================================

@pytest.fixture
def frozen_time():
    """Freeze time for consistent testing"""
    from freezegun import freeze_time
    frozen = freeze_time("2025-01-22 12:00:00")
    frozen.start()
    yield datetime(2025, 1, 22, 12, 0, 0)
    frozen.stop()


# ============================================================================
# Environment Variables
# ============================================================================

@pytest.fixture(autouse=True)
def test_env_vars(monkeypatch):
    """Set test environment variables for all tests"""
    test_vars = {
        "LEXI_JWT_SECRET": "test_secret_key",
        "LEXI_API_KEY": "test_api_key",
        "LEXI_OLLAMA_URL": "http://localhost:11434",
        "LEXI_QDRANT_HOST": "localhost",
        "LEXI_QDRANT_PORT": "6333",
        "LEXI_MEMORY_COLLECTION": "test_lexi_memory",
        "LEXI_EMBEDDING_MODEL": "nomic-embed-text",
        "LEXI_LLM_MODEL": "gemma3:4b-it-qat",
        "LEXI_FEATURE_MEMORY_CACHING": "true",
        "LEXI_FEATURE_AUDIT_LOGGING": "false"
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)


# ============================================================================
# Logging Configuration
# ============================================================================

@pytest.fixture(autouse=True)
def configure_test_logging(caplog):
    """Configure logging for tests"""
    import logging

    # Set log level for tests
    caplog.set_level(logging.DEBUG)

    # Create logs directory if needed
    os.makedirs("tests/logs", exist_ok=True)


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup code runs after test
    await asyncio.sleep(0)  # Allow pending tasks to complete


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Cleanup test data after session"""
    yield
    # Session cleanup
    import shutil
    if os.path.exists("htmlcov"):
        pass  # Keep coverage reports


# ============================================================================
# Assertion Helpers
# ============================================================================

@pytest.fixture
def assert_valid_jwt():
    """Helper to validate JWT tokens"""
    import jwt

    def validate(token: str, secret: str = "test_secret_key"):
        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            assert "sub" in payload  # user_id
            assert "exp" in payload  # expiration
            assert "type" in payload  # token type
            return payload
        except jwt.InvalidTokenError as e:
            pytest.fail(f"Invalid JWT: {e}")

    return validate


@pytest.fixture
def assert_secure_password():
    """Helper to validate password security"""
    def validate(password: str):
        assert len(password) >= 8, "Password too short"
        assert any(c.isupper() for c in password), "No uppercase letter"
        assert any(c.islower() for c in password), "No lowercase letter"
        assert any(c.isdigit() for c in password), "No digit"
        assert any(c in "!@#$%^&*" for c in password), "No special character"

    return validate


# ============================================================================
# Performance Testing
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer"""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

        def assert_less_than(self, max_duration: float):
            assert self.duration < max_duration, \
                f"Operation took {self.duration}s, expected <{max_duration}s"

    return Timer()


# ============================================================================
# Mock Component Bundles
# ============================================================================

@pytest.fixture
def mock_component_bundle(mock_ollama_embeddings, mock_ollama_chat, mock_vectorstore):
    """Mock ComponentBundle from bootstrap"""
    bundle = MagicMock()
    bundle.embeddings = mock_ollama_embeddings
    bundle.chat_client = mock_ollama_chat
    bundle.vectorstore = mock_vectorstore
    bundle.memory = MagicMock()
    bundle.config = MagicMock()
    return bundle


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Pytest configuration hook"""
    # Register custom markers
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_authentication" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)


# ============================================================================
# Test Reporting
# ============================================================================

@pytest.fixture(scope="session")
def test_run_metadata():
    """Metadata about test run"""
    return {
        "start_time": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform
    }


# ============================================================================
# Hooks Integration (Claude Flow)
# ============================================================================

@pytest.fixture(autouse=True)
async def claude_flow_hooks():
    """Integrate with Claude Flow hooks"""
    # Pre-test hook
    # Could call: npx claude-flow@alpha hooks pre-task --description "test"

    yield

    # Post-test hook
    # Could call: npx claude-flow@alpha hooks post-task --task-id "test"
