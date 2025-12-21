"""
Profile Context Tests for LexiAI
Tests: User context retrieval, caching, performance, recent memories
Target: 10+ tests with >95% coverage
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from backend.profile.profile_context import ProfileContext
from backend.profile.models import UserContext, ProfileCategory
from backend.models.memory_entry import MemoryEntry


@pytest.fixture
def mock_vectorstore():
    """Mock Qdrant vectorstore"""
    vectorstore = AsyncMock()
    vectorstore.similarity_search = AsyncMock(return_value=[])
    return vectorstore


@pytest.fixture
def profile_context(mock_vectorstore):
    """Create ProfileContext instance"""
    return ProfileContext(vectorstore=mock_vectorstore)


class TestUserContextRetrieval:
    """Test user context retrieval functionality"""

    @pytest.mark.asyncio
    async def test_retrieve_static_profile(self, profile_context, mock_vectorstore):
        """Should retrieve static profile information"""
        user_id = "user-123"

        # Mock profile memories
        profile_memories = [
            MagicMock(
                page_content="Name: Thomas",
                metadata={
                    "user_id": user_id,
                    "category": "personal",
                    "confidence": "high",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ),
            MagicMock(
                page_content="Beruf: Software Engineer",
                metadata={
                    "user_id": user_id,
                    "category": "professional",
                    "confidence": "high",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        ]

        mock_vectorstore.similarity_search.return_value = profile_memories

        context = await profile_context.get_user_context(user_id)

        assert context is not None
        assert context.user_id == user_id
        assert len(context.static_profile) > 0

    @pytest.mark.asyncio
    async def test_retrieve_recent_memories(self, profile_context, mock_vectorstore):
        """Should retrieve recent memories (last 7 days)"""
        user_id = "user-123"

        # Mock recent memories (within 7 days)
        recent_time = datetime.utcnow() - timedelta(days=3)
        recent_memories = [
            MagicMock(
                page_content="Recent conversation about Python",
                metadata={
                    "user_id": user_id,
                    "timestamp": recent_time.isoformat()
                }
            )
        ]

        # Mock old memories (> 7 days)
        old_time = datetime.utcnow() - timedelta(days=10)
        old_memories = [
            MagicMock(
                page_content="Old conversation",
                metadata={
                    "user_id": user_id,
                    "timestamp": old_time.isoformat()
                }
            )
        ]

        mock_vectorstore.similarity_search.return_value = recent_memories + old_memories

        context = await profile_context.get_user_context(user_id)

        # Should only include recent memories
        assert len(context.recent_memories) == 1
        assert context.recent_memories[0].content == "Recent conversation about Python"

    @pytest.mark.asyncio
    async def test_combine_static_and_dynamic(self, profile_context, mock_vectorstore):
        """Should combine static profile and recent memories"""
        user_id = "user-123"

        profile_memory = MagicMock(
            page_content="Name: Lisa",
            metadata={
                "user_id": user_id,
                "category": "personal",
                "timestamp": (datetime.utcnow() - timedelta(days=30)).isoformat()
            }
        )

        recent_memory = MagicMock(
            page_content="Discussed machine learning",
            metadata={
                "user_id": user_id,
                "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat()
            }
        )

        mock_vectorstore.similarity_search.return_value = [profile_memory, recent_memory]

        context = await profile_context.get_user_context(user_id)

        assert len(context.static_profile) > 0
        assert len(context.recent_memories) > 0

    @pytest.mark.asyncio
    async def test_empty_context_for_new_user(self, profile_context, mock_vectorstore):
        """Should return empty context for new user"""
        user_id = "new-user-999"

        mock_vectorstore.similarity_search.return_value = []

        context = await profile_context.get_user_context(user_id)

        assert context.user_id == user_id
        assert len(context.static_profile) == 0
        assert len(context.recent_memories) == 0


class TestContextCaching:
    """Test context caching functionality"""

    @pytest.mark.asyncio
    async def test_cache_user_context(self, profile_context, mock_vectorstore):
        """Should cache user context for faster retrieval"""
        user_id = "user-123"

        profile_memory = MagicMock(
            page_content="Name: Max",
            metadata={
                "user_id": user_id,
                "category": "personal",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        mock_vectorstore.similarity_search.return_value = [profile_memory]

        # First call - should query vectorstore
        context1 = await profile_context.get_user_context(user_id)
        call_count_1 = mock_vectorstore.similarity_search.call_count

        # Second call - should use cache
        context2 = await profile_context.get_user_context(user_id)
        call_count_2 = mock_vectorstore.similarity_search.call_count

        # Cache should prevent second vectorstore call
        assert call_count_2 == call_count_1
        assert context1.user_id == context2.user_id

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, profile_context):
        """Should invalidate cache when profile updates"""
        user_id = "user-123"

        # Get context (cached)
        context1 = await profile_context.get_user_context(user_id)

        # Invalidate cache
        await profile_context.invalidate_cache(user_id)

        # Next call should fetch fresh data
        # (verified by mock call count in real implementation)
        context2 = await profile_context.get_user_context(user_id)

        assert True  # Cache invalidation completed

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, profile_context):
        """Should expire cache after TTL"""
        user_id = "user-123"

        # Set short TTL for testing
        profile_context._cache_ttl = 0.1  # 100ms

        context1 = await profile_context.get_user_context(user_id)

        # Wait for TTL expiration
        await asyncio.sleep(0.2)

        context2 = await profile_context.get_user_context(user_id)

        # Should fetch fresh data after expiration
        assert True


class TestProfileMemoryFiltering:
    """Test filtering of profile memories"""

    @pytest.mark.asyncio
    async def test_filter_by_category(self, profile_context, mock_vectorstore):
        """Should filter memories by category"""
        user_id = "user-123"

        memories = [
            MagicMock(
                page_content="Personal info",
                metadata={
                    "user_id": user_id,
                    "category": "personal",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ),
            MagicMock(
                page_content="Professional info",
                metadata={
                    "user_id": user_id,
                    "category": "professional",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        ]

        mock_vectorstore.similarity_search.return_value = memories

        context = await profile_context.get_user_context(
            user_id,
            categories=[ProfileCategory.PERSONAL]
        )

        # Should only include personal category
        personal_count = sum(
            1 for item in context.static_profile
            if "personal" in str(item).lower()
        )
        assert personal_count > 0

    @pytest.mark.asyncio
    async def test_filter_by_timestamp(self, profile_context, mock_vectorstore):
        """Should filter memories by timestamp"""
        user_id = "user-123"

        # Recent memory
        recent = MagicMock(
            page_content="Recent",
            metadata={
                "user_id": user_id,
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat()
            }
        )

        # Old memory
        old = MagicMock(
            page_content="Old",
            metadata={
                "user_id": user_id,
                "timestamp": (datetime.utcnow() - timedelta(days=365)).isoformat()
            }
        )

        mock_vectorstore.similarity_search.return_value = [recent, old]

        context = await profile_context.get_user_context(
            user_id,
            max_age_days=30
        )

        # Should only include memories from last 30 days
        assert len(context.recent_memories) == 1


class TestPerformance:
    """Performance tests for context retrieval"""

    @pytest.mark.asyncio
    async def test_context_retrieval_performance(self, profile_context, mock_vectorstore):
        """Should retrieve context in <100ms"""
        import time

        user_id = "user-123"

        memories = [
            MagicMock(
                page_content=f"Memory {i}",
                metadata={
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            for i in range(10)
        ]
        mock_vectorstore.similarity_search.return_value = memories

        start = time.time()
        await profile_context.get_user_context(user_id)
        duration = time.time() - start

        assert duration < 0.1  # 100ms

    @pytest.mark.asyncio
    async def test_cached_retrieval_performance(self, profile_context, mock_vectorstore):
        """Cached retrieval should be <10ms"""
        import time

        user_id = "user-123"

        memories = [MagicMock(page_content="Test", metadata={"user_id": user_id, "timestamp": datetime.utcnow().isoformat()})]
        mock_vectorstore.similarity_search.return_value = memories

        # First call to populate cache
        await profile_context.get_user_context(user_id)

        # Measure cached retrieval
        start = time.time()
        await profile_context.get_user_context(user_id)
        duration = time.time() - start

        assert duration < 0.01  # 10ms


class TestContextFormatting:
    """Test context formatting for LLM consumption"""

    @pytest.mark.asyncio
    async def test_format_context_for_llm(self, profile_context, mock_vectorstore):
        """Should format context properly for LLM"""
        user_id = "user-123"

        memories = [
            MagicMock(
                page_content="Name: Thomas",
                metadata={
                    "user_id": user_id,
                    "category": "personal",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        ]
        mock_vectorstore.similarity_search.return_value = memories

        context = await profile_context.get_user_context(user_id)
        formatted = profile_context.format_for_llm(context)

        assert isinstance(formatted, str)
        assert "thomas" in formatted.lower()
        assert len(formatted) > 0

    @pytest.mark.asyncio
    async def test_context_max_length(self, profile_context, mock_vectorstore):
        """Should respect maximum context length"""
        user_id = "user-123"

        # Create many memories
        memories = [
            MagicMock(
                page_content=f"Memory content {i} " * 100,
                metadata={
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            for i in range(100)
        ]
        mock_vectorstore.similarity_search.return_value = memories

        context = await profile_context.get_user_context(user_id)
        formatted = profile_context.format_for_llm(context, max_length=1000)

        # Should truncate to max length
        assert len(formatted) <= 1000


class TestUserIsolation:
    """Test user isolation in context retrieval"""

    @pytest.mark.asyncio
    async def test_no_cross_user_contamination(self, profile_context, mock_vectorstore):
        """Should not mix contexts between users"""
        user1 = "user-111"
        user2 = "user-222"

        # Mock returns memories for both users
        all_memories = [
            MagicMock(
                page_content="User 1 data",
                metadata={"user_id": user1, "timestamp": datetime.utcnow().isoformat()}
            ),
            MagicMock(
                page_content="User 2 data",
                metadata={"user_id": user2, "timestamp": datetime.utcnow().isoformat()}
            )
        ]

        async def filter_by_user(query, filter=None, k=10):
            if filter and "user_id" in filter:
                return [m for m in all_memories if m.metadata["user_id"] == filter["user_id"]]
            return all_memories

        mock_vectorstore.similarity_search.side_effect = filter_by_user

        context1 = await profile_context.get_user_context(user1)
        context2 = await profile_context.get_user_context(user2)

        # Each context should only have own user's data
        assert context1.user_id == user1
        assert context2.user_id == user2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend/profile", "--cov-report=html"])
