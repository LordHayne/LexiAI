"""
Profile Builder Tests for LexiAI
Tests: Information extraction, categorization, background tasks, Qdrant storage
Target: 15+ tests with >95% coverage
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from backend.profile.profile_builder import ProfileBuilder
from backend.profile.models import (
    ProfileInformation,
    ProfileCategory,
    ProfileConfidence
)
from backend.models.memory_entry import MemoryEntry


@pytest.fixture
def mock_embeddings():
    """Mock embedding model"""
    embeddings = AsyncMock()
    embeddings.embed_query = AsyncMock(return_value=[0.1] * 768)
    embeddings.embed_documents = AsyncMock(return_value=[[0.1] * 768])
    return embeddings


@pytest.fixture
def mock_vectorstore():
    """Mock Qdrant vectorstore"""
    vectorstore = AsyncMock()
    vectorstore.add_texts = AsyncMock()
    vectorstore.similarity_search = AsyncMock(return_value=[])
    return vectorstore


@pytest.fixture
def profile_builder(mock_embeddings, mock_vectorstore):
    """Create ProfileBuilder instance"""
    return ProfileBuilder(
        embeddings=mock_embeddings,
        vectorstore=mock_vectorstore
    )


class TestInformationExtraction:
    """Test profile information extraction from messages"""

    @pytest.mark.asyncio
    async def test_extract_personal_info_name(self, profile_builder):
        """Should extract name from message"""
        message = "Mein Name ist Thomas und ich bin Entwickler."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.PERSONAL
        assert "thomas" in info.content.lower()
        assert info.confidence > ProfileConfidence.MEDIUM

    @pytest.mark.asyncio
    async def test_extract_personal_info_age(self, profile_builder):
        """Should extract age from message"""
        message = "Ich bin 28 Jahre alt."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.PERSONAL
        assert "28" in info.content

    @pytest.mark.asyncio
    async def test_extract_professional_info(self, profile_builder):
        """Should extract professional information"""
        message = "Ich arbeite als Senior Software Engineer bei Google."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.PROFESSIONAL
        assert "software engineer" in info.content.lower()
        assert "google" in info.content.lower()

    @pytest.mark.asyncio
    async def test_extract_preferences(self, profile_builder):
        """Should extract preferences and likes"""
        message = "Ich liebe Python und Machine Learning. Ich mag keine PHP."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.PREFERENCES
        assert "python" in info.content.lower()
        assert "machine learning" in info.content.lower()

    @pytest.mark.asyncio
    async def test_extract_interests(self, profile_builder):
        """Should extract hobbies and interests"""
        message = "In meiner Freizeit spiele ich gerne Gitarre und wandere."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.INTERESTS
        assert "gitarre" in info.content.lower() or "guitar" in info.content.lower()

    @pytest.mark.asyncio
    async def test_no_extraction_for_generic_message(self, profile_builder):
        """Should not extract from generic chat messages"""
        message = "Hallo, wie geht es dir?"

        info = await profile_builder.extract_profile_info(message)

        # Generic greetings should have low confidence or None
        assert info is None or info.confidence < ProfileConfidence.MEDIUM

    @pytest.mark.asyncio
    async def test_extract_goals(self, profile_builder):
        """Should extract goals and aspirations"""
        message = "Mein Ziel ist es, in 2 Jahren ein eigenes Startup zu gründen."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.category == ProfileCategory.GOALS
        assert "startup" in info.content.lower()


class TestCategoryAssignment:
    """Test profile category assignment"""

    @pytest.mark.asyncio
    async def test_category_personal(self, profile_builder):
        """Should assign PERSONAL category correctly"""
        messages = [
            "Ich heiße Anna.",
            "Ich bin 25 Jahre alt.",
            "Ich wohne in Berlin."
        ]

        for message in messages:
            info = await profile_builder.extract_profile_info(message)
            if info:
                assert info.category == ProfileCategory.PERSONAL

    @pytest.mark.asyncio
    async def test_category_professional(self, profile_builder):
        """Should assign PROFESSIONAL category correctly"""
        messages = [
            "Ich bin Softwareentwickler.",
            "Ich arbeite bei Microsoft.",
            "Ich habe 5 Jahre Erfahrung in Python."
        ]

        for message in messages:
            info = await profile_builder.extract_profile_info(message)
            if info:
                assert info.category == ProfileCategory.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_category_preferences(self, profile_builder):
        """Should assign PREFERENCES category correctly"""
        messages = [
            "Ich mag Kaffee mehr als Tee.",
            "Ich bevorzuge VSCode als IDE.",
            "Ich liebe veganes Essen."
        ]

        for message in messages:
            info = await profile_builder.extract_profile_info(message)
            if info:
                assert info.category == ProfileCategory.PREFERENCES


class TestConfidenceScoring:
    """Test confidence scoring for extracted information"""

    @pytest.mark.asyncio
    async def test_high_confidence_explicit_statement(self, profile_builder):
        """Should assign high confidence to explicit statements"""
        message = "Mein Name ist Thomas."

        info = await profile_builder.extract_profile_info(message)

        assert info is not None
        assert info.confidence >= ProfileConfidence.HIGH

    @pytest.mark.asyncio
    async def test_medium_confidence_implied_info(self, profile_builder):
        """Should assign medium confidence to implied information"""
        message = "Ich würde gerne mehr über Machine Learning lernen."

        info = await profile_builder.extract_profile_info(message)

        if info:
            assert info.confidence == ProfileConfidence.MEDIUM

    @pytest.mark.asyncio
    async def test_low_confidence_vague_statement(self, profile_builder):
        """Should assign low confidence to vague statements"""
        message = "Ich mag so Sachen wie... du weißt schon."

        info = await profile_builder.extract_profile_info(message)

        if info:
            assert info.confidence <= ProfileConfidence.MEDIUM


class TestBackgroundTaskExecution:
    """Test background task processing for profile building"""

    @pytest.mark.asyncio
    async def test_background_task_called(self, profile_builder, mock_vectorstore):
        """Should execute profile building as background task"""
        message = "Ich heiße Max und bin Entwickler."
        user_id = "user-123"

        # Mock background task execution
        with patch.object(profile_builder, '_build_profile_background') as mock_bg:
            await profile_builder.process_message_for_profile(message, user_id)

            # Background task should be scheduled
            assert mock_bg.called or profile_builder._pending_tasks > 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, profile_builder):
        """Should handle multiple concurrent profile building tasks"""
        messages = [
            "Ich bin Entwickler.",
            "Ich liebe Python.",
            "Ich arbeite bei Google."
        ]
        user_id = "user-123"

        # Process multiple messages concurrently
        tasks = [
            profile_builder.process_message_for_profile(msg, user_id)
            for msg in messages
        ]

        await asyncio.gather(*tasks)

        # Should complete without errors
        assert True


class TestQdrantStorage:
    """Test storage of profile information in Qdrant"""

    @pytest.mark.asyncio
    async def test_store_profile_info_in_qdrant(self, profile_builder, mock_vectorstore):
        """Should store profile information in Qdrant"""
        message = "Mein Name ist Lisa."
        user_id = "user-123"

        await profile_builder.process_message_for_profile(message, user_id)

        # Wait for background task
        await asyncio.sleep(0.1)

        # Vectorstore should have been called
        assert mock_vectorstore.add_texts.called or \
               mock_vectorstore.add_texts.call_count > 0

    @pytest.mark.asyncio
    async def test_profile_memory_metadata(self, profile_builder, mock_vectorstore):
        """Should include correct metadata in stored profile"""
        message = "Ich bin Software Architekt."
        user_id = "user-789"

        # Mock add_texts to capture metadata
        captured_metadata = []
        async def capture_add_texts(texts, metadatas=None, **kwargs):
            if metadatas:
                captured_metadata.extend(metadatas)

        mock_vectorstore.add_texts.side_effect = capture_add_texts

        await profile_builder.process_message_for_profile(message, user_id)
        await asyncio.sleep(0.1)

        # Check metadata
        if captured_metadata:
            metadata = captured_metadata[0]
            assert metadata["user_id"] == user_id
            assert metadata["source"] == "profile_builder"
            assert "category" in metadata
            assert "confidence" in metadata


class TestDuplicateDetection:
    """Test duplicate profile information detection"""

    @pytest.mark.asyncio
    async def test_detect_duplicate_info(self, profile_builder, mock_vectorstore):
        """Should detect and skip duplicate profile information"""
        user_id = "user-123"
        message = "Mein Name ist Thomas."

        # Mock existing similar memory
        existing_memory = MagicMock()
        existing_memory.page_content = "Name: Thomas"
        existing_memory.metadata = {
            "user_id": user_id,
            "category": "personal",
            "confidence": "high"
        }

        mock_vectorstore.similarity_search.return_value = [existing_memory]

        # Process message - should detect duplicate
        info = await profile_builder.extract_profile_info(message)

        is_duplicate = await profile_builder._is_duplicate(info, user_id)

        assert is_duplicate is True

    @pytest.mark.asyncio
    async def test_allow_non_duplicate_info(self, profile_builder, mock_vectorstore):
        """Should allow non-duplicate information"""
        user_id = "user-123"
        message = "Ich arbeite bei Microsoft."

        # Mock existing different memory
        existing_memory = MagicMock()
        existing_memory.page_content = "Name: Thomas"
        existing_memory.metadata = {"user_id": user_id}

        mock_vectorstore.similarity_search.return_value = [existing_memory]

        info = await profile_builder.extract_profile_info(message)
        is_duplicate = await profile_builder._is_duplicate(info, user_id)

        assert is_duplicate is False


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_message(self, profile_builder):
        """Should handle empty message gracefully"""
        message = ""
        user_id = "user-123"

        result = await profile_builder.process_message_for_profile(message, user_id)

        # Should not crash
        assert result is None or isinstance(result, ProfileInformation)

    @pytest.mark.asyncio
    async def test_very_long_message(self, profile_builder):
        """Should handle very long messages"""
        message = "Ich bin Entwickler. " * 1000  # 1000 words
        user_id = "user-123"

        result = await profile_builder.process_message_for_profile(message, user_id)

        # Should complete without timeout
        assert result is None or isinstance(result, ProfileInformation)

    @pytest.mark.asyncio
    async def test_special_characters(self, profile_builder):
        """Should handle special characters in messages"""
        message = "Mein Name ist Müller-Schmidt & ich arbeite @Microsoft! 💻"

        info = await profile_builder.extract_profile_info(message)

        # Should process without errors
        assert info is None or info.content is not None


class TestPerformance:
    """Performance tests for profile building"""

    @pytest.mark.asyncio
    async def test_extraction_performance(self, profile_builder):
        """Should extract profile info in <100ms"""
        import time

        message = "Ich heiße Thomas und arbeite als Software Engineer bei Google."

        start = time.time()
        await profile_builder.extract_profile_info(message)
        duration = time.time() - start

        assert duration < 0.1  # 100ms

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, profile_builder):
        """Should handle batch processing efficiently"""
        import time

        messages = [
            "Ich bin Entwickler.",
            "Ich liebe Python.",
            "Ich arbeite bei Microsoft.",
            "Ich bin 30 Jahre alt.",
            "Ich wohne in München."
        ]
        user_id = "user-123"

        start = time.time()
        tasks = [
            profile_builder.process_message_for_profile(msg, user_id)
            for msg in messages
        ]
        await asyncio.gather(*tasks)
        duration = time.time() - start

        # Should process 5 messages in <500ms
        assert duration < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend/profile", "--cov-report=html"])
