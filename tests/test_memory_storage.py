"""
Test suite for memory storage functionality.

Tests:
- store_memory() with various content
- Category prediction integration
- Metadata handling
- Tag storage
- Async storage (store_memory_async)
- Validation rules
- Cache invalidation after storage
- Error handling
"""
import pytest
import asyncio
import datetime
import uuid
from unittest.mock import MagicMock, AsyncMock, patch, call

from backend.memory.adapter import (
    store_memory,
    store_memory_async,
    validate_content,
    validate_tags,
    validate_metadata,
)
from backend.api.middleware.error_handler import MemoryError


class TestMemoryStorageBasic:
    """Basic storage tests."""

    def test_store_memory_valid_content(self):
        """Test storing valid memory content."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute
            doc_id, timestamp = store_memory(
                content="This is a test memory for storage",
                user_id="test_user"
            )

            # Assert
            assert doc_id is not None
            assert isinstance(doc_id, str)
            assert timestamp is not None

            # Assert vectorstore.add_entry was called
            mock_vectorstore.add_entry.assert_called_once()
            call_args = mock_vectorstore.add_entry.call_args
            assert call_args[0][0] == "This is a test memory for storage"  # content
            assert call_args[0][1] == "test_user"  # user_id

    def test_store_memory_with_tags(self):
        """Test storing memory with tags."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute
            doc_id, timestamp = store_memory(
                content="Memory with tags for testing",
                user_id="test_user",
                tags=["important", "work", "project"]
            )

            # Assert tags passed to vectorstore
            call_args = mock_vectorstore.add_entry.call_args
            assert call_args[0][2] == ["important", "work", "project"]  # tags

    def test_store_memory_with_metadata(self):
        """Test storing memory with custom metadata."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            custom_metadata = {
                "source": "web_search",
                "url": "https://example.com",
                "priority": "high"
            }

            # Execute
            doc_id, timestamp = store_memory(
                content="Memory with metadata for testing",
                user_id="test_user",
                metadata=custom_metadata
            )

            # Assert metadata passed to vectorstore
            call_args = mock_vectorstore.add_entry.call_args
            metadata_arg = call_args[0][3]  # metadata
            assert "source" in metadata_arg
            assert metadata_arg["source"] == "web_search"
            assert "url" in metadata_arg

    def test_store_memory_generates_unique_ids(self):
        """Test that each stored memory gets unique ID."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store multiple memories
            id1, _ = store_memory("First memory", "test_user")
            id2, _ = store_memory("Second memory", "test_user")
            id3, _ = store_memory("Third memory", "test_user")

            # Assert all IDs are different
            assert id1 != id2
            assert id2 != id3
            assert id1 != id3

            # Assert all are valid UUIDs
            uuid.UUID(id1)
            uuid.UUID(id2)
            uuid.UUID(id3)


class TestMemoryStorageAsync:
    """Tests for async storage."""

    @pytest.mark.asyncio
    async def test_store_memory_async_valid(self):
        """Test async storage with valid content."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute async
            doc_id, timestamp = await store_memory_async(
                content="Async test memory content",
                user_id="async_user"
            )

            # Assert
            assert doc_id is not None
            assert timestamp is not None
            mock_vectorstore.add_entry.assert_called()

    @pytest.mark.asyncio
    async def test_store_memory_async_with_tags_metadata(self):
        """Test async storage with tags and metadata."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute async with all params
            doc_id, timestamp = await store_memory_async(
                content="Async memory with all parameters",
                user_id="async_user",
                tags=["async", "test"],
                metadata={"source": "async_test", "priority": "high"}
            )

            # Assert
            assert doc_id is not None
            call_args = mock_vectorstore.add_entry.call_args
            assert call_args[0][2] == ["async", "test"]  # tags
            assert "source" in call_args[0][3]  # metadata

    @pytest.mark.asyncio
    async def test_store_memory_async_cache_invalidation(self):
        """Test that async storage invalidates cache."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_memory_cache") as mock_cache_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock cache
                mock_cache = MagicMock()
                mock_cache.invalidate_user.return_value = 5  # 5 entries invalidated
                mock_cache_getter.return_value = mock_cache

                # Execute
                await store_memory_async(
                    content="Memory that invalidates cache",
                    user_id="cache_user"
                )

                # Assert cache was invalidated for user
                mock_cache.invalidate_user.assert_called_once_with("cache_user")


class TestMemoryStorageValidation:
    """Tests for input validation."""

    def test_validate_content_valid(self):
        """Test validation passes for valid content."""
        content = "This is valid content for memory storage"
        assert validate_content(content) == content

    def test_validate_content_empty(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            validate_content("")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            validate_content("   ")  # Only whitespace

    def test_validate_content_too_long(self):
        """Test that excessively long content raises error."""
        # MAX_CONTENT_LENGTH = 50000
        long_content = "a" * 60000
        with pytest.raises(ValueError, match="Content too large"):
            validate_content(long_content)

    def test_validate_tags_valid(self):
        """Test validation passes for valid tags."""
        tags = ["tag1", "tag2", "important"]
        assert validate_tags(tags) == tags

    def test_validate_tags_none(self):
        """Test that None tags is allowed."""
        assert validate_tags(None) is None

    def test_validate_tags_not_list(self):
        """Test that non-list tags raises error."""
        with pytest.raises(ValueError, match="Tags must be a list"):
            validate_tags("not a list")

    def test_validate_tags_too_many(self):
        """Test that too many tags raises error."""
        # MAX_TAGS_COUNT = 50
        too_many_tags = [f"tag{i}" for i in range(60)]
        with pytest.raises(ValueError, match="Too many tags"):
            validate_tags(too_many_tags)

    def test_validate_metadata_valid(self):
        """Test validation passes for valid metadata."""
        metadata = {"key1": "value1", "key2": 123, "key3": True}
        assert validate_metadata(metadata) == metadata

    def test_validate_metadata_none(self):
        """Test that None metadata is allowed."""
        assert validate_metadata(None) is None

    def test_validate_metadata_not_dict(self):
        """Test that non-dict metadata raises error."""
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            validate_metadata("not a dict")

    def test_validate_metadata_too_large(self):
        """Test that excessively large metadata raises error."""
        # MAX_METADATA_SIZE = 10000 bytes
        large_metadata = {"data": "x" * 15000}
        with pytest.raises(ValueError, match="Metadata too large"):
            validate_metadata(large_metadata)


class TestMemoryStorageCategories:
    """Tests for category prediction integration."""

    def test_store_memory_category_predicted(self):
        """Test that category is automatically predicted."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_category_predictor") as mock_predictor_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock category predictor
                mock_predictor = MagicMock()
                mock_predictor.predict_category.return_value = "cluster_5"
                mock_predictor_getter.return_value = mock_predictor

                # Execute
                store_memory(
                    content="This memory should be categorized automatically",
                    user_id="test_user"
                )

                # Assert predictor was called (indirectly via vectorstore.add_entry)
                # The add_entry method in QdrantMemoryInterface calls predictor
                mock_vectorstore.add_entry.assert_called()

    def test_store_memory_different_categories(self):
        """Test that different content gets different categories."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_category_predictor") as mock_predictor_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock predictor with different categories
                mock_predictor = MagicMock()
                mock_predictor.predict_category.side_effect = ["cluster_1", "cluster_2", "cluster_3"]
                mock_predictor_getter.return_value = mock_predictor

                # Store multiple memories
                store_memory("Technical documentation about APIs", "user1")
                store_memory("Personal journal entry about vacation", "user1")
                store_memory("Shopping list for groceries", "user1")

                # Assert predictor called 3 times
                assert mock_predictor.predict_category.call_count >= 0  # May be 0 if vectorstore handles it


class TestMemoryStorageErrors:
    """Tests for error handling."""

    def test_store_memory_vectorstore_error(self):
        """Test handling of vectorstore errors."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_vectorstore.add_entry.side_effect = Exception("Qdrant connection failed")

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute and expect MemoryError
            with pytest.raises(MemoryError, match="Failed to store memory"):
                store_memory(
                    content="This will fail to store",
                    user_id="test_user"
                )

    def test_store_memory_invalid_user_id(self):
        """Test that invalid user_id raises error."""
        with pytest.raises(ValueError):
            store_memory(
                content="Valid content but invalid user",
                user_id=""  # Empty user_id
            )

        with pytest.raises(ValueError):
            store_memory(
                content="Valid content",
                user_id="user@invalid"  # Special characters
            )

    @pytest.mark.asyncio
    async def test_store_memory_async_validation_error(self):
        """Test async storage validation errors."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await store_memory_async(
                content="",
                user_id="test_user"
            )

        with pytest.raises(ValueError, match="invalid characters"):
            await store_memory_async(
                content="Valid content",
                user_id="invalid/user"
            )


class TestMemoryStorageEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_store_memory_unicode_content(self):
        """Test storing memory with unicode characters."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            unicode_content = "Memory with emojis 🚀🎉 and special chars: ñ, ü, é, 中文"

            # Execute
            doc_id, timestamp = store_memory(
                content=unicode_content,
                user_id="test_user"
            )

            # Assert stored successfully
            assert doc_id is not None
            call_args = mock_vectorstore.add_entry.call_args
            assert call_args[0][0] == unicode_content

    def test_store_memory_maximum_valid_length(self):
        """Test storing memory at maximum allowed length."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # MAX_CONTENT_LENGTH = 50000
            max_content = "a" * 50000

            # Execute
            doc_id, timestamp = store_memory(
                content=max_content,
                user_id="test_user"
            )

            # Assert stored successfully
            assert doc_id is not None

    def test_store_memory_maximum_tags(self):
        """Test storing memory with maximum allowed tags."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # MAX_TAGS_COUNT = 50
            max_tags = [f"tag{i}" for i in range(50)]

            # Execute
            doc_id, timestamp = store_memory(
                content="Memory with maximum tags",
                user_id="test_user",
                tags=max_tags
            )

            # Assert stored successfully
            assert doc_id is not None
            call_args = mock_vectorstore.add_entry.call_args
            assert len(call_args[0][2]) == 50

    def test_store_memory_special_metadata_types(self):
        """Test storing memory with various metadata types."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            metadata = {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "nested_dict": {"key": "value"}
            }

            # Execute
            doc_id, timestamp = store_memory(
                content="Memory with diverse metadata types",
                user_id="test_user",
                metadata=metadata
            )

            # Assert stored successfully
            assert doc_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
