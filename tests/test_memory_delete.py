"""
Tests for Memory Delete/Forget functionality.

Tests the delete_memory() and delete_memories_by_content() functions
with various scenarios including error handling and validation.
"""
import pytest
import uuid
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from backend.memory.adapter import (
    delete_memory,
    delete_memories_by_content,
    MemoryError
)
from backend.models.memory_entry import MemoryEntry


@pytest.fixture
def mock_components():
    """Mock component bundle for testing."""
    mock_bundle = MagicMock()
    mock_vectorstore = MagicMock()
    mock_bundle.vectorstore = mock_vectorstore
    return mock_bundle, mock_vectorstore


@pytest.fixture
def sample_memory_entries():
    """Sample memory entries for testing."""
    return [
        MemoryEntry(
            id=str(uuid.uuid4()),
            content="Python is a programming language",
            timestamp=datetime.now(timezone.utc),
            category="programming",
            tags=["python", "programming"],
            source="chat",
            relevance=0.95
        ),
        MemoryEntry(
            id=str(uuid.uuid4()),
            content="I love coding in Python",
            timestamp=datetime.now(timezone.utc),
            category="programming",
            tags=["python", "preferences"],
            source="chat",
            relevance=0.88
        ),
        MemoryEntry(
            id=str(uuid.uuid4()),
            content="JavaScript is also a programming language",
            timestamp=datetime.now(timezone.utc),
            category="programming",
            tags=["javascript", "programming"],
            source="chat",
            relevance=0.72
        )
    ]


class TestDeleteMemory:
    """Test delete_memory() function."""

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.get_memory_cache')
    def test_delete_memory_success(self, mock_cache, mock_components_func):
        """Test successful deletion of a memory entry."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        memory_id = str(uuid.uuid4())
        user_id = "test-user"

        # Execute
        result = delete_memory(memory_id, user_id)

        # Verify
        assert result is True
        mock_vectorstore.delete_entry.assert_called_once()
        mock_cache_instance.invalidate_user.assert_called_once_with(user_id)

    @patch('backend.memory.adapter.get_cached_components')
    def test_delete_memory_invalid_id(self, mock_components_func):
        """Test deletion with invalid memory ID."""
        # Setup
        mock_bundle, _ = mock_components()
        mock_components_func.return_value = mock_bundle

        invalid_id = "not-a-uuid"
        user_id = "test-user"

        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid memory_id format"):
            delete_memory(invalid_id, user_id)

    @patch('backend.memory.adapter.get_cached_components')
    def test_delete_memory_invalid_user_id(self, mock_components_func):
        """Test deletion with invalid user ID."""
        # Setup
        mock_bundle, _ = mock_components()
        mock_components_func.return_value = mock_bundle

        memory_id = str(uuid.uuid4())
        invalid_user_id = ""  # Empty user ID

        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid user_id"):
            delete_memory(memory_id, invalid_user_id)

    @patch('backend.memory.adapter.get_cached_components')
    def test_delete_memory_vectorstore_error(self, mock_components_func):
        """Test deletion when vectorstore raises an error."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_vectorstore.delete_entry.side_effect = RuntimeError("Database error")

        memory_id = str(uuid.uuid4())
        user_id = "test-user"

        # Execute & Verify
        with pytest.raises(MemoryError, match="Failed to delete memory"):
            delete_memory(memory_id, user_id)


class TestDeleteMemoriesByContent:
    """Test delete_memories_by_content() function."""

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    @patch('backend.memory.adapter.get_memory_cache')
    def test_delete_by_content_success(
        self, mock_cache, mock_retrieve, mock_components_func, sample_memory_entries
    ):
        """Test successful deletion by content query."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        # Return Python-related memories
        python_memories = [sample_memory_entries[0], sample_memory_entries[1]]
        mock_retrieve.return_value = python_memories

        query = "Python programming"
        user_id = "test-user"

        # Execute
        deleted_ids = delete_memories_by_content(query, user_id)

        # Verify
        assert len(deleted_ids) == 2
        assert all(isinstance(id, str) for id in deleted_ids)
        assert mock_vectorstore.delete_entry.call_count == 2
        mock_cache_instance.invalidate_user.assert_called_once_with(user_id)

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    def test_delete_by_content_no_matches(
        self, mock_retrieve, mock_components_func
    ):
        """Test deletion when no memories match the query."""
        # Setup
        mock_bundle, _ = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_retrieve.return_value = []  # No matches

        query = "nonexistent topic"
        user_id = "test-user"

        # Execute
        deleted_ids = delete_memories_by_content(query, user_id)

        # Verify
        assert len(deleted_ids) == 0

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    def test_delete_by_content_invalid_threshold(
        self, mock_retrieve, mock_components_func
    ):
        """Test deletion with invalid similarity threshold."""
        # Setup
        mock_bundle, _ = mock_components()
        mock_components_func.return_value = mock_bundle

        query = "Python"
        user_id = "test-user"
        invalid_threshold = 1.5  # > 1.0

        # Execute & Verify
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            delete_memories_by_content(query, user_id, similarity_threshold=invalid_threshold)

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    @patch('backend.memory.adapter.get_memory_cache')
    def test_delete_by_content_partial_failure(
        self, mock_cache, mock_retrieve, mock_components_func, sample_memory_entries
    ):
        """Test deletion when some entries fail to delete."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        python_memories = [sample_memory_entries[0], sample_memory_entries[1]]
        mock_retrieve.return_value = python_memories

        # Make second deletion fail
        def delete_side_effect(uuid_id):
            if str(uuid_id) == python_memories[1].id:
                raise RuntimeError("Delete failed")

        mock_vectorstore.delete_entry.side_effect = delete_side_effect

        query = "Python"
        user_id = "test-user"

        # Execute
        deleted_ids = delete_memories_by_content(query, user_id)

        # Verify - should only delete first entry
        assert len(deleted_ids) == 1
        assert deleted_ids[0] == python_memories[0].id

    @patch('backend.memory.adapter.get_cached_components')
    def test_delete_by_content_invalid_query(self, mock_components_func):
        """Test deletion with invalid query."""
        # Setup
        mock_bundle, _ = mock_components()
        mock_components_func.return_value = mock_bundle

        empty_query = ""  # Empty query
        user_id = "test-user"

        # Execute & Verify
        with pytest.raises(ValueError, match="Content cannot be empty"):
            delete_memories_by_content(empty_query, user_id)

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    @patch('backend.memory.adapter.get_memory_cache')
    def test_delete_by_content_custom_threshold(
        self, mock_cache, mock_retrieve, mock_components_func, sample_memory_entries
    ):
        """Test deletion with custom similarity threshold."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        # Only high-relevance memory
        high_relevance_memory = [sample_memory_entries[0]]
        mock_retrieve.return_value = high_relevance_memory

        query = "Python"
        user_id = "test-user"
        custom_threshold = 0.90  # High threshold

        # Execute
        deleted_ids = delete_memories_by_content(
            query, user_id, similarity_threshold=custom_threshold
        )

        # Verify retrieve was called with custom threshold
        mock_retrieve.assert_called_once_with(
            user_id=user_id,
            query=query,
            tags=None,
            limit=50,
            score_threshold=custom_threshold
        )

        assert len(deleted_ids) == 1


class TestMemoryDeleteIntegration:
    """Integration tests for memory deletion."""

    @patch('backend.memory.adapter.get_cached_components')
    @patch('backend.memory.adapter.retrieve_memories_direct')
    @patch('backend.memory.adapter.get_memory_cache')
    def test_delete_flow_validation_to_deletion(
        self, mock_cache, mock_retrieve, mock_components_func, sample_memory_entries
    ):
        """Test complete flow from validation to deletion."""
        # Setup
        mock_bundle, mock_vectorstore = mock_components()
        mock_components_func.return_value = mock_bundle
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        python_memories = sample_memory_entries[:2]
        mock_retrieve.return_value = python_memories

        query = "Python programming knowledge"
        user_id = "integration-test-user"

        # Execute
        deleted_ids = delete_memories_by_content(query, user_id, similarity_threshold=0.70)

        # Verify complete flow
        # 1. Validation happened (no exceptions)
        assert isinstance(deleted_ids, list)

        # 2. Retrieval was called
        mock_retrieve.assert_called_once()

        # 3. Deletion was called for each memory
        assert mock_vectorstore.delete_entry.call_count == len(python_memories)

        # 4. Cache was invalidated
        mock_cache_instance.invalidate_user.assert_called_once_with(user_id)

        # 5. Correct IDs were returned
        assert len(deleted_ids) == len(python_memories)
        assert set(deleted_ids) == {m.id for m in python_memories}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
