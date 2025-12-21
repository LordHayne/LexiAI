"""
Test suite for memory retrieval functionality.

Tests:
- retrieve_memories() with various user_ids
- Semantic search with queries
- Tag filtering
- Score threshold filtering
- Limit validation
- Caching behavior
- Error handling
"""
import pytest
import asyncio
import datetime
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from backend.memory.adapter import (
    retrieve_memories,
    retrieve_memories_with_cache,
    retrieve_memories_direct,
    validate_user_id,
    validate_limit,
)
from backend.models.memory_entry import MemoryEntry
from backend.api.middleware.error_handler import MemoryError


class TestMemoryRetrievalBasic:
    """Basic retrieval tests with mocked vectorstore."""

    def test_retrieve_memories_valid_user(self):
        """Test basic retrieval with valid user_id."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            # Setup mock vectorstore
            mock_vectorstore = MagicMock()
            mock_doc = MagicMock()
            mock_doc.page_content = "Test memory content"
            mock_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "category": "test_category",
                "tags": ["test_tag"],
                "source": "test_source",
                "score": 0.95
            }
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute
            results = retrieve_memories(user_id="test_user", limit=10)

            # Assert
            assert isinstance(results, list)
            assert len(results) > 0
            assert isinstance(results[0], MemoryEntry)
            assert results[0].content == "Test memory content"
            assert results[0].category == "test_category"

    def test_retrieve_memories_multiple_users(self):
        """Test that different user_ids retrieve different memories."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            # User 1 memories
            mock_doc1 = MagicMock()
            mock_doc1.page_content = "User 1 memory"
            mock_doc1.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "category": "user1_cat",
                "tags": [],
                "score": 0.9
            }

            # User 2 memories
            mock_doc2 = MagicMock()
            mock_doc2.page_content = "User 2 memory"
            mock_doc2.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "category": "user2_cat",
                "tags": [],
                "score": 0.85
            }

            # Mock different responses based on filter
            def mock_search(query, k, filter):
                user_id = filter["filter"]["must"][0]["match"]["value"]
                if user_id == "user1":
                    return [mock_doc1]
                elif user_id == "user2":
                    return [mock_doc2]
                return []

            mock_vectorstore.similarity_search.side_effect = mock_search

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute for both users
            results_user1 = retrieve_memories(user_id="user1", limit=5)
            results_user2 = retrieve_memories(user_id="user2", limit=5)

            # Assert different results
            assert len(results_user1) == 1
            assert len(results_user2) == 1
            assert results_user1[0].content == "User 1 memory"
            assert results_user2[0].content == "User 2 memory"
            assert results_user1[0].category != results_user2[0].category

    def test_retrieve_memories_with_query(self):
        """Test semantic search with query string."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            # Create relevant and irrelevant docs
            relevant_doc = MagicMock()
            relevant_doc.page_content = "Python machine learning tutorial"
            relevant_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score": 0.95
            }

            irrelevant_doc = MagicMock()
            irrelevant_doc.page_content = "Cooking recipe"
            irrelevant_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score": 0.3
            }

            mock_vectorstore.similarity_search.return_value = [relevant_doc]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute with query
            results = retrieve_memories(
                user_id="test_user",
                query="machine learning",
                limit=5
            )

            # Assert query was used
            mock_vectorstore.similarity_search.assert_called_once()
            args = mock_vectorstore.similarity_search.call_args
            assert args[0][0] == "machine learning"  # query argument
            assert len(results) > 0
            assert "Python" in results[0].content or "machine learning" in results[0].content


class TestMemoryRetrievalFiltering:
    """Tests for filtering functionality."""

    def test_retrieve_with_tag_filter(self):
        """Test retrieval with tag filtering."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            # Doc with matching tag
            doc_with_tag = MagicMock()
            doc_with_tag.page_content = "Memory with important tag"
            doc_with_tag.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["important", "work"],
                "score": 0.9
            }

            # Doc without matching tag
            doc_no_tag = MagicMock()
            doc_no_tag.page_content = "Memory without tag"
            doc_no_tag.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["personal"],
                "score": 0.8
            }

            mock_vectorstore.similarity_search.return_value = [doc_with_tag, doc_no_tag]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute with tag filter
            results = retrieve_memories(
                user_id="test_user",
                tags=["important"],
                limit=10
            )

            # Assert only tagged memory returned
            assert len(results) == 1
            assert "important" in results[0].tags

    def test_retrieve_with_score_threshold(self):
        """Test retrieval with minimum score threshold."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            # High relevance doc
            high_score_doc = MagicMock()
            high_score_doc.page_content = "Highly relevant memory"
            high_score_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score": 0.95
            }

            # Low relevance doc (should be filtered out)
            low_score_doc = MagicMock()
            low_score_doc.page_content = "Low relevance memory"
            low_score_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score": 0.4
            }

            # Mock vectorstore filters by score_threshold
            mock_vectorstore.similarity_search.return_value = [high_score_doc]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute with score threshold
            results = retrieve_memories(
                user_id="test_user",
                score_threshold=0.7,
                limit=10
            )

            # Assert vectorstore called with score_threshold
            call_kwargs = mock_vectorstore.similarity_search.call_args.kwargs
            assert call_kwargs.get("score_threshold") == 0.7

            # Assert only high score returned
            assert len(results) >= 1
            assert all(r.relevance >= 0.7 for r in results if r.relevance is not None)

    def test_retrieve_with_limit_clamping(self):
        """Test that limit is clamped to maximum."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search.return_value = []

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute with excessive limit
            results = retrieve_memories(
                user_id="test_user",
                limit=99999  # Exceeds MAX_RETRIEVAL_LIMIT (100)
            )

            # Assert limit was clamped
            call_kwargs = mock_vectorstore.similarity_search.call_args.kwargs
            actual_k = call_kwargs.get("k")
            # Should be clamped to MAX_RETRIEVAL_LIMIT * RETRIEVAL_OVERSELECTION_FACTOR
            assert actual_k <= 100 * 2  # Max 200


class TestMemoryRetrievalCaching:
    """Tests for caching behavior."""

    def test_cache_hit(self):
        """Test that cache is used on repeated queries."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_memory_cache") as mock_cache_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock cache returning cached data
                mock_cache = MagicMock()
                cached_data = [{
                    "id": str(uuid.uuid4()),
                    "content": "Cached memory",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "category": "cached",
                    "tags": [],
                    "source": "cache",
                    "relevance": 0.9
                }]
                mock_cache.get.return_value = cached_data
                mock_cache_getter.return_value = mock_cache

                # Execute
                results = retrieve_memories_with_cache(
                    user_id="test_user",
                    query="test query",
                    limit=5,
                    use_cache=True
                )

                # Assert cache was checked
                mock_cache.get.assert_called_once()

                # Assert vectorstore NOT called (cache hit)
                mock_vectorstore.similarity_search.assert_not_called()

                # Assert cached data returned
                assert len(results) == 1
                assert results[0].content == "Cached memory"

    def test_cache_miss(self):
        """Test that vectorstore is queried on cache miss."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_memory_cache") as mock_cache_getter:
                mock_vectorstore = MagicMock()

                mock_doc = MagicMock()
                mock_doc.page_content = "Fresh memory"
                mock_doc.metadata = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "score": 0.85
                }
                mock_vectorstore.similarity_search.return_value = [mock_doc]

                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock cache miss
                mock_cache = MagicMock()
                mock_cache.get.return_value = None
                mock_cache_getter.return_value = mock_cache

                # Execute
                results = retrieve_memories_with_cache(
                    user_id="test_user",
                    query="new query",
                    limit=5,
                    use_cache=True
                )

                # Assert cache was checked
                mock_cache.get.assert_called_once()

                # Assert vectorstore WAS called (cache miss)
                mock_vectorstore.similarity_search.assert_called_once()

                # Assert result stored in cache
                mock_cache.store.assert_called_once()

                # Assert fresh data returned
                assert len(results) == 1
                assert results[0].content == "Fresh memory"


class TestMemoryRetrievalValidation:
    """Tests for input validation."""

    def test_invalid_user_id_empty(self):
        """Test that empty user_id raises error."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_user_id("")

    def test_invalid_user_id_too_long(self):
        """Test that excessively long user_id raises error."""
        long_id = "a" * 300  # Exceeds MAX_USER_ID_LENGTH (255)
        with pytest.raises(ValueError, match="too long"):
            validate_user_id(long_id)

    def test_invalid_user_id_special_chars(self):
        """Test that user_id with special characters raises error."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_user_id("user@domain.com")

        with pytest.raises(ValueError, match="invalid characters"):
            validate_user_id("user/path")

    def test_valid_user_id(self):
        """Test that valid user_id passes validation."""
        assert validate_user_id("user_123") == "user_123"
        assert validate_user_id("test-user") == "test-user"
        assert validate_user_id("User123") == "User123"

    def test_limit_validation_negative(self):
        """Test that negative limit is corrected."""
        assert validate_limit(-5) == 10  # DEFAULT_RETRIEVAL_LIMIT

    def test_limit_validation_zero(self):
        """Test that zero limit is corrected."""
        assert validate_limit(0) == 10  # DEFAULT_RETRIEVAL_LIMIT

    def test_limit_validation_excessive(self):
        """Test that excessive limit is clamped."""
        result = validate_limit(999)
        assert result <= 100  # MAX_RETRIEVAL_LIMIT


class TestMemoryRetrievalErrors:
    """Tests for error handling."""

    def test_retrieve_vectorstore_error(self):
        """Test handling of vectorstore errors."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search.side_effect = Exception("Qdrant connection failed")

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute and expect MemoryError
            with pytest.raises(MemoryError, match="Failed to retrieve memories"):
                retrieve_memories(user_id="test_user", limit=5)

    def test_retrieve_empty_results(self):
        """Test handling of empty results."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search.return_value = []

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute
            results = retrieve_memories(user_id="test_user", limit=5)

            # Assert empty list returned (not error)
            assert isinstance(results, list)
            assert len(results) == 0

    def test_retrieve_malformed_timestamp(self):
        """Test handling of malformed timestamp in metadata."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            mock_doc = MagicMock()
            mock_doc.page_content = "Memory with bad timestamp"
            mock_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": "invalid-date-format",  # Malformed
                "score": 0.8
            }
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute - should handle gracefully
            results = retrieve_memories(user_id="test_user", limit=5)

            # Assert fallback timestamp used (doesn't crash)
            assert len(results) == 1
            assert isinstance(results[0].timestamp, datetime.datetime)


class TestMemoryRetrievalCorrections:
    """Tests for correction memory boosting."""

    def test_correction_memory_boosted(self):
        """Test that correction memories get relevance boost."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()

            # Regular memory
            regular_doc = MagicMock()
            regular_doc.page_content = "Regular memory"
            regular_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": [],
                "source": "user_input",
                "score": 0.8
            }

            # Correction memory
            correction_doc = MagicMock()
            correction_doc.page_content = "Correction memory"
            correction_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["correction"],
                "source": "self_correction",
                "score": 0.7  # Lower base score
            }

            mock_vectorstore.similarity_search.return_value = [regular_doc, correction_doc]

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Execute
            results = retrieve_memories(user_id="test_user", limit=10)

            # Find correction memory
            correction_result = next(r for r in results if "correction" in (r.tags or []))

            # Assert correction got boosted score
            # CORRECTION_BOOST_FACTOR = 1.5
            # 0.7 * 1.5 = 1.05
            assert correction_result.relevance > 0.8  # Higher than regular despite lower base
            assert correction_result.source == "self_correction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
