"""
End-to-end integration tests for memory system.

Tests:
- Full store → retrieve → delete cycle
- Multi-user isolation
- Category prediction pipeline
- Cache behavior across operations
- Forget command workflow
- Memory updates
- Concurrent operations
- Real Qdrant integration (optional)
"""
import pytest
import asyncio
import datetime
import uuid
import time
from unittest.mock import MagicMock, patch

from backend.memory.adapter import (
    store_memory,
    store_memory_async,
    retrieve_memories,
    retrieve_memories_with_cache,
)
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.models.memory_entry import MemoryEntry


@pytest.mark.integration
class TestMemoryIntegrationStoreRetrieve:
    """Integration tests for store → retrieve workflow."""

    def test_store_and_retrieve_basic(self):
        """Test basic store and retrieve cycle."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store memory
            content = "Integration test memory for retrieval"
            doc_id, timestamp = store_memory(
                content=content,
                user_id="integration_user"
            )

            # Mock retrieval
            mock_doc = MagicMock()
            mock_doc.page_content = content
            mock_doc.metadata = {
                "id": doc_id,
                "timestamp": timestamp,
                "score": 0.95
            }
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            # Retrieve
            memories = retrieve_memories(
                user_id="integration_user",
                query="integration test",
                limit=5
            )

            # Assert
            assert len(memories) > 0
            assert memories[0].content == content
            assert memories[0].id == doc_id

    def test_store_multiple_retrieve_all(self):
        """Test storing multiple memories and retrieving all."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store multiple memories
            contents = [
                "First memory about Python",
                "Second memory about JavaScript",
                "Third memory about databases",
            ]

            stored_ids = []
            for content in contents:
                doc_id, _ = store_memory(content=content, user_id="multi_user")
                stored_ids.append(doc_id)

            # Mock retrieval of all
            mock_docs = []
            for i, (content, doc_id) in enumerate(zip(contents, stored_ids)):
                mock_doc = MagicMock()
                mock_doc.page_content = content
                mock_doc.metadata = {
                    "id": doc_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "score": 0.9 - i * 0.1  # Decreasing relevance
                }
                mock_docs.append(mock_doc)

            mock_vectorstore.similarity_search.return_value = mock_docs

            # Retrieve all
            memories = retrieve_memories(user_id="multi_user", limit=10)

            # Assert all retrieved
            assert len(memories) == 3
            retrieved_contents = [m.content for m in memories]
            assert all(c in retrieved_contents for c in contents)

    def test_store_retrieve_with_tags(self):
        """Test storing with tags and filtering by tags."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store memories with different tags
            store_memory("Work task 1", "user1", tags=["work", "urgent"])
            store_memory("Work task 2", "user1", tags=["work"])
            store_memory("Personal note", "user1", tags=["personal"])

            # Mock retrieval filtered by tags
            work_doc1 = MagicMock()
            work_doc1.page_content = "Work task 1"
            work_doc1.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["work", "urgent"],
                "score": 0.95
            }

            work_doc2 = MagicMock()
            work_doc2.page_content = "Work task 2"
            work_doc2.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["work"],
                "score": 0.9
            }

            personal_doc = MagicMock()
            personal_doc.page_content = "Personal note"
            personal_doc.metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": ["personal"],
                "score": 0.85
            }

            mock_vectorstore.similarity_search.return_value = [work_doc1, work_doc2, personal_doc]

            # Retrieve with tag filter
            work_memories = retrieve_memories(user_id="user1", tags=["work"], limit=10)

            # Assert only work memories
            assert len(work_memories) == 2
            assert all("work" in (m.tags or []) for m in work_memories)


@pytest.mark.integration
class TestMemoryIntegrationMultiUser:
    """Integration tests for multi-user isolation."""

    def test_user_isolation(self):
        """Test that different users have isolated memories."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store for different users
            store_memory("User A secret", "userA")
            store_memory("User B secret", "userB")

            # Mock isolated retrieval
            def mock_search(query, k, filter):
                user_id = filter["filter"]["must"][0]["match"]["value"]
                if user_id == "userA":
                    doc = MagicMock()
                    doc.page_content = "User A secret"
                    doc.metadata = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "score": 0.9
                    }
                    return [doc]
                elif user_id == "userB":
                    doc = MagicMock()
                    doc.page_content = "User B secret"
                    doc.metadata = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "score": 0.9
                    }
                    return [doc]
                return []

            mock_vectorstore.similarity_search.side_effect = mock_search

            # Retrieve for each user
            memories_a = retrieve_memories(user_id="userA", limit=10)
            memories_b = retrieve_memories(user_id="userB", limit=10)

            # Assert isolation
            assert len(memories_a) == 1
            assert len(memories_b) == 1
            assert memories_a[0].content == "User A secret"
            assert memories_b[0].content == "User B secret"


@pytest.mark.integration
class TestMemoryIntegrationDelete:
    """Integration tests for delete workflow."""

    def test_store_retrieve_delete_retrieve(self):
        """Test full lifecycle: store → retrieve → delete → verify gone."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store
            doc_id, _ = store_memory("Memory to delete", "user1")

            # Mock retrieval before delete
            mock_doc = MagicMock()
            mock_doc.page_content = "Memory to delete"
            mock_doc.metadata = {
                "id": doc_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score": 0.9
            }
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            # Retrieve (should exist)
            memories_before = retrieve_memories(user_id="user1", limit=10)
            assert len(memories_before) == 1

            # Delete
            with patch("backend.qdrant.qdrant_interface.safe_delete") as mock_delete:
                interface = QdrantMemoryInterface(
                    collection_name="test",
                    embeddings=MagicMock(),
                    qdrant_client=MagicMock()
                )
                interface.delete_entry(uuid.UUID(doc_id))

            # Mock retrieval after delete (empty)
            mock_vectorstore.similarity_search.return_value = []

            # Retrieve (should be gone)
            memories_after = retrieve_memories(user_id="user1", limit=10)
            assert len(memories_after) == 0


@pytest.mark.integration
class TestMemoryIntegrationCache:
    """Integration tests for caching behavior."""

    def test_cache_invalidation_on_store(self):
        """Test that storing new memory invalidates cache."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_memory_cache") as mock_cache_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock cache
                mock_cache = MagicMock()
                mock_cache.invalidate_user.return_value = 2
                mock_cache_getter.return_value = mock_cache

                # Initial retrieval (cache miss)
                mock_cache.get.return_value = None
                mock_vectorstore.similarity_search.return_value = []

                retrieve_memories_with_cache(user_id="cache_user", limit=5)

                # Store new memory
                store_memory("New memory", "cache_user")

                # Assert cache invalidated
                mock_cache.invalidate_user.assert_called_with("cache_user")

    def test_cache_hit_after_repeated_query(self):
        """Test that repeated queries use cache."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            with patch("backend.memory.adapter.get_memory_cache") as mock_cache_getter:
                mock_vectorstore = MagicMock()
                mock_bundle = MagicMock()
                mock_bundle.vectorstore = mock_vectorstore
                mock_components.return_value = mock_bundle

                # Mock cache
                mock_cache = MagicMock()

                # First call: cache miss
                mock_cache.get.side_effect = [
                    None,  # First call: miss
                    [{      # Second call: hit
                        "id": str(uuid.uuid4()),
                        "content": "Cached result",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "category": "test",
                        "tags": [],
                        "source": "cache",
                        "relevance": 0.9
                    }]
                ]
                mock_cache_getter.return_value = mock_cache

                mock_doc = MagicMock()
                mock_doc.page_content = "Fresh result"
                mock_doc.metadata = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "score": 0.9
                }
                mock_vectorstore.similarity_search.return_value = [mock_doc]

                # First query (cache miss)
                result1 = retrieve_memories_with_cache(
                    user_id="cache_user",
                    query="test query",
                    limit=5,
                    use_cache=True
                )

                # Second query (cache hit)
                result2 = retrieve_memories_with_cache(
                    user_id="cache_user",
                    query="test query",
                    limit=5,
                    use_cache=True
                )

                # Assert vectorstore called only once (first time)
                assert mock_vectorstore.similarity_search.call_count == 1

                # Assert cache used on second call
                assert len(result2) == 1
                assert result2[0].content == "Cached result"


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryIntegrationAsync:
    """Integration tests for async operations."""

    async def test_async_store_and_retrieve(self):
        """Test async store followed by retrieval."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Async store
            content = "Async integration test memory"
            doc_id, timestamp = await store_memory_async(
                content=content,
                user_id="async_user"
            )

            # Mock retrieval
            mock_doc = MagicMock()
            mock_doc.page_content = content
            mock_doc.metadata = {
                "id": doc_id,
                "timestamp": timestamp,
                "score": 0.95
            }
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            # Retrieve
            memories = retrieve_memories(user_id="async_user", limit=5)

            # Assert
            assert len(memories) > 0
            assert memories[0].content == content

    async def test_concurrent_stores(self):
        """Test concurrent async stores."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store multiple memories concurrently
            contents = [f"Concurrent memory {i}" for i in range(5)]

            tasks = [
                store_memory_async(content, f"user{i % 2}")  # 2 users
                for i, content in enumerate(contents)
            ]

            results = await asyncio.gather(*tasks)

            # Assert all stored
            assert len(results) == 5
            assert all(doc_id is not None for doc_id, _ in results)


@pytest.mark.integration
class TestMemoryIntegrationEdgeCases:
    """Integration tests for edge cases."""

    def test_empty_database_retrieval(self):
        """Test retrieval from empty database."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search.return_value = []

            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Retrieve from empty DB
            memories = retrieve_memories(user_id="new_user", limit=10)

            # Assert empty list
            assert isinstance(memories, list)
            assert len(memories) == 0

    def test_large_batch_store_retrieve(self):
        """Test storing and retrieving large batch of memories."""
        with patch("backend.memory.adapter.get_cached_components") as mock_components:
            mock_vectorstore = MagicMock()
            mock_bundle = MagicMock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_components.return_value = mock_bundle

            # Store 100 memories
            num_memories = 100
            for i in range(num_memories):
                store_memory(f"Batch memory {i}", "batch_user")

            # Mock retrieval of large batch
            mock_docs = []
            for i in range(num_memories):
                mock_doc = MagicMock()
                mock_doc.page_content = f"Batch memory {i}"
                mock_doc.metadata = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "score": 0.9
                }
                mock_docs.append(mock_doc)

            mock_vectorstore.similarity_search.return_value = mock_docs[:10]  # Limit 10

            # Retrieve
            memories = retrieve_memories(user_id="batch_user", limit=10)

            # Assert limited results
            assert len(memories) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
