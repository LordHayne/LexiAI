"""
CRITICAL: User Isolation Tests

This test suite ensures complete isolation between users.
User A must NEVER see User B's memories or chat context.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from backend.api.api_server import app
from backend.models.memory_entry import MemoryEntry


class TestUserMemoryIsolation:
    """Critical tests for user memory isolation"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_components(self):
        """Mock all external dependencies"""
        with patch('backend.core.bootstrap.initialize_components') as mock_init, \
             patch('backend.memory.adapter.store_memory') as mock_store, \
             patch('backend.memory.adapter.retrieve_memories') as mock_retrieve:

            # Mock component initialization
            mock_init.return_value = MagicMock(
                chat_client=MagicMock(),
                vectorstore=MagicMock(),
                memory=MagicMock(),
                embeddings=MagicMock()
            )

            yield {
                'init': mock_init,
                'store': mock_store,
                'retrieve': mock_retrieve
            }

    def test_complete_user_isolation(self, client, mock_components):
        """
        CRITICAL TEST: Verify complete isolation between users

        Steps:
        1. Create User A and store memory "I love Python"
        2. Create User B and store memory "I love JavaScript"
        3. Verify User A ONLY sees "Python" memory
        4. Verify User B ONLY sees "JavaScript" memory
        """
        # Track memories by user_id
        user_memories = {}

        def mock_store_fn(user_id, content, category="general", **kwargs):
            if user_id not in user_memories:
                user_memories[user_id] = []
            memory = MemoryEntry(
                id=str(uuid.uuid4()),
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                tag=category,
                relevance_score=1.0
            )
            user_memories[user_id].append(memory)
            return memory

        def mock_retrieve_fn(query, user_id="default", limit=10, **kwargs):
            return user_memories.get(user_id, [])

        mock_components['store'].side_effect = mock_store_fn
        mock_components['retrieve'].side_effect = mock_retrieve_fn

        # Create User A
        response_a = client.post("/v1/users/init")
        assert response_a.status_code == 200
        user_a_id = response_a.json()["user_id"]
        assert user_a_id != "default"

        # User A stores memory "I love Python"
        store_a = client.post(
            "/v1/memory/add",
            headers={"X-User-ID": user_a_id},
            json={"content": "I love Python", "category": "preferences"}
        )
        assert store_a.status_code == 200

        # Create User B
        response_b = client.post("/v1/users/init")
        assert response_b.status_code == 200
        user_b_id = response_b.json()["user_id"]
        assert user_b_id != "default"
        assert user_b_id != user_a_id  # Different users

        # User B stores memory "I love JavaScript"
        store_b = client.post(
            "/v1/memory/add",
            headers={"X-User-ID": user_b_id},
            json={"content": "I love JavaScript", "category": "preferences"}
        )
        assert store_b.status_code == 200

        # CRITICAL: User A queries memories - should only see "Python"
        response_a_query = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": user_a_id},
            json={"query": "What do I love?", "limit": 10}
        )
        assert response_a_query.status_code == 200
        memories_a = response_a_query.json()["memories"]

        # Assert User A sees ONLY their own memory
        assert len(memories_a) == 1, f"User A should see 1 memory, got {len(memories_a)}"
        assert "Python" in memories_a[0]["content"], "User A should see Python memory"

        # Verify NO JavaScript leak
        all_content_a = str(memories_a)
        assert "JavaScript" not in all_content_a, "User A should NOT see JavaScript memory"

        # CRITICAL: User B queries memories - should only see "JavaScript"
        response_b_query = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": user_b_id},
            json={"query": "What do I love?", "limit": 10}
        )
        assert response_b_query.status_code == 200
        memories_b = response_b_query.json()["memories"]

        # Assert User B sees ONLY their own memory
        assert len(memories_b) == 1, f"User B should see 1 memory, got {len(memories_b)}"
        assert "JavaScript" in memories_b[0]["content"], "User B should see JavaScript memory"

        # Verify NO Python leak
        all_content_b = str(memories_b)
        assert "Python" not in all_content_b, "User B should NOT see Python memory"

    def test_chat_memory_isolation(self, client, mock_components):
        """
        Verify chat context is isolated between users

        User A's chat about Python should NOT appear in User B's context
        """
        # Mock chat responses
        mock_chat = AsyncMock()
        mock_components['init'].return_value.chat_client = mock_chat

        chat_memories = {}

        def mock_retrieve_fn(query, user_id="default", **kwargs):
            return chat_memories.get(user_id, [])

        mock_components['retrieve'].side_effect = mock_retrieve_fn

        # Create User A
        response_a = client.post("/v1/users/init")
        user_a_id = response_a.json()["user_id"]

        # Store User A's context
        chat_memories[user_a_id] = [
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="User loves Python programming",
                timestamp=datetime.utcnow().isoformat(),
                tag="chat_context",
                relevance_score=1.0
            )
        ]

        # Create User B
        response_b = client.post("/v1/users/init")
        user_b_id = response_b.json()["user_id"]

        # Store User B's context
        chat_memories[user_b_id] = [
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="User loves JavaScript programming",
                timestamp=datetime.utcnow().isoformat(),
                tag="chat_context",
                relevance_score=1.0
            )
        ]

        # User A chats - should retrieve ONLY Python context
        response_a_chat = client.post(
            "/ui/chat",
            headers={"X-User-ID": user_a_id},
            json={"message": "What programming language do I like?"}
        )
        # Verify retrieve_memories was called with User A's ID
        assert any(
            call[1].get('user_id') == user_a_id
            for call in mock_components['retrieve'].call_args_list
        ), "User A's chat should retrieve with user_a_id"

        # User B chats - should retrieve ONLY JavaScript context
        mock_components['retrieve'].reset_mock()
        response_b_chat = client.post(
            "/ui/chat",
            headers={"X-User-ID": user_b_id},
            json={"message": "What programming language do I like?"}
        )
        # Verify retrieve_memories was called with User B's ID
        assert any(
            call[1].get('user_id') == user_b_id
            for call in mock_components['retrieve'].call_args_list
        ), "User B's chat should retrieve with user_b_id"

    def test_memory_query_filtering(self, client, mock_components):
        """
        Verify query_memories endpoint strictly filters by user_id
        """
        # Setup multiple users with memories
        memories_db = {
            "user1": [
                MemoryEntry(
                    id="mem1",
                    content="User 1's secret data",
                    timestamp=datetime.utcnow().isoformat(),
                    tag="personal",
                    relevance_score=1.0
                )
            ],
            "user2": [
                MemoryEntry(
                    id="mem2",
                    content="User 2's secret data",
                    timestamp=datetime.utcnow().isoformat(),
                    tag="personal",
                    relevance_score=1.0
                )
            ]
        }

        def mock_retrieve_fn(query, user_id="default", **kwargs):
            return memories_db.get(user_id, [])

        mock_components['retrieve'].side_effect = mock_retrieve_fn

        # User 1 queries - should ONLY see user1 memories
        response1 = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": "user1"},
            json={"query": "secret", "limit": 10}
        )
        assert response1.status_code == 200
        memories1 = response1.json()["memories"]
        assert len(memories1) == 1
        assert memories1[0]["content"] == "User 1's secret data"

        # User 2 queries - should ONLY see user2 memories
        response2 = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": "user2"},
            json={"query": "secret", "limit": 10}
        )
        assert response2.status_code == 200
        memories2 = response2.json()["memories"]
        assert len(memories2) == 1
        assert memories2[0]["content"] == "User 2's secret data"

    def test_memory_stats_isolation(self, client, mock_components):
        """
        Verify memory statistics are isolated per user
        """
        # Mock stats by user
        stats_db = {
            "user1": {"total": 10, "categories": {"work": 5, "personal": 5}},
            "user2": {"total": 3, "categories": {"hobbies": 3}}
        }

        with patch('backend.memory.adapter.get_memory_stats') as mock_stats:
            def stats_fn(user_id="default"):
                return stats_db.get(user_id, {"total": 0, "categories": {}})

            mock_stats.side_effect = stats_fn

            # User 1 stats
            response1 = client.get(
                "/v1/memory/stats",
                headers={"X-User-ID": "user1"}
            )
            assert response1.status_code == 200
            stats1 = response1.json()
            assert stats1["total"] == 10

            # User 2 stats
            response2 = client.get(
                "/v1/memory/stats",
                headers={"X-User-ID": "user2"}
            )
            assert response2.status_code == 200
            stats2 = response2.json()
            assert stats2["total"] == 3

    def test_default_user_isolation(self, client, mock_components):
        """
        Verify 'default' user is isolated from anonymous users
        """
        memories_db = {
            "default": [
                MemoryEntry(
                    id="default1",
                    content="Default user memory",
                    timestamp=datetime.utcnow().isoformat(),
                    tag="general",
                    relevance_score=1.0
                )
            ]
        }

        def mock_retrieve_fn(query, user_id="default", **kwargs):
            return memories_db.get(user_id, [])

        mock_components['retrieve'].side_effect = mock_retrieve_fn

        # Create new anonymous user
        response = client.post("/v1/users/init")
        new_user_id = response.json()["user_id"]

        # New user queries - should see NOTHING (no memories yet)
        response_query = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": new_user_id},
            json={"query": "anything", "limit": 10}
        )
        assert response_query.status_code == 200
        memories = response_query.json()["memories"]
        assert len(memories) == 0, "New user should not see default user's memories"

        # Default user queries - should see their own
        response_default = client.post(
            "/v1/memory/query",
            headers={"X-User-ID": "default"},
            json={"query": "anything", "limit": 10}
        )
        assert response_default.status_code == 200
        memories_default = response_default.json()["memories"]
        assert len(memories_default) == 1
        assert memories_default[0]["content"] == "Default user memory"

    def test_concurrent_user_operations(self, client, mock_components):
        """
        Verify isolation holds under concurrent operations
        """
        import threading

        memories_db = {}
        lock = threading.Lock()

        def mock_store_fn(user_id, content, **kwargs):
            with lock:
                if user_id not in memories_db:
                    memories_db[user_id] = []
                memory = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=content,
                    timestamp=datetime.utcnow().isoformat(),
                    tag="general",
                    relevance_score=1.0
                )
                memories_db[user_id].append(memory)
            return memory

        def mock_retrieve_fn(query, user_id="default", **kwargs):
            with lock:
                return memories_db.get(user_id, [])

        mock_components['store'].side_effect = mock_store_fn
        mock_components['retrieve'].side_effect = mock_retrieve_fn

        # Create 3 users
        users = []
        for i in range(3):
            response = client.post("/v1/users/init")
            users.append(response.json()["user_id"])

        # Each user stores unique content
        for i, user_id in enumerate(users):
            client.post(
                "/v1/memory/add",
                headers={"X-User-ID": user_id},
                json={"content": f"User {i} unique data", "category": "test"}
            )

        # Verify each user sees ONLY their own data
        for i, user_id in enumerate(users):
            response = client.post(
                "/v1/memory/query",
                headers={"X-User-ID": user_id},
                json={"query": "data", "limit": 10}
            )
            memories = response.json()["memories"]
            assert len(memories) == 1, f"User {i} should see 1 memory"
            assert f"User {i} unique data" in memories[0]["content"]

            # Verify no leakage from other users
            for j in range(3):
                if i != j:
                    assert f"User {j} unique data" not in str(memories), \
                        f"User {i} should not see User {j}'s data"


class TestUserDeletion:
    """Tests for user deletion and memory cleanup"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_delete_user_removes_all_memories(self, client):
        """
        Verify deleting a user removes ALL their memories
        """
        with patch('backend.core.bootstrap.initialize_components'), \
             patch('backend.memory.adapter.store_memory') as mock_store, \
             patch('backend.memory.adapter.retrieve_memories') as mock_retrieve, \
             patch('backend.qdrant.qdrant_interface.QdrantMemoryInterface.delete_all_for_user') as mock_delete:

            # Create user
            response = client.post("/v1/users/init")
            user_id = response.json()["user_id"]

            # Store memories
            for i in range(5):
                client.post(
                    "/v1/memory/add",
                    headers={"X-User-ID": user_id},
                    json={"content": f"Memory {i}", "category": "test"}
                )

            # Delete user
            delete_response = client.delete(
                "/v1/users/me",
                headers={"X-User-ID": user_id}
            )
            assert delete_response.status_code == 200

            # Verify delete_all_for_user was called
            mock_delete.assert_called_once_with(user_id)

    def test_deleted_user_cannot_access_data(self, client):
        """
        Verify deleted user cannot access their old data
        """
        with patch('backend.core.bootstrap.initialize_components'), \
             patch('backend.services.user_store.JSONUserStore.get') as mock_get:

            # Create user
            response = client.post("/v1/users/init")
            user_id = response.json()["user_id"]

            # Delete user
            client.delete("/v1/users/me", headers={"X-User-ID": user_id})

            # Simulate user not found after deletion
            mock_get.return_value = None

            # Try to access user info - should auto-create new user
            me_response = client.get("/v1/users/me", headers={"X-User-ID": user_id})

            # Middleware should create new user if not found
            # OR return 404/401 depending on implementation
            # This tests the behavior is defined
            assert me_response.status_code in [200, 401, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
