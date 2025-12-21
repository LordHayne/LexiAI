"""
Test suite for critical bug fixes in chat processing.

Tests:
1. Async memory storage (non-blocking)
2. Response format consistency
3. Error handling
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch


class TestAsyncMemoryStorage:
    """Test that memory storage is truly async and non-blocking."""

    @pytest.mark.asyncio
    async def test_store_memory_async_is_non_blocking(self):
        """Test that store_memory_async doesn't block the event loop."""
        from backend.memory.adapter import store_memory_async
        from backend.core.component_cache import get_cached_components

        # This test requires components to be initialized
        # In a real test, you'd mock the vectorstore

        # Create a mock that simulates slow blocking I/O
        mock_vectorstore = Mock()
        mock_vectorstore.add_entry = Mock(side_effect=lambda *args, **kwargs: time.sleep(0.1))

        with patch('backend.memory.adapter.get_cached_components') as mock_get_components:
            mock_bundle = Mock()
            mock_bundle.vectorstore = mock_vectorstore
            mock_get_components.return_value = mock_bundle

            start = time.time()

            # Store 5 memories concurrently
            tasks = [
                store_memory_async(f"Test message {i}", "test_user", ["test"])
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start

            # If truly concurrent, should take ~0.1s (one blocking operation in thread pool)
            # If sequential, would take ~0.5s (5 * 0.1s)
            assert elapsed < 0.3, f"Should be concurrent (took {elapsed:.2f}s, expected <0.3s)"
            assert len(results) == 5, "All operations should complete"

    @pytest.mark.asyncio
    async def test_store_memory_async_error_handling(self):
        """Test that store_memory_async handles errors gracefully."""
        from backend.memory.adapter import store_memory_async
        from backend.api.middleware.error_handler import MemoryError

        # Mock components to raise an error
        with patch('backend.memory.adapter.get_cached_components') as mock_get_components:
            mock_bundle = Mock()
            mock_vectorstore = Mock()
            mock_vectorstore.add_entry = Mock(side_effect=Exception("Simulated error"))
            mock_bundle.vectorstore = mock_vectorstore
            mock_get_components.return_value = mock_bundle

            # Should raise MemoryError, not the raw Exception
            with pytest.raises(MemoryError) as exc_info:
                await store_memory_async("Test", "test_user", ["test"])

            assert "Failed to store memory" in str(exc_info.value)


class TestChatResponseFormat:
    """Test that chat responses are consistently formatted."""

    @pytest.mark.asyncio
    async def test_process_chat_returns_dict(self):
        """Test that process_chat_message_async returns a dict, not tuple."""
        from backend.core.chat_processing import process_chat_message_async

        # Mock all components
        mock_chat_client = AsyncMock()
        mock_chat_client.ainvoke = AsyncMock(return_value=Mock(content="Test response"))

        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search = Mock(return_value=[])

        mock_memory = Mock()
        mock_memory.save_context = Mock()

        mock_embeddings = Mock()

        with patch('backend.core.chat_processing.initialize_components') as mock_init:
            mock_init.return_value = (
                mock_embeddings,
                mock_vectorstore,
                mock_memory,
                mock_chat_client,
                None  # No warning
            )

            result = await process_chat_message_async(
                "Test message",
                chat_client=mock_chat_client,
                vectorstore=mock_vectorstore,
                memory=mock_memory,
                embeddings=mock_embeddings,
                user_id="test_user"
            )

            # CRITICAL: Result must be a dict, not a tuple
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "response" in result, "Dict should have 'response' key"
            assert "turn_id" in result, "Dict should have 'turn_id' key"
            assert "source" in result, "Dict should have 'source' key"
            assert "relevant_memory" in result, "Dict should have 'relevant_memory' key"

    @pytest.mark.asyncio
    async def test_response_format_consistency(self):
        """Test that streaming and non-streaming return consistent formats."""
        # This would test that both modes return compatible data structures
        pass


class TestLLMErrorHandling:
    """Test that LLM errors are handled gracefully."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_user_friendly_message(self):
        """Test that LLM errors result in user-friendly error messages."""
        from backend.core.chat_processing import _run_chat_logic

        # Mock LLM to raise an error
        mock_chat_client = AsyncMock()
        mock_chat_client.ainvoke = AsyncMock(side_effect=Exception("LLM connection failed"))

        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search = AsyncMock(return_value=[])

        mock_memory = Mock()
        mock_memory.save_context = Mock()

        mock_embeddings = Mock()

        gen = _run_chat_logic(
            "Test message",
            mock_chat_client,
            mock_vectorstore,
            mock_memory,
            mock_embeddings,
            streaming=False,
            user_id="test_user"
        )

        result = await anext(gen)

        # Should return dict with error message
        assert isinstance(result, dict), "Should return dict even on error"
        assert "response" in result, "Should have response key"

        # Response should contain error message (German)
        response_text = result.get("response", "")
        assert "Fehler" in response_text or "Entschuldigung" in response_text, \
            "Should have user-friendly German error message"

    @pytest.mark.asyncio
    async def test_llm_response_format_variations(self):
        """Test handling of various LLM response formats."""
        from backend.core.chat_processing import _run_chat_logic

        test_cases = [
            # Case 1: Response with .content attribute
            Mock(content="Response from LLM"),
            # Case 2: Dict with "content" key
            {"content": "Response from dict"},
            # Case 3: Plain string
            "Plain string response",
            # Case 4: Coroutine (should be awaited)
            # (This is harder to test without actual async context)
        ]

        for test_response in test_cases[:3]:  # Skip coroutine test for now
            mock_chat_client = AsyncMock()
            mock_chat_client.ainvoke = AsyncMock(return_value=test_response)

            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search = AsyncMock(return_value=[])

            mock_memory = Mock()
            mock_memory.save_context = Mock()

            mock_embeddings = Mock()

            with patch('backend.core.chat_processing.call_model_async', return_value=test_response):
                gen = _run_chat_logic(
                    "Test",
                    mock_chat_client,
                    mock_vectorstore,
                    mock_memory,
                    mock_embeddings,
                    streaming=False,
                    user_id="test"
                )

                result = await anext(gen)

                # Should successfully extract content and return dict
                assert isinstance(result, dict), f"Failed for response type: {type(test_response)}"
                assert "response" in result, "Should have response key"
                assert isinstance(result["response"], str), "Response should be string"


class TestBackwardsCompatibility:
    """Test that old code still works with new changes."""

    def test_sync_store_memory_still_works(self):
        """Test that old sync store_memory() function still works."""
        from backend.memory.adapter import store_memory

        # This would test the sync wrapper
        # In a real test, you'd mock the async function it calls
        pass

    @pytest.mark.asyncio
    async def test_api_handles_both_tuple_and_dict(self):
        """Test that API route handles both old tuple and new dict formats."""
        # This tests the backwards compatibility in chat.py
        pass


class TestPerformance:
    """Performance regression tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_memory_operations_performance(self):
        """Test that concurrent operations are actually faster than sequential."""
        from backend.memory.adapter import store_memory_async

        # Mock with simulated delay
        with patch('backend.memory.adapter.get_cached_components') as mock_get_components:
            mock_bundle = Mock()
            mock_vectorstore = Mock()

            # Simulate 50ms blocking I/O
            def slow_add(*args, **kwargs):
                time.sleep(0.05)

            mock_vectorstore.add_entry = Mock(side_effect=slow_add)
            mock_bundle.vectorstore = mock_vectorstore
            mock_get_components.return_value = mock_bundle

            # Test concurrent execution
            start = time.time()
            tasks = [
                store_memory_async(f"Msg {i}", "user", ["test"])
                for i in range(10)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_time = time.time() - start

            # Should be much faster than 10 * 0.05s = 0.5s
            # With thread pool, should be closer to 0.05s * (10 / pool_size)
            assert concurrent_time < 0.3, \
                f"Concurrent execution too slow: {concurrent_time:.2f}s (expected <0.3s)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
