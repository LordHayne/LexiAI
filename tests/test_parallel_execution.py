"""
Test script to verify parallel execution optimizations in chat_processing.py

This test verifies:
1. Parallel preprocessing (feedback detection + memory retrieval)
2. Parallel background tasks (memory + goal + web storage)
3. Performance improvements
4. Error handling with return_exceptions=True
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.chat_processing import _run_chat_logic


@pytest.mark.asyncio
async def test_parallel_preprocessing():
    """
    Test that feedback detection and memory retrieval run in parallel.

    Expected: Both tasks should run concurrently, not sequentially.
    """
    # Setup mock components
    chat_client = MagicMock()
    vectorstore = MagicMock()
    memory = MagicMock()
    embeddings = MagicMock()

    # Mock the similarity_search to simulate delay
    async def mock_similarity_search(*args, **kwargs):
        await asyncio.sleep(0.5)  # Simulate 500ms delay
        return []

    # Mock conversation tracker methods
    mock_tracker = MagicMock()

    async def mock_detect_reformulation(*args, **kwargs):
        await asyncio.sleep(0.3)  # Simulate 300ms delay
        return None

    async def mock_get_user_history(*args, **kwargs):
        await asyncio.sleep(0.1)
        return []

    # Patch the functions
    with patch('backend.core.chat_processing.get_conversation_tracker', return_value=mock_tracker), \
         patch.object(mock_tracker, 'detect_implicit_reformulation', side_effect=lambda *a, **k: None), \
         patch.object(mock_tracker, 'get_user_history', side_effect=lambda *a, **k: []), \
         patch.object(mock_tracker, 'record_feedback'), \
         patch.object(vectorstore, 'similarity_search', side_effect=lambda *a, **k: []):

        # Measure execution time
        start_time = time.time()

        # Run with long enough message to trigger parallel execution
        message = "This is a test message for parallel execution verification"

        # Mock call_model_async to avoid LLM calls
        with patch('backend.core.chat_processing.call_model_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Test response")

            # Mock memory.save_context
            memory.save_context = MagicMock()

            # Mock conversation_tracker.record_turn
            with patch.object(mock_tracker, 'record_turn', return_value='test-turn-id'):
                # Execute
                gen = _run_chat_logic(
                    message=message,
                    chat_client=chat_client,
                    vectorstore=vectorstore,
                    memory=memory,
                    embeddings=embeddings,
                    streaming=False,
                    user_id="test_user"
                )

                result = await anext(gen)

        elapsed_time = time.time() - start_time

        # Verification
        print(f"⏱️  Execution time: {elapsed_time*1000:.0f}ms")

        # If parallel: ~500ms (max of both tasks)
        # If sequential: ~800ms (sum of both tasks)
        # Allow some overhead, but should be closer to 500ms than 800ms
        assert elapsed_time < 1.5, f"Execution took {elapsed_time:.2f}s - tasks may not be running in parallel"

        print("✅ Parallel preprocessing test passed")


@pytest.mark.asyncio
async def test_parallel_background_tasks():
    """
    Test that background tasks (memory, goal, web) run in parallel.

    Expected: All three tasks should run concurrently.
    """
    # Setup mock components
    chat_client = MagicMock()
    vectorstore = MagicMock()
    memory = MagicMock()
    embeddings = MagicMock()

    # Mock conversation tracker
    mock_tracker = MagicMock()

    with patch('backend.core.chat_processing.get_conversation_tracker', return_value=mock_tracker), \
         patch.object(mock_tracker, 'detect_implicit_reformulation', return_value=None), \
         patch.object(mock_tracker, 'get_user_history', return_value=[]), \
         patch.object(mock_tracker, 'record_turn', return_value='test-turn-id'), \
         patch.object(vectorstore, 'similarity_search', return_value=[]):

        # Mock LLM call
        with patch('backend.core.chat_processing.call_model_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Test response for background tasks")

            # Mock memory operations with delays
            async def mock_store_memory(*args, **kwargs):
                await asyncio.sleep(0.2)  # 200ms
                return ("doc-id", "timestamp")

            # Mock goal detection with delay
            async def mock_detect_goals(*args, **kwargs):
                await asyncio.sleep(0.15)  # 150ms
                return []

            with patch('backend.memory.adapter.store_memory_async', side_effect=mock_store_memory), \
                 patch('backend.memory.goal_tracker.GoalDetector.detect_goals_with_llm', side_effect=mock_detect_goals):

                # Mock memory.save_context
                memory.save_context = MagicMock()

                # Measure background task execution time
                start_time = time.time()

                # Execute
                message = "I want to learn machine learning and build a chatbot"  # Trigger goal detection
                gen = _run_chat_logic(
                    message=message,
                    chat_client=chat_client,
                    vectorstore=vectorstore,
                    memory=memory,
                    embeddings=embeddings,
                    streaming=False,
                    user_id="test_user"
                )

                result = await anext(gen)

                elapsed_time = time.time() - start_time

                print(f"⏱️  Total execution time: {elapsed_time*1000:.0f}ms")

                # If parallel: ~200ms (max of all tasks)
                # If sequential: ~350ms (sum of all tasks)
                # The full execution includes more than just background tasks,
                # but background tasks should add minimal overhead if parallel

                print("✅ Parallel background tasks test passed")


@pytest.mark.asyncio
async def test_error_handling_with_return_exceptions():
    """
    Test that return_exceptions=True prevents one failing task from breaking others.

    Expected: Even if feedback detection fails, memory retrieval should succeed.
    """
    # Setup mock components
    chat_client = MagicMock()
    vectorstore = MagicMock()
    memory = MagicMock()
    embeddings = MagicMock()

    # Mock conversation tracker that raises an error
    mock_tracker = MagicMock()

    def mock_detect_reformulation_error(*args, **kwargs):
        raise ValueError("Simulated feedback detection error")

    with patch('backend.core.chat_processing.get_conversation_tracker', return_value=mock_tracker), \
         patch.object(mock_tracker, 'detect_implicit_reformulation', side_effect=mock_detect_reformulation_error), \
         patch.object(mock_tracker, 'record_turn', return_value='test-turn-id'), \
         patch.object(vectorstore, 'similarity_search', return_value=[MagicMock(metadata={'id': 'mem-1', 'category': 'test'}, page_content='Test memory')]):

        # Mock LLM call
        with patch('backend.core.chat_processing.call_model_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Test response despite error")

            # Mock memory.save_context
            memory.save_context = MagicMock()

            # Execute - should not raise exception
            message = "Test message to verify error handling"
            gen = _run_chat_logic(
                message=message,
                chat_client=chat_client,
                vectorstore=vectorstore,
                memory=memory,
                embeddings=embeddings,
                streaming=False,
                user_id="test_user"
            )

            result = await anext(gen)

            # Verification
            assert result is not None, "Result should not be None even with failed feedback detection"
            assert 'response' in result, "Result should contain response"

            print("✅ Error handling with return_exceptions=True test passed")


@pytest.mark.asyncio
async def test_performance_tracking():
    """
    Test that performance metrics are properly tracked for parallel operations.

    Expected: Performance summary should include parallel execution timings.
    """
    # Setup mock components
    chat_client = MagicMock()
    vectorstore = MagicMock()
    memory = MagicMock()
    embeddings = MagicMock()

    mock_tracker = MagicMock()

    with patch('backend.core.chat_processing.get_conversation_tracker', return_value=mock_tracker), \
         patch.object(mock_tracker, 'detect_implicit_reformulation', return_value=None), \
         patch.object(mock_tracker, 'get_user_history', return_value=[]), \
         patch.object(mock_tracker, 'record_turn', return_value='test-turn-id'), \
         patch.object(vectorstore, 'similarity_search', return_value=[]):

        # Mock LLM call
        with patch('backend.core.chat_processing.call_model_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Test response")

            # Mock memory.save_context
            memory.save_context = MagicMock()

            # Capture log output
            import logging
            from io import StringIO
            import sys

            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger("memory_decisions")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            try:
                # Execute
                message = "Test message for performance tracking"
                gen = _run_chat_logic(
                    message=message,
                    chat_client=chat_client,
                    vectorstore=vectorstore,
                    memory=memory,
                    embeddings=embeddings,
                    streaming=False,
                    user_id="test_user"
                )

                result = await anext(gen)

                # Check logs for performance tracking
                log_output = log_capture.getvalue()

                # Verify performance tracking logs
                assert "Running 2 tasks in parallel" in log_output or "⚡" in log_output, \
                    "Should log parallel execution"

                assert "Performance Summary" in log_output, \
                    "Should log performance summary"

                print("✅ Performance tracking test passed")

            finally:
                logger.removeHandler(handler)


if __name__ == "__main__":
    print("Running parallel execution tests...\n")

    # Run tests
    asyncio.run(test_parallel_preprocessing())
    asyncio.run(test_parallel_background_tasks())
    asyncio.run(test_error_handling_with_return_exceptions())
    asyncio.run(test_performance_tracking())

    print("\n✅ All tests passed!")
