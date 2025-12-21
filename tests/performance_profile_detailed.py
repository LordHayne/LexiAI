"""
Detailed performance profiling for LexiAI chat processing.

Adds timing to every major step to identify bottlenecks.
"""

import asyncio
import time
import logging
from backend.core.bootstrap import initialize_components_bundle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("perf_profile")

async def profile_chat_message(message: str, components):
    """Profile a single chat message with detailed timing."""

    print(f"\n{'='*60}")
    print(f"Profiling: {message}")
    print(f"{'='*60}\n")

    from backend.core.chat_processing import _run_chat_logic

    # Overall timing
    start_total = time.time()

    # Step-by-step timing
    timings = {}

    # Import and setup
    try:
        # Call the chat logic (this is what the API calls)
        async for chunk in _run_chat_logic(
            message=message,
            chat_client=components.chat_client,
            vectorstore=components.vectorstore,
            memory=components.memory,
            embeddings=components.embeddings,
            streaming=False,
            user_id="perf_test"
        ):
            response = chunk
            break
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        response = {"response": f"Error: {e}"}

    total_time = (time.time() - start_total) * 1000  # ms

    print(f"\n📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.0f}ms ({total_time/1000:.2f}s)")
    print(f"{'='*60}\n")

    return response, total_time


async def main():
    """Run performance profiling."""

    print("Initializing components...")
    init_start = time.time()
    components = initialize_components_bundle(force_recreate=False)
    init_time = (time.time() - init_start) * 1000
    print(f"✅ Components initialized in {init_time:.0f}ms\n")

    # Test messages (simple ones to avoid web search)
    test_messages = [
        "Hallo!",
        "Was ist Python?",
        "Erkläre mir Rekursion.",
    ]

    results = []

    for msg in test_messages:
        response, elapsed = await profile_chat_message(msg, components)
        results.append({
            "message": msg,
            "time_ms": elapsed,
            "response_length": len(response.get("response", ""))
        })

        # Small delay between tests
        await asyncio.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}\n")

    total = 0
    for result in results:
        print(f"Message: {result['message'][:40]}")
        print(f"  Time: {result['time_ms']:.0f}ms ({result['time_ms']/1000:.2f}s)")
        print(f"  Response length: {result['response_length']} chars")
        print()
        total += result['time_ms']

    avg = total / len(results) if results else 0
    print(f"Average: {avg:.0f}ms ({avg/1000:.2f}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
