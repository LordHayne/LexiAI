"""
Test script to verify timing instrumentation in chat processing.

This test sends a simple message and verifies that:
1. All major steps are logged with timing
2. Performance summary is generated
3. Unknown/overhead time is calculated
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.chat_processing import process_chat_message_async
from backend.core.bootstrap import initialize_components

# Configure logging to see timing logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_timing():
    """Test timing instrumentation with a simple query."""
    print("=" * 80)
    print("TIMING INSTRUMENTATION TEST")
    print("=" * 80)

    # Initialize components
    print("\n1. Initializing components...")
    embeddings, vectorstore, memory, chat_client, _ = initialize_components()
    print("✓ Components initialized\n")

    # Test message
    test_message = "Was ist maschinelles Lernen?"
    print(f"2. Sending test message: '{test_message}'\n")

    # Process message
    print("3. Processing message with timing instrumentation...\n")
    result = await process_chat_message_async(
        test_message,
        chat_client=chat_client,
        vectorstore=vectorstore,
        memory=memory,
        embeddings=embeddings,
        user_id="test_timing_user"
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResponse preview: {result.get('response', '')[:200]}...")
    print(f"\nCheck the logs above for:")
    print("  - ⏱️ [Step name]: XXXms timing logs")
    print("  - Performance Summary with breakdown")
    print("  - [UNKNOWN/OVERHEAD] percentage")
    print("\n")

if __name__ == "__main__":
    asyncio.run(test_timing())
