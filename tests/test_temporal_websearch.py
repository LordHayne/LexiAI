"""
Focused test for temporal query web search detection
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.bootstrap import initialize_components_bundle
from backend.core.chat_processing import process_chat_message_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable DEBUG logging for web search decision module
logging.getLogger("backend.core.llm_web_search_decision").setLevel(logging.DEBUG)
logging.getLogger("backend.core.chat_processing").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


async def test_temporal_query():
    """Test temporal query that SHOULD trigger web search"""

    print("\n" + "="*80)
    print("TEMPORAL QUERY WEB SEARCH TEST")
    print("="*80)

    # Initialize
    logger.info("Initializing components...")
    components = initialize_components_bundle()
    logger.info("✅ Components ready")

    # Test query with temporal indicator
    query = "Was sind die neuesten Python Features in 2025?"

    print(f"\n📝 Query: '{query}'")
    print("✅ Expected: Web search SHOULD be triggered (has '2025' and 'neuesten')")
    print("\n" + "-"*80)
    print("PROCESSING...")
    print("-"*80 + "\n")

    # Process
    response = await process_chat_message_async(
        message=query,
        chat_client=components.chat_client,
        vectorstore=components.vectorstore,
        memory=components.memory,
        embeddings=components.embeddings,
        user_id="temporal_test"
    )

    print("\n" + "-"*80)
    print("RESULT")
    print("-"*80)

    if isinstance(response, dict):
        response_text = response.get("response", "")
        metadata = response.get("metadata", {})

        web_search_triggered = metadata.get("web_search_used", False)

        print(f"📊 Web Search Triggered: {web_search_triggered}")
        print(f"📝 Response Length: {len(response_text)} chars")
        print(f"💬 Response Preview: {response_text[:200]}...")

        if web_search_triggered:
            print("\n✅ SUCCESS: Web search was triggered for temporal query")
            return True
        else:
            print("\n❌ FAIL: Web search was NOT triggered (should be for '2025')")
            print("\n🔍 DEBUG INFO:")
            print(f"   - Query contains '2025': {'2025' in query}")
            print(f"   - Query contains 'neuesten': {'neuesten' in query.lower()}")
            print(f"   - Metadata: {metadata}")
            return False
    else:
        print(f"❌ Unexpected response format: {type(response)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_temporal_query())
    sys.exit(0 if success else 1)
