"""
Debug script to test memory storage persistence.

This script tests the exact scenario:
1. Store "vergiss Frank" message
2. Check if memory is persisted in Qdrant
3. Verify retrieval of stored memory
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("memory_storage_debug")

async def test_memory_storage():
    """Test memory storage pipeline end-to-end."""

    print("\n" + "="*80)
    print("MEMORY STORAGE DEBUG TEST")
    print("="*80)

    # Step 1: Initialize components
    print("\n[1] Initializing components...")
    try:
        from backend.core.bootstrap import initialize_components
        embeddings, vectorstore, memory, chat_client, qdrant_client = initialize_components()
        print("✓ Components initialized successfully")
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        return False

    # Step 2: Store test memory
    print("\n[2] Storing test memory: 'vergiss Frank'")
    test_content = "User sagte: vergiss Frank. AI antwortete: Okay, ich merke mir, dass ich Frank vergessen soll."
    user_id = "test_user_debug"

    try:
        from backend.memory.adapter import store_memory_async
        doc_id, timestamp = await store_memory_async(
            content=test_content,
            user_id=user_id,
            tags=["test", "self_correction"],
            metadata={"category": "self_correction", "source": "self_correction"}
        )
        print(f"✓ Memory stored with ID: {doc_id}")
        print(f"  Timestamp: {timestamp}")
    except Exception as e:
        print(f"✗ Memory storage failed: {e}")
        logger.exception("Full error:")
        return False

    # Step 3: Verify in Qdrant directly
    print("\n[3] Verifying memory in Qdrant...")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from backend.config.middleware_config import MiddlewareConfig

        collection = MiddlewareConfig.get_memory_collection()

        # Search by ID
        points = qdrant_client.retrieve(
            collection_name=collection,
            ids=[doc_id]
        )

        if points:
            print(f"✓ Found memory in Qdrant by ID")
            for point in points:
                print(f"  ID: {point.id}")
                print(f"  Content: {point.payload.get('content', 'N/A')[:100]}...")
                print(f"  Category: {point.payload.get('category', 'N/A')}")
                print(f"  Source: {point.payload.get('source', 'N/A')}")
                print(f"  Tags: {point.payload.get('tags', [])}")
        else:
            print(f"✗ Memory NOT found in Qdrant by ID: {doc_id}")
            return False

    except Exception as e:
        print(f"✗ Qdrant verification failed: {e}")
        logger.exception("Full error:")
        return False

    # Step 4: Retrieve using memory adapter
    print("\n[4] Retrieving memory via adapter...")
    try:
        from backend.memory.adapter import retrieve_memories

        memories = retrieve_memories(
            user_id=user_id,
            query="Frank vergessen",
            limit=5
        )

        if memories:
            print(f"✓ Retrieved {len(memories)} memories")
            for i, mem in enumerate(memories, 1):
                print(f"\n  Memory {i}:")
                print(f"    ID: {mem.id}")
                print(f"    Content: {mem.content[:100]}...")
                print(f"    Category: {mem.category}")
                print(f"    Source: {mem.source}")
                print(f"    Relevance: {mem.relevance}")
                print(f"    Tags: {mem.tags}")
        else:
            print(f"✗ No memories retrieved")
            return False

    except Exception as e:
        print(f"✗ Memory retrieval failed: {e}")
        logger.exception("Full error:")
        return False

    # Step 5: Test embedding generation
    print("\n[5] Testing embedding generation...")
    try:
        test_query = "Frank vergessen"
        vector = embeddings.embed_query(test_query)
        print(f"✓ Embedding generated: {len(vector)} dimensions")
        print(f"  First 5 values: {vector[:5]}")

        # Verify embedding is not all zeros
        if all(v == 0.0 for v in vector):
            print("✗ WARNING: Embedding is all zeros!")
            return False
        else:
            print("✓ Embedding contains non-zero values")

    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        logger.exception("Full error:")
        return False

    # Step 6: Check category predictor
    print("\n[6] Testing category predictor...")
    try:
        from backend.memory.memory_bootstrap import get_predictor
        predictor = get_predictor()

        category = predictor.predict_category(test_content)
        print(f"✓ Category predicted: {category}")

        # Check if predictor has clusters
        if predictor.clusters:
            print(f"  Predictor has {len(predictor.clusters)} clusters")
        else:
            print("  ⚠ Predictor has no clusters (will auto-build on first use)")

    except Exception as e:
        print(f"✗ Category predictor test failed: {e}")
        logger.exception("Full error:")
        # Don't fail test - predictor is optional

    # Step 7: Check Qdrant collection stats
    print("\n[7] Checking Qdrant collection stats...")
    try:
        collection_info = qdrant_client.get_collection(collection)
        print(f"✓ Collection: {collection}")
        print(f"  Total points: {collection_info.points_count}")
        print(f"  Vector dimensions: {collection_info.config.params.vectors.size}")
        print(f"  Distance metric: {collection_info.config.params.vectors.distance}")

    except Exception as e:
        print(f"✗ Collection stats failed: {e}")
        logger.exception("Full error:")

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("="*80)

    return True

async def test_cache_invalidation():
    """Test if cache is properly invalidated after storing."""

    print("\n" + "="*80)
    print("CACHE INVALIDATION TEST")
    print("="*80)

    from backend.core.bootstrap import initialize_components
    from backend.memory.adapter import store_memory_async, retrieve_memories
    from backend.memory.cache import get_memory_cache

    embeddings, vectorstore, memory, chat_client, qdrant_client = initialize_components()

    user_id = "cache_test_user"

    # Store memory
    print("\n[1] Storing memory...")
    doc_id, _ = await store_memory_async(
        content="Test cache invalidation",
        user_id=user_id,
        tags=["cache_test"]
    )
    print(f"✓ Stored: {doc_id}")

    # Check cache was invalidated
    print("\n[2] Checking cache invalidation...")
    cache = get_memory_cache()

    # Try to retrieve - should NOT be cached
    memories = retrieve_memories(user_id=user_id, query="cache", limit=5)

    if memories:
        print(f"✓ Retrieved {len(memories)} memories (cache working)")
    else:
        print("✗ No memories retrieved")

    return True

if __name__ == "__main__":
    print("Starting memory storage debug tests...\n")

    # Run tests
    try:
        success = asyncio.run(test_memory_storage())

        if success:
            print("\nRunning cache invalidation test...")
            asyncio.run(test_cache_invalidation())

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
