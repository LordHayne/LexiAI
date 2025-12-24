#!/usr/bin/env python3
"""
Quick test for Hybrid Search functionality.
"""

import sys
sys.path.insert(0, '/Users/thomas/Desktop/LexiAI_new')

from backend.core.bootstrap import ComponentInitializer

print("🔍 Testing Hybrid Search...")
print("=" * 60)

# Initialize components
initializer = ComponentInitializer(force_recreate=False)
bundle = initializer.initialize_all()

vectorstore = bundle.vectorstore

# Test query
test_query = "machine learning frameworks"

print(f"\n📊 Test Query: '{test_query}'")
print("-" * 60)

# 1. Semantic Search Only
print("\n1️⃣ SEMANTIC SEARCH ONLY:")
semantic_results = vectorstore.query_memories(test_query, limit=5)
print(f"   Found {len(semantic_results)} results")
for i, entry in enumerate(semantic_results[:3], 1):
    content_preview = entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
    print(f"   {i}. [Score: {entry.relevance:.3f}] {content_preview}")

# 2. Hybrid Search
print("\n2️⃣ HYBRID SEARCH (Semantic + Keyword):")
try:
    hybrid_results = vectorstore.hybrid_search(
        test_query,
        limit=5,
        dense_weight=0.7,
        sparse_weight=0.3,
        fusion_strategy="rrf"
    )
    print(f"   Found {len(hybrid_results)} results")
    for i, entry in enumerate(hybrid_results[:3], 1):
        content_preview = entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
        print(f"   {i}. [Score: {entry.relevance:.3f}] {content_preview}")

    print("\n✅ Hybrid search working!")

except Exception as e:
    print(f"\n❌ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
