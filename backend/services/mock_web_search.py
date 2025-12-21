"""
Mock Web Search Service for testing without Tavily API key
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MockWebSearchService:
    """
    Mock web search service that simulates web search for testing.
    Always returns enabled=True and mock results.
    """

    def __init__(self):
        self.enabled = True
        logger.info("✅ Mock Web Search Service initialized (for testing)")

    def is_enabled(self) -> bool:
        """Always return True for mock service"""
        return True

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return mock search results for testing.
        """
        logger.info(f"🔍 Mock web search for: '{query}'")

        # Simulated results
        mock_results = [
            {
                "title": f"Mock Result 1 for: {query}",
                "url": "https://example.com/result1",
                "content": f"This is a mock search result for the query: {query}. Python 3.14 introduces pattern matching enhancements.",
                "score": 0.95
            },
            {
                "title": f"Mock Result 2 for: {query}",
                "url": "https://example.com/result2",
                "content": f"Another mock result about {query}. Features include improved type hints and performance optimizations.",
                "score": 0.88
            }
        ]

        return {
            "query": query,
            "results": mock_results[:max_results],
            "answer": f"Mock AI summary for '{query}': Based on the latest information, there are several new features.",
            "timestamp": datetime.utcnow().isoformat(),
            "sources": ["https://example.com/result1", "https://example.com/result2"]
        }

    def search_batch(self, queries: list, **kwargs) -> Dict[str, Any]:
        """Mock batch search"""
        logger.info(f"🔍 Mock batch search for {len(queries)} queries")
        return {
            "results": [
                {"query": q, "mock": True} for q in queries
            ]
        }


# Global singleton
_mock_service = None


def get_mock_web_search_service():
    """Get or create the global mock web search service"""
    global _mock_service
    if _mock_service is None:
        _mock_service = MockWebSearchService()
    return _mock_service
