"""
Web Search Service using Tavily API

Provides web search capabilities for Lexi to access current information from the internet.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import os

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily not installed. Web search will not be available.")


logger = logging.getLogger(__name__)


class WebSearchService:
    """
    Service for performing web searches using Tavily API.

    Features:
    - Semantic search optimized for AI applications
    - Content extraction and summarization
    - Source citations
    - Configurable search depth and result count
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search service.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.enabled = TAVILY_AVAILABLE and bool(self.api_key)

        if not TAVILY_AVAILABLE:
            logger.warning("Tavily package not installed. Install with: pip install tavily-python")
            self.client = None
        elif not self.api_key:
            logger.warning("Tavily API key not configured. Set TAVILY_API_KEY environment variable.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("✅ Tavily Web Search Service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.client = None
                self.enabled = False

    def is_enabled(self) -> bool:
        """Check if web search is enabled and available."""
        return self.enabled and self.client is not None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        include_images: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily.

        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 5)
            search_depth: "basic" or "advanced" (default: "basic")
            include_raw_content: Include full page content (default: False)
            include_images: Include image results (default: False)

        Returns:
            Dictionary containing search results with structure:
            {
                "query": str,
                "results": List[Dict],
                "answer": str (AI-generated summary),
                "timestamp": str,
                "sources": List[str]
            }
        """
        if not self.is_enabled():
            logger.warning("Web search is not enabled")
            return {
                "query": query,
                "results": [],
                "answer": "Web-Suche ist nicht verfügbar. Bitte Tavily API-Key konfigurieren.",
                "timestamp": datetime.utcnow().isoformat(),
                "sources": [],
                "error": "Web search not configured"
            }

        try:
            logger.info(f"🔍 Performing web search: '{query}' (max_results={max_results}, depth={search_depth})")

            # Perform search using Tavily
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
                include_images=include_images,
                **kwargs
            )

            # Extract and format results
            results = []
            sources = []

            for result in response.get("results", []):
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "published_date": result.get("published_date")
                }
                results.append(formatted_result)
                sources.append(result.get("url", ""))

            # Get AI-generated answer if available
            answer = response.get("answer", "")

            search_result = {
                "query": query,
                "results": results,
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat(),
                "sources": sources,
                "result_count": len(results)
            }

            logger.info(f"✅ Web search completed: {len(results)} results found")
            return search_result

        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return {
                "query": query,
                "results": [],
                "answer": f"Fehler bei der Web-Suche: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "sources": [],
                "error": str(e)
            }

    def search_sync(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of search (for non-async contexts).

        Args:
            query: The search query
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced"

        Returns:
            Dictionary containing search results
        """
        if not self.is_enabled():
            logger.warning("Web search is not enabled")
            return {
                "query": query,
                "results": [],
                "answer": "Web-Suche ist nicht verfügbar.",
                "timestamp": datetime.utcnow().isoformat(),
                "sources": []
            }

        try:
            logger.info(f"🔍 Performing sync web search: '{query}'")

            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                **kwargs
            )

            results = []
            sources = []

            for result in response.get("results", []):
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                }
                results.append(formatted_result)
                sources.append(result.get("url", ""))

            return {
                "query": query,
                "results": results,
                "answer": response.get("answer", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "sources": sources,
                "result_count": len(results)
            }

        except Exception as e:
            logger.error(f"Error performing sync web search: {e}")
            return {
                "query": query,
                "results": [],
                "answer": f"Fehler: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "sources": []
            }

    def format_search_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """
        Format search results for inclusion in LLM context.

        Args:
            search_result: Result from search() method

        Returns:
            Formatted string for LLM consumption
        """
        if not search_result.get("results"):
            return f"Keine Suchergebnisse für: {search_result.get('query', '')}"

        formatted = f"🔍 Web-Suchergebnisse für: '{search_result['query']}'\n\n"

        # Add AI-generated answer if available
        if search_result.get("answer"):
            formatted += f"📝 Zusammenfassung:\n{search_result['answer']}\n\n"

        # Add individual results
        formatted += "📄 Quellen:\n"
        for i, result in enumerate(search_result["results"], 1):
            formatted += f"\n{i}. {result['title']}\n"
            formatted += f"   URL: {result['url']}\n"
            formatted += f"   {result['content'][:300]}...\n"

        formatted += f"\nℹ️ Insgesamt {search_result['result_count']} Ergebnisse gefunden"
        formatted += f" (Stand: {search_result['timestamp']})\n"

        return formatted

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the web search service.

        Returns:
            Dictionary with service status information
        """
        return {
            "enabled": self.enabled,
            "tavily_available": TAVILY_AVAILABLE,
            "api_key_configured": bool(self.api_key),
            "client_initialized": self.client is not None,
            "status": "ok" if self.enabled else "disabled"
        }


# Singleton instance
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service(api_key: Optional[str] = None) -> WebSearchService:
    """
    Get or create the singleton web search service instance.

    Args:
        api_key: Optional Tavily API key

    Returns:
        WebSearchService instance
    """
    global _web_search_service

    if _web_search_service is None:
        _web_search_service = WebSearchService(api_key=api_key)

    return _web_search_service


def reset_web_search_service():
    """Reset the singleton instance (useful for testing)."""
    global _web_search_service
    _web_search_service = None
