import logging
import httpx
import time
from typing import List, Optional
from backend.utils.config import get_config_value
from backend.embeddings.embedding_interface import EmbeddingModel

logger = logging.getLogger("EmbeddingModel")


class OllamaEmbeddingModel(EmbeddingModel):
    """
    Ollama Embedding Model with connection pooling.

    PERFORMANCE: Uses persistent HTTP clients with connection pooling
    to reduce overhead and improve performance.
    """

    def __init__(self):
        self.model = get_config_value("embedding_model", default="nomic-embed-text")
        self.url = get_config_value("embedding_url", default="http://localhost:11434")
        self.dim = int(get_config_value("embedding_dim", default=768))

        # Persistent HTTP clients with connection pooling
        timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._limits = limits

    def _get_sync_client(self) -> httpx.Client:
        """Get or create persistent sync HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                timeout=self._timeout,
                limits=self._limits
            )
        return self._sync_client

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create persistent async HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=self._limits
            )
        return self._async_client

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for text using persistent HTTP client.

        PERFORMANCE: Reuses HTTP connection across requests.
        """
        start = time.time()
        try:
            client = self._get_sync_client()
            response = client.post(
                f"{self.url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            elapsed_ms = (time.time() - start) * 1000
            if isinstance(embedding, list) and len(embedding) == self.dim and all(isinstance(x, (float, int)) for x in embedding):
                logger.debug(f"⏱️ Embedding query ({len(text)} chars): {elapsed_ms:.0f}ms")
                return embedding
            else:
                logger.error("Ungültiges Embedding-Format: %s", embedding)
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Fehler beim Abrufen des Embeddings ({elapsed_ms:.0f}ms): {e}")
        return [0.0 for _ in range(self.dim)]

    async def aembed_query(self, text: str) -> List[float]:
        """
        Generate embedding for text using persistent async HTTP client.

        PERFORMANCE: Reuses HTTP connection across requests.
        """
        start = time.time()
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            elapsed_ms = (time.time() - start) * 1000
            if isinstance(embedding, list) and len(embedding) == self.dim and all(isinstance(x, (float, int)) for x in embedding):
                logger.debug(f"⏱️ Async embedding query ({len(text)} chars): {elapsed_ms:.0f}ms")
                return embedding
            else:
                logger.error("Ungültiges Async-Embedding-Format: %s", embedding)
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Async-Fehler beim Abrufen des Embeddings ({elapsed_ms:.0f}ms): {e}")
        return [0.0 for _ in range(self.dim)]

    def close(self):
        """Close HTTP clients (cleanup)."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self):
        """Close async HTTP client (cleanup)."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self._sync_client and not self._sync_client.is_closed:
                self._sync_client.close()
        except Exception:
            pass
