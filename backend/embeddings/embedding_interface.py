from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Synchrones Embedding für ein Text-Query"""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch-Verarbeitung von Embeddings (optional überschreibbar)"""
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Optional: Asynchrones Embedding für ein Text-Query"""
        raise NotImplementedError("Async embedding nicht implementiert")
