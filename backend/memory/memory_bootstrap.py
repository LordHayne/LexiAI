# memory_bootstrap.py

from backend.memory import ClusteredCategoryPredictor
from backend.embeddings.embedding_model import OllamaEmbeddingModel
from backend.qdrant.qdrant_interface import QdrantMemoryInterface

def get_predictor():
    embedding_model = OllamaEmbeddingModel()
    collection_name = "lexi_memory"

    qdrant_interface = QdrantMemoryInterface(
        collection_name=collection_name,
        embeddings=embedding_model
    )

    return ClusteredCategoryPredictor(qdrant=qdrant_interface, embedding_model=embedding_model)
