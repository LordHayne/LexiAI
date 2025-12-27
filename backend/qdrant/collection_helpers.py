import logging

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger("qdrant_collection_helpers")


def ensure_collection(
    client,
    collection_name: str,
    vector_size: int = 1,
    distance: models.Distance = models.Distance.COSINE,
) -> bool:
    """
    Ensure collection exists with the expected vector size.

    Returns True if collection was created, False if it already existed.
    """
    try:
        info = client.get_collection(collection_name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != vector_size:
            logger.warning(
                "Collection '%s' has vector size %s (expected %s)",
                collection_name,
                existing_dim,
                vector_size,
            )
        return False
    except UnexpectedResponse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        return True
