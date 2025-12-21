# backend/qdrant/__init__.py

from .client_wrapper import create_qdrant_client
from .qdrant_interface import QdrantMemoryInterface

__all__ = ["create_qdrant_client", "QdrantMemoryInterface"]
