import time
import logging
from typing import Dict, TYPE_CHECKING
import requests
from qdrant_client import QdrantClient

# LAZY IMPORT: langchain_ollama blockiert wenn Ollama nicht läuft
# Runtime imports werden in den Funktionen gemacht wenn benötigt
if TYPE_CHECKING:
    from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.config.middleware_config import MiddlewareConfig
from backend.api.v1.models.response_models import ComponentStatus

logger = logging.getLogger("lexi_middleware.core_adapter")

def check_lexi_components_health() -> Dict[str, ComponentStatus]:
    # RUNTIME IMPORT: Nur importieren wenn Health Check läuft
    from langchain_ollama import OllamaEmbeddings

    components = {}

    # Check Ollama LLM service
    try:
        start_time = time.time()
        llm_url = f"{MiddlewareConfig.get_ollama_base_url()}/api/tags"
        response = requests.get(llm_url, timeout=5)
        latency = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name') for model in models if model.get('name')]
            components["llm_service"] = ComponentStatus(
                status="ok",
                latency_ms=latency,
                model=MiddlewareConfig.get_default_llm_model(),
                message=f"Available models: {', '.join(model_names[:3])}..."
            )
        else:
            components["llm_service"] = ComponentStatus(
                status="error",
                latency_ms=latency,
                message=f"Ollama API returned status {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Error checking Ollama service: {str(e)}")
        components["llm_service"] = ComponentStatus(
            status="error",
            message=f"Failed to connect to Ollama: {str(e)}"
        )

    # Check Ollama Embeddings service
    try:
        start_time = time.time()
        embeddings = OllamaEmbeddings(
            base_url=MiddlewareConfig.get_embedding_base_url(),
            model=MiddlewareConfig.get_default_embedding_model()
        )
        sample_vector = embeddings.embed_query("Test query for health check")
        latency = int((time.time() - start_time) * 1000)

        components["embedding_service"] = ComponentStatus(
            status="ok",
            latency_ms=latency,
            model=embeddings.model
        )
    except Exception as e:
        logger.error(f"Error checking embedding service: {str(e)}")
        components["embedding_service"] = ComponentStatus(
            status="error",
            message=f"Failed to generate embeddings: {str(e)}"
        )

    # Check Qdrant database and dimension consistency
    try:
        start_time = time.time()
        host = MiddlewareConfig.get_qdrant_host()
        if host.startswith(("http://", "https://")):
            host = host.split("://")[1]

        client = QdrantClient(host=host, port=MiddlewareConfig.get_qdrant_port())
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        latency = int((time.time() - start_time) * 1000)

        collection_name = MiddlewareConfig.get_memory_collection()
        if collection_name in collection_names:
            collection_info = client.get_collection(collection_name)
            configured_dimensions = collection_info.config.params.vectors.size
            embeddings = OllamaEmbeddings(
                base_url=MiddlewareConfig.get_embedding_base_url(),
                model=MiddlewareConfig.get_default_embedding_model()
            )
            sample_vector = embeddings.embed_query("Test query for dimension detection")
            actual_dimensions = len(sample_vector)

            if actual_dimensions == configured_dimensions:
                components["database"] = ComponentStatus(
                    status="ok",
                    latency_ms=latency,
                    message=f"Collection '{collection_name}' found with matching dimensions ({actual_dimensions})"
                )
                components["dimensions"] = ComponentStatus(
                    status="ok",
                    actual=actual_dimensions,
                    configured=configured_dimensions,
                    message=f"Embedding dimensions match: {actual_dimensions}"
                )
            else:
                components["database"] = ComponentStatus(
                    status="warning",
                    latency_ms=latency,
                    message=f"Collection '{collection_name}' found, but dimension mismatch"
                )
                components["dimensions"] = ComponentStatus(
                    status="warning",
                    actual=actual_dimensions,
                    configured=configured_dimensions,
                    mismatch=True,
                    message=f"Dimension mismatch: Collection has {configured_dimensions}, embeddings are {actual_dimensions}"
                )
        else:
            components["database"] = ComponentStatus(
                status="warning",
                latency_ms=latency,
                message=f"Collection '{collection_name}' not found"
            )
            components["dimensions"] = ComponentStatus(
                status="warning",
                message="Cannot check dimensions - collection not available"
            )
    except Exception as e:
        logger.error(f"Error checking Qdrant database or dimensions: {str(e)}")
        components["database"] = ComponentStatus(
            status="error",
            message=f"Failed to connect to Qdrant: {str(e)}"
        )
        components["dimensions"] = ComponentStatus(
            status="error",
            message=f"Failed to check dimensions: {str(e)}"
        )

    return components

def initialize_lexi_components():
    try:
        from backend.core.bootstrap import initialize_components
        return initialize_components()
    except Exception as e:
        logger.error(f"Error initializing Lexi components: {str(e)}")
        raise
