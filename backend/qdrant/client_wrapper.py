# Refactored client_wrapper.py and qdrant_interface.py

# -------------------------- client_wrapper.py --------------------------

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, PointStruct
from backend.utils.config import get_config_value
import logging
import backoff
import os
import threading
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("qdrant_client")

_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _get_upsert_wait_default() -> bool:
    return _parse_bool(get_config_value("qdrant_upsert_wait", default="false"), default=False)


def _build_url(host: str, port: int) -> str:
    host = host.strip()
    if not host:
        host = "http://localhost"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    parsed = urlparse(host)
    netloc = parsed.netloc
    if parsed.port is None:
        hostname = parsed.hostname or parsed.netloc
        netloc = f"{hostname}:{port}"
    return f"{parsed.scheme}://{netloc}{parsed.path}"


def create_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client with optimized connection settings.

    PERFORMANCE: Uses connection pooling and optimized timeouts.
    """
    host = os.environ.get("LEXI_QDRANT_HOST") or get_config_value("qdrant_host", default="http://localhost")
    port = os.environ.get("LEXI_QDRANT_PORT") or get_config_value("qdrant_port", default=6333)
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 6333
    url = _build_url(str(host), port)

    logger.info(f"Creating Qdrant client for {url}")
    try:
        client_config = {
            "url": url,
            "timeout": 30.0,
            "prefer_grpc": False,
            # Connection pooling settings
            "https": None,  # Will use httpx defaults
        }

        api_key = os.environ.get("LEXI_QDRANT_API_KEY") or get_config_value("qdrant_api_key", default=None)
        if api_key:
            client_config["api_key"] = api_key

        client = QdrantClient(**client_config)

        # Verify connection
        try:
            client.get_collections()
            logger.info("Qdrant client connection verified")
        except Exception as e:
            logger.warning(f"Could not verify Qdrant connection: {e}")

        return client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client: {e}")
        raise


def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant client with thread-safe singleton pattern.

    PERFORMANCE: Uses double-checked locking for thread-safety.
    """
    global _client

    # Fast path: client already exists
    if _client is not None:
        return _client

    # Slow path: need to create client
    with _client_lock:
        # Double-check: another thread might have created it
        if _client is None:
            _client = create_qdrant_client()

    return _client


class QdrantClientProxy:
    def __getattr__(self, name):
        return getattr(get_qdrant_client(), name)

    def __call__(self, *args, **kwargs):
        return get_qdrant_client()(*args, **kwargs)


client = QdrantClientProxy()


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None,
                      on_backoff=lambda d: logger.warning(f"Upsert retry {d['tries']}/3"))
def safe_upsert(*args, **kwargs):
    if "wait" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["wait"] = _get_upsert_wait_default()
    return get_qdrant_client().upsert(*args, **kwargs)


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None,
                      on_backoff=lambda d: logger.warning(f"Search retry {d['tries']}/3"))
def safe_search(*args, **kwargs):
    client = get_qdrant_client()
    if hasattr(client, "search"):
        return client.search(*args, **kwargs)

    # qdrant-client >=1.16 uses query_points instead of search
    if "query_vector" in kwargs and "query" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["query"] = kwargs.pop("query_vector")
    if "query_filter" in kwargs and "filter" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["filter"] = kwargs.pop("query_filter")

    response = client.query_points(*args, **kwargs)
    return response.points


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None,
                      on_backoff=lambda d: logger.warning(f"Delete retry {d['tries']}/3"))
def safe_delete(*args, **kwargs):
    return get_qdrant_client().delete(*args, **kwargs)


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None,
                      on_backoff=lambda d: logger.warning(f"Scroll retry {d['tries']}/3"))
def safe_scroll(*args, **kwargs):
    return get_qdrant_client().scroll(*args, **kwargs)


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None,
                      on_backoff=lambda d: logger.warning(f"Payload retry {d['tries']}/3"))
def safe_set_payload(*args, **kwargs):
    return get_qdrant_client().set_payload(*args, **kwargs)


def get_batch_size() -> int:
    return int(get_config_value("qdrant_batch_size", default=100))


def health_check() -> bool:
    try:
        get_qdrant_client().get_collections()
        logger.info("Qdrant health check passed")
        return True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return False


def reset_client():
    global _client
    if _client:
        logger.info("Resetting Qdrant client instance")
    _client = None
