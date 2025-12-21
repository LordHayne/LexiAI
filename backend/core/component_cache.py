"""
Global component cache for reusing initialized components.
This prevents reinitializing Ollama and Qdrant connections on every request.

THREAD-SAFETY: Uses threading.Lock to prevent race conditions when multiple
threads/requests attempt to initialize components simultaneously.
"""
import logging
import threading
from typing import Optional
from backend.core.bootstrap import ComponentBundle, initialize_components_bundle

logger = logging.getLogger("lexi_middleware.component_cache")

class ComponentCache:
    """
    Thread-safe singleton cache for application components.

    Uses double-checked locking to ensure thread-safety while maintaining
    performance by only acquiring the lock when necessary.
    """

    _instance: Optional['ComponentCache'] = None
    _bundle: Optional[ComponentBundle] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_bundle(self, force_recreate: bool = False) -> ComponentBundle:
        """
        Get cached component bundle or initialize new one (thread-safe).

        Uses double-checked locking pattern:
        1. Check if initialization needed (without lock - fast path)
        2. If needed, acquire lock
        3. Check again (another thread might have initialized while waiting)
        4. Initialize if still needed

        Args:
            force_recreate: If True, recreate components even if cached

        Returns:
            ComponentBundle with all initialized components
        """
        # Fast path: return cached bundle if available and not forcing recreation
        if self._bundle is not None and not force_recreate:
            return self._bundle

        # Slow path: need to initialize
        with self._lock:
            # Double-check: another thread might have initialized while we waited for lock
            if self._bundle is None or force_recreate:
                logger.info("Initializing component bundle (thread-safe)...")
                self._bundle = initialize_components_bundle(force_recreate=force_recreate)
                logger.info("Component bundle initialized and cached")

        return self._bundle

    def clear(self):
        """Clear the cache (useful for testing or force refresh) - thread-safe."""
        with self._lock:
            logger.info("Clearing component cache")
            self._bundle = None


# Global instance
_cache = ComponentCache()


def get_cached_components(force_recreate: bool = False) -> ComponentBundle:
    """
    Get cached components or initialize if not yet cached.

    This function should be used instead of initialize_components() in request handlers
    to avoid reinitializing components on every request.

    Args:
        force_recreate: If True, recreate components even if cached

    Returns:
        ComponentBundle with all initialized components
    """
    # Check environment variable for force_recreate on first initialization
    import os
    if _cache._bundle is None:
        env_force_recreate = os.environ.get("LEXI_FORCE_RECREATE", "False").lower() == "true"
        force_recreate = force_recreate or env_force_recreate
        if env_force_recreate:
            logger.info("LEXI_FORCE_RECREATE environment variable detected - forcing recreation")

    return _cache.get_bundle(force_recreate=force_recreate)


def clear_component_cache():
    """Clear the component cache (useful for testing)."""
    _cache.clear()
