"""
Worker Bootstrap Module

Initializes and starts the worker coordinator during application startup.
Integrates with existing FastAPI lifespan management.
"""

import logging
from typing import Optional
import yaml
from pathlib import Path

from backend.workers.qdrant_optimizer import WorkerCoordinator
from backend.config.middleware_config import Config

logger = logging.getLogger("worker_bootstrap")


async def initialize_worker_coordinator(
    qdrant_client,
    embeddings,
    memory_adapter=None,
    config_path: Optional[str] = None
) -> Optional[WorkerCoordinator]:
    """
    Initialize and start the worker coordinator.

    Args:
        qdrant_client: Qdrant client instance
        embeddings: Embedding model instance
        memory_adapter: Memory adapter for coordination (optional)
        config_path: Path to workers_config.yaml (optional)

    Returns:
        Initialized WorkerCoordinator or None if disabled
    """
    try:
        # Load worker configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "workers_config.yaml"

        if not Path(config_path).exists():
            logger.warning(f"Worker config not found at {config_path}, workers disabled")
            return None

        with open(config_path, 'r') as f:
            worker_config = yaml.safe_load(f)

        # Check if workers are enabled
        if not worker_config.get("workers", {}).get("enabled", False):
            logger.info("Workers disabled in configuration")
            return None

        # Add database settings from main config
        config = Config()
        worker_config["collection_name"] = config.memory_collection
        worker_config["expected_dimension"] = config.memory_dimension

        # Initialize coordinator
        coordinator = WorkerCoordinator(
            config=worker_config,
            qdrant_client=qdrant_client,
            embeddings=embeddings,
            memory_adapter=memory_adapter
        )

        # Start scheduler
        await coordinator.start()

        logger.info("✅ Worker coordinator initialized and started")

        return coordinator

    except Exception as e:
        logger.error(f"Failed to initialize worker coordinator: {e}", exc_info=True)
        return None


async def shutdown_worker_coordinator(coordinator: Optional[WorkerCoordinator]):
    """
    Gracefully shutdown the worker coordinator.

    Args:
        coordinator: WorkerCoordinator instance to shutdown
    """
    if coordinator is None:
        return

    try:
        await coordinator.stop()
        logger.info("✅ Worker coordinator stopped")
    except Exception as e:
        logger.error(f"Error stopping worker coordinator: {e}", exc_info=True)
