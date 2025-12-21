#!/usr/bin/env python3
"""
Qdrant Storage Optimizer
========================

Optimiert die Qdrant-Speicherung basierend auf Diagnose-Ergebnissen:
- Erstellt fehlende Collections
- Erstellt Payload-Indices
- Optimiert HNSW-Parameter
- Bereinigt inkonsistente Daten
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List
from qdrant_client.models import (
    VectorParams, Distance, HnswConfigDiff,
    PayloadSchemaType, PointStruct
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qdrant_optimizer")


class QdrantOptimizer:
    """Optimiert Qdrant-Speicherung."""

    def __init__(self, dry_run: bool = False):
        """
        Args:
            dry_run: Wenn True, zeigt nur an was gemacht würde
        """
        from backend.qdrant.client_wrapper import get_qdrant_client
        from backend.config.middleware_config import MiddlewareConfig

        self.client = get_qdrant_client()
        self.config = MiddlewareConfig
        self.dry_run = dry_run

        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

    def create_missing_collections(self):
        """Erstellt fehlende Collections."""
        logger.info("\n📦 Creating Missing Collections...")

        # Get embedding dimensions
        embedding_dim = int(os.environ.get("LEXI_MEMORY_DIMENSION", "768"))

        collections_to_create = {
            "lexi_feedback": {
                "description": "Feedback for self-correction",
                "vector_size": embedding_dim
            },
            "lexi_turns": {
                "description": "Conversation turn tracking",
                "vector_size": embedding_dim
            },
            "lexi_knowledge_gaps": {
                "description": "Knowledge gap detection",
                "vector_size": embedding_dim
            }
        }

        existing_collections = {
            col.name for col in self.client.get_collections().collections
        }

        for col_name, col_config in collections_to_create.items():
            if col_name in existing_collections:
                logger.info(f"  ✓ {col_name} already exists")
                continue

            logger.info(f"  Creating {col_name}...")

            if self.dry_run:
                logger.info(f"    [DRY RUN] Would create collection with {col_config['vector_size']}-dim vectors")
                continue

            try:
                # Optimized HNSW configuration
                hnsw_config = HnswConfigDiff(
                    m=32,
                    ef_construct=200,
                    full_scan_threshold=10000
                )

                vector_config = VectorParams(
                    size=col_config["vector_size"],
                    distance=Distance.COSINE,
                    hnsw_config=hnsw_config
                )

                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=vector_config
                )

                logger.info(f"    ✅ Created {col_name}")

            except Exception as e:
                logger.error(f"    ❌ Failed to create {col_name}: {e}")

    def create_payload_indices(self):
        """Erstellt fehlende Payload-Indices."""
        logger.info("\n🔍 Creating Payload Indices...")

        indices_config = {
            "lexi_memory": ["user_id", "category", "tags", "source", "timestamp"],
            "lexi_feedback": ["turn_id", "feedback_type", "timestamp"],
            "lexi_turns": ["turn_id", "user_id", "timestamp"],
            "lexi_knowledge_gaps": ["gap_id", "category", "status", "timestamp"]
        }

        for col_name, fields in indices_config.items():
            # Check if collection exists
            try:
                info = self.client.get_collection(col_name)
            except Exception:
                logger.info(f"  ⏭️  Skipping {col_name} (collection doesn't exist)")
                continue

            logger.info(f"\n  Processing {col_name}...")

            # Get existing indices
            existing_indices = []
            if hasattr(info.config, 'payload_schema') and info.config.payload_schema:
                existing_indices = list(info.config.payload_schema.keys())

            for field in fields:
                if field in existing_indices:
                    logger.info(f"    ✓ Index on '{field}' already exists")
                    continue

                logger.info(f"    Creating index on '{field}'...")

                if self.dry_run:
                    logger.info(f"      [DRY RUN] Would create index")
                    continue

                try:
                    self.client.create_payload_index(
                        collection_name=col_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info(f"      ✅ Index created")

                except Exception as e:
                    logger.error(f"      ❌ Failed to create index: {e}")

    def optimize_hnsw_parameters(self):
        """Optimiert HNSW-Parameter (erfordert Collection-Neuanlage)."""
        logger.info("\n⚡ Optimizing HNSW Parameters...")
        logger.info("  ℹ️  Note: Requires recreating collections (not done automatically)")
        logger.info("  To recreate with optimal HNSW, use: python start_middleware.py --force-recreate")

    def cleanup_orphaned_data(self):
        """Bereinigt verwaiste Daten."""
        logger.info("\n🧹 Cleaning Up Orphaned Data...")

        # Check for feedback without corresponding turns
        # Check for turns without user_id
        # etc.

        logger.info("  ℹ️  Orphaned data cleanup not yet implemented")

    def validate_data_integrity(self):
        """Validiert Datenintegrität."""
        logger.info("\n✓ Validating Data Integrity...")

        for col_name in ["lexi_memory", "lexi_feedback", "lexi_turns", "lexi_knowledge_gaps"]:
            try:
                info = self.client.get_collection(col_name)
                points_count = info.points_count

                if points_count == 0:
                    logger.info(f"  {col_name}: Empty collection")
                    continue

                # Sample some points
                scroll_result = self.client.scroll(
                    collection_name=col_name,
                    limit=10,
                    with_payload=True
                )

                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

                if points:
                    sample = points[0].payload
                    logger.info(f"  {col_name}: {points_count:,} points")
                    logger.info(f"    Sample keys: {list(sample.keys())}")

            except Exception as e:
                logger.info(f"  {col_name}: Not found or error - {e}")

    def run_full_optimization(self):
        """Führt vollständige Optimierung durch."""
        logger.info("="*60)
        logger.info("Starting Qdrant Storage Optimization")
        logger.info("="*60)

        self.create_missing_collections()
        self.create_payload_indices()
        self.optimize_hnsw_parameters()
        self.validate_data_integrity()

        logger.info("\n" + "="*60)
        logger.info("Optimization Complete")
        logger.info("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize Qdrant storage")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force optimization without confirmation"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.force:
        print("\n⚠️  This will modify your Qdrant database!")
        print("   Run with --dry-run first to see what would change.")
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    try:
        optimizer = QdrantOptimizer(dry_run=args.dry_run)
        optimizer.run_full_optimization()
        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
