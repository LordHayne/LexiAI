#!/usr/bin/env python3
"""
Qdrant Data Repair Script
==========================

Repariert Datenintegrität in Qdrant-Collections:
- Fügt fehlende user_id und category zu Memories hinzu
- Repariert lexi_knowledge_gaps Collection (Vector Size)
- Validiert und bereinigt inkonsistente Daten
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qdrant_data_repair")


class QdrantDataRepair:
    """Repariert Datenintegrität in Qdrant."""

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

    def fix_missing_user_ids(self):
        """Fügt fehlende user_id zu Memories hinzu."""
        logger.info("\n👤 Fixing Missing user_id Fields...")

        try:
            # Scroll through all points
            scroll_result = self.client.scroll(
                collection_name="lexi_memory",
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

            points_without_user_id = []

            for point in points:
                payload = point.payload or {}

                if "user_id" not in payload:
                    points_without_user_id.append(point.id)

            if not points_without_user_id:
                logger.info("  ✅ All points have user_id")
                return

            logger.info(f"  Found {len(points_without_user_id)} points without user_id")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would add user_id='default' to {len(points_without_user_id)} points")
                return

            # Add user_id to points
            for point_id in points_without_user_id:
                self.client.set_payload(
                    collection_name="lexi_memory",
                    payload={"user_id": "default"},
                    points=[point_id]
                )

            logger.info(f"  ✅ Added user_id to {len(points_without_user_id)} points")

        except Exception as e:
            logger.error(f"  ❌ Error fixing user_id: {e}")

    def fix_missing_categories(self):
        """Fügt fehlende category zu Memories hinzu."""
        logger.info("\n🏷️  Fixing Missing category Fields...")

        try:
            scroll_result = self.client.scroll(
                collection_name="lexi_memory",
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

            points_without_category = []

            for point in points:
                payload = point.payload or {}

                if "category" not in payload or payload.get("category") is None:
                    points_without_category.append(point.id)

            if not points_without_category:
                logger.info("  ✅ All points have category")
                return

            logger.info(f"  Found {len(points_without_category)} points without category")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would add category='unkategorisiert' to {len(points_without_category)} points")
                return

            # Add category to points
            for point_id in points_without_category:
                self.client.set_payload(
                    collection_name="lexi_memory",
                    payload={"category": "unkategorisiert"},
                    points=[point_id]
                )

            logger.info(f"  ✅ Added category to {len(points_without_category)} points")

        except Exception as e:
            logger.error(f"  ❌ Error fixing category: {e}")

    def fix_knowledge_gaps_vector_size(self):
        """Repariert lexi_knowledge_gaps Collection (Vector Size)."""
        logger.info("\n🔧 Fixing lexi_knowledge_gaps Vector Size...")

        try:
            # Check current vector size
            info = self.client.get_collection("lexi_knowledge_gaps")
            current_size = info.config.params.vectors.size

            if current_size == 768:
                logger.info("  ✅ Vector size already correct (768)")
                return

            logger.info(f"  Current vector size: {current_size}")
            logger.info(f"  Expected vector size: 768")

            if self.dry_run:
                logger.info("  [DRY RUN] Would recreate collection with vector size 768")
                return

            # Recreate collection with correct vector size
            logger.info("  Recreating collection...")

            from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

            hnsw_config = HnswConfigDiff(
                m=32,
                ef_construct=200,
                full_scan_threshold=10000
            )

            vector_config = VectorParams(
                size=768,
                distance=Distance.COSINE,
                hnsw_config=hnsw_config
            )

            self.client.recreate_collection(
                collection_name="lexi_knowledge_gaps",
                vectors_config=vector_config
            )

            logger.info("  ✅ Collection recreated with correct vector size")

        except Exception as e:
            logger.error(f"  ❌ Error fixing vector size: {e}")

    def validate_all_collections(self):
        """Validiert alle Collections."""
        logger.info("\n✓ Validating All Collections...")

        collections = {
            "lexi_memory": ["content", "user_id", "category", "timestamp"],
            "lexi_feedback": ["turn_id", "feedback_type", "timestamp"],
            "lexi_turns": ["turn_id", "user_id", "user_message", "ai_response", "timestamp"],
            "lexi_knowledge_gaps": ["gap_id", "query", "category", "timestamp", "status"]
        }

        for col_name, required_fields in collections.items():
            try:
                info = self.client.get_collection(col_name)
                points_count = info.points_count

                if points_count == 0:
                    logger.info(f"  {col_name}: Empty (no validation needed)")
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

                    missing_fields = set(required_fields) - set(sample.keys())

                    if missing_fields:
                        logger.warning(f"  {col_name}: ⚠️  Missing fields: {missing_fields}")
                    else:
                        logger.info(f"  {col_name}: ✅ All required fields present ({points_count} points)")

            except Exception as e:
                logger.error(f"  {col_name}: ❌ Error - {e}")

    def run_full_repair(self):
        """Führt vollständige Datenreparatur durch."""
        logger.info("="*60)
        logger.info("Starting Qdrant Data Repair")
        logger.info("="*60)

        self.fix_missing_user_ids()
        self.fix_missing_categories()
        self.fix_knowledge_gaps_vector_size()
        self.validate_all_collections()

        logger.info("\n" + "="*60)
        logger.info("Data Repair Complete")
        logger.info("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Repair Qdrant data integrity issues")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force repair without confirmation"
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
        repair = QdrantDataRepair(dry_run=args.dry_run)
        repair.run_full_repair()
        return 0

    except Exception as e:
        logger.error(f"Repair failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
