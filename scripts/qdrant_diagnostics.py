#!/usr/bin/env python3
"""
Qdrant Storage Diagnostics & Optimization Script
=================================================

Überprüft und optimiert die Qdrant-Speicherung für alle LexiAI-Features:
- Memory Storage (lexi_memory)
- Feedback Storage (lexi_feedback)
- Conversation Turns (lexi_turns)
- Knowledge Gaps (lexi_knowledge_gaps)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import json

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qdrant_diagnostics")

# Expected Collections
EXPECTED_COLLECTIONS = {
    "lexi_memory": {
        "description": "Main memory storage",
        "required_fields": ["content", "timestamp", "user_id", "category", "tags", "source", "relevance"],
        "required_indices": ["user_id", "category", "tags", "source", "timestamp"]
    },
    "lexi_feedback": {
        "description": "Feedback storage for self-correction",
        "required_fields": ["turn_id", "feedback_type", "timestamp", "user_comment", "confidence"],
        "required_indices": ["turn_id", "feedback_type", "timestamp"]
    },
    "lexi_turns": {
        "description": "Conversation turn tracking",
        "required_fields": ["turn_id", "user_id", "user_message", "ai_response", "timestamp"],
        "required_indices": ["turn_id", "user_id", "timestamp"]
    },
    "lexi_knowledge_gaps": {
        "description": "Knowledge gap detection",
        "required_fields": ["gap_id", "query", "category", "timestamp", "status"],
        "required_indices": ["gap_id", "category", "status", "timestamp"]
    }
}


class QdrantDiagnostics:
    """Diagnose und Optimierung der Qdrant-Speicherung."""

    def __init__(self):
        """Initialisierung mit Qdrant-Client."""
        from backend.qdrant.client_wrapper import get_qdrant_client
        from backend.config.middleware_config import MiddlewareConfig

        self.client = get_qdrant_client()
        self.config = MiddlewareConfig
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collections": {},
            "issues": [],
            "optimizations": [],
            "summary": {}
        }

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Führt vollständige Diagnostik durch."""
        logger.info("="*60)
        logger.info("Starting Qdrant Storage Diagnostics")
        logger.info("="*60)

        # 1. Check Qdrant Connection
        if not self._check_connection():
            logger.error("❌ Qdrant connection failed - aborting diagnostics")
            return self.results

        # 2. Check Collections
        self._check_collections()

        # 3. Check Indices
        self._check_indices()

        # 4. Check Data Integrity
        self._check_data_integrity()

        # 5. Performance Analysis
        self._analyze_performance()

        # 6. Generate Optimization Recommendations
        self._generate_recommendations()

        # 7. Summary
        self._generate_summary()

        logger.info("="*60)
        logger.info("Diagnostics Complete")
        logger.info("="*60)

        return self.results

    def _check_connection(self) -> bool:
        """Prüft Qdrant-Verbindung."""
        logger.info("\n[1/7] Checking Qdrant Connection...")
        try:
            collections = self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant successfully")
            logger.info(f"   Found {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            self.results["issues"].append({
                "severity": "critical",
                "component": "connection",
                "message": f"Cannot connect to Qdrant: {e}"
            })
            return False

    def _check_collections(self):
        """Prüft ob alle Collections existieren."""
        logger.info("\n[2/7] Checking Collections...")

        try:
            existing_collections = {
                col.name for col in self.client.get_collections().collections
            }

            for col_name, col_config in EXPECTED_COLLECTIONS.items():
                logger.info(f"\nChecking collection: {col_name}")
                logger.info(f"  Description: {col_config['description']}")

                if col_name not in existing_collections:
                    logger.warning(f"  ⚠️  Collection missing - will need creation")
                    self.results["issues"].append({
                        "severity": "warning",
                        "component": "collection",
                        "collection": col_name,
                        "message": f"Collection '{col_name}' does not exist"
                    })
                    self.results["collections"][col_name] = {
                        "exists": False,
                        "needs_creation": True
                    }
                else:
                    # Get collection info
                    info = self.client.get_collection(col_name)
                    points_count = info.points_count
                    vectors_config = info.config.params.vectors

                    logger.info(f"  ✅ Collection exists")
                    logger.info(f"     Points: {points_count:,}")
                    logger.info(f"     Vector size: {vectors_config.size}")
                    logger.info(f"     Distance: {vectors_config.distance}")

                    self.results["collections"][col_name] = {
                        "exists": True,
                        "points_count": points_count,
                        "vector_size": vectors_config.size,
                        "distance": str(vectors_config.distance)
                    }

        except Exception as e:
            logger.error(f"❌ Error checking collections: {e}")
            self.results["issues"].append({
                "severity": "error",
                "component": "collections",
                "message": str(e)
            })

    def _check_indices(self):
        """Prüft Payload-Indices."""
        logger.info("\n[3/7] Checking Payload Indices...")

        for col_name, col_config in EXPECTED_COLLECTIONS.items():
            if col_name not in self.results["collections"]:
                continue

            if not self.results["collections"][col_name]["exists"]:
                continue

            logger.info(f"\nChecking indices for: {col_name}")

            try:
                # Get collection info
                info = self.client.get_collection(col_name)

                # Check if collection has payload schema
                if hasattr(info.config, 'payload_schema') and info.config.payload_schema:
                    indexed_fields = list(info.config.payload_schema.keys())
                    logger.info(f"  Indexed fields: {indexed_fields}")
                else:
                    logger.warning(f"  ⚠️  No payload indices found")
                    indexed_fields = []

                # Check required indices
                required_indices = col_config["required_indices"]
                missing_indices = set(required_indices) - set(indexed_fields)

                if missing_indices:
                    logger.warning(f"  ⚠️  Missing indices: {missing_indices}")
                    self.results["optimizations"].append({
                        "type": "index",
                        "collection": col_name,
                        "action": "create_indices",
                        "fields": list(missing_indices),
                        "benefit": "10-100x faster filtered queries"
                    })
                else:
                    logger.info(f"  ✅ All required indices present")

                self.results["collections"][col_name]["indices"] = {
                    "present": indexed_fields,
                    "missing": list(missing_indices)
                }

            except Exception as e:
                logger.error(f"  ❌ Error checking indices: {e}")

    def _check_data_integrity(self):
        """Prüft Datenintegrität."""
        logger.info("\n[4/7] Checking Data Integrity...")

        for col_name, col_config in EXPECTED_COLLECTIONS.items():
            if not self.results["collections"].get(col_name, {}).get("exists"):
                continue

            logger.info(f"\nChecking data integrity: {col_name}")

            try:
                # Sample some points
                scroll_result = self.client.scroll(
                    collection_name=col_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )

                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

                if not points:
                    logger.info(f"  ℹ️  Collection is empty")
                    continue

                # Check required fields
                required_fields = col_config["required_fields"]
                sample_point = points[0]
                payload = sample_point.payload or {}

                missing_fields = set(required_fields) - set(payload.keys())

                if missing_fields:
                    logger.warning(f"  ⚠️  Sample point missing fields: {missing_fields}")
                    self.results["issues"].append({
                        "severity": "warning",
                        "component": "data_integrity",
                        "collection": col_name,
                        "message": f"Points may be missing fields: {missing_fields}"
                    })
                else:
                    logger.info(f"  ✅ Sample point has all required fields")

                # Check for null/empty values
                null_fields = [k for k, v in payload.items() if v is None or v == ""]
                if null_fields:
                    logger.info(f"  ℹ️  Null/empty fields in sample: {null_fields}")

                self.results["collections"][col_name]["sample_payload"] = payload

            except Exception as e:
                logger.error(f"  ❌ Error checking data integrity: {e}")

    def _analyze_performance(self):
        """Analysiert Performance-Metriken."""
        logger.info("\n[5/7] Analyzing Performance...")

        for col_name in self.results["collections"].keys():
            if not self.results["collections"][col_name].get("exists"):
                continue

            logger.info(f"\nAnalyzing performance: {col_name}")

            try:
                info = self.client.get_collection(col_name)
                points_count = info.points_count

                # Check HNSW parameters
                hnsw_config = info.config.params.vectors.hnsw_config

                if hnsw_config:
                    m = hnsw_config.m
                    ef_construct = hnsw_config.ef_construct

                    logger.info(f"  HNSW Config:")
                    logger.info(f"    m: {m}")
                    logger.info(f"    ef_construct: {ef_construct}")

                    # Check if optimized
                    if m < 32 or ef_construct < 200:
                        logger.warning(f"  ⚠️  HNSW parameters below optimal")
                        self.results["optimizations"].append({
                            "type": "hnsw",
                            "collection": col_name,
                            "current": {"m": m, "ef_construct": ef_construct},
                            "recommended": {"m": 32, "ef_construct": 200},
                            "benefit": "Better recall/precision trade-off"
                        })
                    else:
                        logger.info(f"  ✅ HNSW parameters optimized")
                else:
                    logger.warning(f"  ⚠️  No HNSW config found")

                # Estimate memory usage
                vector_size = info.config.params.vectors.size
                estimated_mb = (points_count * vector_size * 4) / (1024 * 1024)  # 4 bytes per float
                logger.info(f"  Estimated memory: ~{estimated_mb:.1f} MB")

                self.results["collections"][col_name]["performance"] = {
                    "estimated_memory_mb": estimated_mb
                }

            except Exception as e:
                logger.error(f"  ❌ Error analyzing performance: {e}")

    def _generate_recommendations(self):
        """Generiert Optimierungs-Empfehlungen."""
        logger.info("\n[6/7] Generating Recommendations...")

        # Collection-specific recommendations
        for col_name, col_data in self.results["collections"].items():
            if not col_data.get("exists"):
                continue

            points_count = col_data.get("points_count", 0)

            # Recommend indices if missing and collection has data
            if points_count > 100:
                missing_indices = col_data.get("indices", {}).get("missing", [])
                if missing_indices:
                    logger.info(f"  📊 {col_name}: Create indices for {missing_indices}")

        # General recommendations
        total_points = sum(
            col.get("points_count", 0)
            for col in self.results["collections"].values()
            if col.get("exists")
        )

        if total_points > 10000:
            logger.info(f"  💡 Consider enabling batch operations for bulk inserts")
            self.results["optimizations"].append({
                "type": "general",
                "recommendation": "Use batch_store_entries() for bulk operations",
                "benefit": "5-10x faster bulk inserts"
            })

    def _generate_summary(self):
        """Generiert Zusammenfassung."""
        logger.info("\n[7/7] Generating Summary...")

        total_collections = len(EXPECTED_COLLECTIONS)
        existing_collections = sum(
            1 for col in self.results["collections"].values()
            if col.get("exists")
        )

        total_points = sum(
            col.get("points_count", 0)
            for col in self.results["collections"].values()
            if col.get("exists")
        )

        critical_issues = sum(
            1 for issue in self.results["issues"]
            if issue["severity"] == "critical"
        )
        warnings = sum(
            1 for issue in self.results["issues"]
            if issue["severity"] == "warning"
        )

        self.results["summary"] = {
            "total_collections": total_collections,
            "existing_collections": existing_collections,
            "missing_collections": total_collections - existing_collections,
            "total_points": total_points,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "optimizations_available": len(self.results["optimizations"])
        }

        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Collections: {existing_collections}/{total_collections} exist")
        logger.info(f"Total Points: {total_points:,}")
        logger.info(f"Critical Issues: {critical_issues}")
        logger.info(f"Warnings: {warnings}")
        logger.info(f"Optimizations Available: {len(self.results['optimizations'])}")
        logger.info("="*60)

    def save_report(self, filepath: str = None):
        """Speichert Diagnose-Report."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"docs/qdrant_diagnostics_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n📄 Report saved to: {filepath}")
        return filepath


def main():
    """Main entry point."""
    try:
        diagnostics = QdrantDiagnostics()
        results = diagnostics.run_full_diagnostics()

        # Save report
        report_path = diagnostics.save_report()

        # Print recommendations
        if diagnostics.results["optimizations"]:
            print("\n" + "="*60)
            print("RECOMMENDED OPTIMIZATIONS")
            print("="*60)
            for i, opt in enumerate(diagnostics.results["optimizations"], 1):
                print(f"\n{i}. {opt['type'].upper()}: {opt.get('collection', 'General')}")
                for key, value in opt.items():
                    if key not in ['type', 'collection']:
                        print(f"   {key}: {value}")

        # Return exit code based on critical issues
        critical_count = diagnostics.results["summary"]["critical_issues"]
        return 1 if critical_count > 0 else 0

    except Exception as e:
        logger.error(f"Diagnostics failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
