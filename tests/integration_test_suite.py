#!/usr/bin/env python3
"""
LexiAI Integration Test Suite
==============================

Vollständige Integration Tests für alle LexiAI Features:
- Memory Storage & Retrieval
- Self-Correction System
- Knowledge Gap Detection
- Web Search Integration
- Category Prediction
- Chat Processing
- Performance Benchmarks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_tests")


class IntegrationTestSuite:
    """Vollständige Integration Test Suite für LexiAI."""

    def __init__(self):
        """Initialisierung."""
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "metrics": {},
            "summary": {}
        }
        self.test_user_id = f"test_user_{uuid4().hex[:8]}"

    async def run_all_tests(self) -> Dict[str, Any]:
        """Führt alle Integration Tests durch."""
        logger.info("="*70)
        logger.info("🧪 STARTING INTEGRATION TEST SUITE")
        logger.info("="*70)
        logger.info(f"Test User ID: {self.test_user_id}")

        # Test execution order
        tests = [
            ("1. Memory Storage & Retrieval", self.test_memory_storage_retrieval),
            ("2. Category Prediction", self.test_category_prediction),
            ("3. Self-Correction System", self.test_self_correction_system),
            ("4. Knowledge Gap Detection", self.test_knowledge_gap_detection),
            ("5. Web Search Integration", self.test_web_search_integration),
            ("6. Chat Processing End-to-End", self.test_chat_processing),
            ("7. Performance Benchmarks", self.test_performance_benchmarks),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*70}")

            try:
                result = await test_func()
                self.results["tests"][test_name] = result

                if result["status"] == "passed":
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    failed += 1
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")

            except Exception as e:
                failed += 1
                logger.error(f"❌ {test_name}: EXCEPTION - {str(e)}", exc_info=True)
                self.results["tests"][test_name] = {
                    "status": "failed",
                    "error": str(e),
                    "exception": True
                }

        # Summary
        self.results["summary"] = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": f"{(passed/len(tests)*100):.1f}%"
        }

        logger.info("\n" + "="*70)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {self.results['summary']['success_rate']}")
        logger.info("="*70)

        return self.results

    async def test_memory_storage_retrieval(self) -> Dict[str, Any]:
        """Test Memory Storage & Retrieval."""
        try:
            from backend.memory.adapter import store_memory_async, retrieve_memories
            from backend.core.component_cache import get_cached_components

            logger.info("Testing memory storage...")

            # Test 1: Store memories
            test_memories = [
                ("Python ist eine Programmiersprache", ["programming", "python"]),
                ("Der Nutzer mag maschinelles Lernen", ["preference", "ml"]),
                ("Die Hauptstadt von Deutschland ist Berlin", ["geography", "facts"]),
            ]

            stored_ids = []
            for content, tags in test_memories:
                mem_id, timestamp = await store_memory_async(
                    content=content,
                    user_id=self.test_user_id,
                    tags=tags,
                    metadata={"test": True}
                )
                stored_ids.append(mem_id)
                logger.info(f"  ✓ Stored memory: {mem_id[:8]}...")

            # Wait a bit for indexing
            await asyncio.sleep(0.5)

            # Test 2: Retrieve memories
            logger.info("Testing memory retrieval...")

            # Query-based retrieval
            memories = retrieve_memories(
                user_id=self.test_user_id,
                query="Was ist Python?",
                limit=5
            )

            if not memories:
                return {
                    "status": "failed",
                    "error": "No memories retrieved",
                    "stored_count": len(stored_ids)
                }

            logger.info(f"  ✓ Retrieved {len(memories)} memories")

            # Test 3: Tag filtering
            memories_with_tag = retrieve_memories(
                user_id=self.test_user_id,
                tags=["programming"],
                limit=5
            )

            logger.info(f"  ✓ Tag filtering: {len(memories_with_tag)} memories with 'programming' tag")

            # Test 4: Score threshold
            memories_high_score = retrieve_memories(
                user_id=self.test_user_id,
                query="Python Programmierung",
                score_threshold=0.5,
                limit=5
            )

            logger.info(f"  ✓ Score filtering: {len(memories_high_score)} memories above threshold")

            return {
                "status": "passed",
                "stored_count": len(stored_ids),
                "retrieved_count": len(memories),
                "tag_filtered_count": len(memories_with_tag),
                "score_filtered_count": len(memories_high_score),
                "memory_ids": stored_ids
            }

        except Exception as e:
            logger.error(f"Memory test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_category_prediction(self) -> Dict[str, Any]:
        """Test Category Prediction."""
        try:
            from backend.memory.category_predictor import ClusteredCategoryPredictor
            from backend.memory.memory_bootstrap import get_predictor

            logger.info("Testing category prediction...")

            # Get predictor
            predictor = get_predictor()

            if not predictor:
                return {
                    "status": "failed",
                    "error": "Could not initialize predictor"
                }

            # Test predictions
            test_contents = [
                "Python ist eine tolle Programmiersprache",
                "Der Nutzer mag Machine Learning",
                "Berlin ist die Hauptstadt von Deutschland"
            ]

            categories = []
            for content in test_contents:
                category = predictor.predict_category(content)
                categories.append(category)
                logger.info(f"  ✓ '{content[:40]}...' → {category}")

            return {
                "status": "passed",
                "predictions": categories,
                "predictor_type": type(predictor).__name__
            }

        except Exception as e:
            logger.error(f"Category prediction test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_self_correction_system(self) -> Dict[str, Any]:
        """Test Self-Correction System (Feedback & Turns)."""
        try:
            from backend.memory.conversation_tracker import get_conversation_tracker
            from backend.models.feedback import FeedbackType

            logger.info("Testing self-correction system...")

            tracker = get_conversation_tracker()

            # Test 1: Record conversation turn
            turn_id = tracker.record_turn(
                user_id=self.test_user_id,
                user_message="Was ist Python?",
                ai_response="Python ist eine Programmiersprache...",
                retrieved_memories=["mem-1", "mem-2"]
            )

            logger.info(f"  ✓ Recorded turn: {turn_id[:8]}...")

            # Test 2: Record explicit feedback
            tracker.record_feedback(
                turn_id=turn_id,
                feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                user_comment="Gute Antwort!"
            )

            logger.info(f"  ✓ Recorded positive feedback")

            # Test 3: Record negative feedback
            turn_id_2 = tracker.record_turn(
                user_id=self.test_user_id,
                user_message="Wer ist der Präsident?",
                ai_response="Ich weiß es nicht genau...",
                retrieved_memories=[]
            )

            tracker.record_feedback(
                turn_id=turn_id_2,
                feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
                user_comment="Das war nicht hilfreich"
            )

            logger.info(f"  ✓ Recorded negative feedback")

            # Test 4: Get feedback stats
            stats = tracker.get_feedback_stats()

            logger.info(f"  ✓ Feedback stats: {stats['total_feedbacks']} feedbacks recorded")

            # Test 5: Get negative turns
            negative_turns = tracker.get_negative_turns(limit=10)

            logger.info(f"  ✓ Found {len(negative_turns)} turns with negative feedback")

            return {
                "status": "passed",
                "turns_recorded": 2,
                "feedbacks_recorded": 2,
                "stats": stats,
                "negative_turns_count": len(negative_turns)
            }

        except Exception as e:
            logger.error(f"Self-correction test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_knowledge_gap_detection(self) -> Dict[str, Any]:
        """Test Knowledge Gap Detection."""
        try:
            logger.info("Testing knowledge gap detection...")

            # Check if knowledge gap infrastructure exists
            from backend.qdrant.client_wrapper import get_qdrant_client

            client = get_qdrant_client()

            # Verify collection exists
            try:
                info = client.get_collection("lexi_knowledge_gaps")
                logger.info(f"  ✓ Knowledge gaps collection exists")
                logger.info(f"    Vector size: {info.config.params.vectors.size}")
                logger.info(f"    Points: {info.points_count}")

                return {
                    "status": "passed",
                    "collection_exists": True,
                    "vector_size": info.config.params.vectors.size,
                    "points_count": info.points_count
                }

            except Exception as e:
                return {
                    "status": "failed",
                    "error": f"Knowledge gaps collection not found: {e}"
                }

        except Exception as e:
            logger.error(f"Knowledge gap detection test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_web_search_integration(self) -> Dict[str, Any]:
        """Test Web Search Integration."""
        try:
            from backend.services.web_search import get_web_search_service

            logger.info("Testing web search integration...")

            service = get_web_search_service()

            if not service.is_enabled():
                logger.warning("  ⚠️  Web search is disabled (API key missing)")
                return {
                    "status": "passed",
                    "enabled": False,
                    "reason": "API key not configured"
                }

            # Test search
            logger.info("  Performing test search...")

            result = await service.search(
                query="Python programming language",
                max_results=3,
                search_depth="basic"
            )

            if result and result.get("results"):
                logger.info(f"  ✓ Search returned {len(result['results'])} results")

                # Check result structure
                first_result = result["results"][0]
                required_fields = ["title", "url", "content"]
                has_all_fields = all(field in first_result for field in required_fields)

                return {
                    "status": "passed",
                    "enabled": True,
                    "results_count": len(result["results"]),
                    "has_required_fields": has_all_fields
                }
            else:
                return {
                    "status": "failed",
                    "error": "Search returned no results"
                }

        except Exception as e:
            logger.error(f"Web search test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_chat_processing(self) -> Dict[str, Any]:
        """Test Chat Processing End-to-End."""
        try:
            from backend.core.chat_processing import process_chat_message_async

            logger.info("Testing chat processing end-to-end...")

            # Test message
            test_message = "Was ist Python und wofür wird es verwendet?"

            logger.info(f"  Test query: '{test_message}'")

            start_time = time.time()

            response = await process_chat_message_async(
                message=test_message,
                user_id=self.test_user_id
            )

            processing_time = (time.time() - start_time) * 1000  # ms

            if not response:
                return {
                    "status": "failed",
                    "error": "No response received"
                }

            # Check response structure
            has_content = bool(response.get("response"))
            has_metadata = "memories_used" in response

            logger.info(f"  ✓ Response received ({processing_time:.0f}ms)")
            logger.info(f"    Memories used: {response.get('memories_used', 0)}")
            logger.info(f"    Response length: {len(response.get('response', ''))} chars")

            return {
                "status": "passed",
                "processing_time_ms": processing_time,
                "has_content": has_content,
                "has_metadata": has_metadata,
                "memories_used": response.get("memories_used", 0),
                "response_length": len(response.get("response", ""))
            }

        except Exception as e:
            logger.error(f"Chat processing test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test Performance Benchmarks."""
        try:
            from backend.memory.adapter import store_memory_async, retrieve_memories

            logger.info("Testing performance benchmarks...")

            metrics = {}

            # Benchmark 1: Memory storage
            logger.info("  Benchmarking memory storage...")

            storage_times = []
            for i in range(5):
                start = time.time()
                await store_memory_async(
                    content=f"Test memory {i} for benchmarking",
                    user_id=self.test_user_id,
                    tags=["benchmark"]
                )
                storage_times.append((time.time() - start) * 1000)

            metrics["storage_avg_ms"] = sum(storage_times) / len(storage_times)
            logger.info(f"    Avg storage time: {metrics['storage_avg_ms']:.1f}ms")

            # Benchmark 2: Memory retrieval
            logger.info("  Benchmarking memory retrieval...")

            retrieval_times = []
            for i in range(5):
                start = time.time()
                retrieve_memories(
                    user_id=self.test_user_id,
                    query=f"test query {i}",
                    limit=5
                )
                retrieval_times.append((time.time() - start) * 1000)

            metrics["retrieval_avg_ms"] = sum(retrieval_times) / len(retrieval_times)
            logger.info(f"    Avg retrieval time: {metrics['retrieval_avg_ms']:.1f}ms")

            # Benchmark 3: Embedding cache
            logger.info("  Checking embedding cache...")

            try:
                from backend.embeddings.embedding_cache import get_cache_stats
                cache_stats = get_cache_stats()
                metrics["embedding_cache"] = cache_stats
                logger.info(f"    Cache size: {cache_stats.get('cache_size', 0)}")
                logger.info(f"    Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            except (ImportError, AttributeError):
                logger.info("    Cache stats not available")
                metrics["embedding_cache"] = {"status": "not_available"}

            # Benchmark 4: Qdrant connection
            logger.info("  Benchmarking Qdrant connection...")

            from backend.qdrant.client_wrapper import get_qdrant_client

            client = get_qdrant_client()

            qdrant_times = []
            for i in range(5):
                start = time.time()
                client.get_collections()
                qdrant_times.append((time.time() - start) * 1000)

            metrics["qdrant_connection_avg_ms"] = sum(qdrant_times) / len(qdrant_times)
            logger.info(f"    Avg Qdrant latency: {metrics['qdrant_connection_avg_ms']:.1f}ms")

            self.results["metrics"] = metrics

            return {
                "status": "passed",
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Performance benchmark test failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    def save_report(self, filepath: str = None) -> str:
        """Speichert Test-Report."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"docs/integration_test_report_{timestamp}.json"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n📄 Test report saved to: {filepath}")
        return filepath


async def main():
    """Main entry point."""
    try:
        suite = IntegrationTestSuite()
        results = await suite.run_all_tests()

        # Save report
        report_path = suite.save_report()

        # Print summary
        print("\n" + "="*70)
        print("🎯 INTEGRATION TEST RESULTS")
        print("="*70)

        for test_name, result in results["tests"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {test_name}: {result['status'].upper()}")

            if result["status"] == "failed":
                print(f"   Error: {result.get('error', 'Unknown')}")

        print("="*70)
        print(f"Success Rate: {results['summary']['success_rate']}")
        print("="*70)

        # Return exit code
        return 0 if results["summary"]["failed"] == 0 else 1

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
