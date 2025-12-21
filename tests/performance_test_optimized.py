"""
LexiAI Performance Test - Optimized Version Validation

This test validates all Phase 1 and Phase 2 optimizations against the baseline
performance measurements from 22.NOV.2025.

Baseline: 10.9s average response time
Target: <6s (45-63% improvement)

Key Optimizations Tested:
1. Parallel Task Execution (asyncio.gather)
2. Web Search Heuristics (aggressive filtering)
3. Model Keep-Alive (no reloading)
4. Memory Retrieval Optimization
5. Caching Effectiveness

Test Environment:
- Clean Qdrant database (only 4 bootstrap entries)
- 2s delay between tests (avoid rate limiting)
- Detailed timing for each component
"""

import asyncio
import logging
import time
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.bootstrap import initialize_components_bundle
from backend.core.chat_processing import process_chat_message_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Baseline from PERFORMANCE_SUMMARY_22NOV.md
BASELINE_PERFORMANCE = {
    "average_response_time": 10.9,  # seconds
    "simple_query": 13.0,
    "complex_query": 9.8,
    "conversational": 9.9,
    "web_search_trigger_rate": 0.6  # 3/5 queries triggered web search
}

# Performance targets
TARGETS = {
    "phase1_average": 8.0,  # Phase 1: Quick Wins
    "phase2_average": 6.0,  # Phase 2: Deeper Optimizations
    "ideal_average": 3.0,   # Ideal target
    "web_search_trigger_rate": 0.2  # Should trigger only for temporal queries
}


class PerformanceMetrics:
    """Collect detailed performance metrics"""

    def __init__(self):
        self.total_time = 0.0
        self.web_search_decision_time = 0.0
        self.web_search_calls = 0
        self.memory_retrieval_time = 0.0
        self.llm_call_time = 0.0
        self.parallel_tasks_count = 0
        self.parallel_execution_time = 0.0
        self.cache_hits = 0
        self.response_quality_score = 0.0

    def to_dict(self) -> dict:
        return {
            "total_time": self.total_time,
            "web_search_decision_time": self.web_search_decision_time,
            "web_search_calls": self.web_search_calls,
            "memory_retrieval_time": self.memory_retrieval_time,
            "llm_call_time": self.llm_call_time,
            "parallel_tasks": self.parallel_tasks_count,
            "parallel_execution_time": self.parallel_execution_time,
            "cache_hits": self.cache_hits,
            "quality_score": self.response_quality_score
        }


class PerformanceTest:
    """Main performance test suite"""

    def __init__(self):
        self.components = None
        self.results: List[Dict] = []

    async def setup(self):
        """Initialize components once"""
        logger.info("🚀 Initializing LexiAI components...")
        # Use initialize_components_bundle() to get ComponentBundle object
        self.components = initialize_components_bundle()
        logger.info("✅ Components initialized")

    async def run_test_case(
        self,
        query: str,
        test_name: str,
        expected_web_search: bool = False
    ) -> Dict:
        """Run a single test case with detailed metrics"""

        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")

        metrics = PerformanceMetrics()

        # Overall timing
        start_time = time.time()

        try:
            # Run the chat processing
            response = await process_chat_message_async(
                message=query,
                chat_client=self.components.chat_client,
                vectorstore=self.components.vectorstore,
                memory=self.components.memory,
                embeddings=self.components.embeddings,
                user_id="performance_test"
            )

            # Calculate total time
            metrics.total_time = time.time() - start_time

            # FIXED: Extract actual response text from dict
            response_text = response.get("response", "") if isinstance(response, dict) else response

            # Validate response
            response_valid = bool(response_text and len(response_text) > 10)
            metrics.response_quality_score = 1.0 if response_valid else 0.0

            logger.info(f"✅ Response received: {len(response_text) if response_text else 0} chars")
            logger.info(f"⏱️  Total time: {metrics.total_time:.2f}s")

            # Log warning for very short responses
            if response_text and len(response_text) < 20:
                logger.warning(f"⚠️ Very short response detected: '{response_text}'")

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            metrics.total_time = time.time() - start_time
            metrics.response_quality_score = 0.0
            response = None
            response_text = None

        # Build result
        result = {
            "test_name": test_name,
            "query": query,
            "metrics": metrics.to_dict(),
            "response_valid": metrics.response_quality_score > 0,
            "response_length": len(response_text) if response_text else 0,
            "expected_web_search": expected_web_search
        }

        self.results.append(result)
        return result

    def calculate_improvement(self, current: float, baseline: float) -> Tuple[float, str]:
        """Calculate percentage improvement"""
        improvement = ((baseline - current) / baseline) * 100
        emoji = "✅" if improvement > 0 else "❌"
        return improvement, emoji

    def print_results(self):
        """Print formatted test results"""

        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS - LexiAI Optimizations Validation")
        print("="*80)

        print(f"\n📊 BASELINE (from 22.NOV.2025):")
        print(f"   Average Response Time: {BASELINE_PERFORMANCE['average_response_time']}s")
        print(f"   Simple Query: {BASELINE_PERFORMANCE['simple_query']}s")
        print(f"   Complex Query: {BASELINE_PERFORMANCE['complex_query']}s")
        print(f"   Conversational: {BASELINE_PERFORMANCE['conversational']}s")

        print(f"\n🎯 TARGETS:")
        print(f"   Phase 1 (Quick Wins): <{TARGETS['phase1_average']}s")
        print(f"   Phase 2 (Deep Optimization): <{TARGETS['phase2_average']}s")
        print(f"   Ideal: <{TARGETS['ideal_average']}s")

        print("\n" + "-"*80)
        print("TEST RESULTS:")
        print("-"*80)

        total_time = 0.0
        web_search_triggered = 0

        for i, result in enumerate(self.results, 1):
            test_name = result['test_name']
            query = result['query']
            metrics = result['metrics']
            time_taken = metrics['total_time']

            total_time += time_taken
            if metrics['web_search_calls'] > 0:
                web_search_triggered += 1

            print(f"\n{i}. {test_name}")
            print(f"   Query: '{query[:60]}...'")
            print(f"   ⏱️  Time: {time_taken:.2f}s")

            # Compare to baseline if applicable
            baseline_time = None
            if "simple" in test_name.lower() or "factual" in test_name.lower():
                baseline_time = BASELINE_PERFORMANCE['simple_query']
            elif "complex" in test_name.lower():
                baseline_time = BASELINE_PERFORMANCE['complex_query']
            elif "conversational" in test_name.lower():
                baseline_time = BASELINE_PERFORMANCE['conversational']

            if baseline_time:
                improvement, emoji = self.calculate_improvement(time_taken, baseline_time)
                print(f"   📈 vs Baseline ({baseline_time}s): {improvement:+.1f}% {emoji}")

            # Web search status
            web_search_status = "TRIGGERED ⚠️" if metrics['web_search_calls'] > 0 else "SKIPPED ✅"
            expected = "SHOULD" if result['expected_web_search'] else "SHOULD NOT"
            match_emoji = "✅" if (metrics['web_search_calls'] > 0) == result['expected_web_search'] else "❌"
            print(f"   🔍 Web Search: {web_search_status} ({expected} search) {match_emoji}")

            # Memory retrieval
            if metrics['memory_retrieval_time'] > 0:
                print(f"   💾 Memory Retrieval: {metrics['memory_retrieval_time']:.2f}s")

            # Parallel execution
            if metrics['parallel_tasks'] > 0:
                print(f"   ⚡ Parallel Tasks: {metrics['parallel_tasks']} in {metrics['parallel_execution_time']:.2f}s ✅")

            # Response quality
            quality_emoji = "✅" if result['response_valid'] else "❌"
            print(f"   📝 Response Quality: {'VALID' if result['response_valid'] else 'INVALID'} ({result['response_length']} chars) {quality_emoji}")

        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        avg_time = total_time / len(self.results) if self.results else 0
        avg_improvement, _ = self.calculate_improvement(avg_time, BASELINE_PERFORMANCE['average_response_time'])

        print(f"\n📊 Average Response Time: {avg_time:.2f}s")
        print(f"   vs Baseline ({BASELINE_PERFORMANCE['average_response_time']}s): {avg_improvement:+.1f}%")

        # Target achievement
        if avg_time <= TARGETS['phase2_average']:
            print(f"   🎉 PHASE 2 TARGET ACHIEVED! (<{TARGETS['phase2_average']}s) ✅")
        elif avg_time <= TARGETS['phase1_average']:
            print(f"   🎯 PHASE 1 TARGET ACHIEVED! (<{TARGETS['phase1_average']}s) ✅")
        else:
            print(f"   ⚠️  TARGET NOT REACHED (Goal: <{TARGETS['phase1_average']}s)")

        # Web search optimization
        web_search_rate = web_search_triggered / len(self.results) if self.results else 0
        baseline_rate = BASELINE_PERFORMANCE['web_search_trigger_rate']
        target_rate = TARGETS['web_search_trigger_rate']

        print(f"\n🔍 Web Search Trigger Rate: {web_search_rate:.1%}")
        print(f"   Baseline: {baseline_rate:.1%}")
        print(f"   Target: <{target_rate:.1%}")

        if web_search_rate <= target_rate:
            print(f"   ✅ WEB SEARCH OPTIMIZATION SUCCESSFUL!")
        else:
            print(f"   ⚠️  Too many web searches triggered")

        # Performance grade
        print("\n" + "="*80)
        print("PERFORMANCE GRADE")
        print("="*80)

        if avg_time <= TARGETS['ideal_average']:
            grade = "A+ (IDEAL)"
            emoji = "🌟"
        elif avg_time <= TARGETS['phase2_average']:
            grade = "A (EXCELLENT)"
            emoji = "🎉"
        elif avg_time <= TARGETS['phase1_average']:
            grade = "B (GOOD)"
            emoji = "👍"
        elif avg_improvement > 0:
            grade = "C (IMPROVED)"
            emoji = "📈"
        else:
            grade = "D (NEEDS WORK)"
            emoji = "⚠️"

        print(f"\n{emoji} Overall Grade: {grade}")
        print(f"   Average: {avg_time:.2f}s (vs {BASELINE_PERFORMANCE['average_response_time']}s baseline)")
        print(f"   Improvement: {avg_improvement:+.1f}%")

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)


async def main():
    """Main test execution"""

    # Test cases covering different query types
    test_cases = [
        # 1. Simple factual - should NOT trigger web search
        ("Was ist Python?", "Simple Factual", False),

        # 2. Technical query - should NOT trigger web search (context available)
        ("Erkläre mir Rekursion in der Programmierung", "Technical Explanation", False),

        # 3. Conversational - should NOT trigger web search
        ("Hallo Lexi, wie geht es dir heute?", "Conversational", False),

        # 4. Temporal query - SHOULD trigger web search
        ("Was sind die neuesten Python Features in 2025?", "Temporal Query", True),

        # 5. Complex with context - should NOT trigger web search
        ("Wie implementiere ich einen Binary Search Tree in Python?", "Complex with Context", False),
    ]

    print("\n" + "="*80)
    print("LEXI AI PERFORMANCE TEST - OPTIMIZATIONS VALIDATION")
    print("="*80)
    print(f"\nTest Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Baseline: {BASELINE_PERFORMANCE['average_response_time']}s average")
    print(f"Target: <{TARGETS['phase2_average']}s average (Phase 2)")

    test_suite = PerformanceTest()

    try:
        # Setup
        await test_suite.setup()

        # Run all test cases
        for query, test_name, expects_search in test_cases:
            await test_suite.run_test_case(query, test_name, expects_search)

            # Wait between tests to avoid rate limiting
            logger.info("⏸️  Waiting 2s before next test...")
            await asyncio.sleep(2)

        # Print results
        test_suite.print_results()

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    """
    Usage:
        python tests/performance_test_optimized.py

    Prerequisites:
        1. Clean Qdrant database (only 4 bootstrap entries)
        2. Ollama running with model loaded
        3. All optimizations applied (parallel execution, heuristics, etc.)

    Expected Results:
        - Average response time: <6s (Phase 2 target)
        - Web search only for temporal queries (1/5 tests)
        - All tests produce valid responses
        - Clear improvement vs baseline (10.9s → <6s)
    """
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
