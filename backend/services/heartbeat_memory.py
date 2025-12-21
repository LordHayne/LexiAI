"""
Intelligenter Heartbeat Service für Memory Management - Thread-safe und performant.

Dieser Service führt periodisch folgende Operationen aus:

ACTIVE MODE (User ist aktiv):
1. Lightweight Maintenance nur: Relevanz-Updates

IDLE MODE (User ist idle >30min):
1. Phase 1: Memory Synthesis - Meta-Wissen generieren
2. Phase 2: Memory Consolidation - Ähnliche Memories zusammenführen
3. Phase 3: Relevance Update - Adaptive Relevanz anpassen
4. Phase 4: Intelligent Cleanup - Ungenutzte Memories löschen
5. Phase 5: Self-Correction - Fehleranalyse
6. Phase 6-8: Goal/Pattern/Knowledge Gap Analysis

Das System unterscheidet zwischen leichten (ACTIVE) und intensiven (IDLE) Tasks.
"""

import logging
import time
import asyncio
import threading
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Callable
from uuid import UUID
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher

from backend.core.component_cache import get_cached_components
from backend.memory.memory_intelligence import (
    get_usage_tracker,
    MemoryConsolidator,
    IntelligentMemoryCleanup,
    update_memory_relevance
)
from backend.memory.activity_tracker import ActivityTracker, is_system_idle
from backend.memory.memory_synthesizer import MemorySynthesizer
from backend.memory.self_correction import analyze_and_correct_failures

logger = logging.getLogger("lexi_middleware.heartbeat")


# ========================================
# Configuration
# ========================================

@dataclass
class HeartbeatConfig:
    """Centralized configuration for heartbeat service."""

    # Timing
    run_interval_seconds: int = 300  # 5 minutes
    idle_threshold_minutes: int = 30

    # Memory age and relevance
    max_age_days: int = 90  # Longer memory retention
    min_relevance: float = 0.1  # More patient with memories

    # Consolidation
    consolidation_threshold: float = 0.85
    enable_consolidation: bool = True
    enable_intelligent_cleanup: bool = True

    # Per-run limits (DB operations) - Erhöht für größere Datasets
    max_new_entries_per_run: int = 50
    max_updates_per_run: int = 200
    max_deletes_per_run: int = 50
    max_db_ops_per_run: int = 300

    # Global limits (total entries)
    max_total_meta_knowledge: int = 50
    max_total_patterns: int = 100
    max_total_knowledge_gaps: int = 50

    # Per-run creation limits (new items per phase)
    max_new_meta_knowledge: int = 5
    max_goal_reminders: int = 3
    max_new_patterns: int = 5
    max_new_knowledge_gaps: int = 5

    # LLM settings
    llm_timeout_seconds: float = 60.0  # Increased from 30s
    llm_max_retries: int = 2

    # Performance
    stop_check_interval: int = 50  # Check stop signal every N iterations
    pattern_dedup_early_termination: float = 0.99  # Stop if perfect match

    @classmethod
    def from_env(cls) -> 'HeartbeatConfig':
        """Load configuration from environment variables."""
        run_interval_seconds = int(os.getenv("LEXI_HEARTBEAT_INTERVAL", "300"))
        idle_threshold_minutes = int(os.getenv("LEXI_IDLE_THRESHOLD_MINUTES", "30"))

        if run_interval_seconds < 60:
            logger.warning(
                "LEXI_HEARTBEAT_INTERVAL too low (%ss). Using minimum 60s.",
                run_interval_seconds,
            )
            run_interval_seconds = 60

        if idle_threshold_minutes < 5:
            logger.warning(
                "LEXI_IDLE_THRESHOLD_MINUTES too low (%s). Using minimum 5 minutes.",
                idle_threshold_minutes,
            )
            idle_threshold_minutes = 5

        return cls(
            run_interval_seconds=run_interval_seconds,
            max_age_days=int(os.getenv("LEXI_MEMORY_MAX_AGE_DAYS", "90")),
            min_relevance=float(os.getenv("LEXI_MIN_RELEVANCE", "0.1")),
            idle_threshold_minutes=idle_threshold_minutes,
            llm_timeout_seconds=float(os.getenv("LEXI_LLM_TIMEOUT", "60")),
        )


# Global configuration instance
_config = HeartbeatConfig.from_env()
_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_thread_lock = threading.Lock()


# ========================================
# Thread-safe State Management
# ========================================

@dataclass
class HeartbeatState:
    """Thread-safe heartbeat state management."""

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # Control flags
    stop_learning: bool = False
    learning_in_progress: bool = False

    # Statistics
    last_run: Optional[str] = None
    last_consolidation: Optional[str] = None
    last_smart_home_consolidation: Optional[str] = None
    last_synthesis: Optional[str] = None
    last_correction: Optional[str] = None
    mode: str = "unknown"

    deleted_count: int = 0
    consolidated_count: int = 0
    synthesized_count: int = 0
    corrections_count: int = 0
    updated_count: int = 0
    total_memories: int = 0
    run_count: int = 0

    errors: List[Dict] = field(default_factory=list)

    def set_stop_learning(self, value: bool):
        """Set stop learning flag (thread-safe)."""
        with self._lock:
            self.stop_learning = value

    def is_stop_learning(self) -> bool:
        """Check stop learning flag (thread-safe)."""
        with self._lock:
            return self.stop_learning

    def set_learning_in_progress(self, value: bool):
        """Set learning in progress flag (thread-safe)."""
        with self._lock:
            self.learning_in_progress = value

    def is_learning_in_progress(self) -> bool:
        """Check learning in progress flag (thread-safe)."""
        with self._lock:
            return self.learning_in_progress

    def update_stats(self, updates: Dict):
        """Update statistics (thread-safe)."""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Limit error list size
            if "errors" in updates:
                self.errors = self.errors[-10:]

    def add_error(self, error: str):
        """Add error to list (thread-safe)."""
        with self._lock:
            self.errors.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error
            })
            self.errors = self.errors[-10:]

    def get_status(self) -> Dict:
        """Get status snapshot (thread-safe)."""
        with self._lock:
            return {
                "last_run": self.last_run,
                "last_consolidation": self.last_consolidation,
                "last_smart_home_consolidation": self.last_smart_home_consolidation,
                "last_synthesis": self.last_synthesis,
                "last_correction": self.last_correction,
                "mode": self.mode,
                "deleted_count": self.deleted_count,
                "consolidated_count": self.consolidated_count,
                "synthesized_count": self.synthesized_count,
                "corrections_count": self.corrections_count,
                "updated_count": self.updated_count,
                "total_memories": self.total_memories,
                "run_count": self.run_count,
                "errors": self.errors.copy()
            }


# Global state instance
_heartbeat_state = HeartbeatState()


# ========================================
# Database Operation Limits
# ========================================

class HeartbeatLimits:
    """
    Tracking and enforcement for DB operations per heartbeat run.
    Prevents uncontrolled growth through hard limits.
    """

    def __init__(self, config: HeartbeatConfig):
        self.config = config
        self.new_entries = 0
        self.updates = 0
        self.deletes = 0

    @property
    def total_ops(self):
        """Get total number of DB operations."""
        return self.new_entries + self.updates + self.deletes

    def can_create(self, count=1) -> bool:
        """Check if we can create more entries."""
        return (self.new_entries + count <= self.config.max_new_entries_per_run and
                self.total_ops + count <= self.config.max_db_ops_per_run)

    def can_update(self, count=1) -> bool:
        """Check if we can update more entries."""
        return (self.updates + count <= self.config.max_updates_per_run and
                self.total_ops + count <= self.config.max_db_ops_per_run)

    def can_delete(self, count=1) -> bool:
        """Check if we can delete more entries."""
        return (self.deletes + count <= self.config.max_deletes_per_run and
                self.total_ops + count <= self.config.max_db_ops_per_run)

    def track_create(self, count=1):
        """Track created entries."""
        self.new_entries += count

    def track_update(self, count=1):
        """Track updated entries."""
        self.updates += count

    def track_delete(self, count=1):
        """Track deleted entries."""
        self.deletes += count

    def get_summary(self) -> str:
        """Get summary string."""
        return f"{self.new_entries} creates, {self.updates} updates, {self.deletes} deletes (total: {self.total_ops})"


# ========================================
# Memory Budget Manager
# ========================================

class MemoryBudgetManager:
    """
    Manages memory creation budgets across all phases.
    Ensures global limits are respected.
    """

    def __init__(self, vectorstore, config: HeartbeatConfig):
        self.vectorstore = vectorstore
        self.config = config
        self._lock = threading.RLock()
        self._cached_counts = {}
        self._cache_time = None

    def _count_by_metadata_flag(self, flag_name: str) -> int:
        """Count entries by metadata flag (with caching)."""
        with self._lock:
            # Cache for 60 seconds
            now = time.time()
            if self._cache_time and (now - self._cache_time) < 60:
                return self._cached_counts.get(flag_name, 0)

            try:
                all_memories = self.vectorstore.get_all_entries(with_vectors=False)
                counts = defaultdict(int)

                for mem in all_memories:
                    if mem.metadata.get("is_meta_knowledge", False):
                        counts["is_meta_knowledge"] += 1
                    if mem.metadata.get("reminder_type") == "goal_reminder":
                        counts["goal_reminders"] += 1
                    # Patterns and gaps are tracked separately

                self._cached_counts = counts
                self._cache_time = now

                return counts.get(flag_name, 0)
            except Exception as e:
                logger.error(f"Error counting by flag {flag_name}: {e}")
                return 0

    def can_create_meta_knowledge(self, count: int = 1) -> bool:
        """Check if we can create meta knowledge entries."""
        current = self._count_by_metadata_flag("is_meta_knowledge")
        return current + count <= self.config.max_total_meta_knowledge

    def get_meta_knowledge_remaining(self) -> int:
        """Get remaining meta knowledge slots."""
        current = self._count_by_metadata_flag("is_meta_knowledge")
        return max(0, self.config.max_total_meta_knowledge - current)


# ========================================
# LLM Retry Helper
# ========================================

async def call_llm_with_retry(func: Callable, max_retries: int = None, timeout: float = None) -> any:
    """
    Call LLM function with retry and timeout.

    Args:
        func: Async function to call
        max_retries: Maximum retry attempts (default from config)
        timeout: Timeout in seconds (default from config)

    Returns:
        Result from function

    Raises:
        asyncio.TimeoutError: If all retries time out
        Exception: If function raises non-timeout exception
    """
    max_retries = max_retries or _config.llm_max_retries
    timeout = timeout or _config.llm_timeout_seconds

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(func(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"LLM timeout attempt {attempt+1}/{max_retries}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"LLM timeout after {max_retries} attempts")
                raise
        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt+1}: {e}")
            raise

    return None


# ========================================
# Main Heartbeat Logic
# ========================================

async def intelligent_memory_maintenance() -> Dict:
    """
    Main heartbeat function with idle/active mode detection.

    Returns:
        Dict with statistics about performed operations
    """
    global _heartbeat_state, _config

    # Reset stop signal
    _heartbeat_state.set_stop_learning(False)

    logger.info("🧠 Starte intelligenten Memory-Heartbeat")
    start_time = time.time()

    try:
        # Check if system is idle
        idle = is_system_idle(minutes=_config.idle_threshold_minutes)

        if idle:
            logger.info(f"😴 IDLE MODE: System ist {_config.idle_threshold_minutes}+ min idle - starte Deep Learning")
            _heartbeat_state.set_learning_in_progress(True)
            try:
                stats = await run_deep_learning_tasks()
            finally:
                _heartbeat_state.set_learning_in_progress(False)
            mode = "IDLE"
        else:
            logger.info("⚡ ACTIVE MODE: User aktiv - nur lightweight maintenance")
            stats = run_lightweight_maintenance()
            mode = "ACTIVE"

        # Check and run daily Smart Home pattern consolidation (23:59 Uhr)
        try:
            sh_consolidated = await check_and_run_daily_consolidation()
            if sh_consolidated > 0:
                stats["smart_home_patterns_consolidated"] = sh_consolidated
                logger.info(f"🏠 Smart Home Patterns konsolidiert: {sh_consolidated}")
        except Exception as e:
            logger.error(f"Fehler bei Smart Home Pattern Consolidation: {e}")

        # Update global state
        _heartbeat_state.update_stats({
            "last_run": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "deleted_count": _heartbeat_state.deleted_count + stats.get("deleted", 0),
            "consolidated_count": _heartbeat_state.consolidated_count + stats.get("consolidated", 0),
            "synthesized_count": _heartbeat_state.synthesized_count + stats.get("synthesized", 0),
            "corrections_count": _heartbeat_state.corrections_count + stats.get("corrections", 0),
            "updated_count": _heartbeat_state.updated_count + stats.get("updated", 0),
            "run_count": _heartbeat_state.run_count + 1
        })

        elapsed = time.time() - start_time
        logger.info(f"✅ Heartbeat ({mode}) abgeschlossen in {elapsed:.2f}s: {stats}")

        return stats

    except Exception as e:
        logger.error(f"❌ Fehler im Heartbeat: {e}", exc_info=True)
        _heartbeat_state.add_error(str(e))
        return {"deleted": 0, "consolidated": 0, "synthesized": 0, "corrections": 0, "updated": 0, "errors": 1}


async def run_deep_learning_tasks() -> Dict:
    """
    Run intensive learning tasks (IDLE mode only).

    Returns:
        Dict with statistics including "stopped_early" flag
    """
    global _heartbeat_state, _config

    # Initialize limits and budget managers
    limits = HeartbeatLimits(_config)

    bundle = get_cached_components()
    vectorstore = bundle.vectorstore
    embeddings = bundle.embeddings
    usage_tracker = get_usage_tracker()

    budget_manager = MemoryBudgetManager(vectorstore, _config)

    # Get all memories
    all_memories = vectorstore.get_all_entries()
    _heartbeat_state.update_stats({"total_memories": len(all_memories)})

    logger.info(f"📊 Deep Learning auf {len(all_memories)} Memories")
    logger.info(f"📊 Limits: {_config.max_new_entries_per_run} creates, {_config.max_updates_per_run} updates, {_config.max_deletes_per_run} deletes")

    stats = {
        "synthesized": 0,
        "consolidated": 0,
        "updated": 0,
        "deleted": 0,
        "errors": 0,
        "corrections": 0,
        "goals_reminded": 0,
        "patterns_detected": 0,
        "knowledge_gaps_found": 0,
        "stopped_early": False,
        "db_operations": 0
    }

    # Phase 1: Memory Synthesis
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 1 - breche ab")
        stats["stopped_early"] = True
        return stats

    if len(all_memories) >= 10:
        logger.info("🧪 Phase 1: Memory Synthesis - Generiere Meta-Wissen")
        stats["synthesized"] = _synthesize_memories(bundle, limits, budget_manager)

    # Phase 2: Memory Consolidation
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 2 - breche ab")
        stats["stopped_early"] = True
        return stats

    if _config.enable_consolidation and len(all_memories) >= 10:
        logger.info("🔗 Phase 2: Memory Consolidation")
        stats["consolidated"] = _consolidate_memories(vectorstore, embeddings, all_memories, limits)
        if stats["consolidated"] > 0:
            all_memories = vectorstore.get_all_entries()

    # Phase 3: Self-Correction
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 3 - breche ab")
        stats["stopped_early"] = True
        return stats

    logger.info("🔍 Phase 3: Self-Correction - Analysiere Fehler")
    corrections = analyze_and_correct_failures(
        stop_check_fn=lambda: _heartbeat_state.is_stop_learning()
    )
    stats["corrections"] = corrections
    if corrections > 0:
        _heartbeat_state.update_stats({"last_correction": datetime.now(timezone.utc).isoformat()})

    # Phase 4: Update adaptive Relevance
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 4 - breche ab")
        stats["stopped_early"] = True
        return stats

    if len(all_memories) > 0:
        logger.info("📈 Phase 4: Update adaptive Relevance Scores")
        stats["updated"] = _update_all_relevances(vectorstore, all_memories, usage_tracker, limits)

    # Phase 5: Intelligent Cleanup
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 5 - breche ab")
        stats["stopped_early"] = True
        return stats

    if _config.enable_intelligent_cleanup and len(all_memories) > 0:
        logger.info("🧹 Phase 5: Intelligent Cleanup")
        stats["deleted"] = _cleanup_memories(vectorstore, all_memories, usage_tracker, limits)
        if stats["deleted"] > 0:
            all_memories = vectorstore.get_all_entries()

    # Phase 6: Goal Analysis
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 6 - breche ab")
        stats["stopped_early"] = True
        return stats

    logger.info("🎯 Phase 6: Goal Analysis - Prüfe Benutzerziele")
    stats["goals_reminded"] = _analyze_goals(bundle, all_memories, limits)

    # Phase 7: Pattern Detection
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 7 - breche ab")
        stats["stopped_early"] = True
        return stats

    if len(all_memories) >= 5:
        logger.info("🔍 Phase 7: Pattern Detection - Erkenne wiederkehrende Themen")
        stats["patterns_detected"] = _detect_patterns(bundle, all_memories, limits)

    # Phase 8: Knowledge Gap Detection
    if _heartbeat_state.is_stop_learning():
        logger.warning("⚠️ Stop-Signal erkannt vor Phase 8 - breche ab")
        stats["stopped_early"] = True
        return stats

    if len(all_memories) >= 5:
        logger.info("🧠 Phase 8: Knowledge Gap Detection - Finde Wissenslücken")
        stats["knowledge_gaps_found"] = await _detect_knowledge_gaps(bundle, all_memories, limits)

    # Finalize stats
    stats["db_operations"] = limits.total_ops

    logger.info(f"📊 DB Operations Summary: {limits.get_summary()}")

    if limits.total_ops >= _config.max_db_ops_per_run:
        logger.warning(f"⚠️ DB Operations limit reached ({limits.total_ops}/{_config.max_db_ops_per_run})")

    return stats


def run_lightweight_maintenance() -> Dict:
    """
    Run lightweight maintenance (ACTIVE mode).

    Returns:
        Dict with statistics
    """
    bundle = get_cached_components()
    vectorstore = bundle.vectorstore
    usage_tracker = get_usage_tracker()
    limits = HeartbeatLimits(_config)

    all_memories = vectorstore.get_all_entries()

    stats = {
        "synthesized": 0,
        "consolidated": 0,
        "corrections": 0,
        "updated": 0,
        "deleted": 0,
        "errors": 0
    }

    # Only relevance updates
    if len(all_memories) > 0:
        logger.info("📈 Lightweight: Update adaptive Relevance Scores")
        stats["updated"] = _update_all_relevances(vectorstore, all_memories, usage_tracker, limits)

    return stats


# ========================================
# Phase Functions (with limit enforcement)
# ========================================

def _synthesize_memories(bundle, limits: HeartbeatLimits, budget_manager: MemoryBudgetManager) -> int:
    """
    Phase 1: Memory Synthesis with budget enforcement.

    Returns:
        Number of synthesized meta-knowledge entries
    """
    global _heartbeat_state, _config

    try:
        # Check budget
        remaining_slots = budget_manager.get_meta_knowledge_remaining()

        if remaining_slots <= 0:
            logger.warning(f"⚠️ Meta-Knowledge budget exhausted - skipping synthesis")
            return 0

        max_new = min(_config.max_new_meta_knowledge, remaining_slots, limits.new_entries_remaining())

        if max_new <= 0:
            logger.warning("⚠️ No slots available for new Meta-Knowledge")
            return 0

        logger.info(f"📊 Meta-Knowledge: creating max {max_new} new entries")

        synthesizer = MemorySynthesizer(
            llm=bundle.chat_client,
            vectorstore=bundle.vectorstore,
            min_cluster_size=3,
            similarity_threshold=0.85,
            max_clusters_per_run=max_new
        )

        # Synthesize for thomas user (TODO: multi-user support)
        results = synthesizer.synthesize_clusters(
            user_id="thomas",
            exclude_meta_knowledge=True
        )

        if results:
            limits.track_create(len(results))
            _heartbeat_state.update_stats({"last_synthesis": datetime.now(timezone.utc).isoformat()})
            logger.info(f"✅ Memory Synthesis: {len(results)} Meta-Wissen Einträge erstellt")

        return len(results)

    except Exception as e:
        logger.error(f"Fehler bei Memory Synthesis: {e}", exc_info=True)
        return 0


def _update_all_relevances(vectorstore, memories, usage_tracker, limits: HeartbeatLimits) -> int:
    """
    Update adaptive relevance with stop signal checks.

    Returns:
        Number of updated memories
    """
    updated_count = 0

    for i, memory in enumerate(memories):
        # Check stop signal periodically
        if i % _config.stop_check_interval == 0:
            if _heartbeat_state.is_stop_learning():
                logger.warning(f"⚠️ Stop signal during relevance update at {i}/{len(memories)}")
                break

        # Check update limit
        if not limits.can_update(1):
            logger.warning(f"⚠️ Update limit reached at {updated_count} updates")
            break

        try:
            new_relevance = update_memory_relevance(memory)
            old_relevance = memory.relevance or 0.5

            # Only update if significant change
            if abs(new_relevance - old_relevance) > 0.05:
                success = vectorstore.update_entry_metadata(
                    memory.id,
                    {"relevance": new_relevance}
                )
                if success:
                    updated_count += 1
                    limits.track_update(1)
                    logger.debug(f"Updated relevance: {memory.id} {old_relevance:.2f} -> {new_relevance:.2f}")

        except Exception as e:
            logger.warning(f"Error updating relevance for {memory.id}: {e}")
            continue

    return updated_count


def _consolidate_memories(
    vectorstore,
    embeddings,
    memories,
    limits: HeartbeatLimits,
    similarity_threshold: float = None,
    min_cluster_size: int = 2,
    allowed_tags: list = None,
    allowed_sources: list = None,
    require_meta_knowledge: bool = False
) -> int:
    """
    Phase 2: Consolidate similar memories with rollback support.

    BUGFIX: Filters out already-consolidated memories to prevent infinite recursion.

    Returns:
        Number of consolidated memories
    """
    consolidated_count = 0

    try:
        # BUGFIX: Filter out already-consolidated memories
        # This prevents infinite recursion where consolidations get consolidated again
        non_consolidated_memories = [
            m for m in memories
            if not (m.content and "Zusammenfassung von" in m.content and "ähnlichen Erinnerungen" in m.content)
            and not getattr(m, "is_meta_knowledge", False)
            and not getattr(m, "superseded", False)
        ]

        if require_meta_knowledge:
            non_consolidated_memories = [
                m for m in memories if getattr(m, "is_meta_knowledge", False)
            ]

        original_count = len(memories)
        filtered_count = len(non_consolidated_memories)

        if original_count != filtered_count:
            logger.info(f"🛡️ Filtered out {original_count - filtered_count} already-consolidated memories")

        if not non_consolidated_memories:
            logger.info("No non-consolidated memories to process")
            return 0

        if allowed_tags:
            allowed_tag_set = set(allowed_tags)
            non_consolidated_memories = [
                m for m in non_consolidated_memories
                if allowed_tag_set.intersection(set(m.tags or []))
            ]

        if allowed_sources:
            allowed_source_set = set(allowed_sources)
            non_consolidated_memories = [
                m for m in non_consolidated_memories
                if (m.source or "") in allowed_source_set
            ]

        if similarity_threshold is None:
            similarity_threshold = _config.consolidation_threshold

        consolidator = MemoryConsolidator()
        similar_groups = consolidator.find_similar_memories(
            non_consolidated_memories,  # Use filtered memories
            similarity_threshold=similarity_threshold
        )
        if min_cluster_size and min_cluster_size > 2:
            similar_groups = [group for group in similar_groups if len(group) >= min_cluster_size]

        logger.info(f"Gefunden: {len(similar_groups)} Gruppen für Konsolidierung")

        for group in similar_groups:
            # Check limits
            if not limits.can_create(1):
                logger.warning("⚠️ Create limit reached, stopping consolidation")
                break
            if not limits.can_delete(len(group)):
                logger.warning(
                    f"⚠️ Delete limit too low for group size {len(group)} - skipping group"
                )
                continue

            # Check stop signal
            if _heartbeat_state.is_stop_learning():
                logger.warning("⚠️ Stop signal during consolidation")
                break

            try:
                # Create consolidated memory
                consolidated = consolidator.consolidate_group(group, embeddings)

                if not consolidated:
                    continue

                # CRITICAL: Store consolidated FIRST, then delete originals
                try:
                    success = vectorstore.store_entry(consolidated)
                    if not success:
                        logger.error("Failed to store consolidated memory - aborting deletion")
                        continue
                except Exception as store_error:
                    logger.error(f"Error storing consolidated memory: {store_error}")
                    continue

                # Track create operation
                limits.track_create(1)

                # Only delete after successful store
                deleted_ids = []
                try:
                    for original in group:
                        try:
                            vectorstore.delete_entry(UUID(original.id))
                            deleted_ids.append(original.id)
                            limits.track_delete(1)
                        except Exception as delete_error:
                            logger.error(f"Failed to delete {original.id}: {delete_error}")
                            # Continue with other deletes, but log the failure

                    consolidated_count += len(deleted_ids)
                    logger.info(f"✅ Konsolidiert: {len(deleted_ids)} Memories -> {consolidated.id}")

                except Exception as e:
                    logger.error(f"Error during deletion phase of consolidation: {e}")
                    # Consolidated memory is already stored, deletions failed
                    # This is acceptable - better to have duplicates than data loss

            except Exception as e:
                logger.warning(f"Fehler bei Konsolidierung einer Gruppe: {e}")
                continue

    except Exception as e:
        logger.error(f"Fehler in Konsolidierungs-Phase: {e}")

    _heartbeat_state.update_stats({"last_consolidation": datetime.now(timezone.utc).isoformat()})
    return consolidated_count


def _consolidate_meta_knowledge(
    vectorstore,
    memories,
    limits: HeartbeatLimits,
    similarity_threshold: float = None,
    min_cluster_size: int = 2
) -> int:
    """
    Merge highly similar meta-knowledge entries into a single canonical entry.
    """
    merged_count = 0
    try:
        meta_memories = [m for m in memories if getattr(m, "is_meta_knowledge", False)]
        if len(meta_memories) < 2:
            return 0

        if similarity_threshold is None:
            similarity_threshold = 0.85

        # Meta-Wissen thematisch clustern (topic key) + text dedup innerhalb des Topics.
        topic_map = defaultdict(list)
        normalized_texts = []
        for mem in meta_memories:
            topic = getattr(mem, "meta_topic", None) or _meta_topic_key(mem.content)
            topic_map[topic].append(mem)
            normalized_texts.append((mem.id, _normalize_meta_text(mem.content)))

        norm_lookup = {mid: text for mid, text in normalized_texts}
        similar_groups = []

        for topic, group_memories in topic_map.items():
            if len(group_memories) < min_cluster_size:
                continue
            used = set()
            for i, mem_i in enumerate(group_memories):
                if mem_i.id in used:
                    continue
                group = [mem_i]
                used.add(mem_i.id)
                text_i = norm_lookup.get(mem_i.id, "")

                for j in range(i + 1, len(group_memories)):
                    mem_j = group_memories[j]
                    if mem_j.id in used:
                        continue
                    text_j = norm_lookup.get(mem_j.id, "")
                    ratio = SequenceMatcher(None, text_i, text_j).ratio()
                    if ratio >= similarity_threshold:
                        group.append(mem_j)
                        used.add(mem_j.id)

                if len(group) >= min_cluster_size:
                    similar_groups.append(group)

        for group in similar_groups:
            if not limits.can_update(1):
                logger.warning("⚠️ Update limit reached during meta consolidation")
                break
            if not limits.can_delete(len(group) - 1):
                logger.warning(
                    f"⚠️ Delete limit too low for meta group size {len(group)} - skipping group"
                )
                continue
            if _heartbeat_state.is_stop_learning():
                logger.warning("⚠️ Stop signal during meta consolidation")
                break

            # Pick canonical entry: prefer longer content, then higher relevance
            group_sorted = sorted(
                group,
                key=lambda m: (len(m.content or ""), m.relevance or 0.0),
                reverse=True
            )
            keeper = group_sorted[0]
            to_remove = group_sorted[1:]

            merged_source_ids = set()
            merged_tags = set(keeper.tags or [])
            merged_tags.update(["meta_knowledge", "synthesized", "meta_consolidated"])

            for mem in group:
                for sid in mem.source_memory_ids or []:
                    merged_source_ids.add(str(sid))

            try:
                success = vectorstore.update_entry_metadata(
                    keeper.id,
                    {
                        "source_memory_ids": list(merged_source_ids),
                        "tags": list(merged_tags),
                        "meta_consolidated": True
                    }
                )
                if success:
                    limits.track_update(1)
                    merged_count += 1
            except Exception as e:
                logger.warning(f"Failed to update meta keeper {keeper.id}: {e}")
                continue

            for mem in to_remove:
                try:
                    vectorstore.delete_entry(UUID(mem.id))
                    limits.track_delete(1)
                except Exception as e:
                    logger.error(f"Failed to delete meta {mem.id}: {e}")

        return merged_count

    except Exception as e:
        logger.error(f"Fehler in Meta-Konsolidierung: {e}")
        return merged_count


def _normalize_meta_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("meta-wissen", "")
    normalized = normalized.replace("meta-wissens-aussage", "")
    normalized = re.sub(r"[^a-z0-9\\s]", "", normalized)
    return re.sub(r"\\s+", " ", normalized).strip()


def _meta_topic_key(text: str) -> str:
    normalized = (text or "").lower()
    patterns = [
        ("greeting", [r"hallo", r"guten morgen", r"guten tag", r"begr[üu]ß"]),
        ("light_control", [r"licht", r"lampe", r"wohnzimmerlicht", r"badezimmerlicht"]),
        ("temperature", [r"temperatur", r"thermostat", r"°c", r"grad"]),
        ("entity_extraction", [r"entit", r"bath", r"bad", r"wc", r"toilet"]),
        ("identity", [r"wer bin ich", r"hei[ßs]e", r"name", r"identit"]),
        ("home_assistant", [r"home assistant", r"ha ", r"smart home"]),
    ]
    for key, pats in patterns:
        if any(re.search(p, normalized) for p in pats):
            return key

    tokens = re.sub(r"[^a-z0-9\\s]", " ", normalized).split()
    tokens = [t for t in tokens if len(t) > 3]
    return "topic_" + "_".join(tokens[:4]) if tokens else "topic_misc"


def _cleanup_memories(
    vectorstore,
    memories,
    usage_tracker,
    limits: HeartbeatLimits,
    max_age_days: int = None,
    min_relevance: float = None,
    unused_after_days: int = None,
    max_unused_relevance: float = None
) -> int:
    """
    Phase 5: Intelligent cleanup with limit enforcement.

    Returns:
        Number of deleted memories
    """
    deleted_count = 0

    try:
        cleanup = IntelligentMemoryCleanup(usage_tracker)

        _mark_superseded_from_meta(vectorstore, memories, limits)
        _prune_meta_knowledge(vectorstore, memories, limits)

        if max_age_days is None:
            max_age_days = _config.max_age_days
        if min_relevance is None:
            min_relevance = _config.min_relevance

        to_delete = cleanup.identify_memories_for_deletion(
            memories,
            max_age_days=max_age_days,
            min_relevance=min_relevance,
            unused_after_days=unused_after_days,
            max_unused_relevance=max_unused_relevance
        )

        logger.info(f"Identifiziert: {len(to_delete)} Memories für Deletion")

        for memory_id in to_delete:
            # Check limit
            if not limits.can_delete(1):
                logger.warning(f"⚠️ Delete limit reached at {deleted_count} deletions")
                break

            # Check stop signal
            if _heartbeat_state.is_stop_learning():
                logger.warning(f"⚠️ Stop signal during cleanup at {deleted_count} deletions")
                break

            try:
                vectorstore.delete_entry(UUID(memory_id))
                deleted_count += 1
                limits.track_delete(1)
            except Exception as e:
                logger.warning(f"Fehler beim Löschen von {memory_id}: {e}")
                continue

    except Exception as e:
        logger.error(f"Fehler in Cleanup-Phase: {e}")

    return deleted_count


def _prune_meta_knowledge(vectorstore, memories, limits: HeartbeatLimits) -> int:
    """
    Enforce a hard cap for meta-knowledge entries (keep most relevant/recent).
    """
    max_meta = _config.max_total_meta_knowledge
    meta_entries = [m for m in memories if getattr(m, "is_meta_knowledge", False)]
    if len(meta_entries) <= max_meta:
        return 0

    # Keep highest relevance first, then most recent
    meta_entries.sort(
        key=lambda m: (m.relevance or 0.0, m.timestamp),
        reverse=True
    )
    to_delete = meta_entries[max_meta:]
    deleted = 0

    for mem in to_delete:
        if not limits.can_delete(1):
            break
        try:
            vectorstore.delete_entry(UUID(mem.id))
            limits.track_delete(1)
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to prune meta memory {mem.id}: {e}")

    if deleted:
        logger.info(f"Pruned {deleted} meta-knowledge entries (cap={max_meta})")
    return deleted


def _mark_superseded_from_meta(vectorstore, memories, limits: HeartbeatLimits) -> int:
    """
    Backfill superseded flags from existing meta-knowledge entries.
    """
    updated_count = 0
    now = datetime.now(timezone.utc).isoformat()
    memory_map = {str(m.id): m for m in memories}

    for memory in memories:
        if not getattr(memory, "is_meta_knowledge", False):
            continue
        if not memory.source_memory_ids:
            continue

        meta_id = str(memory.id)
        for source_id in memory.source_memory_ids:
            source = memory_map.get(str(source_id))
            if not source:
                continue
            if getattr(source, "superseded", False):
                continue

            tags = set(source.tags or [])
            source_value = (source.source or "").lower()
            if not tags.intersection({"chat", "conversation"}):
                if source_value not in {"chat", "conversation"}:
                    continue
            if tags.intersection({"pattern", "aggregated"}):
                continue

            if not limits.can_update(1):
                return updated_count

            new_relevance = source.relevance or 1.0
            if new_relevance > 0.2:
                new_relevance = 0.2

            try:
                success = vectorstore.update_entry_metadata(
                    source.id,
                    {
                        "superseded": True,
                        "superseded_at": now,
                        "superseded_by": meta_id,
                        "relevance": new_relevance
                    }
                )
                if success:
                    updated_count += 1
                    limits.track_update(1)
            except Exception as e:
                logger.warning(f"Fehler beim Backfill von superseded: {source.id} ({e})")

    return updated_count


def _analyze_goals(bundle, all_memories, limits: HeartbeatLimits) -> int:
    """
    Phase 6: Goal Analysis with duplicate detection and limits.

    Returns:
        Number of goal reminders created
    """
    try:
        from backend.memory.goal_tracker import get_goal_tracker, GoalStatus
        from backend.memory.adapter import store_memory

        tracker = get_goal_tracker(bundle.vectorstore)

        # Get goals needing reminder (TODO: multi-user support)
        user_id = "default"
        goals_needing_reminder = tracker.get_goals_needing_reminder(
            user_id=user_id,
            days_since_mention=7
        )

        if not goals_needing_reminder:
            logger.info("✅ Keine Goals benötigen Erinnerung")
            return 0

        logger.info(f"📌 {len(goals_needing_reminder)} Goals brauchen Aufmerksamkeit")

        # Pre-filter: Get existing goal reminders
        reminder_memories = [
            mem for mem in all_memories
            if isinstance(mem.metadata, dict) and
               mem.metadata.get("reminder_type") == "goal_reminder"
        ]

        recent_goal_reminders = defaultdict(list)
        now = datetime.now(timezone.utc)

        for mem in reminder_memories:
            goal_id = mem.metadata.get("goal_id")
            if goal_id and (now - mem.timestamp).days < 7:
                recent_goal_reminders[goal_id].append(mem)

        reminders_created = 0

        for goal in goals_needing_reminder[:_config.max_goal_reminders]:
            # Check limit
            if not limits.can_create(1):
                logger.warning("⚠️ Create limit reached for goal reminders")
                break

            # Check stop signal
            if _heartbeat_state.is_stop_learning():
                logger.warning("⚠️ Stop signal during goal analysis")
                break

            try:
                # Check for recent reminder
                if goal.id in recent_goal_reminders:
                    logger.debug(f"⏭️  Skipping duplicate reminder for goal {goal.id}")
                    continue

                # Check total reminder count for this goal
                total_reminders = len([
                    mem for mem in reminder_memories
                    if mem.metadata.get("goal_id") == goal.id
                ])

                if total_reminders >= 3:
                    logger.debug(f"⏭️  Max reminders reached for goal {goal.id}")
                    continue

                # Analyze progress
                progress_detected = _analyze_goal_progress(goal, all_memories)

                # Create reminder
                if progress_detected:
                    reminder_text = (
                        f"🎯 Proaktive Erinnerung: Dein Ziel '{goal.content}' "
                        f"wurde seit {(now - goal.last_mentioned).days} Tagen nicht erwähnt. "
                        f"Ich habe jedoch Fortschritt in deinen Aktivitäten bemerkt. "
                        f"Möchtest du dein Ziel aktualisieren?"
                    )
                else:
                    reminder_text = (
                        f"🎯 Proaktive Erinnerung: Dein Ziel '{goal.content}' "
                        f"wurde seit {(now - goal.last_mentioned).days} Tagen nicht erwähnt. "
                        f"Wie steht es damit?"
                    )

                store_memory(
                    content=reminder_text,
                    user_id=user_id,
                    tags=["proactive_reminder", "goal_reminder"],
                    metadata={
                        "is_proactive_suggestion": True,
                        "goal_id": goal.id,
                        "goal_category": goal.category,
                        "reminder_type": "goal_reminder"
                    }
                )

                limits.track_create(1)
                reminders_created += 1
                logger.info(f"📬 Proaktive Erinnerung erstellt für Goal: {goal.content[:50]}...")

            except Exception as e:
                logger.warning(f"Fehler bei Goal Reminder für {goal.id}: {e}")

        return reminders_created

    except Exception as e:
        logger.error(f"Fehler in Goal Analysis: {e}")
        return 0


def _analyze_goal_progress(goal, all_memories) -> bool:
    """
    Analyze if progress was made on a goal using improved text matching.

    Returns:
        True if progress detected
    """
    try:
        import re

        # Better tokenization (4+ character words only)
        goal_text = goal.content.lower()
        goal_words = set(re.findall(r'\b\w{4,}\b', goal_text))

        if not goal_words:
            return False

        relevant_score = 0.0

        for memory in all_memories:
            # Only memories after last goal mention
            if goal.last_mentioned and memory.timestamp > goal.last_mentioned:
                memory_text = memory.content.lower()
                memory_words = set(re.findall(r'\b\w{4,}\b', memory_text))

                # Calculate overlap
                overlap = len(goal_words & memory_words)
                if overlap > 0:
                    overlap_ratio = overlap / len(goal_words)

                    # Weight by recency
                    days_old = (datetime.now(timezone.utc) - memory.timestamp).days
                    recency_weight = max(0.1, 1.0 - (days_old / 30))

                    relevant_score += overlap_ratio * recency_weight

        # Progress detected if score exceeds threshold
        return relevant_score > 1.5

    except Exception as e:
        logger.warning(f"Fehler bei Progress-Analyse: {e}")
        return False


def _detect_patterns(bundle, all_memories, limits: HeartbeatLimits) -> int:
    """
    Phase 7: Pattern Detection with optimized deduplication.

    Returns:
        Number of new patterns detected
    """
    try:
        from backend.memory.pattern_detector import get_pattern_tracker, PatternAnalyzer

        tracker = get_pattern_tracker(bundle.vectorstore)

        user_id = "default"  # TODO: multi-user support
        existing_patterns = tracker.get_all_patterns(user_id)

        # Check global limit
        if len(existing_patterns) >= _config.max_total_patterns:
            logger.warning(f"⚠️ Pattern limit reached ({len(existing_patterns)}/{_config.max_total_patterns})")
            return 0

        # Clear old patterns
        tracker.clear_old_patterns(user_id, older_than_days=90)

        # Detect new patterns
        topic_patterns = PatternAnalyzer.detect_topic_patterns(
            memories=all_memories,
            min_cluster_size=3,
            similarity_threshold=0.75
        )

        interest_patterns = PatternAnalyzer.detect_interest_patterns(
            memories=all_memories,
            min_frequency=3
        )

        all_new_patterns = topic_patterns + interest_patterns

        if not all_new_patterns:
            logger.info("✅ Keine neuen Patterns erkannt")
            return 0

        # Pre-compute keyword sets for existing patterns (performance optimization)
        existing_keyword_sets = {
            idx: set(pattern.keywords)
            for idx, pattern in enumerate(existing_patterns)
        }

        patterns_to_save = []
        patterns_to_update = []

        for pattern in all_new_patterns[:_config.max_new_patterns]:
            # Check limit
            if not limits.can_create(1):
                logger.warning("⚠️ Create limit reached for patterns")
                break

            # Check stop signal
            if _heartbeat_state.is_stop_learning():
                logger.warning("⚠️ Stop signal during pattern detection")
                break

            # Optimized duplicate detection with early termination
            pattern_keywords = set(pattern.keywords)
            best_match = None
            best_similarity = 0.0

            for idx, existing in enumerate(existing_patterns):
                existing_keywords = existing_keyword_sets[idx]

                # Early termination: no overlap
                if not (pattern_keywords & existing_keywords):
                    continue

                # Jaccard similarity
                intersection = len(pattern_keywords & existing_keywords)
                union = len(pattern_keywords | existing_keywords)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing

                    # Early termination: perfect match
                    if similarity >= _config.pattern_dedup_early_termination:
                        break

            # Merge or create new
            if best_match and best_similarity > 0.6:
                # Update existing pattern
                best_match.frequency += pattern.frequency
                best_match.last_seen = pattern.last_seen

                # Deduplicate related_memory_ids
                existing_ids = set(best_match.related_memory_ids)
                new_ids = set(pattern.related_memory_ids)
                combined_ids = list(existing_ids | new_ids)[:100]  # Max 100 IDs
                best_match.related_memory_ids = combined_ids

                patterns_to_update.append(best_match)
                logger.debug(f"Merging pattern '{pattern.name}' into '{best_match.name}' (similarity={best_similarity:.2f})")
            else:
                # New pattern
                patterns_to_save.append(pattern)

        # Batch save
        saved = 0

        for pattern in patterns_to_save:
            # Check limit again
            if len(existing_patterns) + saved >= _config.max_total_patterns:
                logger.warning(f"⚠️ Stopping pattern save at global limit ({_config.max_total_patterns})")
                break

            if tracker.save_pattern(pattern):
                saved += 1
                limits.track_create(1)
                logger.info(f"🔍 New pattern detected: {pattern.name} (freq={pattern.frequency})")

        # Update existing patterns
        for pattern in patterns_to_update:
            tracker.save_pattern(pattern)
            logger.debug(f"📊 Updated pattern: {pattern.name}")

        logger.info(f"✅ Pattern detection: {saved} new, {len(patterns_to_update)} updated")
        return saved

    except Exception as e:
        logger.error(f"Error in pattern detection: {e}", exc_info=True)
        return 0


async def _detect_knowledge_gaps(bundle, all_memories, limits: HeartbeatLimits) -> int:
    """
    Phase 8: Knowledge Gap Detection with LLM retry and deduplication.

    Returns:
        Number of knowledge gaps found
    """
    try:
        from backend.memory.knowledge_gap_detector import (
            get_knowledge_gap_tracker,
            KnowledgeGapAnalyzer
        )
        from backend.memory.pattern_detector import get_pattern_tracker
        from backend.memory.goal_tracker import get_goal_tracker

        gap_tracker = get_knowledge_gap_tracker(bundle.vectorstore)
        pattern_tracker = get_pattern_tracker(bundle.vectorstore)
        goal_tracker = get_goal_tracker(bundle.vectorstore)

        user_id = "default"  # TODO: multi-user support

        # Get patterns and goals
        patterns = pattern_tracker.get_all_patterns(user_id)
        goals = goal_tracker.get_all_goals(user_id, status=None)

        # Get existing gaps
        existing_gaps = gap_tracker.get_all_gaps(user_id, include_dismissed=True)
        active_gaps = [g for g in existing_gaps if not g.dismissed]

        # Check global limit
        if len(active_gaps) >= _config.max_total_knowledge_gaps:
            logger.warning(f"⚠️ Knowledge gap limit reached ({len(active_gaps)}/{_config.max_total_knowledge_gaps})")
            return 0

        # Clear old gaps (>14 days)
        old_gaps = [g for g in existing_gaps if (datetime.now(timezone.utc) - g.created_at).days > 14]
        for gap in old_gaps:
            gap_tracker.delete_gap(gap.id)

        # Detect various types of gaps
        topic_gaps = KnowledgeGapAnalyzer.detect_topic_knowledge_gaps(patterns, goals, all_memories)
        goal_gaps = KnowledgeGapAnalyzer.detect_goal_prerequisite_gaps(goals, all_memories)
        depth_gaps = KnowledgeGapAnalyzer.detect_interest_depth_gaps(patterns, all_memories)

        # LLM detection with retry
        llm_gaps = []
        try:
            llm_gaps = await call_llm_with_retry(
                lambda: KnowledgeGapAnalyzer.generate_contextual_gap_with_llm(
                    patterns, goals, all_memories, bundle.chat_client
                )
            )
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ LLM gap detection timed out ({_config.llm_timeout_seconds}s)")
        except Exception as e:
            logger.warning(f"LLM gap detection failed: {e}")

        all_gaps = topic_gaps + goal_gaps + depth_gaps + (llm_gaps or [])

        if not all_gaps:
            logger.info("✅ Keine neuen Wissenslücken erkannt")
            return 0

        # Sort by priority
        all_gaps.sort(key=lambda g: g.priority, reverse=True)
        saved = 0

        for gap in all_gaps[:_config.max_new_knowledge_gaps]:
            # Check limit
            if not limits.can_create(1):
                logger.warning("⚠️ Create limit reached for knowledge gaps")
                break

            # Check stop signal
            if _heartbeat_state.is_stop_learning():
                logger.warning("⚠️ Stop signal during knowledge gap detection")
                break

            # Improved duplicate detection with Jaccard similarity
            is_duplicate = False

            gap_words = set(gap.title.lower().split())

            for existing in active_gaps:
                existing_words = set(existing.title.lower().split())

                # Jaccard similarity
                intersection = len(gap_words & existing_words)
                union = len(gap_words | existing_words)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > 0.8:
                    is_duplicate = True
                    logger.debug(f"⏭️  Duplicate gap: '{gap.title}' similar to '{existing.title}' (sim={similarity:.2f})")
                    break

            if not is_duplicate:
                # Check global limit again
                if len(active_gaps) + saved >= _config.max_total_knowledge_gaps:
                    logger.warning(f"⚠️ Stopping gap save at global limit ({_config.max_total_knowledge_gaps})")
                    break

                if gap_tracker.save_gap(gap):
                    saved += 1
                    limits.track_create(1)
                    logger.info(f"🧠 Knowledge gap detected: {gap.title} (priority={gap.priority:.2f})")

        logger.info(f"✅ Knowledge gap detection: {saved} new gaps saved")
        return saved

    except Exception as e:
        logger.error(f"Error in knowledge gap detection: {e}", exc_info=True)
        return 0


# ========================================
# Event Loop and Control
# ========================================

async def run_heartbeat_loop_async():
    """
    Async heartbeat loop - should run in dedicated task.
    """
    logger.info("🚀 Starte Heartbeat Loop (async)")

    while True:
        try:
            await intelligent_memory_maintenance()
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled gracefully")
            break
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler im Heartbeat Loop: {e}", exc_info=True)

        try:
            await asyncio.sleep(_config.run_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled during sleep")
            break


def run_heartbeat_loop():
    """
    Synchronous wrapper for backward compatibility.

    This function blocks - should run in separate thread/process.
    """
    try:
        asyncio.run(run_heartbeat_loop_async())
    except KeyboardInterrupt:
        logger.info("Heartbeat loop stopped by user")


def start_heartbeat_service() -> bool:
    """
    Start the heartbeat loop in a dedicated daemon thread if not already running.

    Returns:
        True if a new thread was started, False if already running.
    """
    global _heartbeat_thread, _config
    with _heartbeat_thread_lock:
        if _heartbeat_thread and _heartbeat_thread.is_alive():
            return False

        _config = HeartbeatConfig.from_env()
        _heartbeat_thread = threading.Thread(
            target=run_heartbeat_loop,
            daemon=True,
            name="HeartbeatService",
        )
        _heartbeat_thread.start()
        return True


def get_heartbeat_config_snapshot() -> Dict[str, int]:
    """Return a minimal config snapshot for UI/logging."""
    return {
        "run_interval_seconds": _config.run_interval_seconds,
        "idle_threshold_minutes": _config.idle_threshold_minutes,
    }


# ========================================
# Smart Home Pattern Consolidation
# ========================================

async def _consolidate_smart_home_patterns() -> int:
    """
    Täglich: Konsolidiere Smart Home Patterns zu Memory (23:59 Uhr).

    Returns:
        Number of patterns consolidated
    """
    try:
        from backend.services.smart_home_pattern_aggregator import get_pattern_aggregator

        aggregator = get_pattern_aggregator()

        # Konsolidiere für alle bekannten User
        # TODO: Get actual user list from database
        users = ["thomas", "default"]  # Placeholder

        total_stored = 0
        for user_id in users:
            try:
                stored = await aggregator.consolidate_to_memory(user_id)
                total_stored += stored
                logger.info(f"✅ Smart Home Patterns konsolidiert für {user_id}: {stored} Patterns")
            except Exception as e:
                logger.error(f"❌ Fehler bei Pattern Consolidation für {user_id}: {e}")
                continue

        logger.info(f"✅ Daily Smart Home Pattern Consolidation: {total_stored} Patterns gespeichert")
        return total_stored

    except Exception as e:
        logger.error(f"❌ Fehler bei Smart Home Pattern Consolidation: {e}", exc_info=True)
        return 0


async def check_and_run_daily_consolidation():
    """
    Prüft ob es Zeit für die tägliche Consolidation ist (23:59 Uhr) und führt sie aus.

    Returns:
        Number of patterns consolidated (0 if not time yet)
    """
    now = datetime.now().astimezone()

    # Prüfe ob zwischen 23:55 und 23:59 Uhr (5-Minuten-Fenster wegen Heartbeat-Intervall)
    if 23 <= now.hour <= 23 and 55 <= now.minute <= 59:
        # Prüfe ob heute schon ausgeführt wurde (verhindere mehrfache Ausführung)
        last_consolidation = _heartbeat_state.last_smart_home_consolidation

        if last_consolidation:
            try:
                last_date = datetime.fromisoformat(last_consolidation).date()
                today = now.date()

                if last_date >= today:
                    logger.debug("⏭️  Smart Home Consolidation heute bereits ausgeführt")
                    return 0
            except Exception as e:
                logger.warning(f"Fehler beim Parsen von last_consolidation: {e}")

        # Führe Consolidation aus
        logger.info("🕐 23:59 Uhr - Starte tägliche Smart Home Pattern Consolidation")
        consolidated = await _consolidate_smart_home_patterns()

        # Update state
        _heartbeat_state.update_stats({
            "last_smart_home_consolidation": now.isoformat()
        })

        return consolidated

    return 0


# ========================================
# Public API
# ========================================

def stop_learning_processes():
    """
    Set stop signal for running learning processes.

    Called by middleware when user request comes in.
    """
    _heartbeat_state.set_stop_learning(True)
    logger.warning("⚠️ Stop-Signal gesetzt - Lernprozesse werden unterbrochen")

def allow_learning_processes():
    """
    Clear stop signal so manual runs are not aborted immediately.
    """
    _heartbeat_state.set_stop_learning(False)
    logger.info("✅ Stop-Signal zurückgesetzt - Lernprozesse erlaubt")


def is_learning_in_progress() -> bool:
    """
    Check if intensive learning processes are running.

    Returns:
        True if learning is active
    """
    return _heartbeat_state.is_learning_in_progress()


def get_heartbeat_status() -> Dict:
    """
    Get current heartbeat service status.

    Returns:
        Dict with statistics and status
    """
    return _heartbeat_state.get_status()


def reset_heartbeat_stats():
    """Reset heartbeat statistics (for testing)."""
    global _heartbeat_state
    _heartbeat_state = HeartbeatState()


# Helper method for limits
def new_entries_remaining(self: HeartbeatLimits) -> int:
    """Get remaining new entry slots."""
    return min(
        self.config.max_new_entries_per_run - self.new_entries,
        self.config.max_db_ops_per_run - self.total_ops
    )

# Add method to HeartbeatLimits class
HeartbeatLimits.new_entries_remaining = new_entries_remaining
