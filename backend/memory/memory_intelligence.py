"""
Intelligentes Memory-System für selbst-lernendes Verhalten.

Dieses Modul implementiert:
1. Adaptive Relevance Scoring (Memories werden relevanter durch Nutzung)
2. Memory Consolidation (Zusammenführung ähnlicher Memories)
3. Intelligent Cleanup (Vergessen basierend auf Nutzung, nicht nur Alter)
4. Usage Tracking (Tracking welche Memories hilfreich waren)
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from uuid import UUID
import numpy as np

from backend.models.memory_entry import MemoryEntry

logger = logging.getLogger("lexi_middleware.memory_intelligence")


class MemoryUsageTracker:
    """Trackt Nutzung von Memories für adaptive Relevance."""

    def __init__(self):
        # Format: {memory_id: {"retrievals": int, "used_in_response": int, "last_used": datetime}}
        self._usage_stats: Dict[str, Dict] = defaultdict(lambda: {
            "retrievals": 0,
            "used_in_response": 0,
            "last_used": None,
            "success_rate": 0.0
        })

    def track_retrieval(self, memory_id: str):
        """Memory wurde abgerufen."""
        self._usage_stats[memory_id]["retrievals"] += 1
        self._usage_stats[memory_id]["last_used"] = datetime.now(timezone.utc)

    def track_usage_in_response(self, memory_id: str, was_helpful: bool = True):
        """Memory wurde in Response verwendet."""
        if was_helpful:
            self._usage_stats[memory_id]["used_in_response"] += 1

        # Update success rate
        retrievals = self._usage_stats[memory_id]["retrievals"]
        if retrievals > 0:
            used = self._usage_stats[memory_id]["used_in_response"]
            self._usage_stats[memory_id]["success_rate"] = used / retrievals

    def get_usage_stats(self, memory_id: str) -> Dict:
        """Hole Nutzungsstatistiken für eine Memory."""
        return self._usage_stats.get(memory_id, {
            "retrievals": 0,
            "used_in_response": 0,
            "last_used": None,
            "success_rate": 0.0
        })

    def calculate_adaptive_relevance(self, memory_id: str, base_relevance: float,
                                    memory_age_days: float) -> float:
        """
        Berechnet adaptive Relevanz basierend auf Nutzung und Alter.

        Formel:
        - Base Relevance: Ursprüngliche Relevanz (z.B. Cosine Similarity)
        - Usage Boost: +0.1 pro erfolgreiche Nutzung (max +0.5)
        - Recency Boost: +0.2 wenn in letzten 7 Tagen genutzt
        - Age Decay: -0.01 pro 30 Tage ohne Nutzung
        - Success Rate Multiplier: 0.5 - 1.5 basierend auf Erfolgsrate
        """
        stats = self.get_usage_stats(memory_id)

        # Usage Boost: Häufig genutzte Memories werden wichtiger
        usage_boost = min(0.5, stats["used_in_response"] * 0.1)

        # Recency Boost: Kürzlich genutzte Memories bleiben relevant
        recency_boost = 0.0
        if stats["last_used"]:
            days_since_use = (datetime.now(timezone.utc) - stats["last_used"]).days
            if days_since_use < 7:
                recency_boost = 0.2
            elif days_since_use < 30:
                recency_boost = 0.1

        # Age Decay: Alte, ungenutzte Memories verlieren Relevanz
        age_decay = 0.0
        if not stats["last_used"]:  # Nie genutzt
            age_decay = -(memory_age_days / 30) * 0.01

        # Success Rate Multiplier: Erfolgreiche Memories werden bevorzugt
        success_multiplier = 1.0
        if stats["retrievals"] >= 3:  # Nur bei genug Daten
            success_multiplier = 0.5 + (stats["success_rate"] * 1.0)

        # Finale Berechnung
        adaptive_relevance = (base_relevance + usage_boost + recency_boost + age_decay) * success_multiplier

        # Clamp zwischen 0 und 1
        return max(0.0, min(1.0, adaptive_relevance))


class MemoryConsolidator:
    """Konsolidiert ähnliche Memories zu generalisiertem Wissen."""

    @staticmethod
    def find_similar_memories(memories: List[MemoryEntry],
                            similarity_threshold: float = 0.85) -> List[List[MemoryEntry]]:
        """
        Findet Gruppen von ähnlichen Memories.

        Returns: Liste von Listen, jede innere Liste enthält ähnliche Memories
        """
        if not memories or len(memories) < 2:
            return []

        # Check if embeddings are available
        memories_with_embeddings = [m for m in memories if m.embedding is not None]
        if len(memories_with_embeddings) < 2:
            logger.warning("Not enough memories with embeddings for consolidation")
            return []

        similar_groups = []
        processed = set()

        for i, mem1 in enumerate(memories_with_embeddings):
            if mem1.id in processed:
                continue

            group = [mem1]
            processed.add(mem1.id)

            for j, mem2 in enumerate(memories_with_embeddings[i+1:], start=i+1):
                if mem2.id in processed:
                    continue

                # Calculate cosine similarity
                similarity = MemoryConsolidator._cosine_similarity(
                    mem1.embedding, mem2.embedding
                )

                if similarity >= similarity_threshold:
                    group.append(mem2)
                    processed.add(mem2.id)

            if len(group) >= 2:  # Nur Gruppen mit mindestens 2 Memories
                similar_groups.append(group)

        return similar_groups

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Berechnet Cosine Similarity zwischen zwei Vektoren."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    @staticmethod
    def consolidate_group(group: List[MemoryEntry], embeddings_model) -> Optional[MemoryEntry]:
        """
        Konsolidiert eine Gruppe ähnlicher Memories zu einer generalierten Memory.

        Args:
            group: Liste von ähnlichen Memories
            embeddings_model: Embedding-Modell zum Erstellen des neuen Embeddings

        Returns:
            Konsolidierte MemoryEntry oder None bei Fehler
        """
        if len(group) < 2:
            return None

        try:
            # Erstelle konsolidierten Content
            contents = [m.content for m in group]
            consolidated_content = f"Zusammenfassung von {len(group)} ähnlichen Erinnerungen:\n"
            consolidated_content += "\n".join(f"- {c}" for c in contents[:3])  # Max 3 Beispiele

            if len(contents) > 3:
                consolidated_content += f"\n... und {len(contents) - 3} weitere ähnliche Einträge"

            # Kombiniere Tags
            all_tags = []
            for m in group:
                if m.tags:
                    all_tags.extend(m.tags)
            unique_tags = list(set(all_tags))

            # Höchste Relevanz der Gruppe
            max_relevance = max(m.relevance or 0.0 for m in group)

            # Älteste Timestamp (früheste Memory)
            oldest_timestamp = min(m.timestamp for m in group)

            # Erstelle neue Embedding
            new_embedding = embeddings_model.embed_query(consolidated_content)

            # Generate new UUID for consolidated memory (Qdrant requires proper UUID)
            from uuid import uuid4
            consolidated_id = uuid4()

            # Erstelle konsolidierte Memory
            consolidated = MemoryEntry(
                id=consolidated_id,
                content=consolidated_content,
                timestamp=oldest_timestamp,
                category=group[0].category,
                tags=unique_tags + ["consolidated"],
                source="memory_consolidation",
                relevance=min(1.0, max_relevance * 1.2),  # Boost für konsolidierte Memories
                embedding=new_embedding,
                source_memory_ids=[str(m.id) for m in group]  # Track original IDs
            )

            return consolidated

        except Exception as e:
            logger.error(f"Error consolidating memory group: {e}")
            return None


class IntelligentMemoryCleanup:
    """Intelligenter Cleanup basierend auf Nutzung, nicht nur Alter."""

    def __init__(self, usage_tracker: MemoryUsageTracker):
        self.usage_tracker = usage_tracker
        self._simple_toggle_patterns = [
            r'^(schalte|schalt|mach|stelle)\s+(das\s+)?(licht|lampe|heizung|thermostat|schalter|steckdose|switch)(\s+(im|in|am|auf)\s+\w+(?:\s+\w+)*)?\s+(ein|aus|an|ab)$',
            r'^(licht|lampe|heizung|thermostat|schalter|steckdose|switch)(\s+(im|in|am|auf)\s+\w+(?:\s+\w+)*)?\s+(ein|aus|an|ab)$',
            r'^(ein|aus|an|ab)schalten\s+\w+$',
            r'^(turn\s+on|turn\s+off)\s+(the\s+)?(light|lamp|heater|thermostat|switch)$',
            r'^(light|lamp|heater|thermostat|switch)\s+(on|off)$'
        ]

    def _extract_user_message(self, content: str) -> str:
        if not content:
            return ""
        if content.startswith("Q:"):
            first_line = content.splitlines()[0]
            return first_line[2:].strip()
        return content.strip()

    def _extract_response_message(self, content: str) -> str:
        if not content:
            return ""
        if "A:" in content:
            for line in content.splitlines():
                if line.startswith("A:"):
                    return line[2:].strip()
        return ""

    def _normalize_message(self, message: str) -> str:
        normalized = re.sub(r"[^\w\s]", "", message.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _is_trivial_toggle(self, message: str, tags: List[str]) -> bool:
        if not message:
            return False
        normalized = self._normalize_message(message)
        if len(normalized.split()) > 8:
            return False
        if self._is_question_like(normalized):
            return False
        if any(re.match(pattern, normalized) for pattern in self._simple_toggle_patterns):
            return True
        if "smart_home" in (tags or []):
            if re.match(r'^\w+\s+(ein|aus|an|ab|einschalten|ausschalten)$', normalized):
                return True
        return False

    def _is_question_like(self, normalized: str) -> bool:
        if "?" in normalized:
            return True
        question_words = ["wie", "was", "wann", "warum", "welche", "wieviel"]
        if any(normalized.startswith(word + " ") for word in question_words):
            return True
        if any(token in normalized for token in ["grad", "°", "prozent", "%"]):
            return True
        return False

    def _is_trivial_action_confirmation(self, message: str, response: str) -> bool:
        if not message or not response:
            return False
        normalized = self._normalize_message(message)
        if len(normalized.split()) > 8:
            return False
        if self._is_question_like(normalized):
            return False
        response_norm = response.lower()
        confirmation_markers = ["ist jetzt", "eingeschaltet", "ausgeschaltet", "eingeschalten", "ausgeschalten", "✓"]
        if not any(marker in response_norm for marker in confirmation_markers):
            return False
        return normalized.endswith(("ein", "aus", "an", "ab", "einschalten", "ausschalten"))

    def _is_trivial_action_refusal(self, message: str, response: str) -> bool:
        if not message or not response:
            return False
        normalized = self._normalize_message(message)
        if len(normalized.split()) > 8:
            return False
        if self._is_question_like(normalized):
            return False
        response_norm = response.lower()
        refusal_markers = [
            "kann",
            "kannst",
            "kann ich",
            "leider nicht",
            "nicht einschalten",
            "nicht ausschalten"
        ]
        if not any(marker in response_norm for marker in refusal_markers):
            return False
        return normalized.endswith(("ein", "aus", "an", "ab", "einschalten", "ausschalten"))

    def identify_memories_for_deletion(self,
                                      memories: List[MemoryEntry],
                                      max_age_days: int = 90,
                                      min_relevance: float = 0.1,
                                      unused_after_days: Optional[int] = None,
                                      max_unused_relevance: Optional[float] = None,
                                      superseded_min_age_days: Optional[int] = None,
                                      superseded_unused_days: Optional[int] = None,
                                      superseded_allowed_tags: Optional[List[str]] = None,
                                      superseded_excluded_tags: Optional[List[str]] = None,
                                      consolidated_min_age_days: Optional[int] = None,
                                      consolidated_unused_days: Optional[int] = None) -> List[str]:
        """
        Identifiziert Memories die gelöscht werden sollten.

        Kriterien:
        - Sehr alte Memories (>90 Tage) MIT niedrige Relevanz (<0.1)
        - Nie genutzte Memories älter als 60 Tage
        - Memories mit schlechter Success Rate (<0.2) und alt (>30 Tage)

        NICHT gelöscht werden:
        - Häufig genutzte Memories (unabhängig vom Alter)
        - Hohe Relevanz (>0.5)
        - Kürzlich genutzt (letzte 7 Tage)
        """
        to_delete = []
        now = datetime.now(timezone.utc)
        superseded_min_age_days = 14 if superseded_min_age_days is None else superseded_min_age_days
        superseded_unused_days = 7 if superseded_unused_days is None else superseded_unused_days
        allowed_superseded_tags = set(superseded_allowed_tags or ["chat", "conversation"])
        excluded_superseded_tags = set(superseded_excluded_tags or ["pattern", "aggregated"])
        consolidated_min_age_days = 7 if consolidated_min_age_days is None else consolidated_min_age_days
        consolidated_unused_days = 7 if consolidated_unused_days is None else consolidated_unused_days

        for memory in memories:
            memory_id = str(memory.id)
            age_days = (now - memory.timestamp).days
            stats = self.usage_tracker.get_usage_stats(memory_id)
            tags = memory.tags or []

            # Berechne adaptive Relevanz
            current_relevance = self.usage_tracker.calculate_adaptive_relevance(
                memory_id, memory.relevance or 0.5, age_days
            )

            # Skip aggregierte/strukturierte Muster
            if "pattern" in tags or "aggregated" in tags:
                continue

            # Triviale Smart-Home Toggles direkt löschen (z.B. "Licht ein/aus")
            message = self._extract_user_message(memory.content)
            response = self._extract_response_message(memory.content)
            if self._is_trivial_toggle(message, tags) or self._is_trivial_action_confirmation(message, response) or self._is_trivial_action_refusal(message, response):
                if age_days >= 1:
                    to_delete.append(memory_id)
                    logger.info(f"Marked trivial toggle for deletion: {memory_id} ('{message}')")
                    continue

            # Superseded: Quelle wurde durch Meta-Wissen ersetzt
            if getattr(memory, "superseded", False):
                tag_set = set(tags)
                if allowed_superseded_tags and not tag_set.intersection(allowed_superseded_tags):
                    pass
                elif tag_set.intersection(excluded_superseded_tags):
                    pass
                elif age_days >= superseded_min_age_days:
                    last_used = stats["last_used"]
                    if not last_used or (now - last_used).days >= superseded_unused_days:
                        to_delete.append(memory_id)
                        logger.info(
                            f"Marked superseded for deletion: {memory_id} (age={age_days}d)"
                        )
                        continue

            # Konsolidierte Memories sind nur Zwischenstufe -> aggressiver löschen
            if "consolidated" in tags or (memory.source == "memory_consolidation"):
                last_used = stats["last_used"]
                if age_days >= consolidated_min_age_days:
                    if not last_used or (now - last_used).days >= consolidated_unused_days:
                        to_delete.append(memory_id)
                        logger.info(
                            f"Marked consolidated for deletion: {memory_id} (age={age_days}d)"
                        )
                        continue

            # Schutzbedingungen: NICHT löschen wenn...
            if current_relevance > 0.5:
                continue  # Hohe Relevanz

            if stats["used_in_response"] >= 3:
                continue  # Häufig genutzt

            if stats["last_used"] and (now - stats["last_used"]).days < 7:
                continue  # Kürzlich genutzt

            # UI-spezifischer Cleanup: nie genutzt + alt
            if unused_after_days is not None and stats["retrievals"] == 0 and age_days > unused_after_days:
                if max_unused_relevance is None or current_relevance <= max_unused_relevance:
                    to_delete.append(memory_id)
                    logger.info(
                        f"Marked for deletion: {memory_id} (unused>{unused_after_days}d, relevance={current_relevance:.2f})"
                    )
                    continue

            # Löschbedingungen
            if age_days > max_age_days and current_relevance < min_relevance:
                to_delete.append(memory_id)
                logger.info(f"Marked for deletion: {memory_id} (age={age_days}d, relevance={current_relevance:.2f})")

            elif stats["retrievals"] == 0 and age_days > 60:
                to_delete.append(memory_id)
                logger.info(f"Marked for deletion: {memory_id} (never used, age={age_days}d)")

            elif stats["success_rate"] < 0.2 and stats["retrievals"] >= 5 and age_days > 30:
                to_delete.append(memory_id)
                logger.info(f"Marked for deletion: {memory_id} (low success rate={stats['success_rate']:.2f})")

        return to_delete

    def should_consolidate_instead_of_delete(self, memory: MemoryEntry) -> bool:
        """Prüft ob Memory konsolidiert statt gelöscht werden sollte."""
        stats = self.usage_tracker.get_usage_stats(str(memory.id))

        # Konsolidiere wenn:
        # - Wurde mindestens einmal genutzt
        # - Hat moderate Relevanz (0.2 - 0.5)
        # - Nicht zu alt (< 180 Tage)
        age_days = (datetime.now(timezone.utc) - memory.timestamp).days

        return (stats["retrievals"] > 0 and
                0.2 <= (memory.relevance or 0.0) <= 0.5 and
                age_days < 180)


# Globale Instanz für Tracking (Singleton)
_global_tracker = MemoryUsageTracker()


def get_usage_tracker() -> MemoryUsageTracker:
    """Hole die globale Usage Tracker Instanz."""
    return _global_tracker


def track_memory_retrieval(memory_ids: List[str]):
    """Helper-Funktion zum Tracken von Memory-Abrufen."""
    tracker = get_usage_tracker()
    for mem_id in memory_ids:
        tracker.track_retrieval(mem_id)


def track_memory_usage(memory_id: str, was_helpful: bool = True):
    """Helper-Funktion zum Tracken von Memory-Nutzung in Responses."""
    tracker = get_usage_tracker()
    tracker.track_usage_in_response(memory_id, was_helpful)


def update_memory_relevance(memory: MemoryEntry) -> float:
    """
    Aktualisiert die Relevanz einer Memory basierend auf Nutzung.

    Returns:
        Neue adaptive Relevanz
    """
    tracker = get_usage_tracker()
    age_days = (datetime.now(timezone.utc) - memory.timestamp).days
    return tracker.calculate_adaptive_relevance(
        str(memory.id),
        memory.relevance or 0.5,
        age_days
    )
