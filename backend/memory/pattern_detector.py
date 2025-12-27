"""
Pattern Detection System für LexiAI

Erkennt wiederkehrende Themen und Trends in Gesprächen.

Features:
1. Topic Clustering - Gruppiert Memories nach Themen
2. Frequency Analysis - Findet häufig erwähnte Themen
3. Trend Detection - Erkennt zeitliche Trends
4. Pattern Storage - Speichert erkannte Patterns persistent
"""

import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN

from backend.models.memory_entry import MemoryEntry
from backend.qdrant.client_wrapper import safe_upsert
from backend.qdrant.collection_helpers import ensure_collection

logger = logging.getLogger("lexi_middleware.pattern_detector")


@dataclass
class Pattern:
    """Repräsentiert ein erkanntes Muster"""
    id: str
    user_id: str
    pattern_type: str  # "topic", "behavior", "interest", "routine"
    name: str  # z.B. "Interesse an Pizza", "Python Lernen"
    description: str  # Detaillierte Beschreibung
    confidence: float  # 0.0 - 1.0
    frequency: int  # Wie oft wurde Pattern beobachtet
    first_seen: datetime
    last_seen: datetime
    related_memory_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Speicherung"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "pattern_type": self.pattern_type,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "related_memory_ids": self.related_memory_ids,
            "keywords": self.keywords,
            "trend": self.trend,
            "metadata": self.metadata,
            "timestamp_ms": int(self.last_seen.timestamp() * 1000)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Erstellt Pattern aus Dictionary"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            pattern_type=data["pattern_type"],
            name=data["name"],
            description=data["description"],
            confidence=data["confidence"],
            frequency=data["frequency"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            related_memory_ids=data.get("related_memory_ids", []),
            keywords=data.get("keywords", []),
            trend=data.get("trend", "stable"),
            metadata=data.get("metadata", {})
        )


class PatternAnalyzer:
    """Analysiert Memories und erkennt Patterns"""

    @staticmethod
    def detect_topic_patterns(
        memories: List[MemoryEntry],
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ) -> List[Pattern]:
        """
        Erkennt Topic-Patterns durch Clustering ähnlicher Memories.

        Args:
            memories: Liste von Memories mit Embeddings
            min_cluster_size: Minimale Cluster-Größe
            similarity_threshold: Cosine similarity threshold

        Returns:
            Liste erkannter Topic-Patterns
        """
        patterns = []

        # Nur Memories mit Embeddings
        memories_with_embeddings = [m for m in memories if m.embedding is not None]

        if len(memories_with_embeddings) < min_cluster_size:
            logger.debug(f"Not enough memories with embeddings for clustering: {len(memories_with_embeddings)}")
            return patterns

        try:
            # Extrahiere Embeddings als NumPy array
            embeddings = np.array([m.embedding for m in memories_with_embeddings])

            # DBSCAN Clustering (epsilon basierend auf similarity threshold)
            # epsilon = 1 - similarity (für cosine distance)
            epsilon = 1.0 - similarity_threshold
            clustering = DBSCAN(
                eps=epsilon,
                min_samples=min_cluster_size,
                metric='cosine'
            ).fit(embeddings)

            labels = clustering.labels_

            # Gruppiere Memories nach Clustern
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:  # -1 = noise
                    clusters[label].append(memories_with_embeddings[idx])

            logger.info(f"Found {len(clusters)} topic clusters")

            # Erstelle Pattern für jeden Cluster
            for cluster_id, cluster_memories in clusters.items():
                if len(cluster_memories) < min_cluster_size:
                    continue

                # Extrahiere Keywords aus Cluster
                keywords = PatternAnalyzer._extract_keywords_from_cluster(cluster_memories)

                # Erstelle Pattern-Name basierend auf häufigsten Keywords
                pattern_name = f"Thema: {', '.join(keywords[:3])}"

                # Berechne Confidence basierend auf Cluster-Größe und Similarity
                confidence = min(1.0, len(cluster_memories) / 10.0)  # Max bei 10 Memories

                # Zeitliche Informationen
                timestamps = [m.timestamp for m in cluster_memories]
                first_seen = min(timestamps)
                last_seen = max(timestamps)

                # Trend-Analyse (basierend auf zeitlicher Verteilung)
                trend = PatternAnalyzer._analyze_trend(timestamps)

                # Erstelle Pattern
                pattern = Pattern(
                    id=str(uuid.uuid4()),
                    user_id=cluster_memories[0].metadata.get("user_id", "default"),
                    pattern_type="topic",
                    name=pattern_name,
                    description=f"Wiederkehrendes Thema mit {len(cluster_memories)} Erwähnungen",
                    confidence=confidence,
                    frequency=len(cluster_memories),
                    first_seen=first_seen,
                    last_seen=last_seen,
                    related_memory_ids=[str(m.id) for m in cluster_memories],
                    keywords=keywords,
                    trend=trend,
                    metadata={
                        "cluster_id": int(cluster_id),
                        "avg_relevance": np.mean([m.relevance or 0.5 for m in cluster_memories])
                    }
                )

                patterns.append(pattern)
                logger.info(f"Detected pattern: {pattern_name} (freq={len(cluster_memories)}, trend={trend})")

        except Exception as e:
            logger.error(f"Error in topic pattern detection: {e}")

        return patterns

    @staticmethod
    def _extract_keywords_from_cluster(memories: List[MemoryEntry], top_n: int = 10) -> List[str]:
        """
        Extrahiert häufigste Keywords aus einem Cluster.

        Einfache Implementierung basierend auf Worthäufigkeit.
        """
        # Stopwords (erweiterte Liste)
        stopwords = {
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
            "und", "oder", "aber", "ist", "sind", "war", "waren", "hat", "haben",
            "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein",
            "von", "zu", "in", "an", "auf", "mit", "für", "über", "um", "bei",
            "wie", "was", "wer", "wo", "wann", "warum", "dass", "wenn", "als",
            "user", "assistant", "the", "a", "an", "and", "or", "is", "are", "was", "were"
        }

        # Sammle alle Wörter
        all_words = []
        for memory in memories:
            words = memory.content.lower().split()
            # Filter: Länge > 3, nicht in Stopwords
            words = [w.strip(".,!?:;") for w in words if len(w) > 3 and w.lower() not in stopwords]
            all_words.extend(words)

        # Zähle Häufigkeiten
        word_counts = Counter(all_words)

        # Top N Keywords
        keywords = [word for word, _ in word_counts.most_common(top_n)]

        return keywords

    @staticmethod
    def _analyze_trend(timestamps: List[datetime]) -> str:
        """
        Analysiert zeitlichen Trend basierend auf Timestamps.

        Returns:
            "increasing", "decreasing", oder "stable"
        """
        if len(timestamps) < 3:
            return "stable"

        try:
            # Sortiere Timestamps
            sorted_timestamps = sorted(timestamps)

            # Teile in zwei Hälften
            mid = len(sorted_timestamps) // 2
            first_half = sorted_timestamps[:mid]
            second_half = sorted_timestamps[mid:]

            # Vergleiche Dichte (Mentions pro Tag)
            first_span = (first_half[-1] - first_half[0]).days + 1
            second_span = (second_half[-1] - second_half[0]).days + 1

            first_density = len(first_half) / max(1, first_span)
            second_density = len(second_half) / max(1, second_span)

            # Trend bestimmen
            if second_density > first_density * 1.3:  # 30% increase
                return "increasing"
            elif second_density < first_density * 0.7:  # 30% decrease
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.warning(f"Error analyzing trend: {e}")
            return "stable"

    @staticmethod
    def detect_interest_patterns(
        memories: List[MemoryEntry],
        min_frequency: int = 3
    ) -> List[Pattern]:
        """
        Erkennt Interessen-Patterns basierend auf häufigen Keywords.

        Einfachere Methode als Clustering, basiert auf Keyword-Frequency.
        """
        patterns = []

        # Extrahiere alle Keywords aus Memories
        keyword_to_memories = defaultdict(list)

        stopwords = {
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
            "und", "oder", "aber", "ist", "sind", "war", "waren", "hat", "haben",
            "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein",
            "user", "assistant"
        }

        for memory in memories:
            words = memory.content.lower().split()
            words = [w.strip(".,!?:;") for w in words if len(w) > 4 and w.lower() not in stopwords]

            for word in words:
                keyword_to_memories[word].append(memory)

        # Finde häufige Keywords
        for keyword, related_memories in keyword_to_memories.items():
            if len(related_memories) < min_frequency:
                continue

            # Zeitliche Informationen
            timestamps = [m.timestamp for m in related_memories]
            first_seen = min(timestamps)
            last_seen = max(timestamps)

            # Trend
            trend = PatternAnalyzer._analyze_trend(timestamps)

            # Confidence basierend auf Häufigkeit
            confidence = min(1.0, len(related_memories) / 10.0)

            pattern = Pattern(
                id=str(uuid.uuid4()),
                user_id=related_memories[0].metadata.get("user_id", "default"),
                pattern_type="interest",
                name=f"Interesse: {keyword.capitalize()}",
                description=f"Häufige Erwähnungen von '{keyword}' ({len(related_memories)}x)",
                confidence=confidence,
                frequency=len(related_memories),
                first_seen=first_seen,
                last_seen=last_seen,
                related_memory_ids=[str(m.id) for m in related_memories],
                keywords=[keyword],
                trend=trend
            )

            patterns.append(pattern)

        # Sortiere nach Frequency und limitiere
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns[:20]  # Top 20 Interessen


class PatternTracker:
    """Verwaltet Pattern-Speicherung in Qdrant"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.patterns_collection = "lexi_patterns"
        self._ensure_collection()

    def _ensure_collection(self):
        """Stellt sicher dass Patterns Collection existiert"""
        try:
            client = self.vectorstore.client
            created = ensure_collection(client, self.patterns_collection)
            if created:
                logger.info(f"✅ Created patterns collection: {self.patterns_collection}")
        except Exception as e:
            logger.error(f"Error ensuring patterns collection: {e}")

    def save_pattern(self, pattern: Pattern) -> bool:
        """Speichert ein Pattern"""
        try:
            from qdrant_client.models import PointStruct

            client = self.vectorstore.client

            point = PointStruct(
                id=pattern.id,
                vector=[0.0],
                payload=pattern.to_dict()
            )

            safe_upsert(
                collection_name=self.patterns_collection,
                points=[point]
            )

            logger.info(f"💾 Pattern saved: {pattern.name}")
            return True

        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
            return False

    def get_all_patterns(self, user_id: str, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Holt alle Patterns eines Users"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

            client = self.vectorstore.client

            base_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if pattern_type:
                base_conditions.append(
                    FieldCondition(key="pattern_type", match=MatchValue(value=pattern_type))
                )

            try:
                cutoff_days = os.environ.get("LEXI_PATTERN_DAYS")
                if cutoff_days is None:
                    cutoff_days = os.environ.get("LEXI_STATS_DAYS", "180")
                cutoff_days = int(cutoff_days)
            except (TypeError, ValueError):
                cutoff_days = 180

            cutoff_ms = None
            if cutoff_days > 0:
                cutoff_dt = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
                cutoff_ms = int(cutoff_dt.timestamp() * 1000)

            if cutoff_ms:
                filter_conditions = base_conditions + [
                    FieldCondition(key="timestamp_ms", range=Range(gte=cutoff_ms))
                ]
            else:
                filter_conditions = base_conditions

            scroll_result = client.scroll(
                collection_name=self.patterns_collection,
                scroll_filter=Filter(must=filter_conditions),
                with_payload=True,
                with_vectors=False,
                limit=1000
            )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            if not points and cutoff_ms:
                scroll_result = client.scroll(
                    collection_name=self.patterns_collection,
                    scroll_filter=Filter(must=base_conditions),
                    with_payload=True,
                    with_vectors=False,
                    limit=1000
                )
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

            patterns = []
            for point in points:
                try:
                    pattern = Pattern.from_dict(point.payload)
                    patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Could not parse pattern {point.id}: {e}")

            logger.info(f"📋 Retrieved {len(patterns)} patterns for user {user_id}")
            return patterns

        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []

    def delete_pattern(self, pattern_id: str) -> bool:
        """Löscht ein Pattern"""
        try:
            client = self.vectorstore.client

            client.delete(
                collection_name=self.patterns_collection,
                points_selector=[pattern_id]
            )

            logger.info(f"🗑️ Pattern deleted: {pattern_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting pattern: {e}")
            return False

    def clear_old_patterns(self, user_id: str, older_than_days: int = 90) -> int:
        """Löscht alte Patterns"""
        try:
            all_patterns = self.get_all_patterns(user_id)

            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            old_patterns = [p for p in all_patterns if p.last_seen < cutoff]

            deleted = 0
            for pattern in old_patterns:
                if self.delete_pattern(pattern.id):
                    deleted += 1

            logger.info(f"🧹 Deleted {deleted} old patterns (older than {older_than_days} days)")
            return deleted

        except Exception as e:
            logger.error(f"Error clearing old patterns: {e}")
            return 0


# Globale Instanz
_global_pattern_tracker: Optional[PatternTracker] = None


def get_pattern_tracker(vectorstore=None) -> PatternTracker:
    """Hole oder erstelle globale PatternTracker Instanz"""
    global _global_pattern_tracker

    if _global_pattern_tracker is None:
        if vectorstore is None:
            from backend.core.component_cache import get_cached_components
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore

        _global_pattern_tracker = PatternTracker(vectorstore)

    return _global_pattern_tracker
