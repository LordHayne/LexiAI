"""
Knowledge Gap Detection System für LexiAI

Erkennt fehlende Informationen und Wissenslücken basierend auf:
1. Erkannten Patterns (Interessen)
2. Gesetzten Goals (Zielen)
3. Bisherigen Gesprächen (Memories)

Features:
1. Gap Detection - Findet Wissenslücken
2. Suggestion Generation - Erstellt hilfreiche Vorschläge
3. Priority Scoring - Bewertet Wichtigkeit von Gaps
"""

import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from backend.qdrant.client_wrapper import safe_upsert
from backend.qdrant.collection_helpers import ensure_collection

logger = logging.getLogger("lexi_middleware.knowledge_gap_detector")


@dataclass
class KnowledgeGap:
    """Repräsentiert eine erkannte Wissenslücke"""
    id: str
    user_id: str
    gap_type: str  # "topic_knowledge", "goal_prerequisite", "interest_depth", "context_missing"
    title: str  # Kurze Beschreibung
    description: str  # Detaillierte Erklärung
    suggestion: str  # Konkreter Vorschlag
    priority: float  # 0.0 - 1.0 (Wichtigkeit)
    confidence: float  # 0.0 - 1.0 (Sicherheit der Erkennung)
    related_pattern_ids: List[str] = field(default_factory=list)
    related_goal_ids: List[str] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dismissed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "gap_type": self.gap_type,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "priority": self.priority,
            "confidence": self.confidence,
            "related_pattern_ids": self.related_pattern_ids,
            "related_goal_ids": self.related_goal_ids,
            "related_memory_ids": self.related_memory_ids,
            "created_at": self.created_at.isoformat(),
            "dismissed": self.dismissed,
            "metadata": self.metadata,
            "timestamp_ms": int(self.created_at.timestamp() * 1000)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGap":
        """Erstellt KnowledgeGap aus Dictionary"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            gap_type=data["gap_type"],
            title=data["title"],
            description=data["description"],
            suggestion=data["suggestion"],
            priority=data["priority"],
            confidence=data["confidence"],
            related_pattern_ids=data.get("related_pattern_ids", []),
            related_goal_ids=data.get("related_goal_ids", []),
            related_memory_ids=data.get("related_memory_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            dismissed=data.get("dismissed", False),
            metadata=data.get("metadata", {})
        )


class KnowledgeGapAnalyzer:
    """Analysiert Patterns, Goals und Memories um Wissenslücken zu finden"""

    # Wissensdomänen mit typischen Wissenslücken
    DOMAIN_PREREQUISITES = {
        "programmieren": [
            "Grundlagen der Programmierung",
            "Debugging-Techniken",
            "Best Practices",
            "Entwicklungsumgebung Setup"
        ],
        "python": [
            "Python Installation",
            "Virtuelle Umgebungen",
            "Package Management (pip)",
            "Wichtige Libraries"
        ],
        "fitness": [
            "Ernährungsgrundlagen",
            "Trainingsplan",
            "Regeneration",
            "Verletzungsprävention"
        ],
        "abnehmen": [
            "Kaloriendefizit",
            "Makronährstoffe",
            "Gesunde Ernährung",
            "Bewegung & Sport"
        ],
        "kochen": [
            "Grundtechniken",
            "Küchenausstattung",
            "Rezepte für Anfänger",
            "Lebensmittelkunde"
        ],
        "sprache": [
            "Grammatik-Grundlagen",
            "Vokabular aufbauen",
            "Sprechpraxis",
            "Lernmethoden"
        ]
    }

    @staticmethod
    def detect_topic_knowledge_gaps(patterns: List, goals: List, memories: List) -> List[KnowledgeGap]:
        """
        Erkennt Wissenslücken basierend auf erkannten Interessen (Patterns).

        Logic:
        - User interessiert sich für Topic X
        - Hat aber noch keine Memories zu Grundlagen
        - → Wissenslücke: Grundlagen fehlen
        """
        gaps = []

        for pattern in patterns:
            if pattern.pattern_type != "interest":
                continue

            # Extrahiere Topic aus Pattern
            topic = pattern.keywords[0] if pattern.keywords else pattern.name.lower()

            # Finde passende Domäne
            domain_found = None
            for domain, prerequisites in KnowledgeGapAnalyzer.DOMAIN_PREREQUISITES.items():
                if domain in topic.lower():
                    domain_found = domain
                    break

            if not domain_found:
                continue

            # Prüfe ob Prerequisites in Memories vorhanden sind
            prerequisites = KnowledgeGapAnalyzer.DOMAIN_PREREQUISITES[domain_found]

            for prereq in prerequisites:
                # Check ob prereq in Memories erwähnt wurde
                mentioned = any(
                    prereq.lower() in memory.content.lower()
                    for memory in memories
                )

                if not mentioned:
                    # Wissenslücke gefunden!
                    gap = KnowledgeGap(
                        id=str(uuid.uuid4()),
                        user_id="default",
                        gap_type="topic_knowledge",
                        title=f"Grundwissen: {prereq}",
                        description=f"Du interessierst dich für {topic}, aber '{prereq}' wurde noch nicht besprochen.",
                        suggestion=f"Möchtest du mehr über '{prereq}' erfahren? Das ist wichtig für {domain_found}.",
                        priority=0.7,
                        confidence=0.8,
                        related_pattern_ids=[pattern.id],
                        metadata={"domain": domain_found, "prerequisite": prereq}
                    )
                    gaps.append(gap)

        return gaps

    @staticmethod
    def detect_goal_prerequisite_gaps(goals: List, memories: List) -> List[KnowledgeGap]:
        """
        Erkennt fehlende Voraussetzungen für gesetzte Ziele.

        Logic:
        - User hat Ziel X
        - Ziel benötigt Wissen Y
        - User hat noch keine Memories zu Y
        - → Wissenslücke
        """
        gaps = []

        for goal in goals:
            if goal.status != "active":
                continue

            goal_content = goal.content.lower()

            # Map Goal zu Domäne
            domain_found = None
            for domain in KnowledgeGapAnalyzer.DOMAIN_PREREQUISITES.keys():
                if domain in goal_content:
                    domain_found = domain
                    break

            if not domain_found:
                # Versuch Keywords zu matchen
                if any(word in goal_content for word in ["abnehmen", "gewicht", "kg"]):
                    domain_found = "abnehmen"
                elif any(word in goal_content for word in ["lernen", "studieren", "kurs"]):
                    domain_found = "sprache"
                elif any(word in goal_content for word in ["programmier", "code", "entwicklung"]):
                    domain_found = "programmieren"

            if domain_found:
                prerequisites = KnowledgeGapAnalyzer.DOMAIN_PREREQUISITES[domain_found]

                for prereq in prerequisites[:2]:  # Max 2 pro Goal
                    mentioned = any(
                        prereq.lower() in memory.content.lower()
                        for memory in memories
                    )

                    if not mentioned:
                        gap = KnowledgeGap(
                            id=str(uuid.uuid4()),
                            user_id="default",
                            gap_type="goal_prerequisite",
                            title=f"Für dein Ziel wichtig: {prereq}",
                            description=f"Um '{goal.content}' zu erreichen, solltest du '{prereq}' kennen.",
                            suggestion=f"Lass uns über '{prereq}' sprechen - das hilft dir bei deinem Ziel!",
                            priority=0.9,  # Höher weil mit konkretem Ziel verknüpft
                            confidence=0.85,
                            related_goal_ids=[goal.id],
                            metadata={"domain": domain_found, "goal_content": goal.content}
                        )
                        gaps.append(gap)

        return gaps

    @staticmethod
    def detect_interest_depth_gaps(patterns: List, memories: List) -> List[KnowledgeGap]:
        """
        Erkennt oberflächliches Wissen bei häufigen Interessen.

        Logic:
        - User erwähnt Topic oft (high frequency)
        - Aber nur oberflächlich (wenig Tiefe in Memories)
        - → Könnte tiefer gehen
        """
        gaps = []

        high_freq_patterns = [p for p in patterns if p.frequency >= 5]

        for pattern in high_freq_patterns:
            # Analysiere Tiefe der Memories
            related_memories = [
                m for m in memories
                if str(m.id) in pattern.related_memory_ids
            ]

            if len(related_memories) < 3:
                continue

            # Einfache Heuristik: Durchschnittliche Länge der Memories
            avg_length = sum(len(m.content) for m in related_memories) / len(related_memories)

            # Wenn Memories kurz sind → oberflächlich
            if avg_length < 100:  # < 100 Zeichen im Durchschnitt
                topic = pattern.keywords[0] if pattern.keywords else pattern.name

                gap = KnowledgeGap(
                    id=str(uuid.uuid4()),
                    user_id="default",
                    gap_type="interest_depth",
                    title=f"Vertiefe dein Wissen: {topic}",
                    description=f"Du erwähnst '{topic}' oft ({pattern.frequency}x), aber wir könnten tiefer gehen.",
                    suggestion=f"Möchtest du mehr Details über {topic} erfahren? Ich kann dir helfen!",
                    priority=0.6,
                    confidence=0.7,
                    related_pattern_ids=[pattern.id],
                    metadata={"frequency": pattern.frequency, "avg_length": avg_length}
                )
                gaps.append(gap)

        return gaps

    @staticmethod
    async def generate_contextual_gap_with_llm(
        patterns: List,
        goals: List,
        memories: List,
        llm_client
    ) -> List[KnowledgeGap]:
        """
        Nutzt LLM um intelligente, kontextuelle Wissenslücken zu finden.

        Das ist die fortgeschrittenste Methode - nutzt AI für bessere Erkennung.
        """
        gaps = []

        try:
            # Erstelle Kontext für LLM
            patterns_summary = ", ".join([p.name for p in patterns[:5]])
            goals_summary = ", ".join([g.content for g in goals if g.status == "active"][:3])
            recent_topics = list(set([
                word for m in memories[-20:]
                for word in m.content.lower().split()
                if len(word) > 5
            ]))[:10]

            prompt = f"""Analysiere die Interessen und Ziele eines Users und finde Wissenslücken.

Erkannte Interessen: {patterns_summary}
Aktive Ziele: {goals_summary}
Kürzlich diskutiert: {', '.join(recent_topics)}

Finde 2-3 relevante Wissenslücken die dem User helfen könnten.
Für jede Lücke gib an:
1. Titel (kurz)
2. Beschreibung (warum relevant)
3. Konkreter Vorschlag (was tun)

Antworte im JSON Format:
[
  {{"title": "...", "description": "...", "suggestion": "..."}}
]
"""

            response = await llm_client.ainvoke([{"role": "user", "content": prompt}])
            response_text = response.content

            # Parse JSON
            import json
            import re

            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                detected_gaps = json.loads(json_match.group(0))

                for gap_data in detected_gaps[:3]:  # Max 3
                    gap = KnowledgeGap(
                        id=str(uuid.uuid4()),
                        user_id="default",
                        gap_type="context_missing",
                        title=gap_data["title"],
                        description=gap_data["description"],
                        suggestion=gap_data["suggestion"],
                        priority=0.8,
                        confidence=0.75,
                        metadata={"generated_by": "llm", "llm_response": response_text[:200]}
                    )
                    gaps.append(gap)
                    logger.info(f"🤖 LLM detected gap: {gap.title}")

        except Exception as e:
            logger.error(f"Error in LLM gap detection: {e}")

        return gaps


class KnowledgeGapTracker:
    """Verwaltet KnowledgeGap-Speicherung in Qdrant"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.gaps_collection = "lexi_knowledge_gaps"
        self._ensure_collection()

    def _ensure_collection(self):
        """Stellt sicher dass Knowledge Gaps Collection existiert"""
        try:
            client = self.vectorstore.client
            created = ensure_collection(client, self.gaps_collection)
            if created:
                logger.info(f"✅ Created knowledge gaps collection: {self.gaps_collection}")
        except Exception as e:
            logger.error(f"Error ensuring gaps collection: {e}")

    def save_gap(self, gap: KnowledgeGap) -> bool:
        """Speichert eine Wissenslücke"""
        try:
            from qdrant_client.models import PointStruct

            client = self.vectorstore.client

            point = PointStruct(
                id=gap.id,
                vector=[0.0],
                payload=gap.to_dict()
            )

            safe_upsert(
                collection_name=self.gaps_collection,
                points=[point]
            )

            logger.info(f"💾 Knowledge gap saved: {gap.title}")
            return True

        except Exception as e:
            logger.error(f"Error saving knowledge gap: {e}")
            return False

    def get_all_gaps(self, user_id: str, include_dismissed: bool = False) -> List[KnowledgeGap]:
        """Holt alle Wissenslücken"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

            client = self.vectorstore.client

            base_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if not include_dismissed:
                base_conditions.append(
                    FieldCondition(key="dismissed", match=MatchValue(value=False))
                )

            try:
                cutoff_days = os.environ.get("LEXI_GAP_DAYS")
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
                collection_name=self.gaps_collection,
                scroll_filter=Filter(must=filter_conditions),
                with_payload=True,
                with_vectors=False,
                limit=1000
            )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            if not points and cutoff_ms:
                scroll_result = client.scroll(
                    collection_name=self.gaps_collection,
                    scroll_filter=Filter(must=base_conditions),
                    with_payload=True,
                    with_vectors=False,
                    limit=1000
                )
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

            gaps = []
            for point in points:
                try:
                    gap = KnowledgeGap.from_dict(point.payload)
                    gaps.append(gap)
                except Exception as e:
                    logger.warning(f"Could not parse gap {point.id}: {e}")

            logger.info(f"📋 Retrieved {len(gaps)} knowledge gaps for user {user_id}")
            return gaps

        except Exception as e:
            logger.error(f"Error retrieving gaps: {e}")
            return []

    def dismiss_gap(self, gap_id: str) -> bool:
        """Markiert eine Wissenslücke als dismissed"""
        try:
            client = self.vectorstore.client

            # Hole Gap
            result = client.retrieve(
                collection_name=self.gaps_collection,
                ids=[gap_id],
                with_payload=True
            )

            if not result:
                return False

            gap = KnowledgeGap.from_dict(result[0].payload)
            gap.dismissed = True

            return self.save_gap(gap)

        except Exception as e:
            logger.error(f"Error dismissing gap: {e}")
            return False

    def delete_gap(self, gap_id: str) -> bool:
        """Löscht eine Wissenslücke"""
        try:
            client = self.vectorstore.client

            client.delete(
                collection_name=self.gaps_collection,
                points_selector=[gap_id]
            )

            logger.info(f"🗑️ Knowledge gap deleted: {gap_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting gap: {e}")
            return False


# Globale Instanz
_global_gap_tracker: Optional[KnowledgeGapTracker] = None


def get_knowledge_gap_tracker(vectorstore=None) -> KnowledgeGapTracker:
    """Hole oder erstelle globale KnowledgeGapTracker Instanz"""
    global _global_gap_tracker

    if _global_gap_tracker is None:
        if vectorstore is None:
            from backend.core.component_cache import get_cached_components
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore

        _global_gap_tracker = KnowledgeGapTracker(vectorstore)

    return _global_gap_tracker
