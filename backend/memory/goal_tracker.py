"""
Goal Tracking System für LexiAI

Dieses Modul erkennt und trackt Benutzerziele aus Gesprächen.

Features:
1. Goal Detection - Erkennt Ziele aus Nachrichten
2. Goal Storage - Speichert Ziele persistent in Qdrant
3. Progress Tracking - Verfolgt Fortschritt über Zeit
4. Proactive Reminders - Erinnert an Ziele zur richtigen Zeit
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("lexi_middleware.goal_tracker")


class GoalStatus(Enum):
    """Status eines Ziels"""
    ACTIVE = "active"           # Ziel ist aktiv
    COMPLETED = "completed"     # Ziel wurde erreicht
    ABANDONED = "abandoned"     # Ziel wurde aufgegeben
    PAUSED = "paused"          # Ziel ist pausiert


class GoalPriority(Enum):
    """Priorität eines Ziels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Goal:
    """Repräsentiert ein Benutzerziel"""
    id: str
    user_id: str
    content: str                    # Beschreibung des Ziels
    category: str                   # z.B. "health", "learning", "work"
    status: GoalStatus
    priority: GoalPriority
    created_at: datetime
    updated_at: datetime
    target_date: Optional[datetime] = None  # Wenn Ziel ein Datum hat
    progress: float = 0.0          # 0.0 - 1.0
    milestones: List[str] = field(default_factory=list)
    mentions: int = 0              # Wie oft wurde Ziel erwähnt
    last_mentioned: Optional[datetime] = None
    source_memory_ids: List[str] = field(default_factory=list)  # Welche Memories führten zu diesem Ziel
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Speicherung"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "category": self.category,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "progress": self.progress,
            "milestones": self.milestones,
            "mentions": self.mentions,
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None,
            "source_memory_ids": self.source_memory_ids,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Erstellt Goal aus Dictionary"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            content=data["content"],
            category=data["category"],
            status=GoalStatus(data["status"]),
            priority=GoalPriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            target_date=datetime.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            progress=data.get("progress", 0.0),
            milestones=data.get("milestones", []),
            mentions=data.get("mentions", 0),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"]) if data.get("last_mentioned") else None,
            source_memory_ids=data.get("source_memory_ids", []),
            metadata=data.get("metadata", {})
        )


class GoalDetector:
    """Erkennt Ziele aus Chat-Nachrichten"""

    # Schlüsselwörter die auf Ziele hinweisen
    GOAL_INDICATORS = [
        "möchte", "will", "plane", "habe vor", "ziel ist",
        "vorhaben", "werde", "möchte gerne", "mein ziel",
        "ich würde gerne", "schaffen", "erreichen"
    ]

    # Kategorien basierend auf Schlüsselwörtern
    CATEGORY_KEYWORDS = {
        "health": ["abnehmen", "sport", "fitness", "gesund", "ernährung", "training", "joggen", "laufen"],
        "learning": ["lernen", "studieren", "kurs", "sprache", "programmieren", "ausbildung", "wissen"],
        "work": ["arbeit", "projekt", "karriere", "job", "bewerbung", "geschäft"],
        "finance": ["sparen", "geld", "investieren", "budget", "finanzen"],
        "personal": ["beziehung", "freunde", "familie", "hobby", "reise"],
        "creative": ["schreiben", "malen", "musik", "kunst", "kreativ"]
    }

    @staticmethod
    def detect_goals_from_text(text: str, user_id: str = "default") -> List[Goal]:
        """
        Einfache regelbasierte Goal-Erkennung.

        Später kann das durch LLM-basierte Erkennung ersetzt werden.
        """
        text_lower = text.lower()
        detected_goals = []

        # Prüfe ob Nachricht ein Ziel enthält
        has_goal_indicator = any(indicator in text_lower for indicator in GoalDetector.GOAL_INDICATORS)

        if not has_goal_indicator:
            return detected_goals

        # Bestimme Kategorie
        category = "general"
        for cat, keywords in GoalDetector.CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                category = cat
                break

        # Bestimme Priorität (einfach: basierend auf Dringlichkeitswörtern)
        priority = GoalPriority.MEDIUM
        if any(word in text_lower for word in ["dringend", "wichtig", "muss"]):
            priority = GoalPriority.HIGH
        elif any(word in text_lower for word in ["irgendwann", "vielleicht", "könnte"]):
            priority = GoalPriority.LOW

        # Erstelle Goal
        goal = Goal(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=text,
            category=category,
            status=GoalStatus.ACTIVE,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            mentions=1,
            last_mentioned=datetime.now(timezone.utc)
        )

        detected_goals.append(goal)
        logger.info(f"📌 Goal detected: {category} - {text[:50]}...")

        return detected_goals

    @staticmethod
    async def detect_goals_with_llm(text: str, user_id: str, chat_client) -> List[Goal]:
        """
        Erweiterte Goal-Erkennung mit LLM.

        Der LLM analysiert die Nachricht und extrahiert strukturierte Ziel-Informationen.
        """
        try:
            prompt = f"""Analysiere die folgende Nachricht und erkenne ob der Benutzer ein Ziel erwähnt.

Nachricht: "{text}"

Wenn ein Ziel erkannt wird, extrahiere:
1. Die Zielbeschreibung (kurz und präzise)
2. Kategorie (health, learning, work, finance, personal, creative, general)
3. Priorität (low, medium, high, urgent)
4. Gibt es ein Zeitrahmen? (Datum wenn möglich)

Antworte im JSON-Format:
{{
    "has_goal": true/false,
    "content": "Zielbeschreibung",
    "category": "kategorie",
    "priority": "priorität",
    "target_date": "YYYY-MM-DD oder null"
}}

Wenn kein Ziel erkannt wird, setze has_goal: false."""

            response = await chat_client.ainvoke([{"role": "user", "content": prompt}])

            # Parse JSON aus Response
            import json
            import re

            # Extrahiere JSON aus Response
            json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if not json_match:
                return []

            result = json.loads(json_match.group(0))

            if not result.get("has_goal", False):
                return []

            # Erstelle Goal aus LLM-Ergebnis
            target_date = None
            if result.get("target_date"):
                try:
                    target_date = datetime.fromisoformat(result["target_date"])
                except Exception:
                    pass

            goal = Goal(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=result.get("content", text),
                category=result.get("category", "general"),
                status=GoalStatus.ACTIVE,
                priority=GoalPriority(result.get("priority", "medium")),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                target_date=target_date,
                mentions=1,
                last_mentioned=datetime.now(timezone.utc)
            )

            logger.info(f"🤖 LLM detected goal: {goal.category} - {goal.content[:50]}...")
            return [goal]

        except Exception as e:
            logger.error(f"Error in LLM goal detection: {e}")
            # Fallback zu regelbasierter Erkennung
            return GoalDetector.detect_goals_from_text(text, user_id)


class GoalTracker:
    """Hauptklasse für Goal Tracking"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.goals_collection = "lexi_goals"  # Separate Collection für Goals
        self._ensure_collection()

    def _ensure_collection(self):
        """Stellt sicher dass Goals Collection existiert"""
        try:
            from backend.qdrant.client_wrapper import QdrantClient
            from backend.qdrant.client_wrapper import safe_upsert
            client = self.vectorstore.client

            # Prüfe ob Collection existiert
            collections = client.get_collections().collections
            if not any(c.name == self.goals_collection for c in collections):
                # Erstelle Collection (Goals brauchen keine Embeddings, nur Metadaten)
                from qdrant_client.models import Distance, VectorParams

                client.create_collection(
                    collection_name=self.goals_collection,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE)  # Dummy vector
                )
                logger.info(f"✅ Created goals collection: {self.goals_collection}")
        except Exception as e:
            logger.error(f"Error ensuring goals collection: {e}")

    def add_goal(self, goal: Goal) -> bool:
        """Fügt ein neues Ziel hinzu"""
        try:
            from qdrant_client.models import PointStruct

            client = self.vectorstore.client

            point = PointStruct(
                id=goal.id,
                vector=[0.0],  # Dummy vector
                payload=goal.to_dict()
            )

            safe_upsert(
                collection_name=self.goals_collection,
                points=[point]
            )

            logger.info(f"💾 Goal saved: {goal.content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error adding goal: {e}")
            return False

    def get_all_goals(self, user_id: str, status: Optional[GoalStatus] = None) -> List[Goal]:
        """Holt alle Ziele eines Benutzers"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self.vectorstore.client

            # Filter für user_id
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            # Optional: Filter für Status
            if status:
                filter_conditions.append(
                    FieldCondition(key="status", match=MatchValue(value=status.value))
                )

            scroll_result = client.scroll(
                collection_name=self.goals_collection,
                scroll_filter=Filter(must=filter_conditions),
                with_payload=True,
                with_vectors=False,
                limit=1000
            )

            # Handle tuple return
            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points

            goals = []
            for point in points:
                try:
                    goal = Goal.from_dict(point.payload)
                    goals.append(goal)
                except Exception as e:
                    logger.warning(f"Could not parse goal {point.id}: {e}")

            logger.info(f"📋 Retrieved {len(goals)} goals for user {user_id}")
            return goals

        except Exception as e:
            logger.error(f"Error retrieving goals: {e}")
            return []

    def update_goal(self, goal: Goal) -> bool:
        """Aktualisiert ein Ziel"""
        goal.updated_at = datetime.now(timezone.utc)
        return self.add_goal(goal)

    def delete_goal(self, goal_id: str) -> bool:
        """Löscht ein Ziel"""
        try:
            client = self.vectorstore.client

            client.delete(
                collection_name=self.goals_collection,
                points_selector=[goal_id]
            )

            logger.info(f"🗑️ Goal deleted: {goal_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting goal: {e}")
            return False

    def increment_mention(self, goal_id: str) -> bool:
        """Erhöht Mention-Counter wenn Ziel erwähnt wird"""
        try:
            # Hole Goal
            client = self.vectorstore.client

            result = client.retrieve(
                collection_name=self.goals_collection,
                ids=[goal_id],
                with_payload=True
            )

            if not result:
                return False

            goal = Goal.from_dict(result[0].payload)
            goal.mentions += 1
            goal.last_mentioned = datetime.now(timezone.utc)
            goal.updated_at = datetime.now(timezone.utc)

            return self.update_goal(goal)

        except Exception as e:
            logger.error(f"Error incrementing mention: {e}")
            return False

    def get_goals_needing_reminder(self, user_id: str, days_since_mention: int = 7) -> List[Goal]:
        """
        Findet Ziele die eine Erinnerung brauchen.

        Kriterien:
        - Status = ACTIVE
        - Lange nicht erwähnt (> days_since_mention)
        - Hohe Priorität bevorzugt
        """
        all_goals = self.get_all_goals(user_id, status=GoalStatus.ACTIVE)

        needs_reminder = []
        now = datetime.now(timezone.utc)

        for goal in all_goals:
            if not goal.last_mentioned:
                continue

            days_passed = (now - goal.last_mentioned).days

            # Hohe Priorität: Erinnere früher
            reminder_threshold = days_since_mention
            if goal.priority == GoalPriority.HIGH:
                reminder_threshold = 3
            elif goal.priority == GoalPriority.URGENT:
                reminder_threshold = 1

            if days_passed >= reminder_threshold:
                needs_reminder.append(goal)

        # Sortiere nach Priorität und Zeit seit letzter Erwähnung
        needs_reminder.sort(key=lambda g: (
            -g.priority.value,  # Hohe Priorität zuerst (nach enum order)
            -(now - g.last_mentioned).days  # Längste Zeit zuerst
        ))

        return needs_reminder


# Globale Instanz (Singleton)
_global_goal_tracker: Optional[GoalTracker] = None


def get_goal_tracker(vectorstore=None) -> GoalTracker:
    """Hole oder erstelle globale GoalTracker Instanz"""
    global _global_goal_tracker

    if _global_goal_tracker is None:
        if vectorstore is None:
            from backend.core.component_cache import get_cached_components
            bundle = get_cached_components()
            vectorstore = bundle.vectorstore

        _global_goal_tracker = GoalTracker(vectorstore)

    return _global_goal_tracker
