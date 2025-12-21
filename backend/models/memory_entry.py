from dataclasses import dataclass
from typing import Optional, List
import datetime

@dataclass
class MemoryEntry:
    id: str
    content: str
    timestamp: datetime.datetime
    user_id: str = "default"  # User isolation support
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    relevance: Optional[float] = None
    embedding: Optional[List[float]] = None

    # Phase 1: Memory Synthesis Fields
    is_meta_knowledge: bool = False  # True wenn aus Synthesis generiert
    source_memory_ids: List[str] = None  # IDs der Original-Memories
    synthesis_timestamp: Optional[str] = None  # Wann wurde synthetisiert
    superseded: bool = False  # True wenn durch Meta-Wissen ersetzt
    superseded_at: Optional[str] = None  # Wann ersetzt
    superseded_by: Optional[str] = None  # ID der Meta-Memory
    meta_topic: Optional[str] = None  # Normalized topic key for meta-knowledge

    def __post_init__(self):
        """Post-initialization: Set default for mutable fields"""
        if self.source_memory_ids is None:
            self.source_memory_ids = []

    @property
    def page_content(self) -> str:
        """Langchain Document compatibility: alias for content"""
        return self.content

    @property
    def metadata(self) -> dict:
        """Langchain Document compatibility: return metadata dict"""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "category": self.category,
            "tags": self.tags or [],
            "source": self.source,
            "relevance": self.relevance,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_meta_knowledge": self.is_meta_knowledge,
            "source_memory_ids": self.source_memory_ids or [],
            "superseded": self.superseded,
            "superseded_at": self.superseded_at,
            "superseded_by": self.superseded_by,
            "meta_topic": self.meta_topic
        }
