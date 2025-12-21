"""
Memory Synthesizer für LexiAI - Phase 1: Idle-Mode Memory Synthesis

Generiert Meta-Wissen aus Clustern ähnlicher Memories während Idle-Zeit.
Transformiert isolierte Erinnerungen zu generalisiertem Wissen.

Beispiel:
    Memory 1: "User mag Pizza mit Salami"
    Memory 2: "User isst gerne Pizza mit Thunfisch"
    Memory 3: "Pizza ist Lieblingsessen"

    → Meta-Wissen: "User liebt Pizza in verschiedenen Varianten (Salami, Thunfisch)"

Verwendung:
    from backend.memory.memory_synthesizer import MemorySynthesizer
    from backend.core.component_cache import get_cached_components

    bundle = get_cached_components()
    synthesizer = MemorySynthesizer(
        llm=bundle.chat_client,
        vectorstore=bundle.vectorstore
    )

    # Finde Clusters und synthetisiere
    synthesized = synthesizer.synthesize_clusters(user_id="default")
    print(f"Synthetisiert: {len(synthesized)} Meta-Wissen Einträge")
"""

from typing import List, Dict, Optional
from datetime import datetime, UTC
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

# LAZY IMPORT: langchain_ollama blockiert wenn Ollama nicht läuft
if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

from backend.models.memory_entry import MemoryEntry
from backend.memory.memory_intelligence import MemoryConsolidator
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.qdrant.client_wrapper import safe_search
from qdrant_client.models import Filter, FieldCondition, MatchValue
from backend.embeddings.embedding_cache import cached_embed_query
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Ergebnis einer Memory Synthesis"""
    meta_knowledge: MemoryEntry
    source_memories: List[MemoryEntry]
    cluster_size: int
    avg_similarity: float


class MemorySynthesizer:
    """
    Synthetisiert Meta-Wissen aus Clustern ähnlicher Memories.

    Nutzt LLM um aus mehreren ähnlichen Memories eine generalisierte
    Wissens-Aussage zu erstellen. Dies reduziert Redundanz und verbessert
    das Verständnis von Mustern.
    """

    def __init__(
        self,
        llm: "ChatOllama",
        vectorstore: QdrantMemoryInterface,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.85,
        max_clusters_per_run: int = 10,
        superseded_relevance: float = 0.2,
        superseded_allowed_tags: Optional[List[str]] = None,
        superseded_excluded_tags: Optional[List[str]] = None,
        meta_dedup_threshold: float = 0.88,
        meta_dedup_text_ratio: float = 0.85
    ):
        """
        Initialisiert den Memory Synthesizer.

        Args:
            llm: LLM für Text-Generierung
            vectorstore: Qdrant Interface für Memory-Zugriff
            min_cluster_size: Minimum Anzahl Memories pro Cluster
            similarity_threshold: Cosine Similarity Schwellwert (0-1)
            max_clusters_per_run: Max. Anzahl Cluster pro Synthesis-Run
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.max_clusters_per_run = max_clusters_per_run
        self.consolidator = MemoryConsolidator()
        self.superseded_relevance = superseded_relevance
        self.superseded_allowed_tags = set(superseded_allowed_tags or ["chat", "conversation"])
        self.superseded_excluded_tags = set(superseded_excluded_tags or ["pattern", "aggregated"])
        self.meta_dedup_threshold = meta_dedup_threshold
        self.meta_dedup_text_ratio = meta_dedup_text_ratio

    def synthesize_clusters(
        self,
        user_id: str,
        exclude_meta_knowledge: bool = True
    ) -> List[SynthesisResult]:
        """
        Findet Cluster und synthetisiert Meta-Wissen.

        Args:
            user_id: User ID für Memory-Filter
            exclude_meta_knowledge: Wenn True, ignoriere bereits synthetisierte Memories

        Returns:
            Liste von SynthesisResult Objekten

        Beispiel:
            results = synthesizer.synthesize_clusters(user_id="default")
            for result in results:
                print(f"Synthetisiert aus {result.cluster_size} Memories:")
                print(f"  {result.meta_knowledge.content}")
        """
        logger.info(f"Starte Memory Synthesis für User: {user_id}")

        # 1. Lade alle Memories
        memories = self._load_all_memories(user_id, exclude_meta_knowledge)
        logger.info(f"Geladen: {len(memories)} memories")

        if len(memories) < self.min_cluster_size:
            logger.info("Nicht genug Memories für Clustering")
            return []

        # 2. Finde Cluster ähnlicher Memories
        clusters = self._find_synthesizable_clusters(memories)
        logger.info(f"Gefunden: {len(clusters)} synthesizable clusters")

        if not clusters:
            return []

        # 3. Synthetisiere Meta-Wissen für jeden Cluster
        results = []
        for i, cluster in enumerate(clusters[:self.max_clusters_per_run]):
            logger.info(f"Synthetisiere Cluster {i+1}/{len(clusters)} ({len(cluster)} memories)")

            try:
                result = self._synthesize_from_cluster(cluster, user_id)
                if result:
                    results.append(result)
                    # Speichere Meta-Wissen in Qdrant
                    stored_id = self._store_meta_knowledge(result.meta_knowledge)
                    if stored_id:
                        result.meta_knowledge.id = stored_id
                        self._mark_sources_superseded(result.source_memories, stored_id)
                        logger.info(f"✅ Synthetisiert: {result.meta_knowledge.content[:100]}...")

            except Exception as e:
                logger.error(f"Fehler bei Cluster {i+1}: {e}", exc_info=True)
                continue

        logger.info(f"Synthesis abgeschlossen: {len(results)} Meta-Wissen Einträge erstellt")
        return results

    def _load_all_memories(
        self,
        user_id: str,
        exclude_meta_knowledge: bool
    ) -> List[MemoryEntry]:
        """
        Lädt alle Memories für einen User.

        Args:
            user_id: User ID
            exclude_meta_knowledge: Ob Meta-Knowledge ausgeschlossen werden soll

        Returns:
            Liste von MemoryEntry Objekten
        """
        # Nutze scroll für alle Entries
        all_entries = []
        offset = None

        while True:
            # Scroll durch Collection
            results = self.vectorstore.client.scroll(
                collection_name=self.vectorstore.collection,
                scroll_filter=None,  # Alle Entries
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            points, next_offset = results

            if not points:
                break

            # Konvertiere zu MemoryEntry
            for point in points:
                # Filter: nur für diesen User
                if point.payload.get("user_id") != user_id:
                    continue

                # Filter: exclude meta_knowledge wenn gewünscht
                if exclude_meta_knowledge and point.payload.get("is_meta_knowledge", False):
                    continue

                # Erstelle MemoryEntry
                entry = MemoryEntry(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    timestamp=datetime.fromisoformat(point.payload.get("timestamp", "")),
                    category=point.payload.get("category"),
                    tags=point.payload.get("tags", []),
                    source=point.payload.get("source"),
                    relevance=point.payload.get("relevance", 0.5),
                    embedding=point.vector if hasattr(point, 'vector') else None,
                    is_meta_knowledge=point.payload.get("is_meta_knowledge", False),
                    source_memory_ids=point.payload.get("source_memory_ids", []),
                    synthesis_timestamp=point.payload.get("synthesis_timestamp"),
                    superseded=point.payload.get("superseded", False),
                    superseded_at=point.payload.get("superseded_at"),
                    superseded_by=point.payload.get("superseded_by")
                )
                all_entries.append(entry)

            offset = next_offset
            if offset is None:
                break

        return all_entries

    def _find_synthesizable_clusters(
        self,
        memories: List[MemoryEntry]
    ) -> List[List[MemoryEntry]]:
        """
        Findet Cluster die synthetisiert werden sollten.

        Args:
            memories: Liste von Memories

        Returns:
            Liste von Clustern (jeder Cluster ist Liste von Memories)
        """
        # Nutze MemoryConsolidator für Clustering
        groups = self.consolidator.find_similar_memories(
            memories=memories,
            similarity_threshold=self.similarity_threshold
        )

        # Filtere: nur Cluster >= min_cluster_size
        synthesizable = [
            group for group in groups
            if len(group) >= self.min_cluster_size
        ]

        return synthesizable

    def _synthesize_from_cluster(
        self,
        cluster: List[MemoryEntry],
        user_id: str
    ) -> Optional[SynthesisResult]:
        """
        Synthetisiert Meta-Wissen aus einem Cluster.

        Args:
            cluster: Cluster von ähnlichen Memories
            user_id: User ID

        Returns:
            SynthesisResult oder None bei Fehler
        """
        # Baue Prompt für LLM
        memories_text = "\n".join([
            f"{i+1}. {mem.content}"
            for i, mem in enumerate(cluster)
        ])

        prompt = f"""Du bist ein intelligentes Gedächtnis-System. Analysiere diese ähnlichen Erinnerungen und erstelle EINE generalisierte Meta-Wissens-Aussage.

Erinnerungen:
{memories_text}

Aufgabe:
- Finde das gemeinsame Muster oder Thema
- Erstelle EINE prägnante Meta-Wissens-Aussage (1-2 Sätze)
- Behalte wichtige Details bei, aber generalisiere das Muster
- Nutze deutsche Sprache

Meta-Wissen:"""

        try:
            # LLM generiert Meta-Wissen
            response = self.llm.invoke(prompt)
            meta_content = response.content.strip()

            # Berechne durchschnittliche Similarity (approximation)
            avg_similarity = self.similarity_threshold + 0.05

            # Erstelle Meta-Knowledge Memory Entry
            meta_memory = MemoryEntry(
                id=f"meta_{datetime.now(UTC).timestamp()}",
                content=meta_content,
                timestamp=datetime.now(UTC),
                category="meta_knowledge",
                tags=["synthesized", "meta_knowledge"],
                source="memory_synthesis",
                relevance=0.9,  # Hohe initiale Relevanz
                embedding=None,  # Wird beim Speichern generiert
                is_meta_knowledge=True,
                source_memory_ids=[mem.id for mem in cluster],
                synthesis_timestamp=datetime.now(UTC).isoformat()
            )

            result = SynthesisResult(
                meta_knowledge=meta_memory,
                source_memories=cluster,
                cluster_size=len(cluster),
                avg_similarity=avg_similarity
            )

            return result

        except Exception as e:
            logger.error(f"Fehler bei LLM Synthesis: {e}", exc_info=True)
            return None

    def _store_meta_knowledge(self, meta_memory: MemoryEntry) -> Optional[str]:
        """
        Speichert Meta-Wissen in Qdrant.

        Args:
            meta_memory: Meta-Knowledge MemoryEntry

        Returns:
            ID der gespeicherten Meta-Memory bei Erfolg, sonst None
        """
        try:
            meta_topic = self._meta_topic_key(meta_memory.content)
            meta_memory.meta_topic = meta_topic

            existing_id = self._find_duplicate_meta_id(meta_memory.content, meta_topic=meta_topic)
            if existing_id:
                logger.info(f"Meta-Knowledge dedup: already exists ({existing_id})")
                self._merge_meta_sources(existing_id, meta_memory.source_memory_ids)
                return existing_id

            # Nutze store_entry von QdrantInterface
            # Das generiert automatisch Embedding
            from backend.memory.adapter import store_memory

            doc_id, _timestamp = store_memory(
                content=meta_memory.content,
                user_id="system",  # Meta-Wissen ist System-generiert
                tags=meta_memory.tags,
                metadata={
                    "is_meta_knowledge": True,
                    "source_memory_ids": meta_memory.source_memory_ids,
                    "synthesis_timestamp": meta_memory.synthesis_timestamp,
                    "relevance": meta_memory.relevance,
                    "meta_topic": meta_topic
                }
            )

            logger.info(f"Meta-Knowledge gespeichert: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Fehler beim Speichern von Meta-Knowledge: {e}", exc_info=True)
            return None

    def _find_duplicate_meta_id(self, content: str, meta_topic: Optional[str] = None) -> Optional[str]:
        """
        Checks for existing meta-knowledge with high semantic/text similarity.
        """
        if not content or len(content.strip()) < 10:
            return None

        try:
            vector = cached_embed_query(self.vectorstore.embeddings, content)
            filters = [FieldCondition(
                key="is_meta_knowledge", match=MatchValue(value=True)
            )]
            if meta_topic:
                filters.append(FieldCondition(key="meta_topic", match=MatchValue(value=meta_topic)))

            results = safe_search(
                collection_name=self.vectorstore.collection,
                query_vector=vector,
                query_filter=Filter(must=filters),
                limit=3,
                score_threshold=self.meta_dedup_threshold,
                with_payload=True
            )

            normalized = self._normalize_text(content)
            for point in results:
                payload = point.payload or {}
                existing_content = payload.get("content", "")
                if not existing_content:
                    continue
                existing_norm = self._normalize_text(existing_content)
                ratio = SequenceMatcher(None, normalized, existing_norm).ratio()
                if ratio >= self.meta_dedup_text_ratio:
                    return str(point.id)
            return None
        except Exception as e:
            logger.warning(f"Meta dedup check failed: {e}")
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\\s]", "", (text or "").lower())
        return re.sub(r"\\s+", " ", normalized).strip()

    @staticmethod
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

    def _merge_meta_sources(self, meta_id: str, source_ids: List[str]) -> None:
        if not source_ids:
            return
        try:
            # Merge new sources into existing meta knowledge entry
            self.vectorstore.update_entry_metadata(
                meta_id,
                {
                    "source_memory_ids": list({str(sid) for sid in source_ids}),
                    "meta_consolidated": True
                }
            )
        except Exception as e:
            logger.warning(f"Failed to merge meta sources for {meta_id}: {e}")

    def _mark_sources_superseded(self, source_memories: List[MemoryEntry], meta_id: str) -> None:
        """
        Markiert Quell-Memories als superseded, damit sie spaeter gezielt bereinigt werden koennen.
        """
        superseded_at = datetime.now(UTC).isoformat()
        for memory in source_memories:
            tags = set(memory.tags or [])
            source_value = (memory.source or "").lower()
            if self.superseded_allowed_tags and not tags.intersection(self.superseded_allowed_tags):
                if source_value not in self.superseded_allowed_tags:
                    continue
            if tags.intersection(self.superseded_excluded_tags):
                continue
            if memory.is_meta_knowledge:
                continue

            new_relevance = memory.relevance or 1.0
            if new_relevance > self.superseded_relevance:
                new_relevance = self.superseded_relevance

            try:
                self.vectorstore.update_entry_metadata(
                    memory.id,
                    {
                        "superseded": True,
                        "superseded_at": superseded_at,
                        "superseded_by": meta_id,
                        "relevance": new_relevance
                    }
                )
            except Exception as e:
                logger.warning(f"Fehler beim Markieren als superseded: {memory.id} ({e})")


# Convenience Function
def synthesize_memories(user_id: str = "default") -> int:
    """
    Convenience Function: Synthetisiert Memories für einen User.

    Args:
        user_id: User ID

    Returns:
        Anzahl synthetisierter Meta-Wissen Einträge

    Verwendung:
        from backend.memory.memory_synthesizer import synthesize_memories

        count = synthesize_memories(user_id="default")
        print(f"Synthetisiert: {count} Meta-Wissen Einträge")
    """
    from backend.core.component_cache import get_cached_components

    bundle = get_cached_components()
    synthesizer = MemorySynthesizer(
        llm=bundle.chat_client,
        vectorstore=bundle.vectorstore
    )

    results = synthesizer.synthesize_clusters(user_id=user_id)
    return len(results)
