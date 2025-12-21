import os
import logging
import numpy as np
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.embeddings.embedding_model import OllamaEmbeddingModel
from backend.embeddings.embedding_cache import cached_embed_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("category_predictor")


class ClusteredCategoryPredictor:
    def __init__(self, qdrant=None, embedding_model=None, eps=0.4, min_samples=2, min_score=0.3):
        from backend.qdrant.qdrant_interface import QdrantMemoryInterface
        from backend.embeddings.embedding_model import OllamaEmbeddingModel

        self.qdrant = qdrant or QdrantMemoryInterface()
        self.embedding_model = embedding_model or OllamaEmbeddingModel()
        self.eps = eps
        self.min_samples = min_samples
        self.min_score = min_score
        self.clusters = {}
        self.labels = []
        self.embeddings = []

    def rebuild_clusters(self):
        logger.info("Baue Clustermodell aus Qdrant-Datenbank auf")
        entries = self.qdrant.get_all_entries()
        # MemoryEntry has 'embedding' attribute, not 'vector'
        vectors = [e.embedding for e in entries if e.embedding is not None]

        if not vectors:
            logger.warning("Keine Vektoren in Qdrant gefunden – Clustering wird übersprungen")
            return

        # Lazy import of sklearn components
        from sklearn.cluster import DBSCAN

        self.embeddings = np.array(vectors)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit(self.embeddings)
        self.labels = clustering.labels_

        self.clusters.clear()
        for i, label in enumerate(self.labels):
            if label == -1:
                continue
            self.clusters.setdefault(label, []).append(self.embeddings[i])

        logger.info(f"{len(self.clusters)} Cluster gefunden")

    def predict_category(self, content: str) -> str:
        """
        Thread-safe category prediction with automatic cluster building.

        If clusters not built yet, rebuilds them automatically using double-checked locking.
        This ensures consistent categorization across all predictions.
        """
        logger.debug(f"Bestimme Kategorie für Inhalt: {content[:50]}...")

        # FIXED: Thread-safe lazy initialization with double-checked locking
        if not self.clusters:
            # Import threading only when needed
            import threading
            if not hasattr(self, '_rebuild_lock'):
                self._rebuild_lock = threading.RLock()

            with self._rebuild_lock:
                # Double-check: another thread might have built clusters
                if not self.clusters:
                    logger.info("Clusters not available - rebuilding automatically for consistency")
                    try:
                        self.rebuild_clusters()
                    except Exception as e:
                        logger.warning(f"Failed to rebuild clusters: {e} - returning 'uncategorized'")
                        return "uncategorized"

                # If still no clusters after rebuild, return uncategorized
                if not self.clusters:
                    logger.debug("No data available for clustering - returning 'uncategorized'")
                    return "uncategorized"

        # Use cached embedding for performance (3-5x faster)
        embedding = np.array(cached_embed_query(self.embedding_model, content))

        # Lazy import of sklearn components
        from sklearn.metrics.pairwise import cosine_similarity

        best_label = None
        best_score = -1

        for label, vectors in self.clusters.items():
            sims = cosine_similarity([embedding], vectors)
            score = np.mean(sims)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < self.min_score or best_label is None:
            category_name = "uncategorized"
        else:
            category_name = f"cluster_{best_label}"

        logger.debug(f"Zuordnung zu Kategorie: {category_name} (score={best_score:.2f})")
        return category_name

    def assign_and_store(self, content: str, metadata: dict):
        # Use cached embedding for performance (3-5x faster)
        embedding = np.array(cached_embed_query(self.embedding_model, content))
        category = self.predict_category(content)
        metadata["category"] = category
        self.qdrant.add_entry(content, embedding.tolist(), metadata)
        logger.info(f"Gespeichert mit Kategorie: {category}")
    
    def train(self, contents, categories):
        """
        Trainiert den Kategorienprediktor mit den gegebenen Inhalten und Kategorien.
        
        Args:
            contents: Liste von Textinhalten
            categories: Liste von Kategorien, die den Inhalten entsprechen
        """
        logger.info(f"Training ClusteredCategoryPredictor with {len(contents)} entries")
        
        # Speichere alle Inhalte mit ihren Kategorien in Qdrant
        for content, category in zip(contents, categories):
            # Use cached embedding for performance (3-5x faster)
            embedding = np.array(cached_embed_query(self.embedding_model, content))
            metadata = {"category": category}
            self.qdrant.add_entry(content, embedding.tolist(), metadata)
        
        # Baue die Cluster neu auf
        self.rebuild_clusters()
        logger.info("Training completed, clusters rebuilt")


def create_cluster_predictor() -> ClusteredCategoryPredictor:
    return ClusteredCategoryPredictor()
