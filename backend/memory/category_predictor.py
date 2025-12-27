import os
import logging
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock
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
        self.cache_dir = Path(os.environ.get("LEXI_CLUSTER_CACHE_DIR", "backend/data"))
        self.cache_path = self.cache_dir / "category_clusters.npz"
        self.meta_path = self.cache_dir / "category_clusters_meta.json"
        self.lock_path = self.cache_dir / "category_clusters.lock"
        self.rebuild_threshold = self._read_env_int("LEXI_CLUSTER_REBUILD_THRESHOLD", 50)
        self.rebuild_interval_seconds = self._read_env_float("LEXI_CLUSTER_REBUILD_INTERVAL_HOURS", 24.0) * 3600.0
        self.check_interval_seconds = self._read_env_float("LEXI_CLUSTER_CHECK_INTERVAL_SECONDS", 60.0)
        self.last_build_count = None
        self.last_build_ts = None
        self.last_check_ts = None
        self._load_cluster_cache()

    @staticmethod
    def _read_env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_total_count(self) -> int:
        try:
            info = self.qdrant.client.get_collection(self.qdrant.collection)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception as exc:
            logger.debug(f"Failed to read collection count: {exc}")
            return 0

    def _load_cluster_cache(self) -> bool:
        if not self.cache_path.exists() or not self.meta_path.exists():
            return False

        try:
            with FileLock(str(self.lock_path), timeout=5):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with np.load(self.cache_path) as data:
                    if "embeddings" not in data.files or "labels" not in data.files:
                        return False
                    embeddings = data["embeddings"]
                    labels = data["labels"]

            collection = meta.get("collection")
            if collection and collection != self.qdrant.collection:
                logger.info("Cluster cache collection mismatch; ignoring cached model")
                return False

            if embeddings is None or labels is None:
                return False

            if len(embeddings) != len(labels):
                logger.warning("Cluster cache size mismatch; ignoring cached model")
                return False

            self.embeddings = embeddings
            self.labels = labels
            self.clusters.clear()
            for idx, label in enumerate(self.labels):
                if int(label) == -1:
                    continue
                self.clusters.setdefault(int(label), []).append(self.embeddings[idx])

            self.last_build_count = meta.get("total_count", len(self.labels))
            built_at = meta.get("built_at")
            if built_at:
                try:
                    self.last_build_ts = datetime.fromisoformat(built_at).timestamp()
                except ValueError:
                    self.last_build_ts = None

            logger.info(f"Loaded {len(self.clusters)} clusters from cache")
            return True
        except Exception as exc:
            logger.warning(f"Failed to load cluster cache: {exc}")
            return False

    def _persist_cluster_cache(self, total_count: int) -> None:
        if self.embeddings is None or self.labels is None:
            return

        embeddings = np.asarray(self.embeddings)
        labels = np.asarray(self.labels)
        if embeddings.size == 0 or labels.size == 0:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 1,
            "collection": self.qdrant.collection,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "total_count": total_count,
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "min_score": self.min_score
        }

        tmp_cache_path = self.cache_path.with_name(f"{self.cache_path.stem}.tmp.npz")
        tmp_meta_path = self.meta_path.with_name(f"{self.meta_path.stem}.tmp.json")

        try:
            with FileLock(str(self.lock_path), timeout=10):
                np.savez_compressed(tmp_cache_path, embeddings=embeddings, labels=labels)
                os.replace(tmp_cache_path, self.cache_path)
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                os.replace(tmp_meta_path, self.meta_path)
        except Exception as exc:
            logger.warning(f"Failed to persist cluster cache: {exc}")

    def _should_rebuild_clusters(self, ignore_check_interval: bool = False) -> bool:
        if self.last_build_count is None and self.last_build_ts is None:
            return True

        now = time.time()
        if not ignore_check_interval:
            if self.last_check_ts and (now - self.last_check_ts) < self.check_interval_seconds:
                return False

            self.last_check_ts = now
        total_count = self._get_total_count()
        if self.last_build_ts and (now - self.last_build_ts) >= self.rebuild_interval_seconds:
            logger.info("Cluster rebuild interval reached")
            return True
        if total_count <= 0:
            return False

        if self.last_build_count is None:
            return True

        if total_count - self.last_build_count >= self.rebuild_threshold:
            logger.info(
                "Cluster rebuild threshold reached (%s new memories)",
                total_count - self.last_build_count
            )
            return True

        return False

    def rebuild_clusters(self):
        logger.info("Baue Clustermodell aus Qdrant-Datenbank auf")
        entries = self.qdrant.get_all_entries()
        total_count = self._get_total_count() or len(entries)
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

        self.last_build_count = total_count
        self.last_build_ts = time.time()
        self._persist_cluster_cache(total_count)
        logger.info(f"{len(self.clusters)} Cluster gefunden")

    def predict_category(self, content: str) -> str:
        """
        Thread-safe category prediction with automatic cluster building.

        If clusters not built yet, rebuilds them automatically using double-checked locking.
        This ensures consistent categorization across all predictions.
        """
        logger.debug(f"Bestimme Kategorie für Inhalt: {content[:50]}...")

        # FIXED: Thread-safe lazy initialization with double-checked locking
        needs_rebuild = self._should_rebuild_clusters() if self.clusters else True
        if not self.clusters or needs_rebuild:
            # Import threading only when needed
            import threading
            if not hasattr(self, '_rebuild_lock'):
                self._rebuild_lock = threading.RLock()

            with self._rebuild_lock:
                # Double-check: another thread might have built clusters
                if not self.clusters:
                    if not self._load_cluster_cache():
                        logger.info("Clusters not available - rebuilding automatically for consistency")
                        try:
                            self.rebuild_clusters()
                        except Exception as e:
                            logger.warning(f"Failed to rebuild clusters: {e} - returning 'uncategorized'")
                            return "uncategorized"
                    elif needs_rebuild and self._should_rebuild_clusters(ignore_check_interval=True):
                        logger.info("Rebuilding clusters based on threshold or interval")
                        try:
                            self.rebuild_clusters()
                        except Exception as e:
                            logger.warning(f"Failed to rebuild clusters: {e} - using cached clusters")
                elif needs_rebuild and self._should_rebuild_clusters(ignore_check_interval=True):
                    logger.info("Rebuilding clusters based on threshold or interval")
                    try:
                        self.rebuild_clusters()
                    except Exception as e:
                        logger.warning(f"Failed to rebuild clusters: {e} - using existing clusters")

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
