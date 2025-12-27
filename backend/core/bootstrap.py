from typing import Tuple, Optional, List, Any, Dict, TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager
import logging
import os

# LAZY IMPORT: langchain_ollama kann beim Import blockieren wenn Ollama nicht läuft
# Daher nur für Type Hints importieren, Runtime-Import erfolgt in den Funktionen
if TYPE_CHECKING:
    from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.qdrant.client_wrapper import get_qdrant_client
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
# Langchain imports sind schnell, aber wir importieren ConversationBufferWindowMemory lazy
if TYPE_CHECKING:
    from langchain_classic.memory import ConversationBufferWindowMemory
from backend.config.middleware_config import MiddlewareConfig

logger = logging.getLogger("lexi_middleware.config")

@dataclass
class ComponentBundle:
    embeddings: "OllamaEmbeddings"  # Type hint as string for lazy import
    vectorstore: QdrantMemoryInterface
    memory: "ConversationBufferWindowMemory"  # Type hint as string for lazy import
    chat_client: "ChatOllama"  # Type hint as string for lazy import
    qdrant_client: QdrantClient
    embedding_dimensions: int
    config_warning: Optional[str] = None
    health_summary: Optional[Dict[str, Any]] = None
    ha_service: Optional[Any] = None  # Home Assistant service (optional integration)

class ConfigurationError(Exception):
    pass

class ComponentInitializer:

    def __init__(self, force_recreate: Optional[bool] = None):
        self.force_recreate = self._resolve_force_recreate(force_recreate)
        self._validate_environment()

    @staticmethod
    def _resolve_force_recreate(force_recreate: Optional[bool]) -> bool:
        if force_recreate is not None:
            return force_recreate
        return os.environ.get("LEXI_FORCE_RECREATE", "False").lower() == "true"

    def _validate_environment(self) -> None:
        try:
            MiddlewareConfig.get_default_embedding_model()
            MiddlewareConfig.get_embedding_base_url()
            MiddlewareConfig.get_qdrant_host()
            MiddlewareConfig.get_qdrant_port()
            MiddlewareConfig.get_memory_collection()
            MiddlewareConfig.get_ollama_base_url()
            MiddlewareConfig.get_default_llm_model()
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def initialize_all(self) -> ComponentBundle:
        logger.info("Starting component initialization...")

        embeddings, dimensions = self._init_embeddings()
        vectorstore, qdrant_client, config_warning = self._init_vectorstore(embeddings, dimensions)
        memory = self._init_memory()
        chat_client = self._init_chat_client()

        self._train_category_predictor(vectorstore)
        self._ensure_vectorstore_populated(vectorstore, embeddings)

        # Initialize Home Assistant service (optional)
        ha_service = self._init_home_assistant()

        health = self._collect_health_summary(embeddings, vectorstore, chat_client)

        return ComponentBundle(
            embeddings=embeddings,
            vectorstore=vectorstore,
            memory=memory,
            chat_client=chat_client,
            qdrant_client=qdrant_client,
            embedding_dimensions=dimensions,
            config_warning=config_warning,
            health_summary=health,
            ha_service=ha_service
        )

    def _init_embeddings(self) -> Tuple["OllamaEmbeddings", int]:
        # LAZY IMPORT: Nur hier importieren um Blockieren beim Modul-Import zu vermeiden
        from langchain_ollama import OllamaEmbeddings

        model = MiddlewareConfig.get_default_embedding_model()
        url = MiddlewareConfig.get_embedding_base_url()
        logger.info(f"Initializing embeddings: {model} at {url}")

        ollama_available = self._ensure_ollama_model(url, model, "embedding")

        try:
            embeddings = OllamaEmbeddings(base_url=url, model=model)

            if ollama_available:
                vector = embeddings.embed_query("Test")
                return embeddings, len(vector)
            else:
                # Ollama nicht verfügbar - Default-Dimension verwenden
                logger.warning(f"⚠️ Ollama not available at {url} - using default dimension")
                default_dim = MiddlewareConfig.get_memory_dimension()
                logger.info(f"   Default embedding dimension: {default_dim}")
                return embeddings, default_dim

        except Exception as e:
            if "model not found" in str(e).lower() or "404" in str(e):
                cmd = self._generate_pull_command(model)
                raise ConfigurationError(f"Model '{model}' not found. Pull it via:\n{cmd}")
            raise ConfigurationError(f"Embedding init failed: {e}")

    @staticmethod
    def _generate_pull_command(model: str) -> str:
        return f"curl -X POST http://localhost:11434/api/pull -d '{{\"model\": \"{model}\", \"stream\": false}}'"

    def _ensure_ollama_model(self, url: str, model: str, label: str) -> bool:
        import requests

        base_url = url.replace("/api", "")
        try:
            tags_response = requests.get(f"{base_url}/api/tags", timeout=3)
            if not tags_response.ok:
                logger.warning(f"⚠️ Ollama not available at {base_url} - skipping {label} model check")
                return False

            models = tags_response.json().get("models", [])
            if any(m.get("name") == model for m in models):
                return True

            logger.warning(f"Model '{model}' not found in Ollama. Pulling before startup...")
            pull_response = requests.post(
                f"{base_url}/api/pull",
                json={"model": model, "stream": False},
                timeout=600,
            )
            if not pull_response.ok:
                logger.error(
                    f"Failed to pull model '{model}' (status {pull_response.status_code}): {pull_response.text}"
                )
                return False

            logger.info(f"✅ Model pulled successfully: {model}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not verify/pull model '{model}': {e}")
            return False

    def _init_vectorstore(self, embeddings: "OllamaEmbeddings", dimensions: int) -> Tuple[QdrantMemoryInterface, QdrantClient, Optional[str]]:
        name = MiddlewareConfig.get_memory_collection()

        logger.info("Connecting to Qdrant via client_wrapper")

        try:
            client = get_qdrant_client()
            warn = self._ensure_collection_exists(client, name, dimensions)

            store = QdrantMemoryInterface(qdrant_client=client, collection_name=name, embeddings=embeddings)
            return store, client, warn
        except Exception as e:
            raise ConfigurationError(f"Vectorstore init failed: {e}")

    def _ensure_collection_exists(self, client: QdrantClient, name: str, dim: int) -> Optional[str]:
        try:
            info = client.get_collection(name)
            existing_dim = info.config.params.vectors.size

            # Force recreate if flag is set (regardless of dimensions)
            if self.force_recreate:
                logger.warning(f"Force recreate enabled - deleting collection '{name}'")
                self._create_collection(client, name, dim, recreate=True)
                return None

            # Check for dimension mismatch
            if existing_dim != dim:
                return self._handle_dimension_mismatch(client, name, existing_dim, dim)
            return None
        except UnexpectedResponse:
            logger.info(f"Creating collection '{name}' with {dim} dims")
            self._create_collection(client, name, dim)
            return None

    def _handle_dimension_mismatch(self, client: QdrantClient, name: str, old: int, new: int) -> Optional[str]:
        logger.warning(f"Qdrant dimension mismatch: {old} != {new}")
        if self.force_recreate:
            self._create_collection(client, name, new, recreate=True)
            return None
        return f"Dimension mismatch {old} vs {new} – use --force-recreate to fix."

    @staticmethod
    def _create_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
        """
        Create collection with optimized HNSW parameters and payload indices.

        PERFORMANCE:
        - HNSW parameters optimized for production (m=32, ef_construct=200)
        - Provides better recall/speed trade-off than defaults
        - Indices on user_id, category, tags provide 10-100x speedup for filtered queries
        """
        # Optimized HNSW configuration
        # m=32: More connections = better recall (default: 16)
        # ef_construct=200: Higher quality index (default: 100)
        hnsw_config = models.HnswConfigDiff(
            m=32,              # Number of edges per node (16-64 recommended)
            ef_construct=200,  # Dynamic candidate list size during construction
            full_scan_threshold=10000  # Use exact search for small collections
        )

        cfg = models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            hnsw_config=hnsw_config  # Apply optimized HNSW settings
        )

        if recreate:
            client.recreate_collection(name, cfg)
            logger.info(f"Collection '{name}' recreated with optimized HNSW (m=32, ef_construct=200)")
        else:
            client.create_collection(name, cfg)
            logger.info(f"Collection '{name}' created with optimized HNSW (m=32, ef_construct=200)")

        # Create payload indices for frequently queried fields
        # This dramatically speeds up filtered queries
        try:
            logger.info(f"Creating payload indices for collection '{name}'...")

            # Index user_id (keyword) - most common filter
            client.create_payload_index(
                collection_name=name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index on user_id")

            # Index category (keyword) - for category-based queries
            client.create_payload_index(
                collection_name=name,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index on category")

            # Index tags (keyword array) - for tag-based filtering
            client.create_payload_index(
                collection_name=name,
                field_name="tags",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index on tags")

            # Index source (keyword) - for source-based queries
            client.create_payload_index(
                collection_name=name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index on source")

            # Index timestamp (integer) - for time-based queries
            client.create_payload_index(
                collection_name=name,
                field_name="timestamp",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index on timestamp")

            # Index timestamp_ms (integer) - for efficient range filters
            client.create_payload_index(
                collection_name=name,
                field_name="timestamp_ms",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            logger.info("✓ Created index on timestamp_ms")

            logger.info(f"✅ All payload indices created for '{name}'")
        except Exception as e:
            # Non-critical: indices are optional optimization
            logger.warning(f"Failed to create payload indices (non-critical): {e}")

    @staticmethod
    def _init_memory():
        """
        Initialize conversation memory with size limits to prevent memory leaks.

        PERFORMANCE: Uses ConversationBufferWindowMemory with k=50 to limit
        memory growth in long-running sessions.
        """
        from langchain_classic.memory import ConversationBufferWindowMemory

        # Use window memory to limit size (keeps last 50 messages = ~25 exchanges)
        # This prevents unbounded memory growth while maintaining sufficient context
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=50,  # Keep last 50 messages
            return_messages=True
        )

    def _init_chat_client(self) -> "ChatOllama":
        # LAZY IMPORT: Nur hier importieren um Blockieren beim Modul-Import zu vermeiden
        from langchain_ollama import ChatOllama

        url = MiddlewareConfig.get_ollama_base_url()
        model = MiddlewareConfig.get_default_llm_model()

        # PERFORMANCE: Keep model loaded indefinitely to avoid reload overhead (3.8s)
        # Use 24h as default (Ollama doesn't support -1)
        keep_alive_duration = os.getenv("LEXI_MODEL_KEEP_ALIVE", "24h")

        logger.info(f"Initializing ChatOllama: {model} at {url}")
        logger.info(f"ChatOllama keep_alive set to: {keep_alive_duration}")

        try:
            ollama_available = self._ensure_ollama_model(url, model, "LLM")

            # OPTIMIERT FÜR NATÜRLICHE ANTWORTEN (statt nur Speed)
            # Balance zwischen Qualität und Performance
            chat_client = ChatOllama(
                base_url=url,
                model=model,
                num_predict=512,   # Erlaubt vollständige Antworten (~400 Wörter)
                num_ctx=4096,      # Mehr Kontext für besseres Verständnis
                temperature=0.85,  # Natürlicher, kreativer (war: 0.6)
                top_k=50,          # Mehr Wortwahl-Varianz (war: 15)
                top_p=0.9,         # Besseres Nucleus Sampling (war: 0.85)
                repeat_penalty=1.15,  # Leicht erhöht gegen Wiederholungen
                keep_alive=keep_alive_duration  # Never unload model
            )

            if ollama_available:
                logger.info("Warming up ChatOllama model...")
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    # Use a more complex warmup to ensure model is fully loaded
                    warmup_response = chat_client.invoke([
                        SystemMessage(content="You are a helpful AI assistant."),
                        HumanMessage(content="Please confirm you are ready. Respond with 'Ready'.")
                    ])

                    logger.info(f"✅ Model warmed up successfully: {warmup_response.content if hasattr(warmup_response, 'content') else 'OK'}")
                    logger.info(f"   Model will stay loaded for: {keep_alive_duration}")

                    # Verify model is actually loaded
                    try:
                        ps_response = requests.get(f"{url.replace('/api', '')}/api/ps", timeout=2)
                        if ps_response.ok:
                            models_loaded = ps_response.json().get('models', [])
                            is_loaded = any(m['name'] == model for m in models_loaded)
                            if is_loaded:
                                logger.info(f"✅ Verified: Model '{model}' is loaded in memory")
                            else:
                                logger.warning(f"⚠️ Model '{model}' not found in loaded models")
                    except Exception as e:
                        logger.debug(f"Could not verify loaded models: {e}")

                except Exception as e:
                    logger.warning(f"Model warmup failed (non-critical): {e}")
            else:
                logger.warning(f"⚠️ Ollama not available at {url} - skipping warmup")
                logger.info("   Server will start anyway. Configure Ollama via WebUI later.")

            return chat_client

        except Exception as e:
            raise ConfigurationError(f"Chat client init failed: {e}")

    def _train_category_predictor(self, store: QdrantMemoryInterface) -> None:
        """
        OPTIMIZATION: Category predictor training is now LAZY-LOADED.

        Training is SKIPPED at bootstrap (saves 30-60s with large datasets).
        Training happens automatically on first predict_category() call.

        To force eager training at bootstrap, set: LEXI_CATEGORY_PREDICTOR_EAGER=true
        """
        import os
        eager_training = os.getenv("LEXI_CATEGORY_PREDICTOR_EAGER", "false").lower() == "true"

        if not eager_training:
            logger.info("✓ Category predictor training SKIPPED (lazy-loading enabled)")
            logger.info("   Training will happen automatically on first use")
            return

        # Eager training (nur wenn explizit aktiviert)
        try:
            entries = store.get_all_entries()
            if not entries:
                return
            # get_all_entries() returns List[MemoryEntry], not Points with .payload
            contents = [entry.content for entry in entries if entry.content]
            categories = [entry.category or "uncategorized" for entry in entries if entry.content]
            from backend.memory.memory_bootstrap import get_predictor
            predictor = get_predictor()
            predictor.train(contents, categories)
            logger.info(f"Category predictor trained with {len(contents)} entries")
        except Exception as e:
            logger.warning(f"Category predictor training failed: {e}")

    def _ensure_vectorstore_populated(self, store: QdrantMemoryInterface, embeddings: "OllamaEmbeddings") -> None:
        try:
            # ✅ Besserer Check: Hole tatsächliche Entry-Anzahl statt Query
            all_entries = store.get_all_entries(with_vectors=False)

            if len(all_entries) == 0:
                logger.info("Vectorstore empty - populating with initial data")
                self._add_initial_data(store, embeddings)
            else:
                logger.debug(f"Vectorstore already populated ({len(all_entries)} entries) - skipping bootstrap")

        except Exception as e:
            logger.warning(f"Vectorstore population check failed: {e}")

    @staticmethod
    def _add_initial_data(store: QdrantMemoryInterface, embeddings: "OllamaEmbeddings") -> None:
        text = "This is an initial memory to populate the vectorstore."

        try:
            # ✅ CHECK: Prüfe ob Bootstrap-Memory bereits existiert
            all_entries = store.get_all_entries(with_vectors=False)
            existing_bootstrap = any(
                mem for mem in all_entries
                if mem.source == "bootstrap"  # Direct attribute access, not metadata.get()
            )

            if existing_bootstrap:
                logger.debug("Bootstrap memory already exists - skipping")
                return

            metadata: Dict[str, Any] = {
                "source": "bootstrap",
                "category": "initial",
                "relevance": 1.0,
                "bootstrap_id": "initial_bootstrap_v1"  # ✅ Eindeutige ID
            }

            store.add_entry(content=text, user_id="system", tags=["system"], metadata=metadata)
            logger.info("✅ Bootstrap memory created")

        except Exception as e:
            logger.warning(f"Failed to add bootstrap memory: {e}")

    def _init_home_assistant(self) -> Optional[Any]:
        """
        Initialize Home Assistant service if feature flag is enabled.

        Returns:
            HomeAssistantService instance or None if disabled/not configured
        """
        from backend.config.feature_flags import FeatureFlags

        if not FeatureFlags.is_enabled("home_assistant"):
            logger.info("Home Assistant integration disabled (feature flag)")
            return None
        if not MiddlewareConfig.get_ha_enabled():
            logger.info("⚠️ Home Assistant not configured (URL/Token missing) - disabling feature")
            FeatureFlags.disable("home_assistant")
            return None

        try:
            from backend.services.home_assistant import get_ha_service
            ha_service = get_ha_service()

            if ha_service.is_enabled():
                logger.info("✅ Home Assistant service initialized and configured")
                return ha_service
            else:
                logger.info("⚠️ Home Assistant not configured (URL/Token missing) - disabling feature")
                FeatureFlags.disable("home_assistant")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize Home Assistant service: {e}")
            return None

    def _collect_health_summary(self, emb: "OllamaEmbeddings", store: QdrantMemoryInterface, chat: "ChatOllama") -> Dict[str, Any]:
        summary = {}
        try:
            summary['embeddings'] = len(emb.embed_query("ping"))
        except Exception as e:
            summary['embeddings'] = f"error: {e}"
        try:
            summary['qdrant_entries'] = len(store.get_all_entries())
        except Exception as e:
            summary['qdrant_entries'] = f"error: {e}"
        try:
            response = chat.invoke([{"role": "user", "content": "ping"}])
            summary['chat_response'] = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            summary['chat_response'] = f"error: {e}"
        return summary

@contextmanager
def component_lifecycle(force_recreate: Optional[bool] = None):
    bundle = ComponentInitializer(force_recreate).initialize_all()
    try:
        yield bundle
    finally:
        logger.info("Component lifecycle completed")

def initialize_components_bundle(force_recreate: Optional[bool] = None) -> ComponentBundle:
    return ComponentInitializer(force_recreate).initialize_all()

def initialize_components(force_recreate: Optional[bool] = None) -> Tuple["OllamaEmbeddings", QdrantMemoryInterface, "ConversationBufferWindowMemory", "ChatOllama", Optional[str]]:
    b = ComponentInitializer(force_recreate).initialize_all()
    return b.embeddings, b.vectorstore, b.memory, b.chat_client, b.config_warning
