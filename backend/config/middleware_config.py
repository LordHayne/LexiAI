"""
Middleware configuration settings.
"""
import os

class MiddlewareConfig:
    """
    Configuration for the Lexi middleware.
    """
    # API configuration
    API_VERSION = "v1"
    API_PREFIX = f"/{API_VERSION}"
    
    # Ollama configuration
    @classmethod
    def get_ollama_base_url(cls):
        # Load .env if not already loaded
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("LEXI_OLLAMA_URL", "http://192.168.1.146:11434")

    @property
    def OLLAMA_BASE_URL(cls):
        return cls.get_ollama_base_url()

    @classmethod
    def get_embedding_base_url(cls):
        # Load .env if not already loaded
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("LEXI_EMBEDDING_URL", cls.get_ollama_base_url())
    
    @property
    def EMBEDDING_BASE_URL(cls):
        return cls.get_embedding_base_url()
    
    # Qdrant configuration
    @classmethod
    def get_qdrant_host(cls):
        # Load .env if not already loaded
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("LEXI_QDRANT_HOST", "192.168.1.146")
    
    @property
    def QDRANT_HOST(cls):
        return cls.get_qdrant_host()
    
    @classmethod
    def get_qdrant_port(cls):
        return int(os.environ.get("LEXI_QDRANT_PORT", "6333"))
    
    @property
    def QDRANT_PORT(cls):
        return cls.get_qdrant_port()
    
    @classmethod
    def get_qdrant_grpc_port(cls):
        return int(os.environ.get("LEXI_QDRANT_GRPC_PORT", "6334"))
    
    @property
    def QDRANT_GRPC_PORT(cls):
        return cls.get_qdrant_grpc_port()
    
    @classmethod
    def get_qdrant_api_key(cls):
        return os.environ.get("LEXI_QDRANT_API_KEY", None)
    
    @property
    def QDRANT_API_KEY(cls):
        return cls.get_qdrant_api_key()
    
    # Memory configuration
    @classmethod
    def get_memory_collection(cls):
        return os.environ.get("LEXI_MEMORY_COLLECTION", "lexi_memory")
    
    @property
    def MEMORY_COLLECTION(cls):
        return cls.get_memory_collection()

    # Chat history configuration
    @classmethod
    def get_chat_history_collection(cls):
        return os.environ.get("LEXI_CHAT_HISTORY_COLLECTION", "lexi_chat_history")

    @property
    def CHAT_HISTORY_COLLECTION(cls):
        return cls.get_chat_history_collection()
    
    @classmethod
    def get_memory_dimension(cls):
        return int(os.environ.get("LEXI_MEMORY_DIMENSION", "768"))  # match existing Qdrant collection
    
    @property
    def MEMORY_DIMENSION(cls):
        return cls.get_memory_dimension()
    
    # Logging configuration
    @classmethod
    def get_log_level(cls):
        return os.environ.get("LEXI_LOG_LEVEL", "INFO")
    
    @property
    def LOG_LEVEL(cls):
        return cls.get_log_level()
    
    @classmethod
    def get_log_file(cls):
        return os.environ.get("LEXI_LOG_FILE", "lexi_middleware.log")
    
    @property
    def LOG_FILE(cls):
        return cls.get_log_file()
    
    @classmethod
    def get_audit_log_file(cls):
        return os.environ.get("LEXI_AUDIT_LOG_FILE", "lexi_audit.log")
    
    @property
    def AUDIT_LOG_FILE(cls):
        return cls.get_audit_log_file()
    
    # Model configuration
    @classmethod
    def get_default_llm_model(cls):
        # Load .env if not already loaded
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("LEXI_LLM_MODEL", "gemma3:4b")
    
    @property
    def DEFAULT_LLM_MODEL(cls):
        return cls.get_default_llm_model()
    
    @classmethod
    def get_default_embedding_model(cls):
        return os.environ.get("LEXI_EMBEDDING_MODEL", "nomic-embed-text")
    
    @property
    def DEFAULT_EMBEDDING_MODEL(cls):
        return cls.get_default_embedding_model()
    
    # Cache configuration
    @classmethod
    def get_cache_enabled(cls):
        return os.environ.get("LEXI_CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
    
    @property
    def CACHE_ENABLED(cls):
        return cls.get_cache_enabled()
    
    @classmethod
    def get_cache_ttl(cls):
        return int(os.environ.get("LEXI_CACHE_TTL", "3600"))  # 1 hour in seconds
    
    @property
    def CACHE_TTL(cls):
        return cls.get_cache_ttl()
    
    # Rate limiting
    @classmethod
    def get_rate_limit_enabled(cls):
        return os.environ.get("LEXI_RATE_LIMIT_ENABLED", "True").lower() in ("true", "1", "yes")

    @property
    def RATE_LIMIT_ENABLED(cls):
        return cls.get_rate_limit_enabled()

    @classmethod
    def get_rate_limit_requests(cls):
        return int(os.environ.get("LEXI_RATE_LIMIT_REQUESTS", "100"))  # requests per window

    @property
    def RATE_LIMIT_REQUESTS(cls):
        return cls.get_rate_limit_requests()

    @classmethod
    def get_rate_limit_window(cls):
        return int(os.environ.get("LEXI_RATE_LIMIT_WINDOW", "3600"))  # window in seconds (1 hour)

    @property
    def RATE_LIMIT_WINDOW(cls):
        return cls.get_rate_limit_window()

    # Web Search configuration (Tavily)
    @classmethod
    def get_tavily_api_key(cls):
        return os.environ.get("TAVILY_API_KEY", None)

    @property
    def TAVILY_API_KEY(cls):
        return cls.get_tavily_api_key()

    @classmethod
    def get_web_search_enabled(cls):
        """Web search is enabled if Tavily API key is configured."""
        return bool(cls.get_tavily_api_key())

    @property
    def WEB_SEARCH_ENABLED(cls):
        return cls.get_web_search_enabled()

    @classmethod
    def get_web_search_max_results(cls):
        return int(os.environ.get("LEXI_WEB_SEARCH_MAX_RESULTS", "5"))

    @property
    def WEB_SEARCH_MAX_RESULTS(cls):
        return cls.get_web_search_max_results()

    # Home Assistant configuration
    @classmethod
    def get_ha_url(cls):
        """Get Home Assistant URL."""
        return os.environ.get("LEXI_HA_URL", "")

    @property
    def HA_URL(cls):
        return cls.get_ha_url()

    @classmethod
    def get_ha_token(cls):
        """Get Home Assistant long-lived access token."""
        return os.environ.get("LEXI_HA_TOKEN", "")

    @property
    def HA_TOKEN(cls):
        return cls.get_ha_token()

    @classmethod
    def get_ha_enabled(cls):
        """Home Assistant is enabled if both URL and token are configured."""
        return bool(cls.get_ha_url() and cls.get_ha_token())

    @property
    def HA_ENABLED(cls):
        return cls.get_ha_enabled()
