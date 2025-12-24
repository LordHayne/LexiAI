"""
Configuration endpoints for the Lexi API.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging
import os
import datetime
from typing import Dict, Any, List, Optional, Set
from pydantic import ValidationError

from backend.api.v1.models.request_models import ConfigUpdateRequest
from backend.api.v1.models.response_models import ConfigResponse, ConfigUpdateResponse
from backend.api.middleware.error_handler import ConfigError
from backend.api.middleware.auth import verify_api_key
from backend.config.feature_flags import FeatureFlags
from backend.config.middleware_config import MiddlewareConfig
from backend.config.persistence import ConfigPersistence
from backend.config.security_config import SecurityConfig
from backend.core.bootstrap import initialize_components, ConfigurationError

# Setup logging
logger = logging.getLogger("lexi_middleware.config")

# Create router
router = APIRouter(tags=["config"])

# Configuration validation constants
VALID_FEATURES = {
    "memory_feedback", "advanced_search", "auto_optimization",
    "debug_mode", "rate_limiting", "caching",
    "streaming", "audit_logging", "user_specific_memory",
    "advanced_memory_search", "auto_memory_tagging", "home_assistant"
}

SENSITIVE_CONFIG_KEYS = {"api_key", "auth_token", "password", "secret", "ha_token", "token"}

def _sanitize_config_for_logging(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration data for safe logging."""
    sanitized = config.copy()
    for key in sanitized:
        if any(sensitive in key.lower() for sensitive in SENSITIVE_CONFIG_KEYS):
            sanitized[key] = "***REDACTED***"
    return sanitized

def _validate_url(url: str) -> bool:
    """Basic URL validation."""
    if not url:
        return False
    return url.startswith(('http://', 'https://'))

def _validate_port(port: Any) -> bool:
    """Validate port number."""
    try:
        port_int = int(port)
        return 1 <= port_int <= 65535
    except (ValueError, TypeError):
        return False

def _validate_memory_threshold(threshold: Any) -> bool:
    """Validate memory threshold value."""
    try:
        threshold_float = float(threshold)
        return 0.0 <= threshold_float <= 1.0
    except (ValueError, TypeError):
        return False

@router.get("/config", response_model=ConfigResponse)
async def get_config():
    try:
        logger.info("Retrieving current configuration")

        config = {
            "llm_model": MiddlewareConfig.get_default_llm_model(),
            "embedding_model": MiddlewareConfig.get_default_embedding_model(),
            "ollama_url": MiddlewareConfig.get_ollama_base_url(),
            "embedding_url": MiddlewareConfig.get_embedding_base_url(),
            "qdrant_host": MiddlewareConfig.get_qdrant_host(),
            "qdrant_port": MiddlewareConfig.get_qdrant_port(),
            "memory_collection": MiddlewareConfig.get_memory_collection(),
            "memory_dimension": MiddlewareConfig.get_memory_dimension(),
            "memory_threshold": float(os.getenv("LEXI_MEMORY_THRESHOLD", "0.65")),
            "feedback_enabled": FeatureFlags.is_enabled("memory_feedback"),
            "features": FeatureFlags.get_all(),
            "cache_enabled": MiddlewareConfig.get_cache_enabled(),
            "rate_limit_enabled": MiddlewareConfig.get_rate_limit_enabled(),
            "tavily_api_key": MiddlewareConfig.get_tavily_api_key(),
            "ha_url": MiddlewareConfig.get_ha_url(),
            "ha_token": MiddlewareConfig.get_ha_token() if MiddlewareConfig.get_ha_token() else "",
            "api_key_enabled": SecurityConfig.API_KEY_ENABLED
        }

        # ⬅️ Hier neu:
        persistent_config = ConfigPersistence.load_config()
        config["system_prompt"] = persistent_config.get("system_prompt", "")
        # Override with persistent tavily_api_key if available
        if "tavily_api_key" in persistent_config:
            config["tavily_api_key"] = persistent_config.get("tavily_api_key", "")

        if "LEXI_DETECTED_EMBEDDING_DIMENSION" in os.environ:
            config["detected_dimension"] = int(os.environ["LEXI_DETECTED_EMBEDDING_DIMENSION"])

        return ConfigResponse(
            success=True,
            config=config,
            timestamp=datetime.datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Unexpected error retrieving configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )

@router.post("/config", dependencies=[Depends(verify_api_key)])
async def update_config(request: Request):
    """
    Update the configuration settings.

    Requires API key authentication for security.

    Args:
        request: FastAPI request object containing configuration updates

    Returns:
        Dict: Success response with updated configuration details
    """
    try:
        request_data = await request.json()
        sanitized_data = _sanitize_config_for_logging(request_data)
        logger.info(f"Updating configuration with data: {sanitized_data}")

        # Validate request data
        validation_errors = []

        # Validate URLs if provided
        for url_key in ["ollama_url", "embedding_url", "ha_url"]:
            if url_key in request_data and request_data[url_key] is not None and request_data[url_key].strip():
                if not _validate_url(request_data[url_key]):
                    validation_errors.append(f"Invalid {url_key}: must be a valid HTTP/HTTPS URL")

        # Validate port if provided
        if "qdrant_port" in request_data and request_data["qdrant_port"] is not None:
            if not _validate_port(request_data["qdrant_port"]):
                validation_errors.append("Invalid qdrant_port: must be between 1 and 65535")

        # Validate memory threshold if provided
        if "memory_threshold" in request_data and request_data["memory_threshold"] is not None:
            if not _validate_memory_threshold(request_data["memory_threshold"]):
                validation_errors.append("Invalid memory_threshold: must be between 0.0 and 1.0")

        # Validate features if provided
        if "features" in request_data and request_data["features"] is not None:
            if not isinstance(request_data["features"], dict):
                validation_errors.append("Features must be a dictionary")
            else:
                invalid_features = set(request_data["features"].keys()) - VALID_FEATURES
                if invalid_features:
                    validation_errors.append(f"Invalid features: {', '.join(invalid_features)}")

        if validation_errors:
            logger.warning(f"Configuration validation failed: {validation_errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": validation_errors, "message": "Configuration validation failed"}
            )

        # Convert request_data to ConfigUpdateRequest format
        try:
            config_request = ConfigUpdateRequest(
                llm_model=request_data.get("llm_model"),
                embedding_model=request_data.get("embedding_model"),
                ollama_url=request_data.get("ollama_url"),
                embedding_url=request_data.get("embedding_url"),
                qdrant_host=request_data.get("qdrant_host"),
                qdrant_port=request_data.get("qdrant_port"),
                api_key=request_data.get("api_key"),
                api_key_enabled=request_data.get("api_key_enabled"),
                tavily_api_key=request_data.get("tavily_api_key"),
                ha_url=request_data.get("ha_url"),
                ha_token=request_data.get("ha_token"),
                memory_threshold=request_data.get("memory_threshold"),
                feedback_enabled=request_data.get("feedback_enabled"),
                features=request_data.get("features"),
                force_recreate_collection=request_data.get("force_recreate_collection", False)
            )
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request format: {str(e)}"
            )

        # Track which settings were updated
        updated_settings = []

        # Check if we need to recreate the vector collection
        force_recreate = config_request.force_recreate_collection
        if force_recreate:
            logger.warning("Force recreate collection option enabled")
            updated_settings.append("force_recreate_collection")

        # Determine if we need to reinitialize components
        # ✅ Nur kritische Felder triggern Reinitialisierung (Performance-Optimierung)
        critical_reinit_triggers = {
            config_request.llm_model,        # Neues LLM braucht neuen Client
            config_request.embedding_model,  # Braucht neue Embeddings
            config_request.qdrant_host,      # Neue DB-Verbindung nötig
            config_request.qdrant_port       # Neue DB-Verbindung nötig
        }
        need_reinit = force_recreate or any(x is not None for x in critical_reinit_triggers)

        # ✅ Nicht-kritische Änderungen: Nur Warnung (kein blocking reinit)
        # embedding_url und ollama_url werden erst bei Server-Restart aktiv
        needs_restart_warning = False
        if config_request.embedding_url is not None:
            from backend.config.middleware_config import MiddlewareConfig
            current_url = MiddlewareConfig.get_embedding_base_url()
            if config_request.embedding_url != current_url:
                needs_restart_warning = True
        if config_request.ollama_url is not None:
            from backend.config.middleware_config import MiddlewareConfig
            current_url = MiddlewareConfig.get_ollama_base_url()
            if config_request.ollama_url != current_url:
                needs_restart_warning = True

        # Update configuration settings
        config_updates = [
            ("llm_model", "LEXI_LLM_MODEL", config_request.llm_model),
            ("embedding_model", "LEXI_EMBEDDING_MODEL", config_request.embedding_model),
            ("ollama_url", "LEXI_OLLAMA_URL", config_request.ollama_url),
            ("embedding_url", "LEXI_EMBEDDING_URL", config_request.embedding_url),
            ("qdrant_host", "LEXI_QDRANT_HOST", config_request.qdrant_host),
            ("qdrant_port", "LEXI_QDRANT_PORT", str(config_request.qdrant_port) if config_request.qdrant_port else None),
            ("api_key", "LEXI_API_KEY", config_request.api_key),
            (
                "api_key_enabled",
                "LEXI_API_KEY_ENABLED",
                "true" if config_request.api_key_enabled else "false"
                if config_request.api_key_enabled is not None
                else None
            ),
            ("tavily_api_key", "TAVILY_API_KEY", config_request.tavily_api_key),
            ("ha_url", "LEXI_HA_URL", config_request.ha_url),
            ("ha_token", "LEXI_HA_TOKEN", config_request.ha_token),
            ("memory_threshold", "LEXI_MEMORY_THRESHOLD", str(config_request.memory_threshold) if config_request.memory_threshold else None)
        ]

        for setting_name, env_var, value in config_updates:
            if value is not None:
                logger.info(f"Updating {setting_name}")
                os.environ[env_var] = value
                updated_settings.append(setting_name)
                if setting_name == "api_key_enabled":
                    SecurityConfig.API_KEY_ENABLED = value == "true"

        # Update feedback settings
        if config_request.feedback_enabled is not None:
            if config_request.feedback_enabled:
                FeatureFlags.enable("memory_feedback")
            else:
                FeatureFlags.disable("memory_feedback")
            updated_settings.append("feedback_enabled")

        # Update feature flags
        if config_request.features is not None:
            for feature_name, enabled in config_request.features.items():
                if enabled:
                    FeatureFlags.enable(feature_name)
                else:
                    FeatureFlags.disable(feature_name)
            updated_settings.append("features")

        # Persist configuration
        persist_config = {
            "llm_model": config_request.llm_model or MiddlewareConfig.get_default_llm_model(),
            "embedding_model": config_request.embedding_model or MiddlewareConfig.get_default_embedding_model(),
            "ollama_url": config_request.ollama_url or MiddlewareConfig.get_ollama_base_url(),
            "embedding_url": config_request.embedding_url or MiddlewareConfig.get_embedding_base_url(),
            "qdrant_host": config_request.qdrant_host or MiddlewareConfig.get_qdrant_host(),
            "qdrant_port": config_request.qdrant_port or MiddlewareConfig.get_qdrant_port(),
            "tavily_api_key": config_request.tavily_api_key or MiddlewareConfig.get_tavily_api_key(),
            "ha_url": config_request.ha_url or MiddlewareConfig.get_ha_url(),
            "ha_token": config_request.ha_token or MiddlewareConfig.get_ha_token(),
            "system_prompt": request_data.get("system_prompt", ""),
            "features": config_request.features or FeatureFlags.get_all(),
            "api_key_enabled": (
                config_request.api_key_enabled
                if config_request.api_key_enabled is not None
                else SecurityConfig.API_KEY_ENABLED
            )
        }

        persistence_success = ConfigPersistence.save_config(persist_config)

        # Reinitialize components if necessary
        reinit_success = None
        if need_reinit:
            try:
                logger.info("Reinitializing components due to configuration changes...")
                from backend.core.component_cache import get_cached_components
                get_cached_components(force_recreate=True)
                reinit_success = True
                logger.info("Components reinitialized successfully")
            except ConfigurationError as e:
                logger.error(f"Failed to reinitialize components: {str(e)}")
                reinit_success = False
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Configuration updated but component reinitialization failed: {str(e)}"
                )

        response = {
            "success": True,
            "message": "Configuration updated successfully",
            "timestamp": datetime.datetime.now().isoformat(),
            "updated_settings": updated_settings,
            "reinitialization_required": need_reinit,
            "reinitialization_success": reinit_success,
            "persistence_success": persistence_success,
            "warning": "Server restart required for URL changes to take effect" if needs_restart_warning else None
        }

        logger.info(f"Configuration update completed: {len(updated_settings)} settings updated")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )

@router.post("/config/reset", dependencies=[Depends(verify_api_key)])
async def reset_config():
    """
    Reset configuration to default values.

    Requires API key authentication for security.

    Returns:
        Dict: Success response with reset configuration
    """
    try:
        logger.info("Resetting configuration to defaults")

        # Define default configuration
        default_config = {
            "llm_model": "llama2",
            "embedding_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
            "embedding_url": "http://localhost:11434",
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
            "memory_threshold": 0.65,
            "features": {
                "memory_feedback": True,
                "streaming": True,
                "caching": True,
                "rate_limiting": True,
                "user_specific_memory": True,
                "advanced_memory_search": True,
                "auto_memory_tagging": True,
                "audit_logging": False,
                "home_assistant": False
            }
        }

        # Reset environment variables
        os.environ["LEXI_LLM_MODEL"] = default_config["llm_model"]
        os.environ["LEXI_EMBEDDING_MODEL"] = default_config["embedding_model"]
        os.environ["LEXI_OLLAMA_URL"] = default_config["ollama_url"]
        os.environ["LEXI_EMBEDDING_URL"] = default_config["embedding_url"]
        os.environ["LEXI_QDRANT_HOST"] = default_config["qdrant_host"]
        os.environ["LEXI_QDRANT_PORT"] = str(default_config["qdrant_port"])
        os.environ["LEXI_MEMORY_THRESHOLD"] = str(default_config["memory_threshold"])

        # Reset feature flags
        for feature, enabled in default_config["features"].items():
            if enabled:
                FeatureFlags.enable(feature)
            else:
                FeatureFlags.disable(feature)

        # Save default configuration
        persistence_success = ConfigPersistence.save_config(default_config)

        response = {
            "success": True,
            "message": "Configuration reset to defaults",
            "timestamp": datetime.datetime.now().isoformat(),
            "default_config": default_config,
            "persistence_success": persistence_success
        }

        logger.info("Configuration reset completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}"
        )


@router.post("/config/test-ha")
async def test_home_assistant_connection(request: Request):
    """
    Test Home Assistant connection.

    Args:
        request: FastAPI request with ha_url and ha_token

    Returns:
        Connection test result
    """
    try:
        data = await request.json()
        ha_url = data.get("ha_url", "").strip()
        ha_token = data.get("ha_token", "").strip()

        if not ha_url or not ha_token:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Home Assistant URL und Token sind erforderlich"
                }
            )

        # Test connection using HomeAssistantService
        from backend.services.home_assistant import HomeAssistantService

        test_service = HomeAssistantService(url=ha_url, token=ha_token)

        if not test_service.is_enabled():
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Home Assistant Service konnte nicht initialisiert werden"
                }
            )

        # Try to list entities as connection test
        result = await test_service.list_entities()

        if result.get("success"):
            entity_count = result.get("count", 0)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"Verbindung erfolgreich! {entity_count} Entitäten gefunden.",
                    "entity_count": entity_count,
                    "entities": result.get("entities", [])[:5]  # First 5 entities
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": result.get("error", "Verbindungstest fehlgeschlagen")
                }
            )

    except Exception as e:
        logger.error(f"Error testing HA connection: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": f"Fehler beim Verbindungstest: {str(e)}"
            }
        )
