"""
Enhanced configuration persistence for the Lexi middleware.
"""
import json
import os
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass
from filelock import FileLock, Timeout

# Setup logging
logger = logging.getLogger("lexi_middleware.config_persistence")

@dataclass
class ConfigValidation:
    """Data class for configuration validation rules."""
    required_keys: set
    valid_types: Dict[str, type]
    url_keys: set
    port_range: tuple = (1, 65535)

class ConfigPersistence:
    """
    Enhanced configuration persistence with validation, backup, and atomic operations.
    """
    # Config file path
    CONFIG_FILE = Path("backend/config/persistent_config.json")
    BACKUP_DIR = Path("backend/config/backups")

    # Sensitive keys that should NEVER be persisted to config files
    # These should only come from environment variables
    SENSITIVE_KEYS = {
        "api_key",
        "LEXI_API_KEY",
        "tavily_api_key",
        "TAVILY_API_KEY",
        "qdrant_api_key",
        "QDRANT_API_KEY",
        "jwt_secret",
        "JWT_SECRET"
    }

    # Validation rules
    VALIDATION = ConfigValidation(
        required_keys={"llm_model", "embedding_model"},
        valid_types={
            "llm_model": str,
            "embedding_model": str,
            "ollama_url": str,
            "embedding_url": str,
            "qdrant_host": str,
            "qdrant_port": int,
            "api_key": str,
            "api_key_enabled": bool,
            "tavily_api_key": str,
            "memory_threshold": (int, float),
            "features": dict,
            "system_prompt": str,
            "audit_log_path": str
        },
        url_keys={"ollama_url", "embedding_url"}
    )
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate configuration data.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required keys
        missing_keys = cls.VALIDATION.required_keys - config.keys()
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        # Validate types
        for key, value in config.items():
            if key in cls.VALIDATION.valid_types:
                expected_type = cls.VALIDATION.valid_types[key]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        errors.append(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        errors.append(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
        
        # Validate URLs
        for key in cls.VALIDATION.url_keys:
            if key in config and config[key]:
                url = config[key]
                if not (url.startswith('http://') or url.startswith('https://')):
                    errors.append(f"Invalid URL format for {key}: {url}")
        
        # Validate port range
        if "qdrant_port" in config:
            port = config["qdrant_port"]
            if not (cls.VALIDATION.port_range[0] <= port <= cls.VALIDATION.port_range[1]):
                errors.append(f"Port out of valid range: {port}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def create_backup(cls) -> Optional[Path]:
        """
        Create a backup of the current configuration file.
        
        Returns:
            Path to backup file or None if failed
        """
        if not cls.CONFIG_FILE.exists():
            return None
            
        try:
            cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            backup_path = cls.BACKUP_DIR / f"config_backup_{timestamp}.json"
            
            # Copy current config to backup
            with open(cls.CONFIG_FILE, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            
            logger.info(f"Configuration backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return None
    
    @classmethod
    def cleanup_old_backups(cls, max_backups: int = 10):
        """
        Remove old backup files, keeping only the most recent ones.

        THREAD-SAFETY: Uses file locking to prevent race conditions during cleanup.

        Args:
            max_backups: Maximum number of backups to keep
        """
        if not cls.BACKUP_DIR.exists():
            return

        lock_file = cls.BACKUP_DIR / ".backup_cleanup.lock"

        try:
            # Ensure backup directory exists
            cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

            # Use file lock to prevent concurrent cleanup
            file_lock = FileLock(str(lock_file), timeout=5)

            with file_lock:
                backup_files = list(cls.BACKUP_DIR.glob("config_backup_*.json"))
                if len(backup_files) <= max_backups:
                    return

                # Sort by modification time (oldest first)
                backup_files.sort(key=lambda x: x.stat().st_mtime)

                # Remove oldest files
                for old_backup in backup_files[:-max_backups]:
                    try:
                        old_backup.unlink()
                        logger.info(f"Removed old backup: {old_backup}")
                    except Exception as e:
                        logger.warning(f"Failed to remove backup {old_backup}: {e}")

        except Timeout:
            logger.warning("Timeout waiting for backup cleanup lock (another process is cleaning)")
        except Exception as e:
            logger.error(f"Error cleaning up backups: {str(e)}")
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], validate: bool = True, backup: bool = True) -> bool:
        """
        Save configuration to a persistent file with validation and backup.

        SECURITY: Sensitive keys (API keys, secrets) are automatically filtered out
        and NOT saved to the config file. They must be provided via environment variables.

        THREAD-SAFETY: Uses file locking to prevent race conditions when multiple
        processes/threads attempt to save configuration simultaneously.

        Args:
            config: Configuration settings to save
            validate: Whether to validate config before saving
            backup: Whether to create a backup before saving

        Returns:
            True if successful, False otherwise
        """
        # File lock for thread-safety
        lock_file = cls.CONFIG_FILE.parent / ".config.lock"
        file_lock = FileLock(str(lock_file), timeout=10)

        try:
            # Acquire lock before any file operations
            with file_lock:
                # Filter out sensitive keys before saving
                filtered_config = {
                    k: v for k, v in config.items()
                    if k not in cls.SENSITIVE_KEYS
                }

                # Log warning if sensitive keys were filtered
                filtered_keys = set(config.keys()) - set(filtered_config.keys())
                if filtered_keys:
                    logger.warning(
                        f"Filtered sensitive keys from config (use environment variables): "
                        f"{', '.join(filtered_keys)}"
                    )

                # Validate configuration if requested
                if validate:
                    is_valid, errors = cls.validate_config(filtered_config)
                    if not is_valid:
                        logger.error(f"Configuration validation failed: {errors}")
                        return False

                # Create backup if requested and file exists
                if backup:
                    cls.create_backup()

                # Ensure directory exists
                cls.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

                # Create process-specific temp file to avoid collisions
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=cls.CONFIG_FILE.parent,
                    prefix='.config_',
                    suffix='.tmp'
                )

                try:
                    # Write to temp file
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(filtered_config, f, indent=2, sort_keys=True)

                    # Atomic replace (works on POSIX and Windows)
                    temp_path_obj = Path(temp_path)
                    if os.name == 'nt':  # Windows
                        # On Windows, need to remove target first
                        if cls.CONFIG_FILE.exists():
                            cls.CONFIG_FILE.unlink()
                        shutil.move(temp_path, cls.CONFIG_FILE)
                    else:  # POSIX (Linux, macOS)
                        # On POSIX, rename is atomic
                        temp_path_obj.rename(cls.CONFIG_FILE)

                    logger.info(f"Configuration saved to {cls.CONFIG_FILE}")

                    # Cleanup old backups
                    cls.cleanup_old_backups()

                    return True

                except Exception as e:
                    # Cleanup temp file on error
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                    raise

        except Timeout:
            logger.error("Timeout waiting for config file lock (another process is writing)")
            return False
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    @classmethod
    def load_config(cls, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration from persistent file with validation.
        
        Args:
            validate: Whether to validate loaded config
            
        Returns:
            Loaded configuration or empty dict if not found/error
        """
        try:
            # Check if file exists
            if not cls.CONFIG_FILE.exists():
                logger.info(f"No configuration file found at {cls.CONFIG_FILE}")
                return {}
            
            # Load from file
            with open(cls.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Validate if requested
            if validate:
                is_valid, errors = cls.validate_config(config)
                if not is_valid:
                    logger.warning(f"Loaded configuration has validation errors: {errors}")
                    # Still return the config but log warnings
            
            logger.info(f"Configuration loaded from {cls.CONFIG_FILE}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    @classmethod
    def get_env_mapping(cls) -> Dict[str, str]:
        """
        Get the mapping of config keys to environment variable names.
        
        Returns:
            Dictionary mapping config keys to env var names
        """
        return {
            "llm_model": "LEXI_LLM_MODEL",
            "embedding_model": "LEXI_EMBEDDING_MODEL",
            "ollama_url": "LEXI_OLLAMA_URL",
            "embedding_url": "LEXI_EMBEDDING_URL",
            "qdrant_host": "LEXI_QDRANT_HOST",
            "qdrant_port": "LEXI_QDRANT_PORT",
            "chat_history_days": "LEXI_CHAT_HISTORY_DAYS",
            "stats_days": "LEXI_STATS_DAYS",
            "memory_synthesis_days": "LEXI_MEMORY_SYNTHESIS_DAYS",
            "optimizer_days": "LEXI_OPTIMIZER_DAYS",
            "fact_confidence": "LEXI_FACT_CONFIDENCE",
            "fact_min_confidence": "LEXI_FACT_MIN_CONFIDENCE",
            "fact_ttl_days": "LEXI_FACT_TTL_DAYS",
            "memory_fallback_threshold": "LEXI_MEMORY_FALLBACK_THRESHOLD",
            "api_key": "LEXI_API_KEY",
            "api_key_enabled": "LEXI_API_KEY_ENABLED",
            "memory_threshold": "LEXI_MEMORY_THRESHOLD",
            "tavily_api_key": "TAVILY_API_KEY",
            "ha_url": "LEXI_HA_URL",
            "ha_token": "LEXI_HA_TOKEN",
        }
    
    @classmethod
    def apply_config(cls, config: Dict[str, Any]) -> int:
        """
        Apply loaded configuration to environment variables.
        
        Args:
            config: Configuration to apply
            
        Returns:
            Number of settings applied
        """
        env_mapping = cls.get_env_mapping()
        applied_count = 0
        
        # Process only environment variable settings
        for key, env_var in env_mapping.items():
            if key in config and config[key] is not None:
                # Convert to string (required for environment variables)
                env_value = str(config[key])
                os.environ[env_var] = env_value

                # SECURITY: Mask sensitive values in logs
                if key in cls.SENSITIVE_KEYS:
                    log_value = "***REDACTED***"
                else:
                    log_value = config[key]

                logger.info(f"Applied config setting: {key}={log_value}")
                applied_count += 1

        if "api_key_enabled" in config and config["api_key_enabled"] is not None:
            from backend.config.security_config import SecurityConfig
            enabled_value = config["api_key_enabled"]
            if isinstance(enabled_value, str):
                enabled = enabled_value.lower() == "true"
            else:
                enabled = bool(enabled_value)
            SecurityConfig.API_KEY_ENABLED = enabled

        # Handle special case: embedding URL fallback
        if "ollama_url" in config:
            if ("embedding_url" not in config or 
                not config.get("embedding_url") or 
                config.get("embedding_url") == config["ollama_url"]):
                os.environ["LEXI_EMBEDDING_URL"] = str(config["ollama_url"])
                logger.info(f"Set embedding URL to match Ollama URL: {config['ollama_url']}")
                applied_count += 1
        
        # Apply feature flags if present
        if "features" in config and isinstance(config["features"], dict):
            for feature, enabled in config["features"].items():
                flag_env = f"LEXI_FEATURE_{feature.upper()}"
                os.environ[flag_env] = "true" if enabled else "false"
                logger.debug(f"Applied feature flag: {feature}={enabled}")
                applied_count += 1
        
        logger.info(f"Applied {applied_count} configuration settings to environment")
        return applied_count
    
    @classmethod
    def get_current_config(cls) -> Dict[str, Any]:
        """
        Get current configuration from environment variables.
        
        Returns:
            Current configuration as dictionary
        """
        env_mapping = cls.get_env_mapping()
        current_config = {}
        
        for key, env_var in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert port back to int
                if key == "qdrant_port":
                    try:
                        current_config[key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid port value in environment: {value}")
                else:
                    current_config[key] = value
        
        # Get feature flags
        features = {}
        for env_var, value in os.environ.items():
            if env_var.startswith("LEXI_FEATURE_"):
                feature_name = env_var.replace("LEXI_FEATURE_", "").lower()
                features[feature_name] = value.lower() == "true"
        
        if features:
            current_config["features"] = features
        
        return current_config
