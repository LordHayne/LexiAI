"""
Feature flags for the Lexi middleware.
"""
import os
import logging
from typing import Dict, List, Set

logger = logging.getLogger("lexi_middleware.feature_flags")

class FeatureFlags:
    """
    Dynamic feature flags to control middleware functionality.
    """
    # Default feature flags
    _flags = {
        # API features
        "streaming": True,              # Enable streaming responses
        "rate_limiting": True,          # Enable rate limiting
        "caching": True,                # Enable response caching

        # Intelligence features (Phase 2)
        "llm_tool_calling": True,       # Enable LLM-based tool calling system (Phase 2.1)

        # Memory features
        "memory_feedback": True,        # Enable feedback for memory quality
        "advanced_memory_search": True, # Enable advanced memory search algorithms
        "user_specific_memory": True,   # Enable user-specific memory storage
        "auto_memory_tagging": True,    # Enable automatic memory tagging
        "memory_caching": True,         # Enable memory retrieval caching
        "batch_operations": True,       # Enable batch memory operations
        "vector_search_optimization": True, # Enable optimized vector search
        "auto_cleanup": False,          # Enable automatic memory cleanup (disabled by default)

        # Security features
        "audit_logging": False,         # Enable detailed audit logging (disabled by default)
        "request_validation": True,     # Enable strict request validation

        # UI features
        "feedback_ui": False,           # Enable feedback UI elements
        "memory_visualization": False,  # Enable memory visualization

        # Integration features
        "home_assistant": True,         # Enable Home Assistant smart home control
    }
    
    # Flags that have been explicitly set
    _explicit_flags: Set[str] = set()
    
    # Flag has been initialized
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """
        Initialize feature flags from environment variables.
        """
        if cls._initialized:
            return
            
        # Get feature flags from environment variables
        for flag in cls._flags.keys():
            env_var = f"LEXI_FEATURE_{flag.upper()}"
            env_value = os.environ.get(env_var)
            
            if env_value is not None:
                value = env_value.lower() in ("true", "1", "yes")
                cls._flags[flag] = value
                cls._explicit_flags.add(flag)
                logger.info(f"Feature flag '{flag}' set to {value} from environment")
                
        cls._initialized = True
        logger.info(f"Feature flags initialized with {len(cls._explicit_flags)} explicit settings")
    
    @classmethod
    def is_enabled(cls, flag: str) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag (str): The feature flag to check
            
        Returns:
            bool: True if enabled, False otherwise
        """
        if not cls._initialized:
            cls.initialize()
            
        if flag not in cls._flags:
            logger.warning(f"Unknown feature flag: {flag}")
            return False
            
        return cls._flags[flag]
    
    @classmethod
    def enable(cls, flag: str) -> bool:
        """
        Enable a feature flag.
        
        Args:
            flag (str): The feature flag to enable
            
        Returns:
            bool: True if successful, False otherwise
        """
        if flag not in cls._flags:
            logger.warning(f"Cannot enable unknown feature flag: {flag}")
            return False
            
        cls._flags[flag] = True
        cls._explicit_flags.add(flag)
        logger.info(f"Feature flag '{flag}' enabled")
        return True
    
    @classmethod
    def disable(cls, flag: str) -> bool:
        """
        Disable a feature flag.
        
        Args:
            flag (str): The feature flag to disable
            
        Returns:
            bool: True if successful, False otherwise
        """
        if flag not in cls._flags:
            logger.warning(f"Cannot disable unknown feature flag: {flag}")
            return False
            
        cls._flags[flag] = False
        cls._explicit_flags.add(flag)
        logger.info(f"Feature flag '{flag}' disabled")
        return True
    
    @classmethod
    def get_all(cls) -> Dict[str, bool]:
        """
        Get all feature flags.
        
        Returns:
            Dict[str, bool]: All feature flags
        """
        if not cls._initialized:
            cls.initialize()
            
        return cls._flags.copy()
    
    @classmethod
    def get_enabled_features(cls) -> List[str]:
        """
        Get a list of enabled features.
        
        Returns:
            List[str]: Enabled features
        """
        if not cls._initialized:
            cls.initialize()
            
        return [flag for flag, enabled in cls._flags.items() if enabled]
