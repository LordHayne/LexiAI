"""
CORS configuration for the Lexi middleware.

SECURITY: Proper CORS configuration is critical for preventing unauthorized access.
This module enforces secure CORS policies with strict validation.
"""
# Load .env FIRST
from backend.utils import env_loader

import os
from typing import List

class CORSConfig:
    """
    Cross-Origin Resource Sharing (CORS) configuration.

    SECURITY DESIGN:
    - Strict origin whitelisting (no wildcards in production)
    - Credentials only with explicit origins
    - Defense against CSRF and unauthorized cross-origin requests
    """
    # Parse comma-separated list of allowed origins from environment
    _ENV_ORIGINS = os.environ.get("LEXI_CORS_ORIGINS", None)

    # Check if we're explicitly in development mode
    _IS_DEV = os.environ.get("ENV", "production").lower() in ("development", "dev", "local")

    # SECURITY: Default to localhost only for development, no wildcard
    # In production, LEXI_CORS_ORIGINS MUST be explicitly set
    _DEFAULT_ORIGINS = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"] if _IS_DEV else []

    ALLOW_ORIGINS = [origin.strip() for origin in _ENV_ORIGINS.split(",")] if _ENV_ORIGINS else _DEFAULT_ORIGINS

    # Validate CORS configuration
    if not ALLOW_ORIGINS and not _IS_DEV:
        raise ValueError(
            "LEXI_CORS_ORIGINS environment variable must be set in production. "
            "Example: LEXI_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com"
        )

    # Allow credentials (cookies)
    ALLOW_CREDENTIALS = os.environ.get("LEXI_CORS_CREDENTIALS", "True").lower() in ("true", "1", "yes")

    # CRITICAL SECURITY CHECK: Wildcard with credentials is a major vulnerability
    if "*" in ALLOW_ORIGINS:
        if ALLOW_CREDENTIALS:
            raise ValueError(
                "SECURITY ERROR: CORS wildcard ('*') cannot be used with credentials enabled. "
                "This would allow ANY website to make authenticated requests to your API! "
                "Either:\n"
                "  1. Set specific origins: LEXI_CORS_ORIGINS=https://yourdomain.com\n"
                "  2. Disable credentials: LEXI_CORS_CREDENTIALS=False (not recommended)\n"
                "Current origins: " + str(ALLOW_ORIGINS)
            )

        # Even without credentials, wildcard is only allowed in development
        if not _IS_DEV:
            raise ValueError(
                "SECURITY ERROR: CORS wildcard ('*') is not allowed in production. "
                "Set specific origins via LEXI_CORS_ORIGINS environment variable.\n"
                "Example: LEXI_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com"
            )

        # In development, allow but warn
        import warnings
        warnings.warn(
            "DEVELOPMENT WARNING: CORS wildcard '*' is enabled. "
            "This should NEVER be used in production!",
            SecurityWarning,
            stacklevel=2
        )
    
    # Allowed HTTP methods
    _DEFAULT_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    _ENV_METHODS = os.environ.get("LEXI_CORS_METHODS", None)
    ALLOW_METHODS = _ENV_METHODS.split(",") if _ENV_METHODS else _DEFAULT_METHODS
    
    # Allowed HTTP headers
    _DEFAULT_HEADERS = [
        "Content-Type", 
        "Authorization", 
        "X-API-Key", 
        "Accept", 
        "Origin", 
        "X-Requested-With", 
        "X-CSRF-Token"
    ]
    _ENV_HEADERS = os.environ.get("LEXI_CORS_HEADERS", None)
    ALLOW_HEADERS = _ENV_HEADERS.split(",") if _ENV_HEADERS else _DEFAULT_HEADERS
    
    # Max age for CORS preflight requests (in seconds)
    MAX_AGE = int(os.environ.get("LEXI_CORS_MAX_AGE", "3600"))  # 1 hour
    
    @classmethod
    def get_allowed_origins(cls) -> List[str]:
        """
        Get the list of allowed origins.
        
        Returns:
            List[str]: Allowed origins
        """
        return cls.ALLOW_ORIGINS
    
    @classmethod
    def is_origin_allowed(cls, origin: str) -> bool:
        """
        Check if the provided origin is allowed.
        
        Args:
            origin (str): The origin to check
            
        Returns:
            bool: True if allowed, False otherwise
        """
        if "*" in cls.ALLOW_ORIGINS:
            return True
            
        return origin in cls.ALLOW_ORIGINS
