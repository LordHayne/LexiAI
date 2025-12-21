"""
Authentication configuration for the Lexi middleware.
"""
# Load .env FIRST
from backend.utils import env_loader

import os
import secrets

class SecurityConfig:
    """
    Security configuration for API authentication.
    """
    # API Key configuration
    API_KEY_ENABLED = os.environ.get("LEXI_API_KEY_ENABLED", "True").lower() in ("true", "1", "yes")
    API_KEY_HEADER = os.environ.get("LEXI_API_KEY_HEADER", "X-API-Key")

    # API key MUST be provided via environment variable in production
    # No default key for security reasons
    API_KEY = os.environ.get("LEXI_API_KEY", None)

    # Validate that API key is set when authentication is enabled
    if API_KEY_ENABLED and not API_KEY:
        raise ValueError(
            "LEXI_API_KEY environment variable must be set when API key authentication is enabled. "
            "Set LEXI_API_KEY_ENABLED=False to disable authentication (NOT recommended for production)."
        )
    
    # JWT configuration
    JWT_ENABLED = os.environ.get("LEXI_JWT_ENABLED", "False").lower() in ("true", "1", "yes")
    JWT_ALGORITHM = os.environ.get("LEXI_JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION = int(os.environ.get("LEXI_JWT_EXPIRATION", "86400"))  # 24 hours in seconds

    # JWT secret MUST be provided via environment variable when JWT is enabled
    # Secret must be persistent across restarts to maintain valid tokens
    JWT_SECRET = os.environ.get("LEXI_JWT_SECRET", None)

    # Validate that JWT secret is set when JWT is enabled
    if JWT_ENABLED and not JWT_SECRET:
        raise ValueError(
            "LEXI_JWT_SECRET environment variable must be set when JWT authentication is enabled. "
            "Generate a secure secret with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; img-src 'self' data:; connect-src 'self' cdn.jsdelivr.net;",
        "Permissions-Policy": "microphone=(self)"
    }
    
    @classmethod
    def get_api_key(cls):
        """
        Get the configured API key.
        
        Returns:
            str: The API key
        """
        return cls.API_KEY
    
    @classmethod
    def verify_api_key(cls, key: str) -> bool:
        """
        Verify if the provided API key is valid using constant-time comparison.

        SECURITY: Uses secrets.compare_digest() to prevent timing attacks
        that could be used to determine the correct API key.

        Args:
            key (str): The API key to verify

        Returns:
            bool: True if valid, False otherwise
        """
        if not cls.API_KEY_ENABLED:
            return True

        # Ensure both key and API_KEY are non-None strings
        if not key or not cls.API_KEY:
            return False

        # Use constant-time comparison to prevent timing attacks
        try:
            return secrets.compare_digest(key.encode('utf-8'), cls.API_KEY.encode('utf-8'))
        except Exception:
            # If encoding fails, keys don't match
            return False
