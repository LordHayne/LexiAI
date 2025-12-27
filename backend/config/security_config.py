"""
Security Configuration for LexiAI

SECURITY: This module centralizes all security-related configuration
including API key management, rate limiting, CORS, and security headers.

IMPORTANT: Review and adjust these settings before production deployment!
"""

import os
import secrets
from typing import List, Optional
from datetime import datetime, timedelta


class SecurityConfig:
    """
    Centralized security configuration.

    SECURITY PRINCIPLES:
    1. Secure by default
    2. Fail closed (deny unless explicitly allowed)
    3. Defense in depth (multiple security layers)
    4. Least privilege (minimal permissions)
    """

    # ============================================================================
    # API KEY MANAGEMENT
    # ============================================================================

    # API key settings
    API_KEY_ENABLED = os.environ.get("LEXI_API_KEY_ENABLED", "True").lower() == "true"
    API_KEY_HEADER = os.environ.get("LEXI_API_KEY_HEADER", "X-API-Key")

    # SECURITY: Minimum API key length for strength validation
    MIN_API_KEY_LENGTH = 32

    # SECURITY: API key rotation settings
    # In production, enforce regular key rotation
    API_KEY_ROTATION_DAYS = int(os.environ.get("LEXI_API_KEY_ROTATION_DAYS", "90"))
    API_KEY_ROTATION_WARNING_DAYS = int(os.environ.get("LEXI_API_KEY_ROTATION_WARNING_DAYS", "7"))

    # Weak/default keys that should be rejected
    FORBIDDEN_API_KEYS = [
        "dev_api_key_change_me_in_production",
        "your-secure-api-key-here",
        "test",
        "admin",
        "password",
        "12345",
        "changeme",
    ]

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Get API key from environment.

        SECURITY: Validates key strength and rejects weak/default keys
        """
        api_key = os.environ.get("LEXI_API_KEY")

        if not api_key:
            return None

        # Check for forbidden keys
        if api_key.lower() in SecurityConfig.FORBIDDEN_API_KEYS:
            raise ValueError(
                f"SECURITY ERROR: Default or weak API key detected! "
                f"Please generate a secure API key using: "
                f"python -c 'import secrets; print(secrets.token_hex(32))'"
            )

        # Check minimum length
        if len(api_key) < SecurityConfig.MIN_API_KEY_LENGTH:
            raise ValueError(
                f"SECURITY ERROR: API key too short! "
                f"Minimum length is {SecurityConfig.MIN_API_KEY_LENGTH} characters. "
                f"Generate a secure key using: "
                f"python -c 'import secrets; print(secrets.token_hex(32))'"
            )

        return api_key

    @staticmethod
    def generate_secure_api_key() -> str:
        """
        Generate a cryptographically secure API key.

        Returns:
            64-character hexadecimal API key (32 bytes)
        """
        return secrets.token_hex(32)

    @staticmethod
    def check_api_key_rotation_needed(key_created_at: datetime) -> bool:
        """
        Check if API key rotation is needed.

        Args:
            key_created_at: When the API key was created

        Returns:
            True if rotation is needed
        """
        age = datetime.utcnow() - key_created_at
        return age.days >= SecurityConfig.API_KEY_ROTATION_DAYS

    @staticmethod
    def check_api_key_rotation_warning(key_created_at: datetime) -> bool:
        """
        Check if API key rotation warning should be shown.

        Args:
            key_created_at: When the API key was created

        Returns:
            True if warning should be shown
        """
        age = datetime.utcnow() - key_created_at
        warning_threshold = SecurityConfig.API_KEY_ROTATION_DAYS - SecurityConfig.API_KEY_ROTATION_WARNING_DAYS
        return age.days >= warning_threshold

    # ============================================================================
    # RATE LIMITING
    # ============================================================================

    # Default rate limits per endpoint type
    RATE_LIMITS = {
        # Critical endpoints (authentication, sensitive operations)
        "critical": os.environ.get("LEXI_RATE_LIMIT_CRITICAL", "5/minute"),

        # Memory write operations
        "memory_write": os.environ.get("LEXI_RATE_LIMIT_MEMORY_WRITE", "10/minute"),

        # Chat endpoints
        "chat": os.environ.get("LEXI_RATE_LIMIT_CHAT", "30/minute"),

        # Read operations
        "read": os.environ.get("LEXI_RATE_LIMIT_READ", "100/minute"),

        # Default for all other endpoints
        "default": os.environ.get("LEXI_RATE_LIMIT_DEFAULT", "100/minute"),
    }

    # Rate limit storage backend
    # Options: memory://, redis://localhost:6379, etc.
    RATE_LIMIT_STORAGE = os.environ.get("LEXI_RATE_LIMIT_STORAGE", "memory://")

    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================

    @staticmethod
    def get_allowed_origins() -> List[str]:
        """
        Get allowed CORS origins.

        SECURITY: In production, this should be a strict whitelist of known domains.
        NEVER use "*" in production!
        """
        env = os.environ.get("ENV", "development").lower()
        origins_str = os.environ.get("LEXI_CORS_ORIGINS", "")

        if origins_str:
            # Parse comma-separated origins
            origins = [origin.strip() for origin in origins_str.split(",")]
        else:
            # Default based on environment
            if env in ["production", "prod"]:
                # SECURITY: In production, no wildcard allowed!
                origins = []
                print("WARNING: No CORS origins configured for production! Set LEXI_CORS_ORIGINS")
            else:
                # Development defaults
                origins = [
                    "http://localhost:3000",
                    "http://localhost:8000",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8000",
                ]

        # SECURITY: Validate no wildcard in production
        if env in ["production", "prod"] and "*" in origins:
            raise ValueError(
                "SECURITY ERROR: Wildcard (*) CORS origin not allowed in production! "
                "Set LEXI_CORS_ORIGINS to a comma-separated list of allowed domains."
            )

        return origins

    # CORS settings
    CORS_ALLOW_CREDENTIALS = os.environ.get("LEXI_CORS_CREDENTIALS", "True").lower() == "true"
    CORS_MAX_AGE = int(os.environ.get("LEXI_CORS_MAX_AGE", "3600"))

    # ============================================================================
    # SECURITY HEADERS
    # ============================================================================

    SECURITY_HEADERS = {
        # Prevents MIME type sniffing
        "X-Content-Type-Options": "nosniff",

        # Prevents clickjacking
        "X-Frame-Options": "DENY",

        # Enables XSS filter in browsers
        "X-XSS-Protection": "1; mode=block",

        # HSTS (only added if HTTPS)
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",

        # Content Security Policy
        # SECURITY: Adjust CSP based on your frontend requirements
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "font-src 'self' data: https://cdn.jsdelivr.net; "
            "connect-src 'self' https://cdn.jsdelivr.net; "
            "media-src 'self' blob:; "
            "frame-ancestors 'none';"
        ),

        # Permissions Policy (formerly Feature-Policy)
        "Permissions-Policy": (
            "geolocation=(), "
            "microphone=(self), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        ),
    }

    # ============================================================================
    # JWT CONFIGURATION (if enabled)
    # ============================================================================

    JWT_ENABLED = os.environ.get("LEXI_JWT_ENABLED", "False").lower() == "true"
    JWT_SECRET = os.environ.get("LEXI_JWT_SECRET")
    JWT_ALGORITHM = os.environ.get("LEXI_JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION = int(os.environ.get("LEXI_JWT_EXPIRATION", "86400"))  # 24 hours

    @staticmethod
    def validate_jwt_config():
        """
        Validate JWT configuration if JWT is enabled.

        Raises:
            ValueError: If JWT configuration is invalid
        """
        if SecurityConfig.JWT_ENABLED:
            if not SecurityConfig.JWT_SECRET:
                raise ValueError(
                    "SECURITY ERROR: JWT enabled but no JWT_SECRET configured! "
                    "Generate a secret using: "
                    "python -c 'import secrets; print(secrets.token_hex(32))'"
                )

            if len(SecurityConfig.JWT_SECRET) < 32:
                raise ValueError(
                    "SECURITY ERROR: JWT_SECRET too short! "
                    "Minimum length is 32 characters for security."
                )

    # ============================================================================
    # INPUT VALIDATION LIMITS
    # ============================================================================

    # Maximum lengths for various input fields (DoS prevention)
    MAX_USER_ID_LENGTH = 255
    MAX_CONTENT_LENGTH = 10000
    MAX_TAG_LENGTH = 50
    MAX_TAGS_COUNT = 10
    MAX_QUERY_LIMIT = 100
    MAX_QUERY_OFFSET = 10000

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    # Session timeout (in seconds)
    SESSION_TIMEOUT = int(os.environ.get("LEXI_SESSION_TIMEOUT", "3600"))  # 1 hour

    # Maximum concurrent sessions per user
    MAX_CONCURRENT_SESSIONS = int(os.environ.get("LEXI_MAX_CONCURRENT_SESSIONS", "5"))

    # ============================================================================
    # AUDIT LOGGING
    # ============================================================================

    AUDIT_LOGGING_ENABLED = os.environ.get("LEXI_AUDIT_LOGGING", "True").lower() == "true"
    AUDIT_LOG_PATH = os.environ.get("LEXI_AUDIT_LOG_PATH", "backend/logs/audit.log")

    # Events to audit
    AUDIT_EVENTS = {
        "authentication_success",
        "authentication_failure",
        "api_key_validation_failure",
        "rate_limit_exceeded",
        "input_validation_failure",
        "config_change",
        "memory_write",
        "memory_delete",
        "security_violation",
    }

    # ============================================================================
    # INITIALIZATION & VALIDATION
    # ============================================================================

    @staticmethod
    def validate_security_config():
        """
        Validate security configuration on startup.

        SECURITY: This should be called during application startup to ensure
        all security settings are properly configured before accepting requests.

        Raises:
            ValueError: If security configuration is invalid
        """
        env = os.environ.get("ENV", "development").lower()

        # Validate API key
        if SecurityConfig.API_KEY_ENABLED:
            try:
                api_key = SecurityConfig.get_api_key()
                if not api_key:
                    if env in ["production", "prod"]:
                        raise ValueError(
                            "SECURITY ERROR: API key authentication enabled but no key configured! "
                            "Set LEXI_API_KEY environment variable."
                        )
                    else:
                        print("WARNING: API key authentication enabled but no key configured!")
            except ValueError as e:
                if env in ["production", "prod"]:
                    raise
                else:
                    print(f"WARNING: {e}")

        # Validate JWT config if enabled
        if SecurityConfig.JWT_ENABLED:
            try:
                SecurityConfig.validate_jwt_config()
            except ValueError as e:
                if env in ["production", "prod"]:
                    raise
                else:
                    print(f"WARNING: {e}")

        # Validate CORS origins
        try:
            origins = SecurityConfig.get_allowed_origins()
            if env in ["production", "prod"] and not origins:
                raise ValueError(
                    "SECURITY ERROR: No CORS origins configured for production! "
                    "Set LEXI_CORS_ORIGINS environment variable."
                )
        except ValueError as e:
            if env in ["production", "prod"]:
                raise
            else:
                print(f"WARNING: {e}")

        # Print security status
        print("\n" + "="*60)
        print("SECURITY CONFIGURATION STATUS")
        print("="*60)
        print(f"Environment: {env.upper()}")
        print(f"API Key Authentication: {'ENABLED' if SecurityConfig.API_KEY_ENABLED else 'DISABLED'}")
        print(f"JWT Authentication: {'ENABLED' if SecurityConfig.JWT_ENABLED else 'DISABLED'}")
        print(f"CORS Origins: {len(SecurityConfig.get_allowed_origins())} configured")
        print(f"Rate Limiting: ENABLED")
        print(f"Audit Logging: {'ENABLED' if SecurityConfig.AUDIT_LOGGING_ENABLED else 'DISABLED'}")
        print("="*60 + "\n")

        # Production warnings
        if env in ["production", "prod"]:
            print("🔒 PRODUCTION MODE - Enhanced security checks active")
            if not SecurityConfig.API_KEY_ENABLED:
                print("⚠️  WARNING: API key authentication is DISABLED in production!")
        else:
            print("🔓 DEVELOPMENT MODE - Some security checks relaxed")
            print("   Enable strict security for production deployment!")
