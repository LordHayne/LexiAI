"""
Input Validation Utilities for LexiAI

SECURITY: This module provides comprehensive input validation and sanitization
to prevent common attack vectors including:
- SQL Injection
- XSS (Cross-Site Scripting)
- Command Injection
- Path Traversal
- Payload/DoS attacks
"""

import re
import html
from typing import Optional, List
from fastapi import HTTPException


class InputValidator:
    """
    Comprehensive input validation and sanitization.

    SECURITY PHILOSOPHY:
    - Whitelist approach: Define what's allowed, reject everything else
    - Defense in depth: Multiple validation layers
    - Fail securely: Reject on uncertainty, don't try to sanitize
    """

    # Dangerous patterns that should never appear in user input
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\b(TABLE|DATABASE)\b)",
        r"(--|\#|/\*|\*/)",  # SQL comments
        r"(\bOR\b.*=.*)",
        r"(';|\"--)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick=
        r"<iframe",
        r"<embed",
        r"<object",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",  # Shell metacharacters
        r"\.\.\/",     # Path traversal
        r"\$\{",       # Variable expansion
    ]

    # Whitelist patterns for specific input types
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    @staticmethod
    def validate_user_id(user_id: str, max_length: int = 255) -> str:
        """
        Validate and sanitize user ID.

        SECURITY: User IDs are used in database queries and must be validated
        to prevent injection attacks and ensure proper access control.

        Args:
            user_id: The user ID to validate
            max_length: Maximum allowed length

        Returns:
            Sanitized user ID

        Raises:
            HTTPException: If validation fails
        """
        if not user_id or not isinstance(user_id, str):
            raise HTTPException(
                status_code=400,
                detail="Invalid user_id: must be a non-empty string"
            )

        # Length check (DoS prevention)
        if len(user_id) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid user_id: exceeds maximum length of {max_length}"
            )

        # Check for dangerous patterns
        InputValidator._check_dangerous_patterns(user_id, "user_id")

        # Whitelist validation: Only allow alphanumeric, underscore, hyphen, @, .
        if not re.match(r'^[a-zA-Z0-9_@.-]+$', user_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid user_id: only alphanumeric characters, underscore, hyphen, @ and . are allowed"
            )

        return user_id

    @staticmethod
    def validate_content(
        content: str,
        field_name: str = "content",
        max_length: int = 10000,
        allow_html: bool = False
    ) -> str:
        """
        Validate and sanitize content fields.

        SECURITY: Content is the primary attack vector. This validates:
        - Length limits (DoS prevention)
        - Dangerous patterns (injection prevention)
        - HTML/XSS (if HTML not allowed)

        Args:
            content: The content to validate
            field_name: Name of the field (for error messages)
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags

        Returns:
            Sanitized content

        Raises:
            HTTPException: If validation fails
        """
        if not isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name}: must be a string"
            )

        # Empty content check
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name}: cannot be empty"
            )

        # Length check (DoS prevention)
        if len(content) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name}: exceeds maximum length of {max_length} characters"
            )

        # Check for dangerous patterns
        InputValidator._check_dangerous_patterns(content, field_name)

        # HTML sanitization
        if not allow_html:
            sanitized = html.escape(content)
            # Check if any HTML was escaped (meaning it contained HTML)
            if sanitized != content:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {field_name}: HTML tags are not allowed"
                )

        return content

    @staticmethod
    def validate_api_key(api_key: str) -> str:
        """
        Validate API key format and strength.

        SECURITY: API keys must meet minimum security requirements:
        - Minimum length (32 characters recommended)
        - No default/weak keys
        - Proper character set

        Args:
            api_key: The API key to validate

        Returns:
            Validated API key

        Raises:
            HTTPException: If validation fails
        """
        if not api_key or not isinstance(api_key, str):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key: must be a non-empty string"
            )

        # Check for default/weak keys
        weak_keys = [
            "dev_api_key_change_me_in_production",
            "your-secure-api-key-here",
            "test",
            "admin",
            "password",
            "12345",
        ]

        if api_key.lower() in weak_keys:
            raise HTTPException(
                status_code=401,
                detail="Security Error: Default or weak API key detected. Please generate a secure API key."
            )

        # Length check (minimum 32 characters for security)
        if len(api_key) < 32:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key: must be at least 32 characters for security"
            )

        # Character set validation (hex or base64 style)
        if not re.match(r'^[a-zA-Z0-9_\-+=/.]+$', api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key format"
            )

        return api_key

    @staticmethod
    def validate_uuid(uuid_str: str, field_name: str = "id") -> str:
        """
        Validate UUID format.

        Args:
            uuid_str: UUID string to validate
            field_name: Name of the field (for error messages)

        Returns:
            Validated UUID string

        Raises:
            HTTPException: If validation fails
        """
        if not uuid_str or not isinstance(uuid_str, str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name}: must be a non-empty string"
            )

        if not InputValidator.UUID_PATTERN.match(uuid_str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name}: must be a valid UUID"
            )

        return uuid_str

    @staticmethod
    def validate_tag_list(tags: Optional[List[str]], max_tags: int = 10, max_tag_length: int = 50) -> Optional[List[str]]:
        """
        Validate list of tags.

        Args:
            tags: List of tags to validate
            max_tags: Maximum number of tags allowed
            max_tag_length: Maximum length per tag

        Returns:
            Validated tag list

        Raises:
            HTTPException: If validation fails
        """
        if tags is None:
            return None

        if not isinstance(tags, list):
            raise HTTPException(
                status_code=400,
                detail="Invalid tags: must be a list"
            )

        if len(tags) > max_tags:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tags: maximum {max_tags} tags allowed"
            )

        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid tags: all tags must be strings"
                )

            if len(tag) > max_tag_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tag: exceeds maximum length of {max_tag_length}"
                )

            # Check for dangerous patterns
            InputValidator._check_dangerous_patterns(tag, "tag")

            # Whitelist: alphanumeric, underscore, hyphen, space
            if not re.match(r'^[a-zA-Z0-9_\- ]+$', tag):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tag '{tag}': only alphanumeric characters, spaces, underscores and hyphens allowed"
                )

            validated_tags.append(tag.strip())

        return validated_tags

    @staticmethod
    def _check_dangerous_patterns(value: str, field_name: str):
        """
        Check for dangerous patterns in input.

        SECURITY: This is a defense-in-depth layer that catches common
        attack patterns even if they pass whitelist validation.

        Args:
            value: Value to check
            field_name: Name of the field (for error messages)

        Raises:
            HTTPException: If dangerous patterns detected
        """
        # SQL Injection patterns
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail=f"Security Error: Potential SQL injection detected in {field_name}"
                )

        # XSS patterns
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail=f"Security Error: Potential XSS detected in {field_name}"
                )

        # Command injection patterns
        for pattern in InputValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Security Error: Potential command injection detected in {field_name}"
                )

    @staticmethod
    def validate_limit(limit: Optional[int], default: int = 10, max_limit: int = 100) -> int:
        """
        Validate limit parameter for pagination.

        SECURITY: Prevents DoS via excessive queries

        Args:
            limit: Requested limit
            default: Default limit if None
            max_limit: Maximum allowed limit

        Returns:
            Validated limit

        Raises:
            HTTPException: If validation fails
        """
        if limit is None:
            return default

        if not isinstance(limit, int):
            raise HTTPException(
                status_code=400,
                detail="Invalid limit: must be an integer"
            )

        if limit < 1:
            raise HTTPException(
                status_code=400,
                detail="Invalid limit: must be at least 1"
            )

        if limit > max_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid limit: maximum {max_limit} allowed (DoS prevention)"
            )

        return limit

    @staticmethod
    def validate_offset(offset: Optional[int], default: int = 0, max_offset: int = 10000) -> int:
        """
        Validate offset parameter for pagination.

        Args:
            offset: Requested offset
            default: Default offset if None
            max_offset: Maximum allowed offset

        Returns:
            Validated offset

        Raises:
            HTTPException: If validation fails
        """
        if offset is None:
            return default

        if not isinstance(offset, int):
            raise HTTPException(
                status_code=400,
                detail="Invalid offset: must be an integer"
            )

        if offset < 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid offset: must be non-negative"
            )

        if offset > max_offset:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid offset: maximum {max_offset} allowed"
            )

        return offset
