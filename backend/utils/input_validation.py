"""
Centralized input validation for LexiAI.

SECURITY: All user inputs should be validated to prevent injection attacks,
XSS, and other security vulnerabilities.
"""
import re
import html
from typing import Any, List, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger("lexi_middleware.input_validation")


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class InputValidator:
    """Centralized input validation with security checks."""

    # Dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript protocol
        r"data:text/html",  # Data URLs with HTML
        r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
        r"<iframe",  # Iframe tags
        r"<embed",  # Embed tags
        r"<object",  # Object tags
        r"eval\s*\(",  # Eval calls
        r"expression\s*\(",  # CSS expressions
        r"\x00",  # Null bytes
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bor\b\s+\d+\s*=\s*\d+)",
        r"(\band\b\s+\d+\s*=\s*\d+)",
        r"(;.*drop\s+table)",
        r"(;\s*exec\s+)",
        r"(;\s*execute\s+)",
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize string input by removing dangerous patterns.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input contains dangerous patterns
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value).__name__}")

        # Check length
        if max_length and len(value) > max_length:
            raise ValidationError(f"Input too long (max {max_length} characters)")

        # Check for dangerous patterns
        value_lower = value.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                raise ValidationError("Input contains potentially harmful content")

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                raise ValidationError("Input contains potentially harmful SQL patterns")

        # HTML escape for safety
        return html.escape(value)

    @classmethod
    def validate_chat_message(cls, message: str) -> str:
        """
        Validate chat message input.

        Args:
            message: Chat message to validate

        Returns:
            Validated and sanitized message

        Raises:
            ValidationError: If validation fails
        """
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")

        # Max message length
        MAX_MESSAGE_LENGTH = 10000

        message = message.strip()

        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValidationError(f"Message too long (max {MAX_MESSAGE_LENGTH} characters)")

        # Allow most chat content, but check for obvious attacks
        # Don't HTML escape chat messages as they're processed by LLM
        message_lower = message.lower()

        # Check for script injection
        if "<script" in message_lower or "javascript:" in message_lower:
            raise ValidationError("Message contains potentially harmful content")

        # Check for null bytes
        if "\x00" in message:
            raise ValidationError("Message contains invalid characters")

        return message

    @classmethod
    def validate_url(cls, url: str) -> str:
        """
        Validate URL input.

        Args:
            url: URL to validate

        Returns:
            Validated URL

        Raises:
            ValidationError: If URL is invalid or dangerous
        """
        if not url or not url.strip():
            raise ValidationError("URL cannot be empty")

        url = url.strip()

        # Basic URL validation
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in ["http", "https", "ws", "wss"]:
            raise ValidationError(f"Invalid URL scheme: {parsed.scheme}")

        # Prevent SSRF to local addresses
        if parsed.hostname:
            hostname_lower = parsed.hostname.lower()
            # Block localhost and private IPs in production
            if hostname_lower in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]:
                import os
                if os.environ.get("ENV", "production").lower() not in ("development", "dev"):
                    raise ValidationError("Local URLs not allowed in production")

            # Block private IP ranges
            if any(hostname_lower.startswith(prefix) for prefix in ["10.", "172.", "192.168."]):
                import os
                if os.environ.get("ENV", "production").lower() not in ("development", "dev"):
                    raise ValidationError("Private IP addresses not allowed in production")

        return url

    @classmethod
    def validate_user_id(cls, user_id: str) -> str:
        """
        Validate user ID.

        Args:
            user_id: User ID to validate

        Returns:
            Validated user ID

        Raises:
            ValidationError: If user ID is invalid
        """
        if not user_id or not user_id.strip():
            raise ValidationError("User ID cannot be empty")

        user_id = user_id.strip()

        # User ID should be alphanumeric with limited special chars
        if not re.match(r'^[a-zA-Z0-9_\-\.@]+$', user_id):
            raise ValidationError("User ID contains invalid characters")

        if len(user_id) > 255:
            raise ValidationError("User ID too long (max 255 characters)")

        return user_id

    @classmethod
    def validate_tags(cls, tags: List[str]) -> List[str]:
        """
        Validate list of tags.

        Args:
            tags: List of tags to validate

        Returns:
            Validated list of tags

        Raises:
            ValidationError: If any tag is invalid
        """
        if not isinstance(tags, list):
            raise ValidationError("Tags must be a list")

        if len(tags) > 50:
            raise ValidationError("Too many tags (max 50)")

        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError(f"Tag must be string, got {type(tag).__name__}")

            tag = tag.strip()

            if not tag:
                continue  # Skip empty tags

            if len(tag) > 100:
                raise ValidationError(f"Tag too long: {tag[:20]}... (max 100 characters)")

            # Tags should be simple alphanumeric
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', tag):
                raise ValidationError(f"Tag contains invalid characters: {tag}")

            validated_tags.append(tag)

        return validated_tags

    @classmethod
    def validate_json_field(cls, field_name: str, value: Any, expected_type: type) -> Any:
        """
        Validate JSON field with type checking.

        Args:
            field_name: Name of the field
            value: Value to validate
            expected_type: Expected type

        Returns:
            Validated value

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Field '{field_name}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        return value

    @classmethod
    def validate_integer(cls, value: Any, min_val: Optional[int] = None,
                        max_val: Optional[int] = None) -> int:
        """
        Validate integer value with optional range checking.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(f"Expected integer, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValidationError(f"Value {value} below minimum {min_val}")

        if max_val is not None and value > max_val:
            raise ValidationError(f"Value {value} above maximum {max_val}")

        return value
