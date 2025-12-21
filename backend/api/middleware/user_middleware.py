"""
User middleware for automatic user_id injection.

Middleware that:
1. Extracts user_id from X-User-ID header or lexi_user_id cookie
2. Creates new anonymous user if neither is present
3. Injects user_id into request.state for downstream handlers
4. Updates user's last_seen timestamp
"""

import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.services.user_store import get_user_store, generate_anonymous_user

logger = logging.getLogger(__name__)


class UserMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic user identification and injection.

    Priority order for user_id:
    1. X-User-ID header (explicit)
    2. lexi_user_id cookie
    3. Create new anonymous user (UUID v4)

    The user_id is injected into request.state.user_id for all downstream handlers.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request and inject user_id.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handler
        """
        user_store = get_user_store()
        user_id = None

        # Priority 1: X-User-ID header
        header_user_id = request.headers.get("X-User-ID")
        if header_user_id:
            user_id = header_user_id
            logger.debug(f"User ID from header: {user_id}")

        # Priority 2: lexi_user_id cookie
        if not user_id:
            cookie_user_id = request.cookies.get("lexi_user_id")
            if cookie_user_id:
                user_id = cookie_user_id
                logger.debug(f"User ID from cookie: {user_id}")

        # Priority 3: Create new anonymous user
        if not user_id:
            new_user = generate_anonymous_user()
            try:
                user_store.create_user(new_user)
                user_id = new_user.user_id
                logger.info(f"Created new anonymous user: {user_id}")
            except Exception as e:
                logger.error(f"Failed to create anonymous user: {e}")
                # Fallback to using the generated ID even if storage failed
                user_id = new_user.user_id

        # Backward compatibility: treat 'default' as anonymous
        if user_id == "default":
            logger.warning("Encountered legacy 'default' user_id, treating as anonymous")
            # Don't change the user_id, just log it

        # Inject user_id into request state
        request.state.user_id = user_id

        # Update last_seen timestamp (non-blocking)
        try:
            user_store.update_last_seen(user_id)
        except Exception as e:
            logger.warning(f"Failed to update last_seen for {user_id}: {e}")

        # Call next handler
        response: Response = await call_next(request)

        return response


def get_user_id_from_request(request: Request) -> str:
    """
    Extract user_id from request state.

    This is a helper function for route handlers.

    Args:
        request: FastAPI request object

    Returns:
        User ID string from request.state.user_id

    Raises:
        AttributeError: If user_id not in request state (middleware not run)
    """
    if not hasattr(request.state, "user_id"):
        raise AttributeError("user_id not found in request.state - is UserMiddleware configured?")

    return request.state.user_id
