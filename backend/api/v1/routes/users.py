"""
User management API routes.

Provides endpoints for:
- User initialization (POST /v1/users/init)
- User profile retrieval (GET /v1/users/me)
- User profile updates (PATCH /v1/users/me)
- User statistics (GET /v1/users/stats)
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import JSONResponse

from backend.models.user import (
    User,
    UserCreateRequest,
    UserUpdateRequest,
    UserResponse,
    UserStatsResponse,
    UserTier
)
from backend.services.user_store import get_user_store, generate_anonymous_user
from backend.api.middleware.user_middleware import get_user_id_from_request
from backend.memory.adapter import get_memory_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/users", tags=["users"])


@router.post("/init", response_model=UserResponse)
async def initialize_user(
    request: Request,
    response: Response,
    user_data: Optional[UserCreateRequest] = None
):
    """
    Initialize a new anonymous user.

    This endpoint:
    1. Creates a new anonymous user with UUID v4
    2. Sets the lexi_user_id cookie
    3. Returns the user object in response body

    The client should store user_id from the response body in LocalStorage.

    Args:
        request: FastAPI request
        response: FastAPI response (for setting cookies)
        user_data: Optional request body with display_name

    Returns:
        UserResponse with created user object
    """
    user_store = get_user_store()

    # Generate new anonymous user
    new_user = generate_anonymous_user()

    # Apply custom display name if provided
    if user_data and user_data.display_name:
        new_user.display_name = user_data.display_name

    # Store in database
    try:
        created_user = user_store.create_user(new_user)
    except ValueError as e:
        # User already exists (UUID collision - extremely rare)
        logger.error(f"UUID collision during user creation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user - please retry")
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Set cookie in response
    # HttpOnly=False allows JavaScript access (needed for LocalStorage sync)
    # SameSite=Lax provides CSRF protection while allowing normal navigation
    # Max-Age=365 days
    response.set_cookie(
        key="lexi_user_id",
        value=created_user.user_id,
        httponly=False,
        samesite="lax",
        max_age=365 * 24 * 60 * 60,  # 365 days in seconds
        path="/"
    )

    logger.info(f"Initialized user {created_user.user_id} with display_name '{created_user.display_name}'")

    return UserResponse(
        user=created_user,
        message="User created successfully"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(request: Request, response: Response):
    """
    Get current user profile.

    Uses user_id from middleware (extracted from cookie or header).

    Args:
        request: FastAPI request with user_id in state

    Returns:
        UserResponse with current user object

    Raises:
        HTTPException: 404 if user not found
    """
    user_id = get_user_id_from_request(request)
    user_store = get_user_store()

    user = user_store.get_user(user_id)

    if user is None:
        # Auto-recover missing user ID by recreating anonymous user
        logger.warning(f"User {user_id} not found in database - recreating anonymous user")
        now = datetime.now(timezone.utc).isoformat()
        user = User(
            user_id=user_id,
            display_name="Anonymous User",
            created_at=now,
            last_seen=now,
            tier=UserTier.ANONYMOUS,
            preferences={},
            email=None
        )
        try:
            user = user_store.create_user(user)
        except Exception as e:
            logger.error(f"Failed to recreate user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to recreate user")

        response.set_cookie(
            key="lexi_user_id",
            value=user.user_id,
            httponly=False,
            samesite="lax",
            max_age=365 * 24 * 60 * 60,
            path="/"
        )

    return UserResponse(user=user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(request: Request, updates: UserUpdateRequest):
    """
    Update current user profile.

    Allows updating:
    - display_name
    - preferences

    Args:
        request: FastAPI request with user_id in state
        updates: UserUpdateRequest with fields to update

    Returns:
        UserResponse with updated user object

    Raises:
        HTTPException: 404 if user not found
    """
    user_id = get_user_id_from_request(request)
    user_store = get_user_store()

    # Build update dictionary (only include non-None fields)
    update_dict = {}
    if updates.display_name is not None:
        update_dict["display_name"] = updates.display_name
    if updates.preferences is not None:
        update_dict["preferences"] = updates.preferences

    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Ensure user exists (auto-recover like /me)
    user = user_store.get_user(user_id)
    if user is None:
        logger.warning(f"User {user_id} not found in database - recreating anonymous user")
        now = datetime.now(timezone.utc).isoformat()
        user = User(
            user_id=user_id,
            display_name="Anonymous User",
            created_at=now,
            last_seen=now,
            tier=UserTier.ANONYMOUS,
            preferences={},
            email=None
        )
        try:
            user_store.create_user(user)
        except Exception as e:
            logger.error(f"Failed to recreate user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to recreate user")

    # Update user
    updated_user = user_store.update_user(user_id, update_dict)

    if updated_user is None:
        logger.warning(f"User {user_id} not found during update")
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Updated user {user_id}: {list(update_dict.keys())}")

    return UserResponse(
        user=updated_user,
        message="User updated successfully"
    )


@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(request: Request):
    """
    Get statistics for current user.

    Returns memory statistics including:
    - Total number of memories
    - Memory count by category
    - Last activity timestamp
    - Account age in days

    Args:
        request: FastAPI request with user_id in state

    Returns:
        UserStatsResponse with statistics

    Raises:
        HTTPException: 404 if user not found
    """
    user_id = get_user_id_from_request(request)
    user_store = get_user_store()

    # Get user to verify existence and get created_at
    user = user_store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Get memory statistics
    try:
        memory_stats = get_memory_stats(user_id)
        total_memories = memory_stats.get("total_memories", 0)
        categories = memory_stats.get("categories", {})
    except Exception as e:
        logger.error(f"Error retrieving memory stats for {user_id}: {e}")
        total_memories = 0
        categories = {}

    # Calculate account age
    try:
        created_at = datetime.fromisoformat(user.created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        account_age_days = (now - created_at).days
    except Exception as e:
        logger.warning(f"Error calculating account age: {e}")
        account_age_days = 0

    # Get last activity (use last_seen from user)
    last_activity = user.last_seen

    return UserStatsResponse(
        user_id=user_id,
        total_memories=total_memories,
        categories=categories,
        last_activity=last_activity,
        account_age_days=account_age_days
    )


@router.delete("/me")
async def delete_current_user(request: Request):
    """
    Delete current user account.

    WARNING: This permanently deletes the user profile.
    Memories associated with this user will remain in the database
    but will become inaccessible.

    Args:
        request: FastAPI request with user_id in state

    Returns:
        Success message

    Raises:
        HTTPException: 404 if user not found
    """
    user_id = get_user_id_from_request(request)
    user_store = get_user_store()

    success = user_store.delete_user(user_id)

    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    logger.warning(f"Deleted user {user_id}")

    return JSONResponse(
        content={
            "message": "User deleted successfully",
            "user_id": user_id
        },
        status_code=200
    )
