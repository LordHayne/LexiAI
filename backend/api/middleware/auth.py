"""
Authentication middleware for the Lexi API.
"""
from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import APIKeyHeader
from typing import Optional
import jwt
from datetime import datetime, timedelta

from backend.config.auth_config import SecurityConfig

# API Key security scheme
api_key_header = APIKeyHeader(name=SecurityConfig.API_KEY_HEADER, auto_error=False)

async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    authorization: Optional[str] = Header(None)
):
    """
    Verify API key or JWT token if enabled.
    This function is used as a dependency for protected API endpoints.

    SECURITY: UI endpoints are NOT automatically bypassed.
    Use verify_ui_auth() for UI endpoints instead.
    """
    # Skip auth for health check only
    if request.url.path == "/v1/health":
        return True

    # If all auth methods are disabled, allow all requests
    # WARNING: This should only be used in development
    if not SecurityConfig.API_KEY_ENABLED and not SecurityConfig.JWT_ENABLED:
        import os
        if os.environ.get("ENV", "production").lower() in ("development", "dev"):
            return True
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured on this server"
            )

    # Check API Key if enabled
    if SecurityConfig.API_KEY_ENABLED:
        if api_key and SecurityConfig.verify_api_key(api_key):
            return True

    # Check JWT if enabled and API key validation failed
    if SecurityConfig.JWT_ENABLED:
        token = None

        # 1. Try Authorization header (backwards compatibility)
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() != "bearer":
                    token = None
            except ValueError:
                pass

        # 2. Try HttpOnly Cookie (preferred for XSS protection)
        if not token:
            token = request.cookies.get("access_token")

        # Verify JWT token if found
        if token:
            try:
                payload = jwt.decode(
                    token,
                    SecurityConfig.JWT_SECRET,
                    algorithms=[SecurityConfig.JWT_ALGORITHM]
                )
                # Check if token is expired
                exp = payload.get("exp")
                if exp and datetime.utcfromtimestamp(exp) > datetime.utcnow():
                    return True
            except jwt.PyJWTError:
                pass

    # If all auth methods failed, raise 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def verify_ui_auth(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    authorization: Optional[str] = Header(None)
):
    """
    Optional authentication for UI endpoints.

    SECURITY: UI endpoints can be optionally protected based on configuration.
    - If LEXI_UI_AUTH_REQUIRED=True: Requires authentication
    - If LEXI_UI_AUTH_REQUIRED=False (default): Allows access but logs

    This allows flexibility for development while enforcing security in production.
    """
    import os

    # Check if UI authentication is required
    ui_auth_required = os.environ.get("LEXI_UI_AUTH_REQUIRED", "False").lower() in ("true", "1", "yes")

    if not ui_auth_required:
        # Log unauthenticated access for security monitoring
        import logging
        logger = logging.getLogger("lexi_middleware.auth")
        logger.info(f"Unauthenticated UI access: {request.url.path} from {request.client.host if request.client else 'unknown'}")
        return False  # Not authenticated but allowed

    # If UI auth is required, use same logic as API auth
    try:
        return await verify_api_key(request, api_key, authorization)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="UI authentication required. Please provide valid credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

def create_jwt_token(user_id: str, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT token with an optional expiration time.
    """
    if not SecurityConfig.JWT_ENABLED:
        raise ValueError("JWT authentication is not enabled")
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=SecurityConfig.JWT_EXPIRATION)
    
    # Create payload
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    # Encode and return token
    return jwt.encode(
        payload,
        SecurityConfig.JWT_SECRET,
        algorithm=SecurityConfig.JWT_ALGORITHM
    )
