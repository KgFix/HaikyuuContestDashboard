"""
FastAPI middleware for JWT authentication and authorization
"""
import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.api.auth import verify_jwt_token

logger = logging.getLogger(__name__)

# Security scheme for Swagger UI
security = HTTPBearer()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token and return user payload
    Use as dependency in protected routes
    """
    token = credentials.credentials
    payload = verify_jwt_token(token)
    return payload


async def get_optional_user(request: Request) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise None
    Use for optional authentication
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        return payload
    except HTTPException:
        return None


# ============================================================================
# Authorization Helpers
# ============================================================================

def verify_club_access(user: dict, club_name: str) -> None:
    """
    Verify user has access to specific club
    Raises HTTPException if unauthorized
    """
    allowed_clubs = user.get('clubs', [])
    
    if club_name not in allowed_clubs:
        logger.warning(
            f"User {user.get('username')} attempted to access club {club_name} "
            f"(allowed: {allowed_clubs})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have access to club: {club_name}"
        )


def verify_user_access(current_user: dict, requested_username: str) -> None:
    """
    Verify user can access specific user data
    Users can access:
    1. Their own data (username matches)
    2. Any user in their accessible clubs
    
    Note: This is a simplified check. For full authorization,
    you'd need to verify the requested user is in one of the current user's clubs.
    """
    # Allow access to own data
    if current_user.get('username') == requested_username:
        return
    
    # For now, if user has any club access, allow viewing other users
    # In production, you'd want to verify the requested user belongs to an accessible club
    if current_user.get('clubs'):
        return
    
    logger.warning(
        f"User {current_user.get('username')} attempted to access user {requested_username}"
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"You do not have access to user data: {requested_username}"
    )


def get_accessible_clubs(user: dict) -> List[str]:
    """Get list of clubs user can access"""
    return user.get('clubs', [])
