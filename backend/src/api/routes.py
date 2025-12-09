"""
API route handlers
"""
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from src.api.models import (
    UserDailySummary,
    ClubDailySummary,
    ClubDailyActivity,
    UserInfo,
)
from src.api.database import (
    get_user_daily_history,
    get_user_club,
    get_all_users,
    get_club_daily_history,
    get_club_daily_activity,
    get_all_clubs,
)
from src.api.middleware import (
    get_current_user,
    verify_club_access,
    verify_user_access,
    get_accessible_clubs,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# User Endpoints
# ============================================================================

@router.get("/user/me", response_model=UserInfo)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    return UserInfo(
        discord_id=current_user.get('sub'),
        username=current_user.get('username'),
        display_name=current_user.get('display_name'),
        avatar_url=None,  # Could be added to JWT if needed
        clubs=current_user.get('clubs', [])
    )


@router.get("/user/clubs", response_model=List[str])
async def get_user_clubs(current_user: dict = Depends(get_current_user)):
    """Get list of clubs the current user has access to"""
    return get_accessible_clubs(current_user)


@router.get("/users", response_model=List[str])
async def list_users(current_user: dict = Depends(get_current_user)):
    """
    Get list of all users
    Only available to authenticated users
    """
    try:
        users = await get_all_users()
        return users
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.get("/user/{username}/history", response_model=List[UserDailySummary])
async def get_user_history(
    username: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's daily performance history
    Requires authentication and authorization
    """
    # Verify user has access to this user's data
    verify_user_access(current_user, username)
    
    try:
        history = await get_user_daily_history(username)
        return history
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user history"
        )


# ============================================================================
# Club Endpoints
# ============================================================================

@router.get("/clubs", response_model=List[str])
async def list_clubs(current_user: dict = Depends(get_current_user)):
    """
    Get list of clubs the current user has access to
    Only returns clubs the user is authorized to view
    """
    # Return only clubs the user has access to
    return get_accessible_clubs(current_user)


@router.get("/club/{club_name}/history", response_model=List[ClubDailySummary])
async def get_club_history(
    club_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get club's daily performance history
    Requires authentication and club access
    """
    # Verify user has access to this club
    verify_club_access(current_user, club_name)
    
    try:
        history = await get_club_daily_history(club_name)
        return history
    except Exception as e:
        logger.error(f"Error getting club history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve club history"
        )


@router.get("/club/{club_name}/activity", response_model=List[ClubDailyActivity])
async def get_club_activity(
    club_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get club's daily activity history
    Requires authentication and club access
    """
    # Verify user has access to this club
    verify_club_access(current_user, club_name)
    
    try:
        activity = await get_club_daily_activity(club_name)
        return activity
    except Exception as e:
        logger.error(f"Error getting club activity: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve club activity"
        )
