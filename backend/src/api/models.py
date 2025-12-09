"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# ============================================================================
# Response Models (matching frontend types.ts)
# ============================================================================

class UserDailySummary(BaseModel):
    """User's daily performance summary"""
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    bestHighestToday: int = Field(..., description="Best score for the day")
    clubName: str = Field(..., description="Club name")
    gameWeek: str = Field(..., description="Game week (YYYY-Www)")
    lastUpdated: str = Field(..., description="ISO datetime of last update")


class ClubDailySummary(BaseModel):
    """Club's daily performance summary"""
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    maxTotalPower: int = Field(..., description="Maximum total power for the day")
    gameWeek: str = Field(..., description="Game week (YYYY-Www)")
    lastUpdated: str = Field(..., description="ISO datetime of last update")


class ClubDailyActivity(BaseModel):
    """Club's daily activity tracking"""
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    users: Dict[str, int] = Field(..., description="Map of username to best score")
    totalUsers: int = Field(..., description="Total number of active users")
    gameWeek: str = Field(..., description="Game week (YYYY-Www)")
    lastUpdated: str = Field(..., description="ISO datetime of last update")


# ============================================================================
# Authentication Models
# ============================================================================

class DiscordUser(BaseModel):
    """Discord user information"""
    id: str = Field(..., description="Discord user ID")
    username: str = Field(..., description="Discord username")
    discriminator: str = Field(..., description="Discord discriminator")
    avatar: Optional[str] = Field(None, description="Avatar hash")
    display_name: str = Field(..., description="Display name or username")


class UserInfo(BaseModel):
    """Authenticated user information"""
    discord_id: str = Field(..., description="Discord user ID")
    username: str = Field(..., description="Discord username")
    display_name: str = Field(..., description="Display name")
    avatar_url: Optional[str] = Field(None, description="Full avatar URL")
    clubs: List[str] = Field(..., description="List of club names user has access to")


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: UserInfo = Field(..., description="User information")


# ============================================================================
# Request Models
# ============================================================================

class OAuthCallbackRequest(BaseModel):
    """OAuth callback data"""
    code: str = Field(..., description="Authorization code from Discord")


# ============================================================================
# Utility Models
# ============================================================================

class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
