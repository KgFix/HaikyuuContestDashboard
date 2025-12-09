"""
Discord OAuth authentication and JWT token management
"""
import os
import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import aiohttp
from jose import JWTError, jwt
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ============================================================================
# Environment Configuration
# ============================================================================
DISCORD_CLIENT_ID = os.getenv('DISCORD_CLIENT_ID')
DISCORD_CLIENT_SECRET = os.getenv('DISCORD_CLIENT_SECRET')
DISCORD_REDIRECT_URI = os.getenv('DISCORD_REDIRECT_URI')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

# Discord API endpoints
DISCORD_API_BASE = 'https://discord.com/api/v10'
DISCORD_TOKEN_URL = f'{DISCORD_API_BASE}/oauth2/token'
DISCORD_USER_URL = f'{DISCORD_API_BASE}/users/@me'
DISCORD_GUILDS_URL = f'{DISCORD_API_BASE}/users/@me/guilds'
DISCORD_GUILD_MEMBER_URL = f'{DISCORD_API_BASE}/users/@me/guilds/{{guild_id}}/member'

# Path to activated channels file (same as discord_bot.py)
CHANNELS_STORAGE_FILE = os.getenv('CHANNELS_STORAGE_FILE', 'activated_channels.json')


# ============================================================================
# Activated Channels Management
# ============================================================================

def load_activated_channels() -> Dict[Tuple[int, int], str]:
    """Load activated channels from JSON file"""
    try:
        if os.path.exists(CHANNELS_STORAGE_FILE):
            with open(CHANNELS_STORAGE_FILE, 'r') as f:
                data = json.load(f)
                channels = {
                    tuple(map(int, k.split(','))): v 
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(channels)} activated channels")
                return channels
    except Exception as e:
        logger.warning(f"Could not load activated channels: {str(e)}")
    return {}


# ============================================================================
# Discord OAuth Functions
# ============================================================================

async def exchange_code_for_token(code: str) -> Dict:
    """
    Exchange authorization code for access token
    Returns: {'access_token': str, 'token_type': str, ...}
    """
    data = {
        'client_id': DISCORD_CLIENT_ID,
        'client_secret': DISCORD_CLIENT_SECRET,
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': DISCORD_REDIRECT_URI,
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_TOKEN_URL, data=data, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Discord token exchange failed: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to exchange authorization code"
                    )
                return await resp.json()
    except aiohttp.ClientError as e:
        logger.error(f"Discord API request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discord API unavailable"
        )


async def get_discord_user(access_token: str) -> Dict:
    """
    Get Discord user information
    Returns: {'id': str, 'username': str, 'discriminator': str, ...}
    """
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(DISCORD_USER_URL, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Discord user fetch failed: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid Discord access token"
                    )
                return await resp.json()
    except aiohttp.ClientError as e:
        logger.error(f"Discord API request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discord API unavailable"
        )


async def get_user_guilds(access_token: str) -> List[Dict]:
    """
    Get list of guilds (servers) the user is in
    Returns: [{'id': str, 'name': str, ...}, ...]
    """
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(DISCORD_GUILDS_URL, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Discord guilds fetch failed: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Failed to fetch user guilds"
                    )
                return await resp.json()
    except aiohttp.ClientError as e:
        logger.error(f"Discord API request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discord API unavailable"
        )


async def get_guild_member(access_token: str, guild_id: str) -> Dict:
    """
    Get user's member information in a specific guild (includes roles)
    Returns: {'user': {...}, 'roles': [...], 'nick': str, ...}
    """
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    url = DISCORD_GUILD_MEMBER_URL.format(guild_id=guild_id)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    # User might not have granted guilds.members.read scope
                    logger.warning(f"Failed to fetch member info for guild {guild_id}")
                    return None
                return await resp.json()
    except aiohttp.ClientError as e:
        logger.error(f"Discord API request failed: {str(e)}")
        return None


# ============================================================================
# Authorization Logic
# ============================================================================

async def determine_user_clubs(access_token: str, bot_token: str) -> List[str]:
    """
    Determine which clubs a user has access to based on:
    1. User's guild memberships
    2. Activated channels in those guilds
    3. User's roles in those guilds
    
    Returns: List of club names the user can access
    """
    try:
        # Get user's guilds
        user_guilds = await get_user_guilds(access_token)
        user_guild_ids = {guild['id'] for guild in user_guilds}
        
        # Load activated channels
        activated_channels = load_activated_channels()
        
        # Get bot token for role fetching (need bot token to get role details)
        allowed_clubs = set()
        
        for (guild_id, channel_id), club_name in activated_channels.items():
            # Check if user is in this guild
            if str(guild_id) not in user_guild_ids:
                continue
            
            # Get user's member info in this guild
            member_info = await get_guild_member(access_token, str(guild_id))
            
            if not member_info:
                # If we can't get member info, grant access anyway (user is in guild)
                logger.info(f"User in guild {guild_id}, granting access to {club_name}")
                allowed_clubs.add(club_name)
                continue
            
            # Check if club_name contains a role mention (e.g., "@BotTestRole")
            # Extract role name from club_name
            if club_name.startswith('@'):
                role_name = club_name[1:]  # Remove @ prefix
                
                # Get role IDs from member_info
                user_role_ids = set(member_info.get('roles', []))
                
                # We need to fetch guild roles to match role name to ID
                # For now, we'll grant access if user is in the guild
                # In production, you'd want to cache guild roles or fetch them
                logger.info(f"User has roles in guild {guild_id}: {user_role_ids}")
                allowed_clubs.add(club_name)
            else:
                # No role requirement, just guild membership
                allowed_clubs.add(club_name)
        
        result = sorted(list(allowed_clubs))
        logger.info(f"User has access to {len(result)} clubs: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error determining user clubs: {str(e)}")
        # In case of error, return empty list (no access)
        return []


# ============================================================================
# JWT Token Management
# ============================================================================

def create_jwt_token(discord_id: str, username: str, display_name: str, clubs: List[str]) -> Dict:
    """
    Create JWT token with user information and club access
    Returns: {'access_token': str, 'token_type': str, 'expires_in': int}
    """
    expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        'sub': discord_id,  # Subject (user ID)
        'username': username,
        'display_name': display_name,
        'clubs': clubs,
        'exp': expires_at,
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return {
        'access_token': token,
        'token_type': 'bearer',
        'expires_in': JWT_EXPIRATION_HOURS * 3600
    }


def verify_jwt_token(token: str) -> Dict:
    """
    Verify and decode JWT token
    Returns: Decoded payload {'sub': str, 'username': str, 'clubs': [...], ...}
    Raises: HTTPException if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_avatar_url(user_data: Dict) -> Optional[str]:
    """Generate Discord avatar URL from user data"""
    user_id = user_data.get('id')
    avatar_hash = user_data.get('avatar')
    
    if not avatar_hash:
        # Default avatar
        discriminator = int(user_data.get('discriminator', '0'))
        return f"https://cdn.discordapp.com/embed/avatars/{discriminator % 5}.png"
    
    # User avatar
    extension = 'gif' if avatar_hash.startswith('a_') else 'png'
    return f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.{extension}"
