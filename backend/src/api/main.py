"""
FastAPI Main Application
"""
import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv

from src.api.models import HealthResponse, TokenResponse, OAuthCallbackRequest, ErrorResponse
from src.api.routes import router as api_router
from src.api import auth
from src.api.middleware import get_current_user

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Haikyuu Contest Dashboard API",
    description="API for Discord-based contest tracking with OAuth authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Configuration
# ============================================================================

# Get allowed origins from environment
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:5173')
allowed_origins = [
    FRONTEND_URL,
    "http://localhost:5173",  # Local development
    "http://localhost:3000",  # Alternative local port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ).model_dump()
    )


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        timestamp=datetime.utcnow()
    )


@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """API health check endpoint (with /api prefix)"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        timestamp=datetime.utcnow()
    )


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.get("/api/auth/discord/login")
async def discord_login():
    """
    Redirect to Discord OAuth login
    Frontend should redirect users to this endpoint to start login flow
    """
    client_id = os.getenv('DISCORD_CLIENT_ID')
    redirect_uri = os.getenv('DISCORD_REDIRECT_URI')
    
    if not client_id or not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Discord OAuth not configured"
        )
    
    # Required scopes for our application
    scopes = ['identify', 'guilds', 'guilds.members.read']
    scope_string = '%20'.join(scopes)
    
    discord_auth_url = (
        f"https://discord.com/api/oauth2/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope={scope_string}"
    )
    
    return RedirectResponse(url=discord_auth_url)


@app.get("/api/auth/discord/callback")
async def discord_callback(code: str):
    """
    OAuth callback endpoint - Discord redirects here after user authorization
    Exchanges code for token and returns JWT
    """
    try:
        # Exchange authorization code for access token
        token_data = await auth.exchange_code_for_token(code)
        access_token = token_data['access_token']
        
        # Get user information
        user_data = await auth.get_discord_user(access_token)
        
        # Get bot token for role verification
        bot_token = os.getenv('DISCORD_BOT_TOKEN')
        
        # Determine which clubs user has access to
        clubs = await auth.determine_user_clubs(access_token, bot_token)
        
        # Create JWT token
        discord_id = user_data['id']
        username = user_data['username']
        display_name = user_data.get('global_name') or username
        
        jwt_data = auth.create_jwt_token(discord_id, username, display_name, clubs)
        
        # Get avatar URL
        avatar_url = auth.get_avatar_url(user_data)
        
        # Prepare response
        user_info = {
            'discord_id': discord_id,
            'username': username,
            'display_name': display_name,
            'avatar_url': avatar_url,
            'clubs': clubs
        }
        
        response = TokenResponse(
            access_token=jwt_data['access_token'],
            token_type=jwt_data['token_type'],
            expires_in=jwt_data['expires_in'],
            user=user_info
        )
        
        # For now, return JSON
        # In production, you might want to redirect to frontend with token in URL
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
        redirect_url = f"{frontend_url}/auth/callback?token={jwt_data['access_token']}"
        
        # Option 1: Redirect to frontend with token (simpler for frontend)
        return RedirectResponse(url=redirect_url)
        
        # Option 2: Return JSON (frontend needs to handle callback URL)
        # return response
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@app.post("/api/auth/logout")
async def logout(current_user: dict = None):
    """
    Logout endpoint
    Client should delete stored JWT token
    """
    return {
        "message": "Logged out successfully",
        "detail": "Please delete your access token on the client side"
    }


@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = get_current_user):
    """Get current authenticated user information"""
    return {
        'discord_id': current_user.get('sub'),
        'username': current_user.get('username'),
        'display_name': current_user.get('display_name'),
        'clubs': current_user.get('clubs', [])
    }


# ============================================================================
# Include API Routes
# ============================================================================

app.include_router(api_router, prefix="/api", tags=["data"])


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Haikyuu Contest Dashboard API")
    logger.info(f"CORS allowed origins: {allowed_origins}")
    
    # Verify required environment variables
    required_vars = [
        'DISCORD_CLIENT_ID',
        'DISCORD_CLIENT_SECRET',
        'DISCORD_REDIRECT_URI',
        'JWT_SECRET_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'DYNAMODB_TABLE_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Haikyuu Contest Dashboard API")


# ============================================================================
# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
