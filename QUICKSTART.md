# API Quick Start Guide

Fastest way to get the API running locally for testing.

## Prerequisites
- Backend Discord bot is already working
- Have Discord Developer application created

## Steps

### 1. Update .env File (backend folder)

Add these lines to your existing `backend/.env`:

```env
# Discord OAuth (get from https://discord.com/developers/applications)
DISCORD_CLIENT_ID=your_client_id_here
DISCORD_CLIENT_SECRET=your_client_secret_here
DISCORD_REDIRECT_URI=http://localhost:8000/api/auth/discord/callback

# JWT (generate with: openssl rand -hex 32 in Git Bash)
JWT_SECRET_KEY=generate_a_random_32_character_string_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API Config
API_PORT=8000
FRONTEND_URL=http://localhost:5173
```

### 2. Install Backend Dependencies

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install python-jose[cryptography] python-multipart pydantic pydantic-settings
```

### 3. Configure Discord OAuth

1. Go to https://discord.com/developers/applications
2. Select your application
3. Go to OAuth2 → General
4. Add redirect URL: `http://localhost:8000/api/auth/discord/callback`
5. Save changes

### 4. Run API Server

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 5. Test API

Open browser to: http://localhost:8000/docs

You should see the Swagger API documentation.

### 6. Setup Frontend

```powershell
cd frontend
npm install
npm install react-router-dom
```

Create `frontend/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

### 7. Run Frontend

```powershell
cd frontend
npm run dev
```

### 8. Test Complete Flow

1. Open http://localhost:5173
2. Click "Login with Discord"
3. Authorize on Discord
4. Should redirect back and show your clubs

## Troubleshooting

**"Missing required environment variables"**
- Check all variables in backend/.env are set

**"Discord OAuth not configured"**
- Verify DISCORD_CLIENT_ID and DISCORD_CLIENT_SECRET in .env
- Make sure no extra spaces

**"CORS error"**
- Make sure API is running on port 8000
- Check FRONTEND_URL in backend/.env

**"No clubs shown after login"**
- Make sure you've used `/activatecontestsubmissions @RoleName` in Discord
- Check that you have the role in the Discord server

## File Structure

After setup, you should have:
```
backend/
  ├── src/
  │   ├── api/          ← NEW API code
  │   │   ├── main.py
  │   │   ├── auth.py
  │   │   ├── routes.py
  │   │   └── ...
  │   ├── shared/       ← NEW shared utilities
  │   └── discord_bot.py
  └── .env              ← UPDATED with new variables

frontend/
  ├── src/
  │   ├── auth/         ← NEW authentication
  │   └── ...
  └── .env              ← NEW file
```

## Common Commands

```powershell
# Backend
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --reload  # Run API

# Frontend  
cd frontend
npm run dev  # Run development server

# Check API health
curl http://localhost:8000/api/health
```

Done! 🎉
