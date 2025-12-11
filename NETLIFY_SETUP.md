# Netlify Environment Configuration

## Problem
When clicking "Login with Discord" on the deployed Netlify site, you're getting a malformed URL that flashes and then returns to /login.

## Root Cause
The `VITE_API_BASE_URL` environment variable is not properly configured in Netlify, causing the frontend to use the wrong API URL.

## Solution

### Step 1: Configure Netlify Environment Variable

1. Go to [Netlify Dashboard](https://app.netlify.com/)
2. Select your site: **haikyuucontestdashboard**
3. Go to **Site settings** → **Environment variables**
4. Click **Add a variable** or **Add new variable**
5. Set the following:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com`
   - **Scopes**: Check both "Production" and "Deploy previews"
6. Click **Create variable** or **Save**

### Step 2: Trigger a Redeploy

After adding the environment variable:

1. Go to **Deploys** tab
2. Click **Trigger deploy** → **Clear cache and deploy site**
3. Wait for the deployment to complete

### Step 3: Verify the Fix

1. Visit your site: https://haikyuucontestdashboard.netlify.app/
2. Click "Login with Discord"
3. You should be redirected to Discord's OAuth page
4. After authorizing, you should be redirected back to your dashboard

## Important Notes

### About Cloudflare Tunnel URLs
- Your current tunnel URL: `https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com`
- Cloudflare Tunnel URLs from the free plan **change on every restart**
- This means you'll need to update the Netlify environment variable each time you restart your tunnel

### For a Permanent Solution
Consider one of these options:

1. **Get a named Cloudflare Tunnel** (requires Cloudflare account):
   - This gives you a permanent URL that doesn't change
   - Follow: [Cloudflare Tunnel Guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)

2. **Deploy backend to AWS EC2** (as per your existing script):
   - Get a permanent domain/IP
   - See: `COMPLETE_DEPLOYMENT_GUIDE.md`

3. **Use ngrok with a static domain** (paid):
   - Get a permanent ngrok URL

## Current Configuration

### Backend Environment Variables (on EC2/local)
```bash
DISCORD_CLIENT_ID=your_client_id
DISCORD_CLIENT_SECRET=your_client_secret
DISCORD_REDIRECT_URI=https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com/api/auth/discord/callback
FRONTEND_URL=https://haikyuucontestdashboard.netlify.app
DISCORD_BOT_TOKEN=your_bot_token
JWT_SECRET_KEY=your_jwt_secret
```

### Frontend Environment Variables (on Netlify)
```bash
VITE_API_BASE_URL=https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com
```

## Troubleshooting

### If login still doesn't work:

1. **Check browser console** for errors:
   - Press F12
   - Go to Console tab
   - Try logging in again
   - Share any error messages

2. **Verify the API is accessible**:
   - Visit: `https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com/health`
   - Should return: `{"status":"healthy","message":"API is running","timestamp":"..."}`

3. **Check Discord OAuth settings**:
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Select your application
   - Go to OAuth2 → Redirects
   - Ensure this URL is listed: `https://haikyuu-api.enforcement-correct-developing-virtue.trycloudflare.com/api/auth/discord/callback`

4. **Check Network tab**:
   - F12 → Network tab
   - Try logging in
   - Look for failed requests or CORS errors

## Testing Locally

To test the full flow locally before deploying:

```bash
# Terminal 1 - Backend
cd backend
python -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

Then visit: http://localhost:5173
