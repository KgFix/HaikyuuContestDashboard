# API & Discord OAuth Deployment Guide

Complete guide for deploying the Haikyuu Contest Dashboard with Discord OAuth authentication.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Discord OAuth Setup](#discord-oauth-setup)
4. [Backend API Deployment](#backend-api-deployment)
5. [Frontend Deployment](#frontend-deployment)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│           AWS EC2 Instance (t4g.small)             │
│                                                     │
│  ┌─────────────────┐      ┌────────────────────┐  │
│  │  Discord Bot    │      │   FastAPI Server   │  │
│  │  (systemd)      │      │   (systemd)        │  │
│  │  Port: N/A      │      │   Port: 8000       │  │
│  └─────────────────┘      └────────────────────┘  │
│           │                         │              │
│           └────────┬────────────────┘              │
│                    │                               │
│            ┌───────▼──────────┐                    │
│            │   AWS DynamoDB   │                    │
│            └──────────────────┘                    │
└─────────────────────────────────────────────────────┘
                       ▲
                       │ HTTPS
                       │
              ┌────────▼─────────┐
              │  Nginx (Reverse  │
              │  Proxy + SSL)    │
              └──────────────────┘
                       ▲
                       │
              ┌────────▼─────────┐
              │  React Frontend  │
              │  (Vercel/Netlify)│
              └──────────────────┘
```

**Authentication Flow:**
1. User clicks "Login with Discord" on frontend
2. Redirects to Discord OAuth
3. User authorizes → Discord redirects to API callback
4. API verifies user's guild memberships and roles
5. API checks against `activated_channels.json`
6. API generates JWT token with club access
7. Frontend stores JWT and uses for all API calls

---

## Prerequisites

### Required Accounts
- ✅ AWS Account (EC2 + DynamoDB already set up)
- ✅ Discord Developer Account
- ⬜ Domain name (optional but recommended for production)
  - Or use EC2 public IP temporarily

### Required Software (on local machine)
- Python 3.11+
- Node.js 18+
- Git
- SSH client
- Text editor

---

## Discord OAuth Setup

### Step 1: Create Discord Application

1. Go to https://discord.com/developers/applications
2. Click **"New Application"**
3. Name it: `Haikyuu Contest Dashboard` (or your preferred name)
4. Click **Create**

### Step 2: Configure OAuth2

1. Navigate to **OAuth2** → **General** in left sidebar
2. Click **Add Redirect** under "Redirects"
3. Add redirects:
   ```
   # For local development
   http://localhost:8000/api/auth/discord/callback
   
   # For production (replace with your domain/IP)
   https://your-ec2-ip-or-domain/api/auth/discord/callback
   ```
4. Click **Save Changes**

### Step 3: Get Credentials

1. Copy **Client ID** (you'll need this)
2. Click **Reset Secret** → **Yes, do it!**
3. Copy **Client Secret** (you'll need this - save it securely!)

⚠️ **Never commit Client Secret to git!**

---

## Backend API Deployment

### Step 1: Update .env File

SSH into your EC2 instance:
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-ip
cd ~/discord-bot
```

Edit your `.env` file:
```bash
nano .env
```

Add these new variables (keep existing ones):
```env
# Existing variables (keep these)
DISCORD_BOT_TOKEN=your_existing_bot_token
AWS_ACCESS_KEY_ID=your_existing_key
AWS_SECRET_ACCESS_KEY=your_existing_secret
AWS_REGION=your_region
DYNAMODB_TABLE_NAME=your_table_name

# NEW - Discord OAuth
DISCORD_CLIENT_ID=your_client_id_from_step_3
DISCORD_CLIENT_SECRET=your_client_secret_from_step_3
DISCORD_REDIRECT_URI=http://your-ec2-ip:8000/api/auth/discord/callback

# NEW - JWT Configuration
JWT_SECRET_KEY=generate_with_command_below
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# NEW - API Configuration
API_PORT=8000
FRONTEND_URL=http://localhost:5173
```

Generate JWT secret:
```bash
openssl rand -hex 32
```
Copy the output and paste it as `JWT_SECRET_KEY` value.

Save and exit: `Ctrl+X` → `Y` → `Enter`

### Step 2: Install New Dependencies

```bash
# Activate virtual environment
source ~/discord-bot/venv/bin/activate

# Install new packages
pip install python-jose[cryptography]==3.3.0 python-multipart==0.0.9 pydantic==2.10.5 pydantic-settings==2.7.1

# Or reinstall all from updated requirements.txt
pip install -r requirements.txt
```

### Step 3: Upload New API Code

**From your local machine** (in PowerShell), upload the new files:

```powershell
# Navigate to your project directory
cd "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\backend"

# Upload new src/api directory
scp -i "your-key.pem" -r src/api ubuntu@your-ec2-ip:~/discord-bot/src/

# Upload new src/shared directory
scp -i "your-key.pem" -r src/shared ubuntu@your-ec2-ip:~/discord-bot/src/

# Upload updated requirements.txt
scp -i "your-key.pem" requirements.txt ubuntu@your-ec2-ip:~/discord-bot/
```

### Step 4: Create API Systemd Service

Back on EC2:
```bash
sudo nano /etc/systemd/system/api.service
```

Add this content:
```ini
[Unit]
Description=Haikyuu Contest Dashboard API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord-bot
Environment="PATH=/home/ubuntu/discord-bot/venv/bin"
ExecStart=/home/ubuntu/discord-bot/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit: `Ctrl+X` → `Y` → `Enter`

### Step 5: Update Security Group

1. Go to AWS EC2 Console
2. Select your instance
3. Click **Security** tab
4. Click on the security group link
5. Click **Edit inbound rules**
6. Click **Add rule**:
   - Type: `Custom TCP`
   - Port range: `8000`
   - Source: `0.0.0.0/0` (or restrict to your IPs)
   - Description: `API Server`
7. Click **Save rules**

### Step 6: Start API Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable API service (start on boot)
sudo systemctl enable api

# Start API service
sudo systemctl start api

# Check status
sudo systemctl status api

# View logs
sudo journalctl -u api -f
```

### Step 7: Test API

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Should return: {"status":"healthy","message":"API is running",...}
```

From your local machine:
```powershell
curl http://your-ec2-ip:8000/api/health
```

---

## Frontend Deployment

### Option A: Local Development First

1. **Install dependencies:**
```powershell
cd "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\frontend"
npm install
```

2. **Create `.env` file:**
```powershell
# Create .env file
New-Item -Path ".env" -ItemType File

# Edit with notepad
notepad .env
```

Add:
```env
VITE_API_BASE_URL=http://your-ec2-ip:8000
```

3. **Run development server:**
```powershell
npm run dev
```

4. **Test locally:**
- Open http://localhost:5173
- Click "Login with Discord"
- Should redirect to Discord → Authorize → Redirect back

### Option B: Deploy to Vercel (Recommended)

1. **Install Vercel CLI:**
```powershell
npm install -g vercel
```

2. **Login to Vercel:**
```powershell
vercel login
```

3. **Deploy:**
```powershell
cd "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\frontend"
vercel
```

Follow prompts:
- Set up and deploy: `Y`
- Scope: (select your account)
- Link to existing project: `N`
- Project name: `haikyuu-contest-dashboard`
- Directory: `./`
- Override settings: `N`

4. **Set environment variables in Vercel:**
```powershell
vercel env add VITE_API_BASE_URL
```
Enter: `http://your-ec2-ip:8000`

5. **Deploy to production:**
```powershell
vercel --prod
```

6. **Update Discord OAuth redirect:**
- Go to Discord Developer Portal
- Add new redirect: `https://your-vercel-url.vercel.app/auth/callback`
- Update `DISCORD_REDIRECT_URI` in backend `.env`
- Update `FRONTEND_URL` in backend `.env`
- Restart API: `sudo systemctl restart api`

---

## Setup Nginx + SSL (Optional but Recommended)

### Install Nginx

```bash
sudo apt update
sudo apt install nginx -y
```

### Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/api
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # or your EC2 IP

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Install SSL (if you have a domain)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

Follow prompts to get free SSL certificate.

---

## Testing

### Test Backend

1. **Health check:**
```bash
curl http://your-ec2-ip:8000/api/health
```

2. **OAuth flow:**
- Visit: `http://your-ec2-ip:8000/api/auth/discord/login`
- Should redirect to Discord

3. **Check logs:**
```bash
sudo journalctl -u api -f
```

### Test Frontend

1. **Open in browser:**
   - Local: http://localhost:5173
   - Production: Your Vercel URL

2. **Test login flow:**
   - Click "Login with Discord"
   - Authorize on Discord
   - Should redirect back to dashboard
   - Should see your clubs in header

3. **Test dashboard:**
   - Select a club you have access to
   - Should see charts and data
   - Try switching between Club and Player views

---

## Troubleshooting

### "Discord OAuth not configured" Error

**Solution:** Check backend `.env` has all Discord variables set.

```bash
# Verify variables
cat ~/discord-bot/.env | grep DISCORD
```

Should show:
- DISCORD_BOT_TOKEN
- DISCORD_CLIENT_ID
- DISCORD_CLIENT_SECRET
- DISCORD_REDIRECT_URI

### "Invalid redirect_uri" from Discord

**Solution:** 
1. Check Discord Developer Portal → OAuth2 → Redirects
2. Ensure exact match: `http://your-ip:8000/api/auth/discord/callback`
3. No trailing slash!

### API returns 401 Unauthorized

**Solutions:**
- Check if JWT is stored in localStorage (F12 → Application → Local Storage)
- Try logging out and back in
- Check API logs: `sudo journalctl -u api -f`

### "You do not have access to club" Error

**Solutions:**
1. Verify you're in the Discord server
2. Verify the channel was activated with `/activatecontestsubmissions @YourRole`
3. Verify you have the role mentioned
4. Check `activated_channels.json`:
```bash
cat ~/discord-bot/activated_channels.json
```

### CORS Errors in Browser

**Solution:** Update `FRONTEND_URL` in backend `.env`:
```env
FRONTEND_URL=https://your-vercel-url.vercel.app
```

Restart API:
```bash
sudo systemctl restart api
```

### API Service Won't Start

**Check logs:**
```bash
sudo journalctl -u api -xe
```

**Common issues:**
- Missing environment variables
- Python import errors (reinstall dependencies)
- Port 8000 already in use

---

## Quick Commands Reference

### Backend (EC2)
```bash
# View API logs
sudo journalctl -u api -f

# Restart API
sudo systemctl restart api

# Check API status
sudo systemctl status api

# View environment variables
cat ~/discord-bot/.env

# Test local API
curl http://localhost:8000/api/health
```

### Frontend (Local)
```powershell
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Deploy to Vercel
vercel --prod
```

---

## Security Checklist

- [ ] JWT_SECRET_KEY is randomly generated (32+ characters)
- [ ] Discord Client Secret is never committed to git
- [ ] `.env` file is in `.gitignore`
- [ ] CORS is configured to only allow your frontend domain
- [ ] SSL/HTTPS is enabled (for production)
- [ ] Security group restricts access appropriately
- [ ] API rate limiting is configured (built into FastAPI)

---

## Next Steps

1. ✅ Test entire flow locally
2. ✅ Deploy backend API to EC2
3. ✅ Deploy frontend to Vercel
4. ⬜ Get domain name (optional)
5. ⬜ Setup SSL with Let's Encrypt
6. ⬜ Configure monitoring/alerts
7. ⬜ Setup automated backups

---

## Support

If you encounter issues:
1. Check logs: `sudo journalctl -u api -f`
2. Verify environment variables
3. Test API endpoints with curl
4. Check browser console (F12) for frontend errors
5. Review Discord Developer Portal settings

---

**Created:** December 2025  
**Version:** 1.0  
**Author:** GitHub Copilot
