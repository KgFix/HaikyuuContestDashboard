# Complete Deployment Guide - EC2 + Netlify

## Prerequisites Checklist

- [ ] EC2 instance running with Discord bot
- [ ] Frontend deployed to Netlify
- [ ] Discord Application created in Developer Portal
- [ ] Backend code uploaded to EC2

## Required Information

Before starting, gather these:

1. **EC2 Public IP:** `_____________` (from AWS Console → EC2 → Instances)
2. **Netlify URL:** `_____________` (from Netlify dashboard, e.g., https://your-app.netlify.app)
3. **Discord Client ID:** (already in `.env`)
4. **Discord Client Secret:** (already in `.env`)

---

## Part 1: Update Discord OAuth Settings

### Step 1: Update Redirect URIs in Discord Developer Portal

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application (ID: `1445878524962410688`)
3. Click **OAuth2** → **General**
4. Under **Redirects**, add:
   ```
   https://YOUR-NETLIFY-URL.netlify.app/auth/callback
   http://YOUR-EC2-IP:8000/api/auth/discord/callback
   ```
5. Click **Save Changes**

---

## Part 2: Update Backend Configuration on EC2

### Step 2: SSH into EC2

```bash
ssh -i "your-key.pem" ubuntu@YOUR-EC2-IP
```

### Step 3: Update `.env` File

```bash
cd ~/discord-bot
nano .env
```

**Update these lines:**
```env
DISCORD_REDIRECT_URI=https://YOUR-NETLIFY-URL.netlify.app/auth/callback
FRONTEND_URL=https://YOUR-NETLIFY-URL.netlify.app
```

Save with `Ctrl+X`, then `Y`, then `Enter`.

### Step 4: Deploy API Service

```bash
# Make deploy script executable
chmod +x deploy-api.sh

# Run deployment
./deploy-api.sh
```

This will:
- Install the systemd service file
- Enable both services to start on boot
- Start the API service
- Show status of both services

---

## Part 3: Update AWS Security Group

### Step 5: Add Port 8000 to Security Group

1. Go to **AWS Console** → **EC2** → **Instances**
2. Select your instance
3. Click **Security** tab
4. Click on the **Security Group** link
5. Click **Edit inbound rules**
6. Click **Add rule**:
   - **Type:** Custom TCP
   - **Port range:** 8000
   - **Source:** 0.0.0.0/0
   - **Description:** API Server
7. Click **Save rules**

---

## Part 4: Testing

### Step 6: Test API from EC2

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Expected response:
# {"status":"healthy","message":"API is running",...}
```

### Step 7: Test API from Your Computer

```powershell
# From Windows PowerShell
curl http://YOUR-EC2-IP:8000/api/health
```

### Step 8: Test Full OAuth Flow

1. Open your Netlify URL in browser: `https://YOUR-NETLIFY-URL.netlify.app`
2. Click **"Login with Discord"**
3. Authorize the application
4. Should redirect back successfully and show dashboard

### Step 9: Verify Discord Bot Still Works

```bash
# On EC2
sudo systemctl status discord-bot
sudo journalctl -u discord-bot -f
```

Test a bot command in Discord (e.g., `/activate_reminder`)

---

## Part 5: Update Frontend Environment Variable

### Step 10: Update Netlify Environment Variable

1. Go to [Netlify Dashboard](https://app.netlify.com/)
2. Select your site
3. Go to **Site settings** → **Environment variables**
4. Update or add:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `http://YOUR-EC2-IP:8000`
5. Click **Save**

### Step 11: Redeploy Frontend

1. Go to **Deploys** tab
2. Click **Trigger deploy** → **Deploy site**
3. Wait for deployment to complete

---

## Quick Reference Commands

### Service Management
```bash
# Check status
sudo systemctl status discord-bot
sudo systemctl status api

# View logs
sudo journalctl -u discord-bot -f
sudo journalctl -u api -f

# Restart services
sudo systemctl restart discord-bot
sudo systemctl restart api

# Stop services
sudo systemctl stop discord-bot
sudo systemctl stop api
```

### Debugging
```bash
# Check if port 8000 is listening
sudo lsof -i :8000

# Check recent API errors
sudo journalctl -u api -xe

# Check bot errors
sudo journalctl -u discord-bot -xe

# Test API locally
curl http://localhost:8000/api/health
curl http://localhost:8000/api/auth/discord
```

---

## Common Issues & Solutions

### Issue: API won't start
```bash
# Check for errors
sudo journalctl -u api -xe

# Common cause: Port already in use
sudo lsof -i :8000
# Kill the process: sudo kill <PID>

# Then restart
sudo systemctl restart api
```

### Issue: OAuth fails with "redirect_uri mismatch"
1. Check Discord Developer Portal redirects match exactly
2. Verify `.env` has correct `DISCORD_REDIRECT_URI`
3. Make sure no trailing slashes in URLs
4. Restart API after changes: `sudo systemctl restart api`

### Issue: Frontend can't reach API
1. Check Security Group has port 8000 open
2. Verify `VITE_API_BASE_URL` in Netlify is correct
3. Test API manually: `curl http://YOUR-EC2-IP:8000/api/health`
4. Check browser console for CORS errors

### Issue: Discord bot stopped working
```bash
# Check if still running
sudo systemctl status discord-bot

# Restart
sudo systemctl restart discord-bot

# Check logs for errors
sudo journalctl -u discord-bot -f
```

---

## Architecture Summary

```
┌─────────────────┐
│   Netlify       │
│   (Frontend)    │  ← User visits this
└────────┬────────┘
         │
         │ OAuth flow & API calls
         ↓
┌─────────────────┐
│   AWS EC2       │
│  ┌───────────┐  │
│  │ API :8000 │  │ ← FastAPI + OAuth
│  └───────────┘  │
│  ┌───────────┐  │
│  │Discord Bot│  │ ← Commands & reminders
│  └───────────┘  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   DynamoDB      │  ← Contest data storage
└─────────────────┘
```

---

## Next Steps (Optional Improvements)

1. **Set up a domain name:**
   - Buy domain (e.g., from Route 53, Namecheap)
   - Point to EC2 IP
   - Use Let's Encrypt for SSL

2. **Add HTTPS to API:**
   - Install nginx as reverse proxy
   - Get SSL certificate with Certbot
   - Update all URLs to https://

3. **Set up monitoring:**
   - Enable CloudWatch for EC2
   - Set up alerts for service failures
   - Monitor API response times

4. **Implement proper secrets management:**
   - Use AWS Secrets Manager
   - Rotate Discord bot token regularly
   - Use IAM roles instead of access keys

---

## Support

If you encounter issues:
1. Check the relevant section in "Common Issues & Solutions"
2. Review service logs: `sudo journalctl -u api -f`
3. Verify all URLs and IDs match between Discord Portal, EC2 `.env`, and Netlify

**Remember:** Both services (bot + API) run on the same EC2 instance and share the same `.env` and data files.
