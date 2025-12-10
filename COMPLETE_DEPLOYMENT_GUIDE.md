# Complete Deployment Guide - EC2 + Netlify

## Prerequisites Checklist

- [ ] EC2 instance running with Discord bot
- [ ] Frontend deployed to Netlify
- [ ] Discord Application created in Developer Portal
- [ ] Backend code uploaded to EC2

## Required Information

✅ All information ready:

1. **EC2 Public IP:** `13.50.5.247`
2. **Netlify URL:** `https://haikyuucontestdashboard.netlify.app`
3. **Discord Client ID:** `1445878524962410688` (already in `.env`)
4. **Discord Client Secret:** (already in `.env`)
5. **SSH Key:** `D:\All\Kirone\Projects\HaikyuuContestDashboardProject\HaikyuuBotKey.pem`

---

## Part 1: Update Discord OAuth Settings

### Step 1: Update Redirect URIs in Discord Developer Portal

1. Go to [Discord Developer Portal](https://discord.com/developers/applications/1445878524962410688/oauth2)
2. Click **OAuth2** → **General**
3. Under **Redirects**, add:
   ```
   https://haikyuucontestdashboard.netlify.app/auth/callback
   ```
4. Click **Save Changes**

---

## Part 2: Deploy to EC2

### Step 2: Run Automated Deployment Script

From your Windows machine, run:

```powershell
cd "D:\All\Kirone\Projects\HaikyuuContestDashboardProject"
.\deploy-to-ec2.ps1
```

This will automatically:
- Upload updated `.env` file with Netlify URL
- Upload `api.service` systemd file
- Upload `deploy-api.sh` script
- Upload API source files
- SSH into EC2 and run deployment
- Install and start the API service
- Show status of both bot and API services

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

### Step 3: Test API from EC2

If you want to manually verify on the EC2 instance:

```bash
ssh -i "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\HaikyuuBotKey.pem" ubuntu@13.50.5.247

# Test health endpoint
curl http://localhost:8000/api/health

# Expected response:
# {"status":"healthy","message":"API is running",...}
```

### Step 4: Test API from Your Computer

```powershell
# From Windows PowerShell
curl http://13.50.5.247:8000/api/health
```

### Step 5: Test Full OAuth Flow

1. Open your Netlify URL in browser: `https://haikyuucontestdashboard.netlify.app`
2. Click **"Login with Discord"**
3. Authorize the application
4. Should redirect back successfully and show dashboard

### Step 6: Verify Discord Bot Still Works

```bash
ssh -i "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\HaikyuuBotKey.pem" ubuntu@13.50.5.247

# On EC2
sudo systemctl status discord-bot
sudo journalctl -u discord-bot -f
```

Test a bot command in Discord (e.g., `/activate_reminder`)

---

## Part 5: Update Frontend Environment Variable

### Step 7: Update Netlify Environment Variable

1. Go to [Netlify Dashboard](https://app.netlify.com/)
2. Select your site: **haikyuucontestdashboard**
3. Go to **Site settings** → **Environment variables**
4. Update or add:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `http://13.50.5.247:8000`
5. Click **Save**

### Step 8: Redeploy Frontend

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
