# Free Permanent Cloudflare Tunnel Setup Guide

This guide will help you set up a **FREE permanent Cloudflare Tunnel** with a custom subdomain that never changes.

## Prerequisites

- A Cloudflare account (free)
- A domain name (optional - you can use a free Cloudflare subdomain)
- Your backend running locally or on a server

## Step-by-Step Setup

### Step 1: Create a Cloudflare Account

1. Go to [Cloudflare](https://dash.cloudflare.com/sign-up)
2. Sign up for a free account
3. Verify your email

### Step 2: Install Cloudflare Tunnel (cloudflared)

#### For Windows (PowerShell):

```powershell
# Download the latest version
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$env:USERPROFILE\cloudflared.exe"

# Add to PATH (optional but recommended)
# Move to a permanent location
New-Item -ItemType Directory -Path "C:\Program Files\cloudflared" -Force
Move-Item -Path "$env:USERPROFILE\cloudflared.exe" -Destination "C:\Program Files\cloudflared\cloudflared.exe" -Force

# Add to system PATH
$env:Path += ";C:\Program Files\cloudflared"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
```

Or simply:
```powershell
# Using winget (if you have Windows Package Manager)
winget install --id Cloudflare.cloudflared
```

#### Verify Installation:

```powershell
cloudflared --version
```

### Step 3: Authenticate with Cloudflare

```powershell
cloudflared tunnel login
```

This will:
1. Open your browser
2. Ask you to select a domain (or you can skip if you don't have one)
3. Download a certificate to your computer
4. Certificate location: `~/.cloudflared/cert.pem` (or `C:\Users\YourName\.cloudflared\cert.pem`)

### Step 4: Create a Named Tunnel

```powershell
# Create a tunnel with a permanent name
cloudflared tunnel create haikyuu-api

# This will output:
# - Tunnel ID (save this!)
# - Credentials file location: ~/.cloudflared/<tunnel-id>.json
```

**Important:** Save the Tunnel ID that's displayed!

### Step 5: Configure the Tunnel

Create a configuration file at `~/.cloudflared/config.yml` (or `C:\Users\YourName\.cloudflared\config.yml`):

```powershell
# Create the config file
New-Item -ItemType File -Path "$env:USERPROFILE\.cloudflared\config.yml" -Force
notepad "$env:USERPROFILE\.cloudflared\config.yml"
```

Add this content to the file:

```yaml
tunnel: haikyuu-api
credentials-file: C:\Users\YourName\.cloudflared\<YOUR-TUNNEL-ID>.json

ingress:
  - hostname: haikyuu-api.trycloudflare.com
    service: http://localhost:8000
  - service: http_status:404
```

**Replace:**
- `<YOUR-TUNNEL-ID>` with your actual tunnel ID from Step 4
- `YourName` with your Windows username
- `haikyuu-api.trycloudflare.com` will be assigned automatically (or you can use a custom domain - see below)

### Step 6: Get Your Permanent Tunnel URL

```powershell
# List your tunnels
cloudflared tunnel list

# Route your tunnel (this assigns a permanent URL)
cloudflared tunnel route dns haikyuu-api haikyuu-api
```

This will give you a permanent URL like: `https://haikyuu-api.trycloudflare.com` or `https://haikyuu-api.<your-domain>.com`

### Step 7: Run the Tunnel

```powershell
# Run the tunnel
cloudflared tunnel run haikyuu-api
```

Or run it in the background:

```powershell
cloudflared tunnel --config "$env:USERPROFILE\.cloudflared\config.yml" run haikyuu-api
```

### Step 8: Set Up as a Windows Service (Optional - Auto-start on boot)

To make the tunnel start automatically when Windows boots:

```powershell
# Install as a service
cloudflared service install

# The service will use the config file at: C:\Users\YourName\.cloudflared\config.yml
# Make sure your config.yml is properly set up before this step
```

Start the service:

```powershell
# Start the service
sc start cloudflared

# Check service status
sc query cloudflared
```

To stop the service:

```powershell
sc stop cloudflared
```

To uninstall the service:

```powershell
cloudflared service uninstall
```

## Alternative: Using a Custom Domain (Free)

If you want a custom domain instead of `.trycloudflare.com`:

### Option 1: Use Cloudflare's Free Subdomain

Cloudflare provides free `*.pages.dev` subdomains:

1. Go to Cloudflare Dashboard
2. Workers & Pages → Create
3. You'll get a subdomain like `your-project.pages.dev`
4. Use this in your tunnel config

### Option 2: Register a Free Domain

Use services like:
- **Freenom** (free .tk, .ml, .ga, .cf, .gq domains)
- **DuckDNS** (free subdomain)
- **No-IP** (free subdomain)

Then add the domain to Cloudflare and use it with your tunnel.

## Update Your Configuration

### Update Backend .env

Once you have your permanent tunnel URL (e.g., `https://haikyuu-api.trycloudflare.com`):

```bash
DISCORD_REDIRECT_URI=https://haikyuu-api.trycloudflare.com/api/auth/discord/callback
FRONTEND_URL=https://haikyuucontestdashboard.netlify.app
```

### Update Discord Developer Portal

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application
3. Go to OAuth2 → Redirects
4. Add: `https://haikyuu-api.trycloudflare.com/api/auth/discord/callback`
5. Save changes

### Update Netlify Environment Variable

1. Go to Netlify Dashboard → Your Site
2. Site settings → Environment variables
3. Update `VITE_API_BASE_URL` to: `https://haikyuu-api.trycloudflare.com`
4. Redeploy

## Testing Your Tunnel

### Test 1: Check Tunnel Status

```powershell
cloudflared tunnel list
```

Should show your tunnel with status "ACTIVE"

### Test 2: Test API Endpoint

```powershell
# Test health endpoint
curl https://haikyuu-api.trycloudflare.com/health
```

Should return:
```json
{"status":"healthy","message":"API is running","timestamp":"..."}
```

### Test 3: Test from Browser

Visit: `https://haikyuu-api.trycloudflare.com/health`

## Complete Startup Script

Create a file `start-tunnel.ps1`:

```powershell
# Start Backend API
Write-Host "Starting Backend API..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; python -m uvicorn src.api.main:app --reload --port 8000"

# Wait for API to start
Start-Sleep -Seconds 3

# Start Cloudflare Tunnel
Write-Host "Starting Cloudflare Tunnel..." -ForegroundColor Green
cloudflared tunnel run haikyuu-api
```

Run it:

```powershell
.\start-tunnel.ps1
```

## Troubleshooting

### Issue: "tunnel credentials file not found"

```powershell
# Check if credentials file exists
Get-ChildItem "$env:USERPROFILE\.cloudflared\"

# Make sure config.yml points to the correct credentials file
```

### Issue: "tunnel already exists"

```powershell
# List existing tunnels
cloudflared tunnel list

# Use existing tunnel or delete and recreate
cloudflared tunnel delete haikyuu-api
cloudflared tunnel create haikyuu-api
```

### Issue: Tunnel not accessible

1. Check if backend is running:
   ```powershell
   curl http://localhost:8000/health
   ```

2. Check tunnel logs:
   ```powershell
   cloudflared tunnel run haikyuu-api --loglevel debug
   ```

3. Check firewall settings (Windows Defender might block connections)

### Issue: CORS errors

Make sure your backend `.env` has:
```bash
FRONTEND_URL=https://haikyuucontestdashboard.netlify.app
```

And your backend `main.py` has proper CORS configuration (which it should already have).

## Benefits of Permanent Tunnel

✅ **Free forever**
✅ **Permanent URL** - Never changes, even after restart
✅ **No port forwarding** needed
✅ **Automatic HTTPS**
✅ **Works behind NAT/Firewall**
✅ **Auto-reconnects** if connection drops
✅ **Can run as Windows Service** for auto-start

## Cost Comparison

| Solution | Cost | Permanent URL | Setup Difficulty |
|----------|------|---------------|------------------|
| Cloudflare Tunnel (Free) | $0/month | ✅ Yes | Easy |
| Cloudflare Tunnel (Named) | $0/month | ✅ Yes | Medium |
| ngrok Free | $0/month | ❌ No | Easy |
| ngrok Static | $8/month | ✅ Yes | Easy |
| AWS EC2 (t2.micro) | ~$8-10/month | ✅ Yes | Hard |

## Next Steps

1. Complete the setup above
2. Update your `.env` files with the permanent URL
3. Update Discord OAuth redirect URIs
4. Update Netlify environment variables
5. Test the complete authentication flow
6. Set up as Windows Service for auto-start (optional)

## Questions?

If you run into any issues during setup, let me know which step is causing problems!
