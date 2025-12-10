# Update Deployment Script
# Run this after making OAuth changes

Write-Host "=== Updating Discord OAuth Configuration ===" -ForegroundColor Green

# Step 1: Update Discord Developer Portal
Write-Host "`n1. Update Discord Developer Portal:" -ForegroundColor Yellow
Write-Host "   - Go to: https://discord.com/developers/applications/1445878524962410688/oauth2"
Write-Host "   - Under 'Redirects', replace the Netlify URL with:"
Write-Host "     http://13.50.5.247:8000/api/auth/discord/callback" -ForegroundColor Cyan
Write-Host "   - Click 'Save Changes'"
Read-Host "`nPress Enter when done"

# Step 2: Update EC2 .env
Write-Host "`n2. Updating EC2 backend .env..." -ForegroundColor Yellow
$envContent = Get-Content "backend\.env" -Raw
Set-Content "backend\.env.temp" -Value $envContent

Write-Host "   Copying .env to EC2..."
scp -i "HaikyuuBotKey.pem" backend\.env ubuntu@13.50.5.247:~/discord-bot/.env

Write-Host "   Restarting API service..."
ssh -i "HaikyuuBotKey.pem" ubuntu@13.50.5.247 "sudo systemctl restart api"

Write-Host "   Checking API status..."
ssh -i "HaikyuuBotKey.pem" ubuntu@13.50.5.247 "sudo systemctl status api --no-pager"

# Step 3: Redeploy frontend
Write-Host "`n3. Redeploying frontend to Netlify..." -ForegroundColor Yellow
git add .
git commit -m "Fix OAuth redirect URI configuration"
git push origin main

Write-Host "`n=== Deployment Complete! ===" -ForegroundColor Green
Write-Host "`nTest the flow:"
Write-Host "1. Visit: https://haikyuucontestdashboard.netlify.app"
Write-Host "2. Click 'Login with Discord'"
Write-Host "3. Authorize on Discord"
Write-Host "4. You should be redirected to the dashboard"
