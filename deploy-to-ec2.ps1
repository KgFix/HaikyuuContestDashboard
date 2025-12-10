# ============================================================================
# Complete Deployment Script for EC2
# ============================================================================
# This script uploads all necessary files to EC2 and deploys the API service
# Run this from your Windows machine in PowerShell
# ============================================================================

$EC2_IP = "13.50.5.247"
$KEY_PATH = "D:\All\Kirone\Projects\HaikyuuContestDashboardProject\HaikyuuBotKey.pem"
$SSH_USER = "ubuntu"
$PROJECT_PATH = "D:\All\Kirone\Projects\HaikyuuContestDashboardProject"

Write-Host "🚀 Starting EC2 Deployment..." -ForegroundColor Green
Write-Host ""

# Step 1: Upload updated .env file
Write-Host "📤 Uploading .env configuration..." -ForegroundColor Yellow
scp -i $KEY_PATH "$PROJECT_PATH\backend\.env" "${SSH_USER}@${EC2_IP}:~/discord-bot/.env"

# Step 2: Upload API service file
Write-Host "📤 Uploading api.service..." -ForegroundColor Yellow
scp -i $KEY_PATH "$PROJECT_PATH\backend\api.service" "${SSH_USER}@${EC2_IP}:~/discord-bot/"

# Step 3: Upload deployment script
Write-Host "📤 Uploading deploy-api.sh..." -ForegroundColor Yellow
scp -i $KEY_PATH "$PROJECT_PATH\backend\deploy-api.sh" "${SSH_USER}@${EC2_IP}:~/discord-bot/"

# Step 4: Upload API source files (if there were changes)
Write-Host "📤 Uploading API source files..." -ForegroundColor Yellow
scp -i $KEY_PATH -r "$PROJECT_PATH\backend\src\api\*" "${SSH_USER}@${EC2_IP}:~/discord-bot/src/api/"

Write-Host ""
Write-Host "✅ Files uploaded successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🔧 Now deploying API service on EC2..." -ForegroundColor Yellow
Write-Host ""

# Step 5: SSH and run deployment
ssh -i $KEY_PATH "${SSH_USER}@${EC2_IP}" @"
cd ~/discord-bot
chmod +x deploy-api.sh
./deploy-api.sh
"@

Write-Host ""
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Update Discord Developer Portal with:" -ForegroundColor White
Write-Host "   https://haikyuucontestdashboard.netlify.app/auth/callback" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Update AWS Security Group to allow port 8000" -ForegroundColor White
Write-Host "   (See SECURITY_GROUP_SETUP.md for instructions)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Update Netlify environment variable:" -ForegroundColor White
Write-Host "   VITE_API_BASE_URL = http://13.50.5.247:8000" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Test API:" -ForegroundColor White
Write-Host "   curl http://13.50.5.247:8000/api/health" -ForegroundColor Gray
Write-Host ""
