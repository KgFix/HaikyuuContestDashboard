#!/bin/bash
# ============================================================================
# AWS EC2 Deployment Script for API Service
# ============================================================================
# This script sets up and starts the API service on your EC2 instance
# Run this on your EC2 instance after uploading the code
# ============================================================================

set -e  # Exit on error

echo "🚀 Starting API Service Deployment..."

# Step 1: Copy service file
echo "📝 Installing systemd service file..."
sudo cp /home/ubuntu/discord-bot/api.service /etc/systemd/system/api.service

# Step 2: Reload systemd
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Step 3: Enable API service (start on boot)
echo "✅ Enabling API service..."
sudo systemctl enable api

# Step 4: Start API service
echo "🚀 Starting API service..."
sudo systemctl start api

# Step 5: Wait a moment for service to start
sleep 2

# Step 6: Check status
echo ""
echo "📊 Service Status:"
echo "=================="
sudo systemctl status discord-bot --no-pager
echo ""
sudo systemctl status api --no-pager

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📝 Useful Commands:"
echo "==================="
echo "View API logs:        sudo journalctl -u api -f"
echo "View bot logs:        sudo journalctl -u discord-bot -f"
echo "Restart API:          sudo systemctl restart api"
echo "Restart bot:          sudo systemctl restart discord-bot"
echo "Check API status:     sudo systemctl status api"
echo "Check bot status:     sudo systemctl status discord-bot"
echo ""
echo "🧪 Test API:"
echo "============"
echo "curl http://localhost:8000/api/health"
echo ""
