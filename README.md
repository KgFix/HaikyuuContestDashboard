# Haikyuu Contest Dashboard

A full-stack application that tracks Haikyuu volleyball game contest performance through Discord integration. The system uses a Discord bot to automatically process match screenshots via OCR, stores player and club statistics in DynamoDB, and displays real-time performance metrics through a React dashboard with interactive visualizations.

## Backend - Discord Bot

The Discord bot processes screenshots using OCR and stores data in DynamoDB.


cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/discord_bot.py


Configure `backend/.env`:
```env
DISCORD_BOT_TOKEN=your_token
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=your_region
DYNAMODB_TABLE_NAME=your_table
```

## Frontend - React Dashboard Template

A clean React template with pre-built dashboard components. **Connect it to your own backend/database.**

### Features
- React 19 + TypeScript
- Recharts for line graph visualizations
- Pre-built dashboard with club/player views
- Responsive design
- Ready to customize

### Setup
```bash
cd frontend
npm install
npm run dev
```

Visit: http://localhost:5173

### Customize Your API Connection

Edit `frontend/src/api.ts` to connect to your own backend endpoints. The template includes:
- Dashboard component with toggle views
- Performance chart component (line graphs)
- TypeScript types for club/player data
- Example API structure

## Project Structure

```
HaikyuuContestDashboardProject/
├── backend/
│   ├── src/
│   │   └── discord_bot.py      # Discord bot with OCR
│   └── requirements.txt
└── frontend/                    # React template
    ├── src/
    │   ├── components/
    │   │   ├── Dashboard.tsx    # Main dashboard UI
    │   │   └── PerformanceChart.tsx  # Line chart component
    │   ├── api.ts              # API client (customize this)
    │   ├── types.ts            # TypeScript types
    │   └── App.tsx
    └── package.json
```

## Technologies

- **Backend**: Python, Discord.py, OpenCV, EasyOCR, Boto3, DynamoDB
- **Frontend**: React 19, TypeScript, Recharts, Axios, Vite

---

**Note:** Backend and frontend are independent. The bot collects data, the frontend is a template you customize to connect to your own data source.


Bot Hosting:
1. Start and Enable Service
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable discord-bot

# Start service now
sudo systemctl start discord-bot

# Check status
sudo systemctl status discord-bot

2. Monitor Bot
# View live logs
sudo journalctl -u discord-bot -f

# View last 100 lines
sudo journalctl -u discord-bot -n 100

# Restart bot
sudo systemctl restart discord-bot

# Stop bot
sudo systemctl stop discord-bot

3. Useful Commands
# SSH back in anytime
ssh -i "haikyuu-bot-key.pem" ubuntu@YOUR_EC2_IP

# Update bot code (after making changes locally):
# 1. Transfer new files via SCP
# 2. Restart service:
sudo systemctl restart discord-bot

# Check resource usage
htop  # (install with: sudo apt install htop)

# View disk space
df -h

# Check memory usage
free -h