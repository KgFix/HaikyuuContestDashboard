# Haikyuu Contest Dashboard - Temporary Test Frontend

This is a temporary standalone React frontend for testing the analytics dashboard UI with mock data.

## Features

- 📊 **Club Performance View**: View club total power over time and daily active users
- 👤 **Player Performance View**: Track individual player scores over time
- 🎲 **Mock Data**: Uses generated test data matching the production schema
- 📈 **Interactive Charts**: Powered by Recharts with responsive design
- 🎨 **Identical UI**: Matches the production frontend design

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3001`

### Build for Production

```bash
npm run build
```

## Mock Data

The application uses mock data generators that create realistic test data:

- **5 Mock Clubs**: Karasuno High, Nekoma High, Aoba Johsai, Shiratorizawa Academy, Fukurodani Academy
- **10 Mock Players**: Hinata_Shoyo, Kageyama_Tobio, Kenma_Kozume, etc.
- **30 Days of Historical Data**: Performance trends with realistic variations
- **Club Activity**: Simulated daily active user counts and power levels
- **Player Scores**: Individual performance tracking with upward trends

## Copying to Production

When ready to integrate with the real backend:

1. Copy UI components from `src/Dashboard.tsx` and `src/PerformanceChart.tsx` to the production frontend
2. Replace `mockApi` imports with real API calls
3. Update authentication and authorization logic
4. Migrate CSS styles from `src/Dashboard.css`

## Project Structure

```
tempfrontend/
├── src/
│   ├── App.tsx              # Main app component
│   ├── Dashboard.tsx        # Dashboard with mock data
│   ├── PerformanceChart.tsx # Recharts chart component
│   ├── mockData.ts          # Mock data generators
│   ├── types.ts             # TypeScript interfaces
│   └── *.css                # Styling
├── package.json
└── vite.config.ts
```

## Notes

- This is a **temporary testing environment** for UI development
- No authentication or real database connections
- Mock data simulates realistic patterns for testing
- Designed to be easily portable to the production frontend
