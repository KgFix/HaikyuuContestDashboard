export interface UserDailySummary {
  date: string;
  bestHighestToday: number;
  clubName: string;
  gameWeek: string;
  lastUpdated: string;
}

export interface ClubDailySummary {
  date: string;
  maxTotalPower: number;
  gameWeek: string;
  lastUpdated: string;
}

export interface ClubDailyActivity {
  date: string;
  users: Record<string, number>;
  totalUsers: number;
  gameWeek: string;
  lastUpdated: string;
}

export type ViewType = 'club' | 'player';

export interface ChartDataPoint {
  date: string;
  value: number;
  label?: string;
}
