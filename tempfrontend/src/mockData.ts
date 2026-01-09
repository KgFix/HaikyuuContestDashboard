import type { UserDailySummary, ClubDailySummary, ClubDailyActivity } from './types';

// Generate dates for the last 30 days
const generateDates = (days: number): string[] => {
  const dates: string[] = [];
  const today = new Date();
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
  }
  
  return dates;
};

const dates = generateDates(30);

// Mock clubs
export const mockClubs = [
  'Karasuno High',
  'Nekoma High',
  'Aoba Johsai',
  'Shiratorizawa Academy',
  'Fukurodani Academy'
];

// Mock users
export const mockUsers = [
  'Hinata_Shoyo',
  'Kageyama_Tobio',
  'Kenma_Kozume',
  'Oikawa_Tooru',
  'Ushijima_Wakatoshi',
  'Bokuto_Kotaro',
  'Kuroo_Tetsuro',
  'Tsukishima_Kei',
  'Nishinoya_Yuu',
  'Sugawara_Koushi'
];

// Generate mock club history data
export const generateClubHistory = (clubName: string): ClubDailySummary[] => {
  const baseValue = Math.floor(Math.random() * 5000000) + 10000000; // 10M - 15M
  
  return dates.map((date, index) => {
    // Add some variation and upward trend
    const variation = Math.random() * 500000 - 250000;
    const trend = index * 50000;
    const value = Math.floor(baseValue + variation + trend);
    
    return {
      date,
      maxTotalPower: value,
      gameWeek: `Week ${Math.floor(index / 7) + 1}`,
      lastUpdated: new Date().toISOString()
    };
  });
};

// Generate mock club activity data
export const generateClubActivity = (clubName: string): ClubDailyActivity[] => {
  const baseUsers = Math.floor(Math.random() * 20) + 30; // 30-50 base users
  
  return dates.map((date, index) => {
    const variation = Math.floor(Math.random() * 10) - 5;
    const totalUsers = Math.max(10, baseUsers + variation);
    
    // Generate random user activity
    const users: Record<string, number> = {};
    const selectedUsers = mockUsers
      .sort(() => Math.random() - 0.5)
      .slice(0, totalUsers);
    
    selectedUsers.forEach(user => {
      users[user] = Math.floor(Math.random() * 1000000) + 500000; // 500K - 1.5M
    });
    
    return {
      date,
      users,
      totalUsers,
      gameWeek: `Week ${Math.floor(index / 7) + 1}`,
      lastUpdated: new Date().toISOString()
    };
  });
};

// Generate mock user history data
export const generateUserHistory = (username: string): UserDailySummary[] => {
  const baseScore = Math.floor(Math.random() * 500000) + 1000000; // 1M - 1.5M
  const clubIndex = Math.floor(Math.random() * mockClubs.length);
  const clubName = mockClubs[clubIndex];
  
  return dates.map((date, index) => {
    // Add some variation and slight upward trend
    const variation = Math.random() * 100000 - 50000;
    const trend = index * 5000;
    const value = Math.floor(baseScore + variation + trend);
    
    return {
      date,
      bestHighestToday: value,
      clubName,
      gameWeek: `Week ${Math.floor(index / 7) + 1}`,
      lastUpdated: new Date().toISOString()
    };
  });
};

// Mock API with simulated delay
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const mockApi = {
  getClubs: async (): Promise<string[]> => {
    await delay(300);
    return mockClubs;
  },

  getUsers: async (): Promise<string[]> => {
    await delay(300);
    return mockUsers;
  },

  getClubHistory: async (clubName: string): Promise<ClubDailySummary[]> => {
    await delay(500);
    return generateClubHistory(clubName);
  },

  getClubActivity: async (clubName: string): Promise<ClubDailyActivity[]> => {
    await delay(500);
    return generateClubActivity(clubName);
  },

  getUserHistory: async (username: string): Promise<UserDailySummary[]> => {
    await delay(500);
    return generateUserHistory(username);
  }
};
