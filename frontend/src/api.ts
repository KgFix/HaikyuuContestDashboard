import axios from 'axios';
import type { UserDailySummary, ClubDailySummary, ClubDailyActivity } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

export const api = {
  // Get all clubs
  getClubs: async (): Promise<string[]> => {
    const response = await axios.get(`${API_BASE_URL}/clubs`);
    return response.data;
  },

  // Get all users
  getUsers: async (): Promise<string[]> => {
    const response = await axios.get(`${API_BASE_URL}/users`);
    return response.data;
  },

  // Get club performance history
  getClubHistory: async (clubName: string): Promise<ClubDailySummary[]> => {
    const response = await axios.get(`${API_BASE_URL}/club/${encodeURIComponent(clubName)}/history`);
    return response.data;
  },

  // Get club activity history
  getClubActivity: async (clubName: string): Promise<ClubDailyActivity[]> => {
    const response = await axios.get(`${API_BASE_URL}/club/${encodeURIComponent(clubName)}/activity`);
    return response.data;
  },

  // Get user performance history
  getUserHistory: async (username: string): Promise<UserDailySummary[]> => {
    const response = await axios.get(`${API_BASE_URL}/user/${encodeURIComponent(username)}/history`);
    return response.data;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; message: string }> => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },
};

