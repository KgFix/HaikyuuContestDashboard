import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type { UserDailySummary, ClubDailySummary, ClubDailyActivity } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const TOKEN_STORAGE_KEY = 'haikyuu_auth_token';

// Create axios instance with interceptors
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
  });

  // Request interceptor to add auth token
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      const token = localStorage.getItem(TOKEN_STORAGE_KEY);
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor to handle auth errors
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Unauthorized - clear auth and redirect to login
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem('haikyuu_user_info');
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );

  return client;
};

const apiClient = createApiClient();

export const api = {
  // Get all clubs (only returns clubs user has access to)
  getClubs: async (): Promise<string[]> => {
    const response = await apiClient.get('/api/clubs');
    return response.data;
  },

  // Get all users
  getUsers: async (): Promise<string[]> => {
    const response = await apiClient.get('/api/users');
    return response.data;
  },

  // Get club performance history
  getClubHistory: async (clubName: string): Promise<ClubDailySummary[]> => {
    const response = await apiClient.get(`/api/club/${encodeURIComponent(clubName)}/history`);
    return response.data;
  },

  // Get club activity history
  getClubActivity: async (clubName: string): Promise<ClubDailyActivity[]> => {
    const response = await apiClient.get(`/api/club/${encodeURIComponent(clubName)}/activity`);
    return response.data;
  },

  // Get user performance history
  getUserHistory: async (username: string): Promise<UserDailySummary[]> => {
    const response = await apiClient.get(`/api/user/${encodeURIComponent(username)}/history`);
    return response.data;
  },

  // Get current user info
  getCurrentUser: async () => {
    const response = await apiClient.get('/api/auth/me');
    return response.data;
  },

  // Get user's accessible clubs
  getUserClubs: async (): Promise<string[]> => {
    const response = await apiClient.get('/api/user/clubs');
    return response.data;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; message: string }> => {
    const response = await apiClient.get('/api/health');
    return response.data;
  },
};

