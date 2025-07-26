import axios from 'axios';
import type {
  ChatMessage,
  ChatResponse,
  Session,
  SessionCreate,
  SessionResponse,
  SessionsListResponse,
  TravelPlan,
  TravelPlanResponse,
  ChatHistory,
} from '../types/api';

// Configure axios instance
const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // 30 second timeout for AI responses
  headers: {
    'Content-Type': 'application/json',
  },
});

// API response interceptor for error handling
api.interceptors.response.use(
  (response: any) => response,
  (error: any) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export class ApiService {
  // Session Management
  static async getSessions(): Promise<SessionsListResponse> {
    const response = await api.get('/sessions');
    return response.data;
  }

  static async createSession(sessionData: SessionCreate): Promise<SessionResponse> {
    const response = await api.post('/sessions', sessionData);
    return response.data;
  }

  static async switchSession(sessionId: string): Promise<SessionResponse> {
    const response = await api.put(`/sessions/${sessionId}/switch`);
    return response.data;
  }

  static async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/sessions/${sessionId}`);
  }

  // Chat Management
  static async sendMessage(message: ChatMessage): Promise<ChatResponse> {
    const response = await api.post('/chat', message);
    return response.data;
  }

  static async getChatHistory(sessionId: string): Promise<ChatHistory> {
    const response = await api.get(`/chat/history/${sessionId}`);
    return response.data;
  }

  static async clearChatHistory(sessionId: string): Promise<void> {
    await api.delete(`/chat/history/${sessionId}`);
  }

  // Travel Plans
  static async getTravelPlan(planId: string): Promise<TravelPlanResponse> {
    const response = await api.get(`/plans/${planId}`);
    return response.data;
  }

  static async getSessionPlans(sessionId: string): Promise<TravelPlan[]> {
    try {
      const response = await api.get(`/plans?session_id=${sessionId}`);
      return response.data.plans || [];
    } catch (error) {
      console.warn('No plans found for session:', sessionId);
      return [];
    }
  }

  // System Health
  static async getSystemHealth(): Promise<any> {
    const response = await api.get('/system/status');
    return response.data;
  }
}

export default ApiService; 