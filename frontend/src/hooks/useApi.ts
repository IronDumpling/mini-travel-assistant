import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import ApiService from '../services/api';
import type { ChatMessage, SessionCreate } from '../types/api';

// Query keys for cache management
export const queryKeys = {
  sessions: ['sessions'] as const,
  chatHistory: (sessionId: string) => ['chatHistory', sessionId] as const,
  travelPlans: (sessionId: string) => ['travelPlans', sessionId] as const,
  systemHealth: ['systemHealth'] as const,
};

// Sessions hooks
export const useSessions = () => {
  return useQuery({
    queryKey: queryKeys.sessions,
    queryFn: ApiService.getSessions,
    staleTime: 30000, // 30 seconds
  });
};

export const useCreateSession = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (sessionData: SessionCreate) => ApiService.createSession(sessionData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions });
    },
  });
};

export const useSwitchSession = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (sessionId: string) => ApiService.switchSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions });
    },
  });
};

export const useDeleteSession = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (sessionId: string) => ApiService.deleteSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions });
    },
  });
};

// Chat hooks
export const useChatHistory = (sessionId: string | null) => {
  return useQuery({
    queryKey: queryKeys.chatHistory(sessionId || ''),
    queryFn: () => sessionId ? ApiService.getChatHistory(sessionId) : null,
    enabled: !!sessionId,
    staleTime: 10000, // 10 seconds
  });
};

export const useSendMessage = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (message: ChatMessage) => ApiService.sendMessage(message),
    onSuccess: (_: any, variables: ChatMessage) => {
      // Invalidate chat history for the session
      if (variables.session_id) {
        queryClient.invalidateQueries({ 
          queryKey: queryKeys.chatHistory(variables.session_id) 
        });
        // Also invalidate travel plans as they might be updated
        queryClient.invalidateQueries({ 
          queryKey: queryKeys.travelPlans(variables.session_id) 
        });
      }
    },
  });
};

// Travel plans hooks
export const useTravelPlans = (sessionId: string | null) => {
  return useQuery({
    queryKey: queryKeys.travelPlans(sessionId || ''),
    queryFn: () => sessionId ? ApiService.getSessionPlans(sessionId) : [],
    enabled: !!sessionId,
    staleTime: 30000, // 30 seconds
  });
};

// System health hook
export const useSystemHealth = () => {
  return useQuery({
    queryKey: queryKeys.systemHealth,
    queryFn: ApiService.getSystemHealth,
    staleTime: 60000, // 1 minute
    retry: 1,
  });
}; 