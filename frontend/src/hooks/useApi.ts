import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import ApiService from '../services/api';
import type { ChatMessage, SessionCreate } from '../types/api';

// Query keys for cache management
export const queryKeys = {
  sessions: ['sessions'] as const,
  chatHistory: (sessionId: string) => ['chatHistory', sessionId] as const,
  travelPlans: (sessionId: string) => ['travelPlans', sessionId] as const,
  planStatus: (sessionId: string) => ['planStatus', sessionId] as const,
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
    queryFn: async () => {
      if (!sessionId) return null;
      
      try {
        const plan = await ApiService.getSessionPlan(sessionId);
        console.log('Received travel plan:', plan);
        
        if (plan && plan.events) {
          plan.events.forEach((event: any, index: number) => {
            console.log(`Event ${index}:`, {
              title: event.title,
              start: event.start_time || event.start,
              end: event.end_time || event.end,
              type: event.event_type || event.type
            });
          });
        }
        
        return plan;
      } catch (error) {
        console.error('Error fetching travel plan:', error);
        return null;
      }
    },
    enabled: !!sessionId,
    staleTime: 30000, // 30 seconds
  });
};

// Plan generation status hook
export const usePlanGenerationStatus = (sessionId: string | null) => {
  return useQuery({
    queryKey: queryKeys.planStatus(sessionId || ''),
    queryFn: () => sessionId ? ApiService.getPlanGenerationStatus(sessionId) : null,
    enabled: !!sessionId,
    refetchInterval: (query) => {
      // Stop polling if plan generation is completed or failed
      if (!query.state.data || !sessionId) return false;
      const status = query.state.data.plan_generation_status;
      return (status === 'pending' || status === 'unknown') ? 2000 : false; // Poll every 2 seconds if pending
    },
    staleTime: 1000, // 1 second
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