import React, { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SessionSidebar } from './components/Sidebar/SessionSidebar';
import { ChatInterface } from './components/Chat/ChatInterface';
import { TravelCalendar } from './components/Calendar/TravelCalendar';
import { useSessions } from './hooks/useApi';

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
    mutations: {
      retry: 0, // Don't retry failed chat messages
      onError: (error: any) => {
        // Only log actual errors, not timeouts
        if (!error?.code || error.code !== 'ECONNABORTED') {
          console.error('Mutation error:', error);
        }
      },
    },
  },
});

const AppContent: React.FC = () => {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const { data: sessionsData } = useSessions();

  // Auto-select first session if none selected
  useEffect(() => {
    if (!currentSessionId && sessionsData?.sessions && sessionsData.sessions.length > 0) {
      setCurrentSessionId(sessionsData.current_session || sessionsData.sessions[0].session_id);
    }
  }, [sessionsData, currentSessionId]);

  return (
    <div className="h-screen flex bg-gray-100">
      {/* Left Sidebar - Session Management */}
      <SessionSidebar
        currentSessionId={currentSessionId}
        onSessionSelect={setCurrentSessionId}
      />

      {/* Middle Panel - Chat Interface */}
      <div className="flex-1 flex flex-col">
        <ChatInterface sessionId={currentSessionId} />
      </div>

      {/* Right Panel - Travel Calendar */}
      <TravelCalendar sessionId={currentSessionId} />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
};

export default App; 