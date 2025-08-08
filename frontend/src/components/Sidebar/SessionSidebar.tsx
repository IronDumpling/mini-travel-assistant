import React, { useState } from 'react';
import { Plus, MessageSquare, Trash2, Settings, User } from 'lucide-react';
import { useSessions, useCreateSession, useSwitchSession, useDeleteSession } from '../../hooks/useApi';
import type { Session } from '../../types/api';

interface SessionSidebarProps {
  currentSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  currentSessionId,
  onSessionSelect,
}) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newSessionTitle, setNewSessionTitle] = useState('');

  const { data: sessionsData, isLoading } = useSessions();
  const createSessionMutation = useCreateSession();
  const switchSessionMutation = useSwitchSession();
  const deleteSessionMutation = useDeleteSession();

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSessionTitle.trim()) return;

    try {
      const response = await createSessionMutation.mutateAsync({
        title: newSessionTitle,
        description: `Travel planning session created on ${new Date().toLocaleDateString()}`,
      });
      
      onSessionSelect(response.session_id);
      setNewSessionTitle('');
      setShowCreateForm(false);
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const handleSwitchSession = async (sessionId: string) => {
    try {
      await switchSessionMutation.mutateAsync(sessionId);
      onSessionSelect(sessionId);
    } catch (error) {
      console.error('Failed to switch session:', error);
    }
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this session?')) return;

    try {
      await deleteSessionMutation.mutateAsync(sessionId);
      if (currentSessionId === sessionId) {
        // If deleting current session, switch to another one
        const remainingSessions = sessionsData?.sessions.filter(s => s.session_id !== sessionId);
        if (remainingSessions && remainingSessions.length > 0) {
          onSessionSelect(remainingSessions[0].session_id);
        } else {
          onSessionSelect('');
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="w-80 bg-gray-900 text-white h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <User className="w-5 h-5" />
          <span className="font-semibold">Travel Assistant</span>
        </div>
        
        <button
          onClick={() => setShowCreateForm(true)}
          className="w-full flex items-center gap-2 p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>New Session</span>
        </button>
      </div>

      {/* Create Session Form */}
      {showCreateForm && (
        <div className="p-4 bg-gray-800 border-b border-gray-700">
          <form onSubmit={handleCreateSession} className="space-y-3">
            <input
              type="text"
              value={newSessionTitle}
              onChange={(e) => setNewSessionTitle(e.target.value)}
              placeholder="Session title..."
              className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={!newSessionTitle.trim() || createSessionMutation.isPending}
                className="flex-1 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm transition-colors"
              >
                {createSessionMutation.isPending ? 'Creating...' : 'Create'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowCreateForm(false);
                  setNewSessionTitle('');
                }}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 text-center text-gray-400">Loading sessions...</div>
        ) : sessionsData?.sessions.length === 0 ? (
          <div className="p-4 text-center text-gray-400">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No sessions yet</p>
            <p className="text-sm">Create your first travel planning session</p>
          </div>
        ) : (
          <div className="p-2">
            {sessionsData?.sessions.map((session: Session) => (
              <div
                key={session.session_id}
                onClick={() => handleSwitchSession(session.session_id)}
                className={`group relative p-3 rounded-lg cursor-pointer transition-colors mb-2 ${
                  session.session_id === currentSessionId
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 hover:bg-gray-700'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium truncate">{session.title}</h3>
                    <p className="text-xs text-gray-400 mt-1">
                      {formatDate(session.updated_at)}
                    </p>
                    {session.description && (
                      <p className="text-xs text-gray-500 mt-1 truncate">
                        {session.description}
                      </p>
                    )}
                  </div>
                  
                  <button
                    onClick={(e) => handleDeleteSession(session.session_id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-600 rounded transition-all"
                    title="Delete session"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                
                {session.session_id === currentSessionId && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-white rounded-r"></div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <Settings className="w-4 h-4" />
          <span>DeepSeek Powered</span>
        </div>
      </div>
    </div>
  );
}; 