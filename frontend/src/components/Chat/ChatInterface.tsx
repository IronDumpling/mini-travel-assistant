import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { useChatHistory, useSendMessage } from '../../hooks/useApi';
import type { ChatMessage as ChatMessageType } from '../../types/api';

interface ChatInterfaceProps {
  sessionId: string | null;
}

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  confidence?: number;
  isLoading?: boolean;
  plan_changes?: {
    success: boolean;
    changes_made: string[];
    events_added: number;
    events_updated: number;
    events_deleted: number;
    metadata_updated: boolean;
    plan_modifications?: {
      reason: string;
      impact: string;
    };
    error?: string;
  };
}

interface LoadingMessageProps {
  processingStartTime: number | null;
  showDetailedProgress: boolean;
}

const LoadingMessage: React.FC<LoadingMessageProps> = ({ processingStartTime, showDetailedProgress }) => {
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (!processingStartTime) return;

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - processingStartTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [processingStartTime]);

  const getProgressMessage = () => {
    if (elapsedTime < 10) return "Analyzing your request...";
    if (elapsedTime < 30) return "Planning your travel itinerary...";
    if (elapsedTime < 60) return "Researching destinations and options...";
    if (elapsedTime < 90) return "Refining recommendations with AI...";
    return "Finalizing your personalized travel plan...";
  };

  return (
    <div className="flex items-start gap-2 text-blue-600">
      <Loader2 className="w-4 h-4 animate-spin mt-0.5" />
      <div className="flex-1">
        <div className="font-medium">{getProgressMessage()}</div>
        {showDetailedProgress && (
          <div className="text-sm text-gray-500 mt-1">
            Processing time: {elapsedTime}s
            {elapsedTime > 30 && (
              <div className="text-xs mt-1">
                ✨ The AI is using advanced refinement to create the best travel plan for you
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const ChatMessage: React.FC<MessageProps> = ({ 
  role, 
  content, 
  timestamp, 
  confidence,
  isLoading = false,
  plan_changes  // Add plan_changes parameter
}) => {
  const formatTimestamp = (ts?: string) => {
    if (!ts) return '';
    return new Date(ts).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className={`flex gap-3 p-4 ${role === 'assistant' ? 'bg-gray-50' : 'bg-white'}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        role === 'assistant' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'
      }`}>
        {role === 'assistant' ? <Bot className="w-4 h-4" /> : <User className="w-4 h-4" />}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium text-gray-900">
            {role === 'assistant' ? 'Travel Assistant' : 'You'}
          </span>
          {timestamp && (
            <span className="text-xs text-gray-500">{formatTimestamp(timestamp)}</span>
          )}
          {confidence !== undefined && (
            <div className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-500" />
              <span className="text-xs text-gray-500">{Math.round(confidence * 100)}%</span>
            </div>
          )}
        </div>
        
        <div className="prose prose-sm max-w-none">
          {isLoading ? (
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Thinking...</span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap text-gray-800">{content}</div>
          )}
        </div>

        {/* Plan Changes Notification */}
        {role === 'assistant' && plan_changes && plan_changes.success && plan_changes.changes_made.length > 0 && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5" />
              <div className="flex-1">
                <div className="font-medium text-blue-800 text-sm mb-1">Travel Plan Updated</div>
                <ul className="text-sm text-blue-700 space-y-1">
                  {plan_changes.changes_made.map((change, idx) => (
                    <li key={idx} className="flex items-center gap-1">
                      <span className="w-1 h-1 bg-blue-500 rounded-full"></span>
                      {change}
                    </li>
                  ))}
                </ul>
                {plan_changes.plan_modifications && (
                  <div className="mt-2 text-xs text-blue-600">
                    <div className="font-medium">Impact:</div>
                    <div>{plan_changes.plan_modifications.impact}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Plan Update Error */}
        {role === 'assistant' && plan_changes && !plan_changes.success && plan_changes.error && (
          <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5" />
              <div className="flex-1">
                <div className="font-medium text-yellow-800 text-sm mb-1">Plan Update Issue</div>
                <div className="text-sm text-yellow-700">{plan_changes.error}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ sessionId }) => {
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);
  const [showDetailedProgress, setShowDetailedProgress] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { data: chatHistory, isLoading: historyLoading } = useChatHistory(sessionId);
  const sendMessageMutation = useSendMessage();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isTyping]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() || !sessionId || sendMessageMutation.isPending) return;

    const userMessage = message.trim();
    setMessage('');
    setIsTyping(true);
    setProcessingStartTime(Date.now());
    setShowDetailedProgress(false);

    // Show detailed progress after 15 seconds
    const progressTimer = setTimeout(() => {
      setShowDetailedProgress(true);
    }, 15000);

    try {
      const chatMessage: ChatMessageType = {
        message: userMessage,
        session_id: sessionId,
        enable_refinement: true,
      };

      await sendMessageMutation.mutateAsync(chatMessage);
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      clearTimeout(progressTimer);
      setIsTyping(false);
      setProcessingStartTime(null);
      setShowDetailedProgress(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  if (!sessionId) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">Welcome to Travel Assistant</h3>
          <p>Select a session or create a new one to start planning your trip</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-gray-900">Travel Planning Chat</h2>
          {sendMessageMutation.isPending && (
            <div className="flex items-center gap-1 text-blue-600">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">
                {showDetailedProgress ? `Processing (${Math.floor((Date.now() - (processingStartTime || Date.now())) / 1000)}s)` : 'Processing...'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {historyLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Loading conversation...</span>
            </div>
          </div>
        ) : chatHistory?.messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500 max-w-md">
              <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">Start Your Travel Journey</h3>
              <p className="mb-4">
                Ask me anything about travel planning! I can help you with:
              </p>
              <ul className="text-left space-y-1 text-sm">
                <li>• Destination recommendations</li>
                <li>• Flight and hotel search</li>
                <li>• Itinerary planning</li>
                <li>• Travel tips and advice</li>
              </ul>
            </div>
          </div>
        ) : (
          <div>
            {chatHistory?.messages.map((msg, index) => (
              <ChatMessage
                key={index}
                role={msg.role}
                content={msg.content}
                timestamp={msg.timestamp}
                confidence={msg.metadata?.confidence}
                plan_changes={msg.metadata?.plan_changes} // Get plan_changes from metadata
              />
            ))}
            {isTyping && (
              <div className="flex gap-3 p-4 bg-gray-50">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-gray-900">Travel Assistant</span>
                  </div>
                  <LoadingMessage 
                    processingStartTime={processingStartTime}
                    showDetailedProgress={showDetailedProgress}
                  />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex-shrink-0 p-4 border-t border-gray-200 bg-white">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about your travel plans..."
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={1}
              disabled={!sessionId || sendMessageMutation.isPending}
            />
            <button
              type="submit"
              disabled={!message.trim() || !sessionId || sendMessageMutation.isPending}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-gray-400 hover:text-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {sendMessageMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </form>
        
        {sendMessageMutation.isPending && !isTyping && processingStartTime && (Date.now() - processingStartTime) > 200000 && (
          <div className="mt-2 flex items-center gap-2 text-orange-600 text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>The AI is taking longer than usual. Your request is still being processed...</span>
          </div>
        )}
        {sendMessageMutation.isError && !isTyping && (
          <div className="mt-2 flex items-center gap-2 text-red-600 text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>Failed to send message. Please try again.</span>
          </div>
        )}
      </div>
    </div>
  );
}; 