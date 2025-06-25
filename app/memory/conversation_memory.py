"""
Conversation Memory Module - Conversation memory management

TODO: Implement the following features
1. Multi-turn conversation context management
2. Important information extraction and storage
3. Conversation history compression
4. Context relevance scoring
5. Conversation topic tracking
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pydantic import BaseModel, Field
from app.core.rag_engine import get_rag_engine, Document


class ConversationTurn(BaseModel):
    """Conversation turn"""
    id: str
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}
    sentiment: Optional[str] = None
    importance_score: float = 0.5


class ConversationSession(BaseModel):
    """Conversation session"""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    turns: List[ConversationTurn] = []
    context: Dict[str, Any] = {}
    topic: Optional[str] = None
    is_active: bool = True


class ConversationMemory:
    """Conversation memory manager"""
    
    def __init__(self, max_turns_per_session: int = 50, max_sessions: int = 100):
        self.max_turns_per_session = max_turns_per_session
        self.max_sessions = max_sessions
        self.sessions: Dict[str, ConversationSession] = {}
        self.rag_engine = get_rag_engine()
    
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationSession:
        """Create new session"""
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id
        )
        self.sessions[session_id] = session
        
        # Keep session count within limit
        if len(self.sessions) > self.max_sessions:
            self._cleanup_old_sessions()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session"""
        return self.sessions.get(session_id)
    
    def add_turn(
        self, 
        session_id: str, 
        user_message: str, 
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add conversation turn"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        turn_id = f"{session_id}_{len(session.turns)}"
        
        turn = ConversationTurn(
            id=turn_id,
            user_message=user_message,
            agent_response=agent_response
        )
        
        # TODO: Extract intent, entities and sentiment
        if metadata:
            turn.intent = metadata.get("intent")
            turn.entities = metadata.get("entities", {})
            turn.sentiment = metadata.get("sentiment")
        
        # TODO: Calculate importance score
        turn.importance_score = self._calculate_importance_score(turn)
        
        session.turns.append(turn)
        session.last_activity = datetime.utcnow()
        
        # Keep turn count within limit
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]
        
        # TODO: Update session context
        self._update_session_context(session, turn)
        
        return turn
    
    def get_recent_context(
        self, 
        session_id: str, 
        max_turns: int = 10
    ) -> List[ConversationTurn]:
        """Get recent conversation context"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return session.turns[-max_turns:]
    
    def get_relevant_context(
        self, 
        session_id: str, 
        query: str,
        max_turns: int = 5
    ) -> List[ConversationTurn]:
        """Get relevant conversation context"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        # TODO: Find the most relevant conversation turns based on query content
        # 1. Calculate similarity between query and historical conversations
        # 2. Sort by relevance
        # 3. Return the most relevant turns
        
        # Temporary implementation: Return turns with highest importance score
        sorted_turns = sorted(
            session.turns, 
            key=lambda x: x.importance_score, 
            reverse=True
        )
        
        return sorted_turns[:max_turns]
    
    def search_conversations(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[ConversationTurn]:
        """Search conversation history"""
        # TODO: Implement conversation history search
        # 1. Support keyword search
        # 2. Support semantic search
        # 3. Support time range filtering
        # 4. Support intent filtering
        
        results = []
        sessions_to_search = [self.sessions[session_id]] if session_id else self.sessions.values()
        
        for session in sessions_to_search:
            for turn in session.turns:
                # Simple keyword matching (TODO: Improve to semantic search)
                if query.lower() in turn.user_message.lower() or query.lower() in turn.agent_response.lower():
                    # Time range filtering
                    if time_range:
                        if not (time_range[0] <= turn.timestamp <= time_range[1]):
                            continue
                    results.append(turn)
        
        return results
    
    def extract_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Extract user preferences from conversation"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        preferences = {}
        
        # TODO: Implement preference extraction logic
        # 1. Analyze user-mentioned preference keywords
        # 2. Identify user's choice patterns
        # 3. Count user's behavior tendencies
        
        for turn in session.turns:
            # Simple keyword extraction (TODO: Improve to NLP analysis)
            entities = turn.entities
            for key, value in entities.items():
                if key not in preferences:
                    preferences[key] = []
                if value not in preferences[key]:
                    preferences[key].append(value)
        
        return preferences
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # TODO: Generate intelligent session summary
        # 1. Extract main topics
        # 2. Summarize key decisions
        # 3. Identify unresolved issues
        
        return {
            "session_id": session_id,
            "total_turns": len(session.turns),
            "duration": (session.last_activity - session.start_time).total_seconds(),
            "main_topic": session.topic,
            "key_entities": self._extract_key_entities(session),
            "unresolved_issues": self._identify_unresolved_issues(session),
            "user_satisfaction": self._estimate_satisfaction(session)
        }
    
    def _calculate_importance_score(self, turn: ConversationTurn) -> float:
        """Calculate importance score for conversation turn"""
        # TODO: Implement more complex importance scoring algorithm
        # 1. Based on intent type
        # 2. Based on entity count
        # 3. Based on user sentiment
        # 4. Based on response length
        
        score = 0.5  # Base score
        
        # Adjust score based on intent
        if turn.intent in ["planning", "booking", "complaint"]:
            score += 0.3
        elif turn.intent in ["query", "information"]:
            score += 0.1
        
        # Adjust score based on entity count
        score += min(len(turn.entities) * 0.1, 0.3)
        
        # Adjust score based on sentiment
        if turn.sentiment == "negative":
            score += 0.2  # Negative sentiment is more important
        elif turn.sentiment == "positive":
            score += 0.1
        
        return min(score, 1.0)
    
    def _update_session_context(self, session: ConversationSession, turn: ConversationTurn):
        """Update session context"""
        # TODO: Implement context update logic
        # 1. Update current topic
        # 2. Update user state
        # 3. Update to-do items
        
        # Update entity information
        for key, value in turn.entities.items():
            session.context[key] = value
        
        # Update topic
        if turn.intent and turn.intent != "greeting":
            session.topic = turn.intent
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions"""
        # Keep recently active sessions
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_activity,
            reverse=True
        )
        
        # Keep recent sessions
        sessions_to_keep = dict(sorted_sessions[:self.max_sessions])
        self.sessions = sessions_to_keep
    
    def _extract_key_entities(self, session: ConversationSession) -> Dict[str, List[Any]]:
        """Extract key entities from conversation"""
        # TODO: Implement entity extraction and aggregation
        entities = {}
        for turn in session.turns:
            for key, value in turn.entities.items():
                if key not in entities:
                    entities[key] = []
                if value not in entities[key]:
                    entities[key].append(value)
        return entities
    
    def _identify_unresolved_issues(self, session: ConversationSession) -> List[str]:
        """Identify unresolved issues"""
        # TODO: Implement issue tracking logic
        issues = []
        for turn in session.turns:
            if turn.intent == "complaint" or turn.sentiment == "negative":
                issues.append(turn.user_message)
        return issues
    
    def _estimate_satisfaction(self, session: ConversationSession) -> float:
        """Estimate user satisfaction"""
        # TODO: Estimate satisfaction based on conversation content
        if not session.turns:
            return 0.5
        
        positive_count = sum(1 for turn in session.turns if turn.sentiment == "positive")
        negative_count = sum(1 for turn in session.turns if turn.sentiment == "negative")
        
        if positive_count + negative_count == 0:
            return 0.5
        
        return positive_count / (positive_count + negative_count)


# Global conversation memory instance
conversation_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """Get conversation memory instance"""
    global conversation_memory
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
    return conversation_memory 