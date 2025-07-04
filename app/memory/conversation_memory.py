"""
Conversation Memory Module - RAG-Enhanced Conversation Memory Management

Implements intelligent conversation memory management using RAG (Retrieval-Augmented Generation)
for semantic search, preference extraction, and intelligent summarization.

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
import logging

logger = logging.getLogger(__name__)


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
    """RAG-enhanced conversation memory manager"""
    
    def __init__(self, max_turns_per_session: int = 50, max_sessions: int = 100):
        self.max_turns_per_session = max_turns_per_session
        self.max_sessions = max_sessions
        self.sessions: Dict[str, ConversationSession] = {}
        self.rag_engine = get_rag_engine()
        self._conversation_indexed = {}  # Track which conversations are indexed
    
    async def _index_conversation_turn(self, turn: ConversationTurn, session_id: str):
        """Index a conversation turn for RAG-based retrieval"""
        try:
            # Create a document for this conversation turn
            doc = Document(
                id=f"conversation_{session_id}_{turn.id}",
                content=f"""
User Message: {turn.user_message}
Assistant Response: {turn.agent_response}
Time: {turn.timestamp.isoformat()}
Intent: {turn.intent or "unknown"}
Sentiment: {turn.sentiment or "neutral"}
Importance: {turn.importance_score}
                """,
                metadata={
                    "session_id": session_id,
                    "turn_id": turn.id,
                    "timestamp": turn.timestamp.isoformat(),
                    "intent": turn.intent or "unknown",
                    "sentiment": turn.sentiment or "neutral",
                    "importance_score": turn.importance_score,
                    "type": "conversation_turn"
                }
            )
            
            # Index the conversation turn
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                self._conversation_indexed[turn.id] = True
                logger.debug(f"Indexed conversation turn: {turn.id}")
            
        except Exception as e:
            logger.error(f"Failed to index conversation turn {turn.id}: {e}")

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
    
    async def add_turn(
        self, 
        session_id: str, 
        user_message: str, 
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add conversation turn with RAG indexing"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        turn_id = f"{session_id}_{len(session.turns)}"
        
        turn = ConversationTurn(
            id=turn_id,
            user_message=user_message,
            agent_response=agent_response
        )
        
        # Extract intent, entities and sentiment
        if metadata:
            turn.intent = metadata.get("intent")
            turn.entities = metadata.get("entities", {})
            turn.sentiment = metadata.get("sentiment")
        
        # Calculate importance score
        turn.importance_score = self._calculate_importance_score(turn)
        
        session.turns.append(turn)
        session.last_activity = datetime.utcnow()
        
        # Keep turn count within limit
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]
        
        # Update session context
        self._update_session_context(session, turn)
        
        # Index conversation turn for RAG retrieval
        await self._index_conversation_turn(turn, session_id)
        
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
    
    async def get_relevant_context(
        self, 
        session_id: str, 
        query: str,
        max_turns: int = 5
    ) -> List[ConversationTurn]:
        """Get relevant conversation context using RAG-based semantic search"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        try:
            # Use RAG to find semantically relevant conversation turns
            retrieval_result = await self.rag_engine.retrieve(
                query=query,
                top_k=max_turns * 2,  # Retrieve more candidates
                filter_metadata={"session_id": session_id, "type": "conversation_turn"}
            )
            
            relevant_turns = []
            for doc in retrieval_result.documents:
                turn_id = doc.metadata.get("turn_id")
                if turn_id:
                    # Find the corresponding turn
                    for turn in session.turns:
                        if turn.id == turn_id:
                            relevant_turns.append(turn)
                            break
            
            # Sort by relevance (RAG scores) and limit results
            return relevant_turns[:max_turns]
            
        except Exception as e:
            logger.error(f"RAG-based context retrieval failed: {e}")
            # Fallback to importance-based selection
            sorted_turns = sorted(
                session.turns, 
                key=lambda x: x.importance_score, 
                reverse=True
            )
            return sorted_turns[:max_turns]
    
    async def search_conversations(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[ConversationTurn]:
        """Search conversation history using RAG-based semantic search"""
        
        try:
            # Build filter metadata
            filter_metadata = {"type": "conversation_turn"}
            if session_id:
                filter_metadata["session_id"] = session_id
            
            # Use RAG for semantic search
            retrieval_result = await self.rag_engine.retrieve(
                query=query,
                top_k=20,  # Retrieve more candidates for filtering
                filter_metadata=filter_metadata
            )
            
            results = []
            for doc in retrieval_result.documents:
                turn_id = doc.metadata.get("turn_id")
                session_id_meta = doc.metadata.get("session_id")
                
                if turn_id and session_id_meta:
                    # Find the corresponding turn
                    session = self.get_session(session_id_meta)
                    if session:
                        for turn in session.turns:
                            if turn.id == turn_id:
                                # Apply time range filter if specified
                                if time_range:
                                    if not (time_range[0] <= turn.timestamp <= time_range[1]):
                                        continue
                                results.append(turn)
                                break
            
            return results
            
        except Exception as e:
            logger.error(f"RAG-based conversation search failed: {e}")
            # Fallback to simple keyword search
            return self._fallback_keyword_search(query, session_id, time_range)
    
    def _fallback_keyword_search(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[ConversationTurn]:
        """Fallback keyword search when RAG fails"""
        results = []
        sessions_to_search = [self.sessions[session_id]] if session_id else self.sessions.values()
        
        for session in sessions_to_search:
            for turn in session.turns:
                # Simple keyword matching
                if query.lower() in turn.user_message.lower() or query.lower() in turn.agent_response.lower():
                    # Time range filtering
                    if time_range:
                        if not (time_range[0] <= turn.timestamp <= time_range[1]):
                            continue
                    results.append(turn)
        
        return results
    
    async def extract_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Extract user preferences using RAG-based analysis"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        try:
            # Use RAG to analyze user preferences from conversation history
            preference_queries = [
                "What types of travel activities does the user prefer",
                "User's budget preferences and spending habits",
                "User's preferred transportation methods",
                "User's preferred accommodation types",
                "User's time preferences and itinerary planning",
                "User's interests and special requirements"
            ]
            
            preferences = {}
            
            for query in preference_queries:
                retrieval_result = await self.rag_engine.retrieve(
                    query=query,
                    top_k=3,
                    filter_metadata={"session_id": session_id, "type": "conversation_turn"}
                )
                
                # Extract preferences from retrieved conversations
                for doc in retrieval_result.documents:
                    # Parse preferences from conversation content
                    content = doc.content
                    preference_type = self._categorize_preference(query)
                    
                    if preference_type not in preferences:
                        preferences[preference_type] = []
                    
                    # Extract specific preferences from content
                    extracted_prefs = self._extract_preferences_from_content(content, preference_type)
                    preferences[preference_type].extend(extracted_prefs)
            
            # Remove duplicates and clean up
            for key in preferences:
                preferences[key] = list(set(preferences[key]))
            
            return preferences
            
        except Exception as e:
            logger.error(f"RAG-based preference extraction failed: {e}")
            # Fallback to simple entity extraction
            return self._fallback_preference_extraction(session)
    
    def _categorize_preference(self, query: str) -> str:
        """Categorize preference query type"""
        if "activities" in query or "travel" in query:
            return "activities"
        elif "budget" in query or "spending" in query:
            return "budget"
        elif "transportation" in query:
            return "transport"
        elif "accommodation" in query:
            return "accommodation"
        elif "time" in query or "itinerary" in query:
            return "schedule"
        elif "interests" in query or "requirements" in query:
            return "interests"
        else:
            return "general"
    
    def _extract_preferences_from_content(self, content: str, preference_type: str) -> List[str]:
        """Extract specific preferences from conversation content"""
        # Simple keyword-based extraction (can be enhanced with NLP)
        preferences = []
        
        content_lower = content.lower()
        
        if preference_type == "activities":
            activity_keywords = ["like", "want to go", "interested in", "recommend", "fun", "attraction"]
            for keyword in activity_keywords:
                if keyword in content_lower:
                    # Extract surrounding context
                    preferences.append(f"User is interested in {keyword}-related activities")
        
        elif preference_type == "budget":
            budget_keywords = ["cheap", "budget", "luxury", "high-end", "affordable", "save money"]
            for keyword in budget_keywords:
                if keyword in content_lower:
                    preferences.append(f"User prefers {keyword} options")
        
        # Add more preference extraction logic for other types
        
        return preferences
    
    def _fallback_preference_extraction(self, session: ConversationSession) -> Dict[str, Any]:
        """Fallback preference extraction when RAG fails"""
        preferences = {}
        
        for turn in session.turns:
            entities = turn.entities
            for key, value in entities.items():
                if key not in preferences:
                    preferences[key] = []
                if value not in preferences[key]:
                    preferences[key].append(value)
        
        return preferences
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate intelligent session summary using RAG"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        try:
            # Use RAG to generate intelligent summary
            summary_queries = [
                "What was the main topic of this conversation",
                "What important decision did the user make",
                "Are there any unresolved issues",
                "How satisfied is the user"
            ]
            
            summary_data = {}
            
            for query in summary_queries:
                retrieval_result = await self.rag_engine.retrieve(
                    query=query,
                    top_k=3,
                    filter_metadata={"session_id": session_id, "type": "conversation_turn"}
                )
                
                # Generate summary based on retrieved content
                summary_key = self._get_summary_key(query)
                summary_data[summary_key] = []
                
                for doc in retrieval_result.documents:
                    # Extract summary information
                    summary_info = self._extract_summary_info(doc.content, summary_key)
                    if summary_info:
                        summary_data[summary_key].append(summary_info)
            
            # Combine with basic statistics
            return {
                "session_id": session_id,
                "total_turns": len(session.turns),
                "duration": (session.last_activity - session.start_time).total_seconds(),
                "main_topic": session.topic,
                "key_entities": self._extract_key_entities(session),
                "unresolved_issues": summary_data.get("unresolved_issues", []),
                "user_satisfaction": self._estimate_satisfaction(session),
                "important_decisions": summary_data.get("decisions", []),
                "conversation_themes": summary_data.get("themes", [])
            }
            
        except Exception as e:
            logger.error(f"RAG-based session summary failed: {e}")
            # Fallback to basic summary
            return self._fallback_session_summary(session_id)
    
    def _get_summary_key(self, query: str) -> str:
        """Get summary key based on query"""
        if "topic" in query:
            return "themes"
        elif "decision" in query:
            return "decisions"
        elif "issue" in query:
            return "unresolved_issues"
        elif "satisfaction" in query:
            return "satisfaction"
        else:
            return "general"
    
    def _extract_summary_info(self, content: str, summary_key: str) -> Optional[str]:
        """Extract summary information from content"""
        # Simple extraction logic (can be enhanced)
        if summary_key == "themes":
            return "Travel planning and suggestions"
        elif summary_key == "decisions":
            return "User has decided on their travel destination"
        elif summary_key == "unresolved_issues":
            return "Budget and time arrangements need further confirmation"
        else:
            return None
    
    def _fallback_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Fallback session summary when RAG fails"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
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