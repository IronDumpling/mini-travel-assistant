"""
Conversation Memory Module - RAG-Enhanced Conversation Memory Management

Implements intelligent conversation memory management using RAG (Retrieval-Augmented Generation)
for semantic search, preference extraction, and intelligent summarization.

Enhanced with full core architecture integration including LLM services, prompt management,
and travel agent preference learning.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.rag_engine import get_rag_engine, Document, DocumentType
from app.core.llm_service import get_llm_service
from app.core.prompt_manager import prompt_manager, PromptType
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ConversationTurn(BaseModel):
    """Enhanced conversation turn with LLM-analyzed metadata"""
    id: str
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}
    sentiment: Optional[str] = None
    importance_score: float = 0.5
    confidence_score: float = 0.0
    topics: List[str] = []
    user_preferences: Dict[str, Any] = {}


class ConversationSession(BaseModel):
    """Enhanced conversation session with intelligent context"""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    turns: List[ConversationTurn] = []
    context: Dict[str, Any] = {}
    topic: Optional[str] = None
    is_active: bool = True
    learned_preferences: Dict[str, Any] = {}
    unresolved_issues: List[str] = []
    satisfaction_trend: List[float] = []


class ConversationMemory:
    """Enhanced RAG-powered conversation memory manager with full core integration"""
    
    def __init__(self, max_turns_per_session: int = 50, max_sessions: int = 100):
        self.max_turns_per_session = max_turns_per_session
        self.max_sessions = max_sessions
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Core service integrations
        self.rag_engine = get_rag_engine()
        self.llm_service = get_llm_service()
        self._conversation_indexed = {}  # Track which conversations are indexed
    
    async def _index_conversation_turn(self, turn: ConversationTurn, session_id: str):
        """Enhanced conversation turn indexing with better metadata"""
        try:
            # Create enriched document for this conversation turn
            doc = Document(
                id=f"conversation_{session_id}_{turn.id}",
                content=f"""
User Intent: {turn.intent or "unknown"}
User Message: {turn.user_message}
Assistant Response: {turn.agent_response}
Conversation Topics: {', '.join(turn.topics)}
User Sentiment: {turn.sentiment or "neutral"}
Extracted Entities: {', '.join([f"{k}: {v}" for k, v in turn.entities.items()])}
User Preferences: {', '.join([f"{k}: {v}" for k, v in turn.user_preferences.items()])}
Timestamp: {turn.timestamp.isoformat()}
Importance Score: {turn.importance_score}
                """,
                metadata={
                    "session_id": session_id,
                    "turn_id": turn.id,
                    "timestamp": turn.timestamp.isoformat(),
                    "intent": turn.intent or "unknown",
                    "sentiment": turn.sentiment or "neutral",
                    "importance_score": turn.importance_score,
                    "confidence_score": turn.confidence_score,
                    "topics": ','.join(turn.topics),
                    "type": "conversation_turn"
                },
                doc_type=DocumentType.CONVERSATION_TURN
            )
            
            # Index the conversation turn
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                self._conversation_indexed[turn.id] = True
                logger.debug(f"✅ Indexed enhanced conversation turn: {turn.id}")
            
        except Exception as e:
            logger.error(f"Failed to index conversation turn {turn.id}: {e}")

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationSession:
        """Create new enhanced session"""
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
        """Enhanced conversation turn addition with LLM analysis"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        turn_id = f"{session_id}_{len(session.turns)}"
        
        turn = ConversationTurn(
            id=turn_id,
            user_message=user_message,
            agent_response=agent_response
        )
        
        # 🔧 Enhanced analysis using LLM services
        try:
            # Analyze intent, sentiment, and extract entities using LLM
            analysis_result = await self._analyze_conversation_turn(user_message, agent_response)
            
            turn.intent = analysis_result.get("intent")
            turn.sentiment = analysis_result.get("sentiment")
            turn.entities = analysis_result.get("entities", {})
            turn.topics = analysis_result.get("topics", [])
            turn.user_preferences = analysis_result.get("user_preferences", {})
            turn.confidence_score = analysis_result.get("confidence_score", 0.5)
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for turn {turn_id}: {e}")
            # Fallback to basic analysis
            turn.intent = self._basic_intent_detection(user_message)
            turn.sentiment = self._basic_sentiment_analysis(user_message)
        
        # Apply metadata if provided
        if metadata:
            if "intent" in metadata:
                turn.intent = metadata["intent"]
            if "sentiment" in metadata:
                turn.sentiment = metadata["sentiment"]
            if "entities" in metadata:
                turn.entities.update(metadata["entities"])
        
        # Calculate enhanced importance score
        turn.importance_score = self._calculate_importance_score(turn)
        
        session.turns.append(turn)
        session.last_activity = datetime.utcnow()
        
        # Keep turn count within limit
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]
        
        # Update session context with enhanced logic
        await self._update_session_context(session, turn)
        
        # Index conversation turn for RAG retrieval
        await self._index_conversation_turn(turn, session_id)
        
        return turn

    async def _analyze_conversation_turn(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """Use LLM to analyze conversation turn for intent, sentiment, entities, and preferences"""
        try:
            if not self.llm_service:
                logger.warning("LLM service not available, using basic analysis")
                return {}

            # Create analysis prompt using prompt manager
            analysis_prompt = f"""
Please analyze the following conversation turn and extract:

User Message: "{user_message}"
Agent Response: "{agent_response}"

Extract and provide:
1. User Intent (planning, query, recommendation, booking, complaint, modification)
2. User Sentiment (positive, negative, neutral, excited, worried)
3. Key Entities (destinations, dates, budget, preferences, etc.)
4. Conversation Topics (main themes discussed)
5. User Preferences (travel style, interests, constraints)
6. Confidence Score (0.0-1.0 for analysis accuracy)

Please respond in JSON format:
{{
    "intent": "intent_type",
    "sentiment": "sentiment_type", 
    "entities": {{"entity_type": "entity_value"}},
    "topics": ["topic1", "topic2"],
    "user_preferences": {{"preference_type": "preference_value"}},
    "confidence_score": 0.0-1.0
}}
"""

            # Get LLM analysis
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=400
            )

            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group(0))
                logger.debug(f"LLM conversation analysis successful: {analysis_result}")
                return analysis_result
            else:
                logger.warning("Failed to parse LLM analysis response")
                return {}

        except Exception as e:
            logger.error(f"LLM conversation analysis failed: {e}")
            return {}

    def _basic_intent_detection(self, user_message: str) -> str:
        """Fallback basic intent detection"""
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["plan", "create", "organize", "arrange"]):
            return "planning"
        elif any(word in user_lower for word in ["recommend", "suggest", "what should"]):
            return "recommendation"
        elif any(word in user_lower for word in ["book", "reserve", "purchase"]):
            return "booking"
        elif any(word in user_lower for word in ["change", "modify", "update"]):
            return "modification"
        elif any(word in user_lower for word in ["problem", "issue", "complaint"]):
            return "complaint"
        else:
            return "query"

    def _basic_sentiment_analysis(self, user_message: str) -> str:
        """Fallback basic sentiment analysis"""
        user_lower = user_message.lower()
        
        positive_words = ["great", "amazing", "wonderful", "excited", "love", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "disappointed", "hate", "frustrated"]
        
        positive_count = sum(1 for word in positive_words if word in user_lower)
        negative_count = sum(1 for word in negative_words if word in user_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
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
        """Enhanced RAG-based relevant context retrieval"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        try:
            # Use RAG to find semantically relevant conversation turns
            retrieval_result = await self.rag_engine.retrieve(
                query=query,
                top_k=max_turns * 2,  # Get more candidates for filtering
                filter_metadata={"session_id": session_id, "type": "conversation_turn"},
                doc_type=DocumentType.CONVERSATION_TURN
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
        """Enhanced conversation search using RAG"""
        
        try:
            # Build filter metadata
            filter_metadata = {"type": "conversation_turn"}
            if session_id:
                filter_metadata["session_id"] = session_id
            
            # Use RAG for semantic search
            retrieval_result = await self.rag_engine.retrieve(
                query=query,
                top_k=20,  # Retrieve more candidates for filtering
                filter_metadata=filter_metadata,
                doc_type=DocumentType.CONVERSATION_TURN
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
        """Enhanced fallback keyword search"""
        results = []
        sessions_to_search = [self.sessions[session_id]] if session_id else self.sessions.values()
        
        query_terms = query.lower().split()
        
        for session in sessions_to_search:
            for turn in session.turns:
                # Enhanced keyword matching including entities and topics
                search_text = f"{turn.user_message} {turn.agent_response} {turn.intent} {' '.join(turn.topics)} {' '.join(turn.entities.values())}"
                search_text = search_text.lower()
                
                # Calculate match score
                match_score = sum(1 for term in query_terms if term in search_text)
                
                if match_score > 0:
                    # Time range filtering
                    if time_range:
                        if not (time_range[0] <= turn.timestamp <= time_range[1]):
                            continue
                    results.append(turn)
        
        # Sort by relevance (recent and high importance first)
        results.sort(key=lambda x: (x.importance_score, x.timestamp), reverse=True)
        return results
    
    async def extract_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Enhanced user preference extraction using LLM and RAG"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        try:
            # Use LLM to analyze user preferences from conversation history
            conversation_text = self._build_conversation_summary(session)
            
            if not self.llm_service:
                return self._fallback_preference_extraction(session)

            # Create preference extraction prompt
            preference_prompt = f"""
Analyze the following conversation history and extract user travel preferences:

{conversation_text}

Please extract and categorize the user's preferences:
1. Destinations (preferred locations, regions)
2. Travel Style (luxury, budget, mid-range, adventure, family)
3. Activities (cultural, outdoor, food, nightlife, shopping)
4. Accommodation (hotel types, amenities)
5. Transportation (flight class, ground transport)
6. Budget Considerations (price sensitivity, spending priorities)
7. Time Preferences (duration, seasons, flexibility)
8. Special Requirements (accessibility, dietary, etc.)

Provide results in JSON format:
{{
    "destinations": ["dest1", "dest2"],
    "travel_style": "style_preference",
    "activities": ["activity1", "activity2"],
    "accommodation": ["pref1", "pref2"],
    "transportation": ["pref1", "pref2"],
    "budget": {{"level": "mid-range", "priorities": ["priority1"]}},
    "time_preferences": {{"duration": "1-2 weeks", "season": "spring"}},
    "special_requirements": ["req1", "req2"],
    "confidence_score": 0.0-1.0
}}
"""

            # Get LLM analysis
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": preference_prompt}],
                temperature=0.2,
                max_tokens=600
            )

            # Parse response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                preferences = json.loads(json_match.group(0))
                
                # Update session's learned preferences
                session.learned_preferences.update(preferences)
                
                logger.info(f"✅ Extracted user preferences for session {session_id}")
                return preferences
            else:
                logger.warning("Failed to parse preference extraction response")
                return self._fallback_preference_extraction(session)
            
        except Exception as e:
            logger.error(f"LLM-based preference extraction failed: {e}")
            return self._fallback_preference_extraction(session)

    def _build_conversation_summary(self, session: ConversationSession) -> str:
        """Build conversation summary for analysis"""
        summary_parts = []
        for turn in session.turns[-10:]:  # Last 10 turns
            summary_parts.append(f"User: {turn.user_message}")
            summary_parts.append(f"Assistant: {turn.agent_response}")
            if turn.intent:
                summary_parts.append(f"Intent: {turn.intent}")
            if turn.entities:
                summary_parts.append(f"Entities: {turn.entities}")
        
        return "\n".join(summary_parts)
    
    def _fallback_preference_extraction(self, session: ConversationSession) -> Dict[str, Any]:
        """Enhanced fallback preference extraction"""
        preferences = {
            "destinations": [],
            "travel_style": "mid-range",
            "activities": [],
            "accommodation": [],
            "transportation": [],
            "budget": {"level": "mid-range", "priorities": []},
            "time_preferences": {},
            "special_requirements": []
        }
        
        for turn in session.turns:
            # Extract from entities
            for key, value in turn.entities.items():
                if key == "destination":
                    if value not in preferences["destinations"]:
                        preferences["destinations"].append(value)
                elif key == "activity":
                    if value not in preferences["activities"]:
                        preferences["activities"].append(value)
            
            # Extract from user preferences in turn
            for key, value in turn.user_preferences.items():
                if key in preferences and isinstance(preferences[key], list):
                    if value not in preferences[key]:
                        preferences[key].append(value)
        
        return preferences
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Enhanced session summary using LLM"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        try:
            if not self.llm_service:
                return self._fallback_session_summary(session_id)

            conversation_text = self._build_conversation_summary(session)
            
            # Create summary prompt
            summary_prompt = f"""
Analyze this conversation session and provide a comprehensive summary:

{conversation_text}

Please provide:
1. Main topics discussed
2. Key decisions made by the user
3. Travel plans or preferences identified
4. Any unresolved issues or questions
5. Overall user satisfaction level
6. Important follow-up actions needed

Provide results in JSON format:
{{
    "main_topics": ["topic1", "topic2"],
    "key_decisions": ["decision1", "decision2"], 
    "travel_plans": {{"destination": "Paris", "duration": "5 days"}},
    "unresolved_issues": ["issue1", "issue2"],
    "satisfaction_level": "high|medium|low",
    "follow_up_actions": ["action1", "action2"],
    "conversation_quality": 0.0-1.0
}}
"""

            # Get LLM analysis
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=500
            )

            # Parse response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                summary_data = json.loads(json_match.group(0))
                
                # Combine with basic statistics
                enhanced_summary = {
                    "session_id": session_id,
                    "total_turns": len(session.turns),
                    "duration_seconds": (session.last_activity - session.start_time).total_seconds(),
                    "main_topic": session.topic,
                    "learned_preferences": session.learned_preferences,
                    **summary_data
                }
                
                return enhanced_summary
            else:
                return self._fallback_session_summary(session_id)
            
        except Exception as e:
            logger.error(f"LLM-based session summary failed: {e}")
            return self._fallback_session_summary(session_id)
    
    def _fallback_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Enhanced fallback session summary"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Extract topics from turns
        all_topics = []
        for turn in session.turns:
            all_topics.extend(turn.topics)
        
        # Get most common topics
        from collections import Counter
        common_topics = [topic for topic, _ in Counter(all_topics).most_common(5)]
        
        return {
            "session_id": session_id,
            "total_turns": len(session.turns),
            "duration_seconds": (session.last_activity - session.start_time).total_seconds(),
            "main_topic": session.topic,
            "common_topics": common_topics,
            "learned_preferences": session.learned_preferences,
            "unresolved_issues": session.unresolved_issues,
            "average_importance": sum(turn.importance_score for turn in session.turns) / len(session.turns) if session.turns else 0,
            "satisfaction_trend": session.satisfaction_trend
        }
    
    def _calculate_importance_score(self, turn: ConversationTurn) -> float:
        """Enhanced importance score calculation"""
        score = 0.3  # Lower base score
        
        # Intent-based scoring
        intent_scores = {
            "planning": 0.4,
            "booking": 0.4,
            "complaint": 0.3,
            "modification": 0.3,
            "recommendation": 0.2,
            "query": 0.1
        }
        score += intent_scores.get(turn.intent, 0.1)
        
        # Entity count scoring
        entity_boost = min(len(turn.entities) * 0.05, 0.2)
        score += entity_boost
        
        # Sentiment scoring
        sentiment_scores = {
            "negative": 0.2,  # Negative feedback is important
            "positive": 0.1,
            "excited": 0.15,
            "worried": 0.15,
            "neutral": 0.0
        }
        score += sentiment_scores.get(turn.sentiment, 0.0)
        
        # Confidence scoring
        if turn.confidence_score > 0.8:
            score += 0.1
        elif turn.confidence_score < 0.5:
            score -= 0.1
        
        # User preferences scoring
        if turn.user_preferences:
            score += min(len(turn.user_preferences) * 0.05, 0.15)
        
        # Topic diversity scoring
        if len(turn.topics) > 2:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _update_session_context(self, session: ConversationSession, turn: ConversationTurn):
        """Enhanced session context update with intelligent topic tracking"""
        
        # Update entity information
        for key, value in turn.entities.items():
            session.context[key] = value
        
        # Update learned preferences
        for key, value in turn.user_preferences.items():
            if key not in session.learned_preferences:
                session.learned_preferences[key] = []
            if isinstance(session.learned_preferences[key], list):
                if value not in session.learned_preferences[key]:
                    session.learned_preferences[key].append(value)
            else:
                session.learned_preferences[key] = value
        
        # Topic tracking with intelligent aggregation
        if turn.topics:
            session.context.setdefault("discussed_topics", [])
            for topic in turn.topics:
                if topic not in session.context["discussed_topics"]:
                    session.context["discussed_topics"].append(topic)
        
        # Update primary topic (most recent significant topic)
        if turn.intent and turn.intent not in ["greeting", "query"] and turn.importance_score > 0.5:
            session.topic = turn.intent
        
        # Track unresolved issues
        if turn.sentiment in ["negative", "worried"] or turn.intent == "complaint":
            issue_description = f"{turn.intent}: {turn.user_message[:100]}..."
            if issue_description not in session.unresolved_issues:
                session.unresolved_issues.append(issue_description)
        
        # Update satisfaction trend
        satisfaction_map = {
            "positive": 0.8,
            "excited": 0.9,
            "neutral": 0.5,
            "negative": 0.2,
            "worried": 0.3
        }
        satisfaction_score = satisfaction_map.get(turn.sentiment, 0.5)
        session.satisfaction_trend.append(satisfaction_score)
        
        # Keep only recent satisfaction data
        if len(session.satisfaction_trend) > 10:
            session.satisfaction_trend = session.satisfaction_trend[-10:]
    
    def _cleanup_old_sessions(self):
        """Enhanced session cleanup with importance weighting"""
        # Sort sessions by activity and importance
        session_scores = []
        for session_id, session in self.sessions.items():
            # Calculate session importance score
            avg_importance = sum(turn.importance_score for turn in session.turns) / len(session.turns) if session.turns else 0
            recency_score = (datetime.utcnow() - session.last_activity).days
            importance_score = avg_importance - (recency_score * 0.01)  # Decay over time
            
            session_scores.append((session_id, importance_score))
        
        # Sort by importance (keep most important sessions)
        session_scores.sort(key=lambda x: x[1], reverse=True)
        sessions_to_keep = dict(session_scores[:self.max_sessions])
        
        # Keep only the important sessions
        sessions_to_keep_dict = {sid: self.sessions[sid] for sid, _ in session_scores[:self.max_sessions]}
        self.sessions = sessions_to_keep_dict
        
        logger.info(f"Cleaned up sessions, kept {len(self.sessions)} most important sessions")


# Enhanced global conversation memory instance
conversation_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """Get enhanced conversation memory instance with full core integration"""
    global conversation_memory
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
        logger.info("✅ Initialized enhanced conversation memory with full core integration")
    return conversation_memory 