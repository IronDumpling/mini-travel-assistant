"""
Session Manager Module - Enhanced single-user session management

Designed for local single-user deployment with full core architecture integration.
Manages conversation sessions, context switching, memory organization, and intelligent
session analysis using RAG and LLM services.

Enhanced Features:
1. ✅ Session lifecycle management (create, activate, archive)
2. ✅ Context switching between different travel planning sessions  
3. ✅ Session-based memory isolation
4. ✅ Automatic session cleanup and archiving
5. ✅ Enhanced session metadata and analytics with RAG integration
"""

import uuid
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

# Enhanced core integrations
from app.core.rag_engine import get_rag_engine, Document
from app.core.llm_service import get_llm_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"

class SessionMessage(BaseModel):
    """Individual message in a session"""
    id: str
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Enhanced fields for better analysis
    intent: Optional[str] = None
    confidence: Optional[float] = None
    topics: List[str] = Field(default_factory=list)
    importance_score: float = 0.5

class SessionMetadata(BaseModel):
    """Session metadata"""
    session_id: str
    title: str
    description: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime
    last_activity: datetime
    total_messages: int = 0
    travel_context: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    messages: List[SessionMessage] = Field(default_factory=list)
    
    # Enhanced metadata fields
    learned_preferences: Dict[str, Any] = Field(default_factory=dict)
    main_topics: List[str] = Field(default_factory=list)
    satisfaction_score: float = 0.5
    complexity_score: float = 0.5
    resolution_status: str = "ongoing"  # ongoing, resolved, needs_followup

class SessionManager:
    """Single-user session manager"""
    
    def __init__(self, storage_path: str = "./data/sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, SessionMetadata] = {}
        self.current_session_id: Optional[str] = None
        self.max_active_sessions = 10  # Reasonable limit for single user
        
        # Enhanced core service integrations
        self.rag_engine = get_rag_engine()
        self.llm_service = get_llm_service()
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(
        self, 
        title: Optional[str] = None, 
        description: Optional[str] = None,
        travel_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc)
        
        # Generate smart title if not provided
        if not title:
            title = f"Travel Planning - {current_time.strftime('%Y-%m-%d %H:%M')}"
        
        metadata = SessionMetadata(
            session_id=session_id,
            title=title,
            description=description,
            created_at=current_time,
            last_activity=current_time,
            travel_context=travel_context or {}
        )
        
        self.sessions[session_id] = metadata
        self.current_session_id = session_id
        
        # Auto-cleanup if too many active sessions
        self._cleanup_old_sessions()
        
        # Save to disk
        self._save_session(metadata)
        
        # Index session for RAG search (async task)
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._index_session_metadata(metadata))
        except RuntimeError:
            # No event loop running, skip indexing for now
            logger.debug("No event loop running, skipping session indexing")
        
        logger.info(f"✅ Created enhanced session: {session_id}")
        return session_id
    
    def get_current_session(self) -> Optional[SessionMetadata]:
        """Get current active session"""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to a different session"""
        if session_id in self.sessions:
            # Update status of previous session (but not last_activity)
            if self.current_session_id:
                prev_session = self.sessions.get(self.current_session_id)
                if prev_session:
                    prev_session.status = SessionStatus.INACTIVE
                    self._save_session(prev_session)
            
            # Switch to new session
            self.current_session_id = session_id
            current_session = self.sessions[session_id]
            current_session.status = SessionStatus.ACTIVE
            # Don't update last_activity when just switching - only when adding messages
            # This prevents session list reordering when user clicks on sessions
            self._save_session(current_session)
            
            logger.info(f"Switched to session: {session_id}")
            return True
        return False
    
    def list_sessions(
        self, 
        status: Optional[SessionStatus] = None,
        limit: int = 20
    ) -> List[SessionMetadata]:
        """List sessions with optional filtering"""
        sessions = list(self.sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # Enhanced sorting by activity, importance, and satisfaction
        def sort_key(session):
            recency_score = session.last_activity.timestamp()
            importance_score = session.complexity_score * len(session.messages)
            satisfaction_bonus = session.satisfaction_score * 0.1
            return recency_score + importance_score + satisfaction_bonus
        
        sessions.sort(key=sort_key, reverse=True)
        
        return sessions[:limit]
    
    async def search_sessions(
        self, 
        query: str, 
        use_rag: bool = True,
        limit: int = 10
    ) -> List[SessionMetadata]:
        """Enhanced session search using RAG semantic search"""
        
        if use_rag and self.rag_engine:
            try:
                # Use RAG to find semantically relevant sessions
                retrieval_result = await self.rag_engine.retrieve(
                    query=query,
                    top_k=limit * 2,
                    filter_metadata={"type": "session_metadata"}
                )
                
                relevant_sessions = []
                for doc in retrieval_result.documents:
                    session_id = doc.metadata.get("session_id")
                    if session_id and session_id in self.sessions:
                        relevant_sessions.append(self.sessions[session_id])
                
                logger.info(f"RAG search found {len(relevant_sessions)} relevant sessions")
                return relevant_sessions[:limit]
                
            except Exception as e:
                logger.warning(f"RAG session search failed: {e}, falling back to keyword search")
        
        # Fallback to enhanced keyword search
        return self._fallback_session_search(query, limit)
    
    def _fallback_session_search(self, query: str, limit: int) -> List[SessionMetadata]:
        """Enhanced fallback session search"""
        query_lower = query.lower()
        matching_sessions = []
        
        for session in self.sessions.values():
            score = 0
            
            # Search in title and description
            if query_lower in session.title.lower():
                score += 3
            if session.description and query_lower in session.description.lower():
                score += 2
            
            # Search in travel context
            context_str = json.dumps(session.travel_context).lower()
            if query_lower in context_str:
                score += 2
            
            # Search in tags and topics
            all_tags = session.tags + session.main_topics
            if any(query_lower in tag.lower() for tag in all_tags):
                score += 2
            
            # Search in message content
            for message in session.messages[-5:]:  # Recent messages only
                if query_lower in message.user_message.lower():
                    score += 1
                if query_lower in message.agent_response.lower():
                    score += 1
            
            if score > 0:
                matching_sessions.append((session, score))
        
        # Sort by relevance score
        matching_sessions.sort(key=lambda x: x[1], reverse=True)
        return [session for session, _ in matching_sessions[:limit]]

    def update_session_context(
        self, 
        session_id: Optional[str] = None, 
        context_updates: Optional[Dict[str, Any]] = None,
        increment_messages: bool = False
    ):
        """Update session context and metadata"""
        target_session_id = session_id or self.current_session_id
        if not target_session_id or target_session_id not in self.sessions:
            return
        
        session = self.sessions[target_session_id]
        session.last_activity = datetime.now(timezone.utc)
        
        if increment_messages:
            session.total_messages += 1
        
        if context_updates:
            session.travel_context.update(context_updates)
            
            # Extract travel-related tags
            self._update_session_tags(session, context_updates)
        
        # Update session analytics
        self._update_session_analytics(session)
        
        self._save_session(session)
    
    def _update_session_tags(self, session: SessionMetadata, context_updates: Dict[str, Any]):
        """Update session tags based on context"""
        new_tags = set(session.tags)
        
        # Extract tags from context updates
        if "destination" in context_updates:
            dest = context_updates["destination"]
            if isinstance(dest, str) and dest not in new_tags:
                new_tags.add(dest.lower())
        
        if "travel_dates" in context_updates:
            new_tags.add("dated_travel")
        
        if "budget" in context_updates:
            budget = context_updates["budget"]
            if isinstance(budget, (int, float)):
                if budget < 1000:
                    new_tags.add("budget_travel")
                elif budget > 5000:
                    new_tags.add("luxury_travel")
                else:
                    new_tags.add("mid_range_travel")
        
        session.tags = list(new_tags)
    
    def _update_session_analytics(self, session: SessionMetadata):
        """Update session analytics and scores"""
        if not session.messages:
            return
            
        # Calculate complexity score based on message content and travel context
        complexity_factors = [
            len(session.travel_context),  # More context = more complex
            len(session.messages),  # More messages = more complex
            len(set(msg.intent for msg in session.messages if msg.intent)),  # Intent diversity
            len(session.main_topics)  # Topic diversity
        ]
        
        session.complexity_score = min(sum(f * 0.1 for f in complexity_factors), 1.0)
        
        # Calculate satisfaction score from recent messages
        recent_messages = session.messages[-5:]
        confidence_scores = [msg.confidence for msg in recent_messages if msg.confidence]
        session.satisfaction_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Update resolution status
        if session.satisfaction_score > 0.8 and len(session.messages) > 3:
            session.resolution_status = "resolved"
        elif session.satisfaction_score < 0.4:
            session.resolution_status = "needs_followup"
        else:
            session.resolution_status = "ongoing"

    def archive_session(self, session_id: str) -> bool:
        """Archive a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.status = SessionStatus.ARCHIVED
            session.last_activity = datetime.now(timezone.utc)
            
            # Generate session summary before archiving (async task)
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._generate_session_summary(session))
            except RuntimeError:
                logger.debug("No event loop running, skipping session summary generation")
            
            # If archiving current session, clear current session
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            self._save_session(session)
            logger.info(f"Archived session: {session_id}")
            return True
        return False
    
    async def _generate_session_summary(self, session: SessionMetadata):
        """Generate intelligent session summary for archived sessions"""
        try:
            if not self.llm_service or not session.messages:
                return
            
            # Build conversation summary
            conversation_text = ""
            for msg in session.messages[-10:]:  # Last 10 messages
                conversation_text += f"User: {msg.user_message}\n"
                conversation_text += f"Assistant: {msg.agent_response}\n\n"
            
            # Create summary prompt
            summary_prompt = f"""
Analyze this travel planning session and create a brief summary:

{conversation_text}

Session Info:
- Duration: {session.total_messages} messages
- Travel Context: {session.travel_context}

Please provide a 2-3 sentence summary focusing on:
1. Main travel goals/destinations discussed
2. Key decisions or preferences identified
3. Current status (resolved, pending, needs follow-up)

Summary:"""

            # Get LLM summary
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            # Update session description with summary
            if response and response.content:
                session.description = response.content.strip()
                self._save_session(session)
                logger.info(f"Generated summary for archived session: {session.session_id}")
                
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session"""
        if session_id in self.sessions:
            # Remove from memory
            del self.sessions[session_id]
            
            # Remove from disk
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Remove from RAG index (async task)
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._remove_session_from_index(session_id))
            except RuntimeError:
                logger.debug("No event loop running, skipping session index cleanup")
            
            # Clear current session if deleted
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    async def _remove_session_from_index(self, session_id: str):
        """Remove session from RAG index"""
        try:
            # Remove session metadata and messages from index
            document_ids = [f"session_{session_id}"]
            
            # Also remove all messages from this session
            session = self.sessions.get(session_id)
            if session:
                message_ids = [f"message_{msg.id}" for msg in session.messages]
                document_ids.extend(message_ids)
            
            await self.rag_engine.vector_store.delete_documents(document_ids)
        except Exception as e:
            logger.warning(f"Failed to remove session from RAG index: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_sessions = len(self.sessions)
        active_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE])
        archived_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ARCHIVED])
        
        total_messages = sum(s.total_messages for s in self.sessions.values())
        
        # Enhanced analytics
        avg_satisfaction = sum(s.satisfaction_score for s in self.sessions.values()) / total_sessions if total_sessions else 0
        avg_complexity = sum(s.complexity_score for s in self.sessions.values()) / total_sessions if total_sessions else 0
        
        resolution_stats = {}
        for session in self.sessions.values():
            status = session.resolution_status
            resolution_stats[status] = resolution_stats.get(status, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "archived_sessions": archived_sessions,
            "current_session_id": self.current_session_id,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / total_sessions if total_sessions else 0,
            "average_satisfaction_score": avg_satisfaction,
            "average_complexity_score": avg_complexity,
            "resolution_stats": resolution_stats,
            "storage_path": str(self.storage_path),
            "rag_integration": True
        }

    def _load_sessions(self):
        """Enhanced session loading with backwards compatibility"""
        if not self.storage_path.exists():
            return
        
        loaded_count = 0
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    
                    # Handle backwards compatibility for enhanced fields
                    if "learned_preferences" not in session_data:
                        session_data["learned_preferences"] = {}
                    if "main_topics" not in session_data:
                        session_data["main_topics"] = []
                    if "satisfaction_score" not in session_data:
                        session_data["satisfaction_score"] = 0.5
                    if "complexity_score" not in session_data:
                        session_data["complexity_score"] = 0.5
                    if "resolution_status" not in session_data:
                        session_data["resolution_status"] = "ongoing"
                    
                    # Enhance existing messages with new fields
                    for msg_data in session_data.get("messages", []):
                        if "intent" not in msg_data:
                            msg_data["intent"] = None
                        if "confidence" not in msg_data:
                            msg_data["confidence"] = None
                        if "topics" not in msg_data:
                            msg_data["topics"] = []
                        if "importance_score" not in msg_data:
                            msg_data["importance_score"] = 0.5
                    
                    metadata = SessionMetadata(**session_data)
                    self.sessions[metadata.session_id] = metadata
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load session {session_file}: {e}")
        
        # Set most recent active session as current
        active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        if active_sessions:
            most_recent = max(active_sessions, key=lambda x: x.last_activity)
            self.current_session_id = most_recent.session_id
        
        logger.info(f"Loaded {loaded_count} sessions from storage")
    
    def _save_session(self, session: SessionMetadata):
        """Enhanced session saving with error handling"""
        session_file = self.storage_path / f"{session.session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.model_dump(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    async def _index_session_metadata(self, session: SessionMetadata):
        """Index session metadata for RAG search"""
        try:
            # Create document for session metadata
            doc = Document(
                id=f"session_{session.session_id}",
                content=f"""
Session Title: {session.title}
Description: {session.description or "No description"}
Travel Context: {json.dumps(session.travel_context)}
Tags: {', '.join(session.tags)}
Main Topics: {', '.join(session.main_topics)}
Status: {session.status}
Resolution Status: {session.resolution_status}
Created: {session.created_at.isoformat()}
Total Messages: {session.total_messages}
                """,
                metadata={
                    "session_id": session.session_id,
                    "type": "session_metadata",
                    "status": session.status,
                    "satisfaction_score": session.satisfaction_score,
                    "complexity_score": session.complexity_score,
                    "resolution_status": session.resolution_status
                }
            )
            
            # Index the session metadata
            success = await self.rag_engine.index_documents([doc])
            if success:
                logger.debug(f"✅ Indexed session metadata: {session.session_id}")
                
        except Exception as e:
            logger.error(f"Failed to index session metadata {session.session_id}: {e}")

    def _cleanup_old_sessions(self):
        """Enhanced session cleanup with intelligent prioritization"""
        active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        
        if len(active_sessions) > self.max_active_sessions:
            # Sort by importance score (activity + satisfaction + complexity)
            def importance_score(session):
                recency = (datetime.now(timezone.utc) - session.last_activity).days
                return (
                    session.satisfaction_score * 0.4 +
                    session.complexity_score * 0.3 +
                    len(session.messages) * 0.001 +
                    max(0, 30 - recency) * 0.01  # Recency bonus
                )
            
            active_sessions.sort(key=importance_score, reverse=True)
            sessions_to_keep = active_sessions[:self.max_active_sessions]
            sessions_to_deactivate = active_sessions[self.max_active_sessions:]
            
            for session in sessions_to_deactivate:
                session.status = SessionStatus.INACTIVE
                self._save_session(session)
            
            logger.info(f"Deactivated {len(sessions_to_deactivate)} sessions, kept {len(sessions_to_keep)} active")

    def add_message(
        self,
        user_message: str,
        agent_response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhanced message addition with intelligent analysis"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            raise ValueError(f"Session {target_session_id} not found")
        
        session = self.sessions[target_session_id]
        
        # Create enhanced message
        message_id = f"{target_session_id}_{len(session.messages)}"
        message = SessionMessage(
            id=message_id,
            user_message=user_message,
            agent_response=agent_response,
            metadata=metadata or {}
        )
        
        # Extract enhanced metadata if available
        if metadata:
            message.intent = metadata.get("intent")
            message.confidence = metadata.get("confidence")
            message.topics = metadata.get("topics", [])
            
            # Calculate importance score
            if message.intent or message.confidence:
                message.importance_score = self._calculate_message_importance(message)
        
        # Add to session
        session.messages.append(message)
        session.total_messages += 1
        session.last_activity = datetime.now(timezone.utc)
        
        # Update session analytics and context
        self._update_session_from_message(session, message, metadata)
        
        # Save session
        self._save_session(session)
        
        # Index message for search (async task)
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._index_message(message, target_session_id))
        except RuntimeError:
            logger.debug("No event loop running, skipping message indexing")
        
        return message_id
    
    def _calculate_message_importance(self, message: SessionMessage) -> float:
        """Calculate message importance score"""
        score = 0.3  # Base score
        
        # Intent-based scoring
        intent_scores = {
            "planning": 0.4, "booking": 0.4, "complaint": 0.3,
            "modification": 0.3, "recommendation": 0.2, "query": 0.1
        }
        if message.intent:
            score += intent_scores.get(message.intent, 0.1)
        
        # Confidence scoring
        if message.confidence:
            if message.confidence > 0.8:
                score += 0.2
            elif message.confidence < 0.5:
                score += 0.1  # Low confidence might indicate issues
        
        # Topic diversity
        if len(message.topics) > 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _update_session_from_message(
        self, 
        session: SessionMetadata, 
        message: SessionMessage, 
        metadata: Optional[Dict[str, Any]]
    ):
        """Update session analytics from new message"""
        
        # Update main topics
        if message.topics:
            for topic in message.topics:
                if topic not in session.main_topics:
                    session.main_topics.append(topic)
        
        # Update travel context if metadata contains travel-related info
        if metadata:
            travel_updates = {}
            for key in ["destination", "travel_dates", "budget", "travelers", "preferences"]:
                if key in metadata:
                    travel_updates[key] = metadata[key]
            
            if travel_updates:
                session.travel_context.update(travel_updates)
                
                # Extract preferences for session
                if "preferences" in travel_updates:
                    prefs = travel_updates["preferences"]
                    if isinstance(prefs, dict):
                        session.learned_preferences.update(prefs)
        
        # Update session analytics
        self._update_session_analytics(session)
    
    async def _index_message(self, message: SessionMessage, session_id: str):
        """Index message content for search"""
        try:
            doc = Document(
                id=f"message_{message.id}",
                content=f"""
User Message: {message.user_message}
Agent Response: {message.agent_response}
Intent: {message.intent or "unknown"}
Topics: {', '.join(message.topics)}
Timestamp: {message.timestamp.isoformat()}
                """,
                metadata={
                    "session_id": session_id,
                    "message_id": message.id,
                    "type": "session_message",
                    "intent": message.intent or "unknown",
                    "importance_score": message.importance_score,
                    "confidence": message.confidence or 0.5
                }
            )
            
            success = await self.rag_engine.index_documents([doc])
            if success:
                logger.debug(f"✅ Indexed session message: {message.id}")
                
        except Exception as e:
            logger.error(f"Failed to index message {message.id}: {e}")

    def get_messages(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[SessionMessage]:
        """Enhanced message retrieval with filtering"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return []
        
        session = self.sessions[target_session_id]
        messages = session.messages[offset:]
        
        if limit:
            messages = messages[:limit]
        
        return messages
    
    def clear_messages(self, session_id: Optional[str] = None) -> bool:
        """Enhanced message clearing with index cleanup"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return False
        
        session = self.sessions[target_session_id]
        
        # Clear from RAG index (async task)
        message_ids = [f"message_{msg.id}" for msg in session.messages]
        if message_ids:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._cleanup_message_index(message_ids))
            except RuntimeError:
                logger.debug("No event loop running, skipping message index cleanup")
        
        # Clear messages
        session.messages.clear()
        session.total_messages = 0
        session.last_activity = datetime.now(timezone.utc)
        
        # Reset analytics
        session.main_topics.clear()
        session.satisfaction_score = 0.5
        session.complexity_score = 0.5
        session.resolution_status = "ongoing"
        
        # Save session
        self._save_session(session)
        return True
    
    async def _cleanup_message_index(self, message_ids: List[str]):
        """Remove messages from RAG index"""
        try:
            await self.rag_engine.vector_store.delete_documents(message_ids)
        except Exception as e:
            logger.warning(f"Failed to cleanup message index: {e}")

    async def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        use_rag: bool = True
    ) -> List[SessionMessage]:
        """Enhanced message search with RAG support"""
        
        if use_rag and self.rag_engine:
            try:
                # Build filter metadata
                filter_metadata = {"type": "session_message"}
                if session_id:
                    filter_metadata["session_id"] = session_id
                
                # Use RAG for semantic search
                retrieval_result = await self.rag_engine.retrieve(
                    query=query,
                    top_k=limit * 2,
                    filter_metadata=filter_metadata
                )
                
                relevant_messages = []
                for doc in retrieval_result.documents:
                    message_id = doc.metadata.get("message_id")
                    session_id_meta = doc.metadata.get("session_id")
                    
                    if message_id and session_id_meta and session_id_meta in self.sessions:
                        session = self.sessions[session_id_meta]
                        for message in session.messages:
                            if message.id == message_id:
                                relevant_messages.append(message)
                                break
                
                logger.info(f"RAG search found {len(relevant_messages)} relevant messages")
                return relevant_messages[:limit]
                
            except Exception as e:
                logger.warning(f"RAG message search failed: {e}, falling back to keyword search")
        
        # Fallback to keyword search
        query_lower = query.lower()
        matching_messages = []
        
        # Determine which sessions to search
        if session_id:
            sessions_to_search = [self.sessions[session_id]] if session_id in self.sessions else []
        else:
            sessions_to_search = self.sessions.values()
        
        for session in sessions_to_search:
            for message in session.messages:
                score = 0
                if query_lower in message.user_message.lower():
                    score += 2
                if query_lower in message.agent_response.lower():
                    score += 2
                if message.intent and query_lower in message.intent.lower():
                    score += 1
                if any(query_lower in topic.lower() for topic in message.topics):
                    score += 1
                
                if score > 0:
                    matching_messages.append((message, score))
        
        # Sort by relevance and recency
        matching_messages.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        return [message for message, _ in matching_messages[:limit]]
    
    def get_session_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced session statistics with analytics"""
        if session_id:
            session = self.sessions.get(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Calculate advanced metrics
            message_lengths = [len(msg.user_message) + len(msg.agent_response) for msg in session.messages]
            avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
            
            intent_distribution = {}
            for msg in session.messages:
                if msg.intent:
                    intent_distribution[msg.intent] = intent_distribution.get(msg.intent, 0) + 1
            
            confidence_scores = [msg.confidence for msg in session.messages if msg.confidence]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                "session_id": session_id,
                "basic_stats": {
                    "total_messages": len(session.messages),
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                    "status": session.status,
                    "resolution_status": session.resolution_status
                },
                "content_analysis": {
                    "main_topics": session.main_topics,
                    "travel_context_keys": list(session.travel_context.keys()),
                    "learned_preferences": session.learned_preferences,
                    "average_message_length": avg_message_length,
                    "intent_distribution": intent_distribution
                },
                "quality_metrics": {
                    "satisfaction_score": session.satisfaction_score,
                    "complexity_score": session.complexity_score,
                    "average_confidence": avg_confidence,
                    "importance_scores": [msg.importance_score for msg in session.messages]
                },
                "storage_info": {
                    "storage_size_kb": self._get_session_file_size(session_id),
                    "indexed_in_rag": True
                }
            }
        else:
            # Global statistics
            total_messages = sum(len(session.messages) for session in self.sessions.values())
            total_size = sum(self._get_session_file_size(sid) for sid in self.sessions.keys())
            
            avg_satisfaction = sum(s.satisfaction_score for s in self.sessions.values()) / len(self.sessions) if self.sessions else 0
            avg_complexity = sum(s.complexity_score for s in self.sessions.values()) / len(self.sessions) if self.sessions else 0
            
            return {
                "global_stats": {
                    "total_sessions": len(self.sessions),
                    "total_messages": total_messages,
                    "total_storage_kb": total_size,
                    "average_messages_per_session": total_messages / len(self.sessions) if self.sessions else 0
                },
                "quality_overview": {
                    "average_satisfaction_score": avg_satisfaction,
                    "average_complexity_score": avg_complexity,
                    "sessions_by_status": {
                        status.value: len([s for s in self.sessions.values() if s.status == status])
                        for status in SessionStatus
                    }
                },
                "time_range": {
                    "oldest_session": min(self.sessions.values(), key=lambda x: x.created_at).created_at if self.sessions else None,
                    "newest_session": max(self.sessions.values(), key=lambda x: x.created_at).created_at if self.sessions else None
                },
                "integration_status": {
                    "rag_enabled": True,
                    "llm_enabled": self.llm_service is not None
                }
            }
    
    def _get_session_file_size(self, session_id: str) -> float:
        """Get session file size in KB"""
        try:
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                return session_file.stat().st_size / 1024
        except Exception:
            pass
        return 0.0


# Enhanced global session manager instance
session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Get enhanced session manager instance with full core integration"""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
        logger.info("✅ Initialized enhanced session manager with RAG and LLM integration")
    return session_manager 