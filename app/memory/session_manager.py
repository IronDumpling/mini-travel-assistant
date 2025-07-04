"""
Session Manager Module - Single-user session management

Designed for local single-user deployment without authentication complexity.
Manages conversation sessions, context switching, and memory organization.

TODO: Implement the following features
1. Session lifecycle management (create, activate, archive)
2. Context switching between different travel planning sessions
3. Session-based memory isolation
4. Automatic session cleanup and archiving
5. Session metadata and analytics
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

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

class SessionManager:
    """Single-user session manager"""
    
    def __init__(self, storage_path: str = "./data/sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, SessionMetadata] = {}
        self.current_session_id: Optional[str] = None
        self.max_active_sessions = 10  # Reasonable limit for single user
        
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
        
        return session_id
    
    def get_current_session(self) -> Optional[SessionMetadata]:
        """Get current active session"""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to a different session"""
        if session_id in self.sessions:
            # Update last activity of previous session
            if self.current_session_id:
                prev_session = self.sessions.get(self.current_session_id)
                if prev_session:
                    prev_session.last_activity = datetime.now(timezone.utc)
                    self._save_session(prev_session)
            
            # Switch to new session
            self.current_session_id = session_id
            current_session = self.sessions[session_id]
            current_session.status = SessionStatus.ACTIVE
            current_session.last_activity = datetime.now(timezone.utc)
            self._save_session(current_session)
            
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
        
        # Sort by last activity (most recent first)
        sessions.sort(key=lambda x: x.last_activity, reverse=True)
        
        return sessions[:limit]
    
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
        
        self._save_session(session)
    
    def archive_session(self, session_id: str) -> bool:
        """Archive a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.status = SessionStatus.ARCHIVED
            session.last_activity = datetime.now(timezone.utc)
            
            # If archiving current session, clear current session
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            self._save_session(session)
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session"""
        if session_id in self.sessions:
            # Remove from memory
            del self.sessions[session_id]
            
            # Remove from disk
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Clear current session if deleted
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            return True
        return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_sessions = len(self.sessions)
        active_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE])
        archived_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ARCHIVED])
        
        total_messages = sum(s.total_messages for s in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "archived_sessions": archived_sessions,
            "current_session_id": self.current_session_id,
            "total_messages": total_messages,
            "storage_path": str(self.storage_path)
        }
    
    def search_sessions(self, query: str) -> List[SessionMetadata]:
        """Search sessions by title, description, or context"""
        query_lower = query.lower()
        matching_sessions = []
        
        for session in self.sessions.values():
            # Search in title and description
            if (query_lower in session.title.lower() or 
                (session.description and query_lower in session.description.lower())):
                matching_sessions.append(session)
                continue
            
            # Search in travel context
            context_str = json.dumps(session.travel_context).lower()
            if query_lower in context_str:
                matching_sessions.append(session)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in session.tags):
                matching_sessions.append(session)
        
        # Sort by relevance (last activity)
        matching_sessions.sort(key=lambda x: x.last_activity, reverse=True)
        return matching_sessions
    
    def _load_sessions(self):
        """Load sessions from disk"""
        if not self.storage_path.exists():
            return
        
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    metadata = SessionMetadata(**session_data)
                    self.sessions[metadata.session_id] = metadata
            except Exception as e:
                print(f"Warning: Failed to load session {session_file}: {e}")
        
        # Set most recent active session as current
        active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        if active_sessions:
            most_recent = max(active_sessions, key=lambda x: x.last_activity)
            self.current_session_id = most_recent.session_id
    
    def _save_session(self, session: SessionMetadata):
        """Save session to disk"""
        session_file = self.storage_path / f"{session.session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.model_dump(), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save session {session.session_id}: {e}")
    
    def _cleanup_old_sessions(self):
        """Cleanup old sessions if too many active"""
        active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        
        if len(active_sessions) > self.max_active_sessions:
            # Sort by last activity and archive oldest
            active_sessions.sort(key=lambda x: x.last_activity)
            sessions_to_archive = active_sessions[:-self.max_active_sessions]
            
            for session in sessions_to_archive:
                session.status = SessionStatus.INACTIVE
                self._save_session(session)

    def add_message(
        self,
        user_message: str,
        agent_response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to the session"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            raise ValueError(f"Session {target_session_id} not found")
        
        session = self.sessions[target_session_id]
        
        # Create message
        message_id = f"{target_session_id}_{len(session.messages)}"
        message = SessionMessage(
            id=message_id,
            user_message=user_message,
            agent_response=agent_response,
            metadata=metadata or {}
        )
        
        # Add to session
        session.messages.append(message)
        session.total_messages += 1
        session.last_activity = datetime.now(timezone.utc)
        
        # Update travel context if metadata contains travel-related info
        if metadata:
            travel_updates = {}
            if "destination" in metadata:
                travel_updates["destination"] = metadata["destination"]
            if "travel_dates" in metadata:
                travel_updates["travel_dates"] = metadata["travel_dates"]
            if "budget" in metadata:
                travel_updates["budget"] = metadata["budget"]
            
            if travel_updates:
                session.travel_context.update(travel_updates)
        
        # Save session
        self._save_session(session)
        
        return message_id
    
    def get_messages(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[SessionMessage]:
        """Get messages from a session"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return []
        
        session = self.sessions[target_session_id]
        messages = session.messages[offset:]
        
        if limit:
            messages = messages[:limit]
        
        return messages
    
    def clear_messages(self, session_id: Optional[str] = None) -> bool:
        """Clear all messages from a session"""
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return False
        
        session = self.sessions[target_session_id]
        session.messages.clear()
        session.total_messages = 0
        session.last_activity = datetime.now(timezone.utc)
        
        # Save session
        self._save_session(session)
        return True

    def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SessionMessage]:
        """Search messages within a session or across all sessions"""
        query_lower = query.lower()
        matching_messages = []
        
        # Determine which sessions to search
        if session_id:
            sessions_to_search = [self.sessions[session_id]] if session_id in self.sessions else []
        else:
            sessions_to_search = self.sessions.values()
        
        for session in sessions_to_search:
            for message in session.messages:
                # Search in user message and agent response
                if (query_lower in message.user_message.lower() or 
                    query_lower in message.agent_response.lower()):
                    matching_messages.append(message)
        
        # Sort by timestamp (most recent first)
        matching_messages.sort(key=lambda x: x.timestamp, reverse=True)
        return matching_messages[:limit]
    
    def get_session_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed statistics for a session or all sessions"""
        if session_id:
            session = self.sessions.get(session_id)
            if not session:
                return {"error": "Session not found"}
            
            return {
                "session_id": session_id,
                "total_messages": len(session.messages),
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "status": session.status,
                "travel_context_keys": list(session.travel_context.keys()),
                "average_response_length": sum(len(msg.agent_response) for msg in session.messages) / len(session.messages) if session.messages else 0,
                "storage_size_kb": self._get_session_file_size(session_id)
            }
        else:
            # Global statistics
            total_messages = sum(len(session.messages) for session in self.sessions.values())
            total_size = sum(self._get_session_file_size(sid) for sid in self.sessions.keys())
            
            return {
                "total_sessions": len(self.sessions),
                "total_messages": total_messages,
                "total_storage_kb": total_size,
                "average_messages_per_session": total_messages / len(self.sessions) if self.sessions else 0,
                "oldest_session": min(self.sessions.values(), key=lambda x: x.created_at).created_at if self.sessions else None,
                "newest_session": max(self.sessions.values(), key=lambda x: x.created_at).created_at if self.sessions else None
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
    
    def backup_session(self, session_id: str, backup_path: Optional[str] = None) -> bool:
        """Create a backup of a specific session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            if not backup_path:
                backup_path = f"./data/backups/session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(session.model_dump(), f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Failed to backup session {session_id}: {e}")
            return False
    
    def export_session_messages(
        self, 
        session_id: str, 
        format: str = "json",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Export session messages in different formats (json, txt, csv)"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        if not output_path:
            output_path = f"./data/exports/session_{session_id}_messages.{format}"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    messages_data = [msg.model_dump() for msg in session.messages]
                    json.dump(messages_data, f, indent=2, default=str)
            
            elif format == "txt":
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Chat History for Session: {session.title}\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"Created: {session.created_at}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, msg in enumerate(session.messages, 1):
                        f.write(f"Message #{i} - {msg.timestamp}\n")
                        f.write(f"User: {msg.user_message}\n")
                        f.write(f"Assistant: {msg.agent_response}\n")
                        f.write("-" * 40 + "\n\n")
            
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'user_message', 'agent_response', 'confidence', 'actions_taken'])
                    for msg in session.messages:
                        writer.writerow([
                            msg.timestamp,
                            msg.user_message,
                            msg.agent_response,
                            msg.metadata.get('confidence', ''),
                            ', '.join(msg.metadata.get('actions_taken', []))
                        ])
            
            return str(output_file)
        
        except Exception as e:
            print(f"Failed to export session messages: {e}")
            return None

# Global session manager instance
session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Get session manager instance"""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager 