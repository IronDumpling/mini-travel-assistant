"""
Session Management Endpoints - Session CRUD operations
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

class SessionCreate(BaseModel):
    """Request model for creating a session"""
    title: Optional[str] = None
    description: Optional[str] = None

class SessionResponse(BaseModel):
    """Response model for session operations"""
    session_id: str
    message: str
    session: Optional[dict] = None

@router.get("/sessions")
async def list_sessions():
    """List all sessions"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        sessions = session_manager.list_sessions()
        return {
            "sessions": [session.model_dump() for session in sessions],
            "current_session": session_manager.current_session_id,
            "total": len(sessions)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve sessions: {str(e)}"}
        )

@router.post("/sessions", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    """Create a new session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        session_id = session_manager.create_session(
            title=session_data.title, 
            description=session_data.description
        )
        session = session_manager.get_current_session()
        
        return SessionResponse(
            session_id=session_id,
            session=session.model_dump() if session else None,
            message="Session created successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create session: {str(e)}"
        )

@router.put("/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    """Switch to a different session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        success = session_manager.switch_session(session_id)
        if success:
            return {"message": f"Switched to session {session_id}"}
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to switch session: {str(e)}"}
        )

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        # Switch to session to get details
        success = session_manager.switch_session(session_id)
        if success:
            session = session_manager.get_current_session()
            return {
                "session": session.model_dump() if session else None,
                "message": f"Session {session_id} details retrieved"
            }
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve session: {str(e)}"}
        )

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        # TODO: Implement session deletion in session manager
        return JSONResponse(
            status_code=501,
            content={"error": "Session deletion not yet implemented"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete session: {str(e)}"}
        )

@router.get("/sessions/{session_id}/statistics")
async def get_session_statistics(session_id: str):
    """Get detailed statistics for a specific session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        stats = session_manager.get_session_statistics(session_id)
        return stats
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get session statistics: {str(e)}"}
        )

@router.get("/sessions/statistics")
async def get_all_sessions_statistics():
    """Get statistics for all sessions"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        stats = session_manager.get_session_statistics()
        return stats
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get sessions statistics: {str(e)}"}
        )

@router.get("/sessions/{session_id}/search")
async def search_session_messages(session_id: str, query: str, limit: int = 10):
    """Search messages within a specific session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        messages = session_manager.search_messages(query, session_id, limit)
        return {
            "session_id": session_id,
            "query": query,
            "results": [msg.model_dump() for msg in messages],
            "total_found": len(messages)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to search messages: {str(e)}"}
        )

@router.get("/sessions/search")
async def search_all_messages(query: str, limit: int = 20):
    """Search messages across all sessions"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        messages = session_manager.search_messages(query, None, limit)
        return {
            "query": query,
            "results": [msg.model_dump() for msg in messages],
            "total_found": len(messages)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to search messages: {str(e)}"}
        )

@router.post("/sessions/{session_id}/backup")
async def backup_session(session_id: str):
    """Create a backup of a specific session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        success = session_manager.backup_session(session_id)
        if success:
            return {"message": f"Session {session_id} backed up successfully"}
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found or backup failed"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to backup session: {str(e)}"}
        )

@router.get("/sessions/{session_id}/export")
async def export_session_messages(session_id: str, format: str = "json"):
    """Export session messages in different formats"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        if format not in ["json", "txt", "csv"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Supported formats: json, txt, csv"}
            )
        
        export_path = session_manager.export_session_messages(session_id, format)
        if export_path:
            return {
                "message": f"Session {session_id} exported successfully",
                "export_path": export_path,
                "format": format
            }
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found or export failed"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to export session: {str(e)}"}
        ) 