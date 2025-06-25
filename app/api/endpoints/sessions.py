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