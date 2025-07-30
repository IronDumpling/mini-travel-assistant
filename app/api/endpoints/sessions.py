"""
Session Management Endpoints - Session CRUD operations with RAG-enhanced search
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from app.memory.conversation_memory import get_conversation_memory
from app.core.logging_config import get_logger
from datetime import datetime

logger = get_logger(__name__)

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
        
        # Check if session exists first
        sessions = session_manager.list_sessions()
        session_exists = any(s.session_id == session_id for s in sessions)
        
        if not session_exists:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )
        
        # Delete the session
        try:
            # If this is the current session, we need to handle switching
            current_session_id = session_manager.current_session_id
            
            # Try to delete using session manager
            success = session_manager.delete_session(session_id)
            
            if success:
                # If we deleted the current session, switch to another one or clear current
                if current_session_id == session_id:
                    remaining_sessions = session_manager.list_sessions()
                    if remaining_sessions:
                        session_manager.switch_session(remaining_sessions[0].session_id)
                    else:
                        session_manager.current_session_id = None
                
                return {
                    "message": f"Session {session_id} deleted successfully",
                    "deleted_session_id": session_id,
                    "was_current_session": current_session_id == session_id
                }
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Failed to delete session - deletion operation failed"}
                )
                
        except AttributeError:
            # If session_manager doesn't have delete_session method, try manual deletion
            logger.warning("Session manager doesn't have delete_session method, attempting manual deletion")
            
            # Try to manually remove from sessions list and files
            import os
            from pathlib import Path
            
            # Remove from memory
            if hasattr(session_manager, 'sessions'):
                session_manager.sessions = {k: v for k, v in session_manager.sessions.items() if k != session_id}
            
            # Try to remove session file if it exists
            data_dir = Path(session_manager.storage_path)
            session_file = data_dir / f"{session_id}.json"
            
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Deleted session file: {session_file}")
            
            # If this was the current session, switch to another one
            if session_manager.current_session_id == session_id:
                remaining_sessions = session_manager.list_sessions()
                if remaining_sessions:
                    session_manager.switch_session(remaining_sessions[0].session_id)
                else:
                    session_manager.current_session_id = None
            
            return {
                "message": f"Session {session_id} deleted successfully (manual deletion)",
                "deleted_session_id": session_id,
                "method": "manual_file_deletion"
            }
            
    except Exception as e:
        logger.error(f"Session deletion failed: {str(e)}")
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

# 🆕 RAG-Enhanced Intelligent Features

@router.get("/sessions/{session_id}/intelligent-search")
async def intelligent_search_session(session_id: str, query: str, limit: int = 10):
    """使用RAG进行智能语义搜索"""
    try:
        conversation_memory = get_conversation_memory()
        
        # 使用RAG进行语义搜索
        results = await conversation_memory.search_conversations(
            query=query,
            session_id=session_id
        )
        
        return {
            "session_id": session_id,
            "query": query,
            "results": [
                {
                    "user_message": turn.user_message,
                    "agent_response": turn.agent_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "importance_score": turn.importance_score,
                    "intent": turn.intent,
                    "sentiment": turn.sentiment
                }
                for turn in results[:limit]
            ],
            "total_found": len(results),
            "search_type": "semantic_rag"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"智能搜索失败: {str(e)}"}
        )

@router.get("/sessions/{session_id}/preferences")
async def extract_user_preferences(session_id: str):
    """提取用户旅行偏好"""
    try:
        conversation_memory = get_conversation_memory()
        
        # 使用RAG分析用户偏好
        preferences = await conversation_memory.extract_user_preferences(session_id)
        
        return {
            "session_id": session_id,
            "preferences": preferences,
            "extraction_method": "rag_analysis",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"偏好提取失败: {str(e)}"}
        )

@router.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """生成智能会话总结"""
    try:
        conversation_memory = get_conversation_memory()
        
        # 生成智能总结
        summary = await conversation_memory.get_session_summary(session_id)
        
        return {
            "session_id": session_id,
            "summary": summary,
            "generation_method": "rag_enhanced",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"总结生成失败: {str(e)}"}
        )

@router.get("/sessions/{session_id}/context")
async def get_relevant_context(session_id: str, query: str, max_turns: int = 5):
    """获取与查询相关的对话上下文"""
    try:
        conversation_memory = get_conversation_memory()
        
        # 获取相关上下文
        context_turns = await conversation_memory.get_relevant_context(
            session_id=session_id,
            query=query,
            max_turns=max_turns
        )
        
        return {
            "session_id": session_id,
            "query": query,
            "relevant_context": [
                {
                    "user_message": turn.user_message,
                    "agent_response": turn.agent_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "importance_score": turn.importance_score,
                    "intent": turn.intent
                }
                for turn in context_turns
            ],
            "context_count": len(context_turns),
            "max_requested": max_turns
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"上下文获取失败: {str(e)}"}
        )

@router.get("/conversations/global-search")
async def global_intelligent_search(query: str, limit: int = 20):
    """跨所有会话的全局智能搜索"""
    try:
        conversation_memory = get_conversation_memory()
        
        # 全局语义搜索
        results = await conversation_memory.search_conversations(
            query=query,
            session_id=None  # 搜索所有会话
        )
        
        return {
            "query": query,
            "results": [
                {
                    "user_message": turn.user_message,
                    "agent_response": turn.agent_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "importance_score": turn.importance_score,
                    "intent": turn.intent,
                    "sentiment": turn.sentiment
                }
                for turn in results[:limit]
            ],
            "total_found": len(results),
            "search_scope": "global",
            "search_type": "semantic_rag"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"全局搜索失败: {str(e)}"}
        ) 