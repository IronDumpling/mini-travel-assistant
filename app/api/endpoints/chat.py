"""
Chat Endpoints - Conversational AI interface
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage
from app.memory.session_manager import get_session_manager
from datetime import datetime

router = APIRouter()

class ChatMessage(BaseModel):
    """Chat message for conversational travel planning"""
    message: str
    session_id: Optional[str] = None
    enable_refinement: bool = True

class ChatResponse(BaseModel):
    """Response from the travel agent"""
    success: bool
    content: str
    confidence: float
    actions_taken: List[str]
    next_steps: List[str]
    session_id: str
    refinement_details: Optional[dict] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """
    Chat with the travel agent using natural language.
    This is the main endpoint that leverages the self-refine loop.
    """
    try:
        # Get or create session
        session_manager = get_session_manager()
        if message.session_id:
            session_manager.switch_session(message.session_id)
        else:
            session_id = session_manager.create_session(
                title=f"Travel Planning - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description="AI-powered travel planning conversation"
            )
            message.session_id = session_id

        # Create travel agent
        agent = TravelAgent()
        agent.configure_refinement(enabled=message.enable_refinement)
        
        # Create agent message
        agent_message = AgentMessage(
            sender="user",
            receiver="travel_agent",
            content=message.message,
            metadata={"session_id": message.session_id}
        )
        
        # Process with self-refinement
        if message.enable_refinement:
            response = await agent.plan_travel(agent_message)
        else:
            response = await agent.process_message(agent_message)
        
        # Store conversation in session
        session_manager.add_message(
            user_message=message.message,
            agent_response=response.content,
            metadata={
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "refinement_used": message.enable_refinement
            }
        )
        
        # Prepare response
        chat_response = ChatResponse(
            success=response.success,
            content=response.content,
            confidence=response.confidence,
            actions_taken=response.actions_taken,
            next_steps=response.next_steps,
            session_id=message.session_id
        )
        
        # Add refinement details if available
        if "refinement_iteration" in response.metadata:
            chat_response.refinement_details = {
                "final_iteration": response.metadata["refinement_iteration"],
                "final_quality_score": response.metadata["quality_score"],
                "refinement_status": response.metadata["refinement_status"],
                "quality_dimensions": agent.get_quality_dimensions()
            }
        
        return chat_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """
    Get chat history for a specific session
    """
    try:
        session_manager = get_session_manager()
        
        # Switch to the requested session
        success = session_manager.switch_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get session details
        session = session_manager.get_current_session()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get recent messages (limited)
        messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
        
        return {
            "session_id": session_id,
            "total_messages": len(session.messages),
            "returned_messages": len(messages),
            "messages": [msg.model_dump() for msg in messages]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session
    """
    try:
        session_manager = get_session_manager()
        
        # Switch to the requested session
        success = session_manager.switch_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Clear messages using session manager
        success = session_manager.clear_messages(session_id)
        if success:
            return {
                "message": f"Chat history cleared for session {session_id}",
                "status": "success"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to clear chat history"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}") 