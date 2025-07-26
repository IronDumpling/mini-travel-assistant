"""
Chat Endpoints - Conversational AI interface
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from app.agents.travel_agent import get_travel_agent
from app.agents.base_agent import AgentMessage
from app.memory.session_manager import get_session_manager
from app.memory.conversation_memory import get_conversation_memory
from app.core.logging_config import get_logger
from datetime import datetime

logger = get_logger(__name__)

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

        # ✅ 使用单例Agent，避免全局配置污染
        agent = get_travel_agent()
        
        # 🆕 获取session历史并传给agent (使用RAG增强的对话记忆)
        session = session_manager.get_current_session()
        conversation_memory = get_conversation_memory()
        conversation_history = []
        
        if session and session.messages:
            # 首先尝试使用RAG搜索获取相关对话上下文
            try:
                relevant_turns = await conversation_memory.get_relevant_context(
                    session_id=message.session_id,
                    query=message.message,
                    max_turns=5
                )
                
                if relevant_turns:
                    # 使用RAG找到的相关对话
                    for turn in relevant_turns:
                        conversation_history.append({
                            "user": turn.user_message,
                            "assistant": turn.agent_response,
                            "timestamp": turn.timestamp.isoformat(),
                            "importance": turn.importance_score,
                            "intent": turn.intent
                        })
                    logger.info(f"使用RAG搜索找到 {len(relevant_turns)} 条相关对话")
                else:
                    # 如果RAG没有找到相关内容，回退到最近的对话
                    recent_messages = session.messages[-3:]  # 最近3条
                    for msg in recent_messages:
                        conversation_history.append({
                            "user": msg.user_message,
                            "assistant": msg.agent_response,
                            "timestamp": msg.timestamp.isoformat()
                        })
                    logger.info(f"RAG未找到相关内容，使用最近 {len(recent_messages)} 条对话")
                        
            except Exception as e:
                logger.warning(f"RAG搜索失败，使用最近对话: {e}")
                # 回退到简单的最近对话
                recent_messages = session.messages[-3:]  # 最近3条
                for msg in recent_messages:
                    conversation_history.append({
                        "user": msg.user_message,
                        "assistant": msg.agent_response,
                        "timestamp": msg.timestamp.isoformat()
                    })
        
        # Create agent message with conversation history
        agent_message = AgentMessage(
            sender="user",
            receiver="travel_agent",
            content=message.message,
            metadata={
                "session_id": message.session_id,
                "conversation_history": conversation_history  # 🆕 传递历史
            }
        )
        
        # Temporary configuration processing - avoid global configuration pollution
        original_refine_enabled = agent.refine_enabled
        
        try:
            # If user's refinement setting is different from current agent setting, temporarily adjust
            if message.enable_refinement != original_refine_enabled:
                logger.info(f"Temporarily adjust refinement setting: {original_refine_enabled} -> {message.enable_refinement}")
                agent.configure_refinement(enabled=message.enable_refinement)
            
            # Process with self-refinement based on user preference
            if message.enable_refinement:
                response = await agent.plan_travel(agent_message)
            else:
                response = await agent.process_message(agent_message)
                
        finally:
            # 🔄 Restore original configuration to avoid affecting other API calls
            if message.enable_refinement != original_refine_enabled:
                logger.info(f"Restore original refinement setting: {message.enable_refinement} -> {original_refine_enabled}")
                agent.configure_refinement(enabled=original_refine_enabled)
        
        # 📝 Store conversation in both systems
        # 1. Store in session manager (basic storage)
        session_manager.add_message(
            user_message=message.message,
            agent_response=response.content,
            metadata={
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "refinement_used": message.enable_refinement
            }
        )
        
        # 2. Store in conversation memory (RAG enhanced analysis and indexing)
        try:
            session = session_manager.get_current_session()
            should_analyze = False
            
            if session and len(session.messages) % 2 == 1:  # Every 2nd message (odd count means just added 2nd)
                should_analyze = True

            enhanced_metadata = {
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "refinement_used": message.enable_refinement,
                "intent": response.metadata.get("intent", {}).get("type"),
                "sentiment": "positive" if response.confidence > 0.7 else "neutral"
            }
            
            await conversation_memory.add_turn(
                session_id=message.session_id,
                user_message=message.message,
                agent_response=response.content,
                metadata=enhanced_metadata,
                skip_analysis=not should_analyze
            )
            
        except Exception as e:
            logger.error(f"Failed to store in conversation memory system: {e}")
        
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
            # Get refinement history to extract iterations for test framework
            refinement_history = response.metadata.get("refinement_history", [])
            actual_refinement_loops = response.metadata.get("actual_refinement_loops", max(0, response.metadata.get("refinement_iteration", 1) - 1))
            
            # Construct iterations array in the format expected by test framework
            iterations = []
            for i, hist_item in enumerate(refinement_history):
                iteration_detail = {
                    "iteration": hist_item.get("iteration", i + 1),
                    "confidence": response.confidence if i == len(refinement_history) - 1 else hist_item.get("score", 0.0),
                    "quality_scores": {"overall": hist_item.get("score", 0.0)},
                    "actions_taken": response.actions_taken if i == len(refinement_history) - 1 else [],
                    "response_time": 0.0,  # Individual iteration time not tracked yet
                    "improvements_made": hist_item.get("suggestions", [])
                }
                iterations.append(iteration_detail)
            
            chat_response.refinement_details = {
                "final_iteration": response.metadata["refinement_iteration"],
                "final_quality_score": response.metadata["quality_score"],
                "refinement_status": response.metadata["refinement_status"],
                "quality_dimensions": agent.get_quality_dimensions(),
                "user_requested_refinement": message.enable_refinement,
                "configuration_isolation": True,  
                # 🔧 Add iterations array for test framework compatibility
                "iterations": iterations,
                "actual_refinement_loops": actual_refinement_loops,
                "total_iterations": response.metadata.get("total_iterations", response.metadata["refinement_iteration"])
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