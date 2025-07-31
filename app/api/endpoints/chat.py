"""
Chat Endpoints - Conversational AI interface
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.agents.travel_agent import get_travel_agent
from app.agents.base_agent import AgentMessage
from app.memory.session_manager import get_session_manager
from app.memory.conversation_memory import get_conversation_memory
from app.core.logging_config import get_logger
from datetime import datetime, timedelta
import asyncio

logger = get_logger(__name__)

async def _async_update_plan(session_id: str, user_message: str, agent_response):
    """Asynchronously update plan in background without blocking chat response"""
    try:
        from app.core.plan_manager import get_plan_manager
        plan_manager = get_plan_manager()
        
        # Check if this is a plan request that needs detailed generation
        is_plan_request = agent_response.metadata.get("is_plan_request", False)
        
        if is_plan_request:
            # Now generating detailed plan in the background
            logger.info(f"Background: Generating detailed plan for session {session_id}")
            
            # Get the execution_result passed
            execution_result = agent_response.metadata.get("execution_result_for_plan", {})
            intent = agent_response.metadata.get("intent", {})
            
            if execution_result:
                # Use travel_agent's LLM plan generation capabilities
                from app.agents.travel_agent import get_travel_agent
                travel_agent = get_travel_agent()
                
                # Generate detailed structured plan
                detailed_plan_response = await travel_agent._generate_plan_aware_response(
                    execution_result, intent, user_message
                )
                
                # Use detailed plan update
                plan_update_result = await plan_manager.update_plan_from_structured_response(
                    session_id=session_id,
                    agent_response=detailed_plan_response
                )
            else:
                # If no execution_result, fall back to basic response parsing
                plan_update_result = await plan_manager.update_plan_from_chat_response(
                    session_id=session_id,
                    user_message=user_message,
                    agent_response=agent_response.content,
                    response_metadata=agent_response.metadata or {}
                )
        else:
            # Standard plan update for non-plan requests
            logger.info(f"Background: Standard plan update for session {session_id}")
            plan_update_result = await plan_manager.update_plan_from_chat_response(
                session_id=session_id,
                user_message=user_message,
                agent_response=agent_response.content,
                response_metadata=agent_response.metadata or {}
            )
        
        logger.info(f"Background plan update completed for session {session_id}: {plan_update_result}")
        
        # Update session with plan generation status
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        session_manager.update_session_context(
            session_id=session_id,
            context_updates={
                "plan_generation_status": "completed",
                "plan_update_result": plan_update_result
            }
        )
        
    except Exception as e:
        logger.error(f"Background plan update failed for session {session_id}: {e}")
        # Update session with error status
        try:
            from app.memory.session_manager import get_session_manager
            session_manager = get_session_manager()
            session_manager.update_session_context(
                session_id=session_id,
                context_updates={
                    "plan_generation_status": "failed",
                    "plan_generation_error": str(e)
                }
            )
        except Exception as update_error:
            logger.error(f"Failed to update session status: {update_error}")

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
    plan_changes: Optional[Dict[str, Any]] = None  # New field for plan updates

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """
    Chat with the travel agent using natural language.
    This is the main endpoint that leverages the self-refine loop.
    """
    try:
        # Get or create session with proper validation
        session_manager = get_session_manager()
        if message.session_id:
            # Validate session exists before switching
            if message.session_id in session_manager.sessions:
                session_manager.switch_session(message.session_id)
                logger.info(f"Switched to existing session: {message.session_id}")
            else:
                logger.warning(f"Session {message.session_id} not found, creating new session")
                session_id = session_manager.create_session(
                    title=f"Travel Planning - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    description="AI-powered travel planning conversation"
                )
                message.session_id = session_id
        else:
            session_id = session_manager.create_session(
                title=f"Travel Planning - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description="AI-powered travel planning conversation"
            )
            message.session_id = session_id
            logger.info(f"Created new session: {session_id}")

        # Use singleton agent to avoid global configuration pollution   
        agent = get_travel_agent()
        
        # Get session history and pass to agent (using RAG enhanced conversation memory)
        session = session_manager.get_current_session()
        conversation_memory = get_conversation_memory()
        conversation_history = []
        
        if session and session.messages:
            # First try to use RAG to get relevant conversation context
            try:
                relevant_turns = await conversation_memory.get_relevant_context(
                    session_id=message.session_id,
                    query=message.message,
                    max_turns=5
                )
                
                if relevant_turns:
                    # Use RAG to find relevant conversation
                    for turn in relevant_turns:
                        conversation_history.append({
                            "user": turn.user_message,
                            "assistant": turn.agent_response,
                            "timestamp": turn.timestamp.isoformat(),
                            "importance": turn.importance_score,
                            "intent": turn.intent
                        })
                    logger.info(f"Found {len(relevant_turns)} relevant conversations using RAG")
                else:
                    # If RAG didn't find relevant content, fall back to recent conversation
                    recent_messages = session.messages[-3:]  # Last 3 messages
                    for msg in recent_messages:
                        conversation_history.append({
                            "user": msg.user_message,
                            "assistant": msg.agent_response,
                            "timestamp": msg.timestamp.isoformat()
                        })
                    logger.info(f"RAG didn't find relevant content, using recent {len(recent_messages)} messages")
                        
            except Exception as e:
                logger.warning(f"RAG search failed, using recent conversation: {e}")
                # Fall back to simple recent conversation
                recent_messages = session.messages[-3:]  # Last 3 messages
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
                "conversation_history": conversation_history  # Pass history
            }
        )
        
        # Pass refinement preference as parameter instead of modifying singleton state
        # This eliminates the race condition where concurrent requests interfere with each other
        if message.enable_refinement:
            response = await agent.plan_travel(agent_message, enable_refinement=True)
        else:
            response = await agent.process_message(agent_message)

        # Schedule plan generation asynchronously (non-blocking)
        plan_update_result = {"success": True, "status": "scheduled", "changes_made": ["Plan generation scheduled"]}
        try:
            # Schedule background plan update without waiting
            asyncio.create_task(
                _async_update_plan(message.session_id, message.message, response)
            )
            logger.info(f"Plan update scheduled for session {message.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to schedule plan update for session {message.session_id}: {e}")
            plan_update_result = {
                "success": False,
                "error": f"Plan scheduling failed: {str(e)}",
                "changes_made": []
            }
        
        # 📝 Store conversation in both systems
        # 1. Store in session manager (basic storage)
        session_manager.add_message(
            user_message=message.message,
            agent_response=response.content,
            metadata={
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "refinement_used": message.enable_refinement,
                "plan_changes": plan_update_result  # Include plan changes in metadata
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
            session_id=message.session_id,
            plan_changes=plan_update_result  # Include plan changes in response
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
        session_messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
        
        # Convert to frontend format
        formatted_messages = []
        for msg in session_messages:
            # Safely get metadata, ensuring it's never None for unpacking
            msg_metadata = getattr(msg, 'metadata', {}) or {}
            
            # Add user message
            formatted_messages.append({
                "role": "user",
                "content": msg.user_message,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg_metadata
            })
            
            # Add assistant response with a slightly later timestamp to maintain chronological order
            response_timestamp = msg.timestamp + timedelta(seconds=1)
            formatted_messages.append({
                "role": "assistant", 
                "content": msg.agent_response,
                "timestamp": response_timestamp.isoformat(),
                "metadata": {
                    "confidence": getattr(msg, 'confidence', None),
                    **msg_metadata
                }
            })
        
        return {
            "conversation_id": session_id,
            "messages": formatted_messages
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

@router.get("/chat/plan-status/{session_id}")
async def get_plan_generation_status(session_id: str):
    """
    Get plan generation status for a specific session
    """
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        session = session_manager.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get plan generation status from travel context
        plan_status = session.travel_context.get("plan_generation_status", "unknown")
        plan_error = session.travel_context.get("plan_generation_error")
        plan_result = session.travel_context.get("plan_update_result", {})
        
        return {
            "session_id": session_id,
            "plan_generation_status": plan_status,
            "plan_generation_error": plan_error,
            "plan_update_result": plan_result,
            "last_updated": session.last_activity.isoformat() if session.last_activity else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plan status: {str(e)}")