from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from app.models.schemas import TravelPreferences, TravelPlan, PlanUpdate
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage, AgentResponse
from app.memory.session_manager import get_session_manager
import json
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

class RefinementConfig(BaseModel):
    """Configuration for self-refinement"""
    enabled: bool = True
    quality_threshold: float = 0.75
    max_iterations: int = 3

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

@router.post("/plans", response_model=TravelPlan)
async def create_travel_plan(preferences: TravelPreferences):
    """
    Create a structured travel plan based on preferences.
    Uses the agent system with self-refinement for better quality.
    """
    try:
        # Convert preferences to natural language message
        message_content = _preferences_to_message(preferences)
        
        # Create travel agent
        agent = TravelAgent()
        
        # Create agent message
        agent_message = AgentMessage(
            sender="api_user",
            receiver="travel_agent",
            content=message_content,
            metadata={"preferences": preferences.model_dump()}
        )
        
        # Process with self-refinement
        response = await agent.plan_travel(agent_message)
        
        # Convert agent response to structured TravelPlan
        # TODO: Implement response parsing to TravelPlan structure
        # For now, return a basic structure
        return TravelPlan(
            id=f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            preferences=preferences,
            daily_plans=[],  # TODO: Parse from agent response
            total_cost=0.0,
            status="generated"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan creation failed: {str(e)}")

@router.get("/plans/{plan_id}", response_model=TravelPlan)
async def get_travel_plan(plan_id: str):
    """
    Get a specific travel plan by ID.
    TODO: Implement plan storage and retrieval
    """
    # TODO: Implement plan storage in session manager or database
    raise HTTPException(status_code=404, detail="Plan storage not yet implemented")

@router.put("/plans/{plan_id}", response_model=TravelPlan)
async def update_travel_plan(plan_id: str, update: PlanUpdate):
    """
    Update an existing travel plan with new preferences or feedback.
    Uses agent system to incorporate feedback and refine the plan.
    """
    try:
        if update.feedback:
            # Use feedback to refine the plan
            agent = TravelAgent()
            
            feedback_message = AgentMessage(
                sender="api_user",
                receiver="travel_agent",
                content=f"Please update the travel plan based on this feedback: {update.feedback}",
                metadata={"plan_id": plan_id, "update": update.model_dump()}
            )
            
            response = await agent.plan_travel(feedback_message)
            
            # TODO: Update stored plan with refined version
            # For now, return updated structure
            return TravelPlan(
                id=plan_id,
                preferences=update.preferences,
                daily_plans=[],  # TODO: Parse from refined response
                total_cost=0.0,
                status="updated",
                feedback=update.feedback
            )
        
        raise HTTPException(status_code=400, detail="No update data provided")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan update failed: {str(e)}")

@router.delete("/plans/{plan_id}")
async def delete_travel_plan(plan_id: str):
    """
    Delete a travel plan.
    TODO: Implement plan deletion from storage
    """
    # TODO: Implement plan deletion
    raise HTTPException(status_code=404, detail="Plan deletion not yet implemented")

@router.post("/agent/configure")
async def configure_agent_refinement(config: RefinementConfig):
    """
    Configure the self-refinement settings for the travel agent.
    """
    try:
        # This would typically be stored per session or user
        # For now, return the configuration
        return {
            "message": "Agent refinement configured",
            "config": config.model_dump(),
            "note": "Configuration applied to new agent instances"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/agent/status")
async def get_agent_status():
    """
    Get the current status and capabilities of the travel agent.
    """
    try:
        agent = TravelAgent()
        status = agent.get_status()
        
        return {
            "agent_info": {
                "name": status["name"],
                "description": status["description"],
                "capabilities": status["capabilities"],
                "tools": status["tools"]
            },
            "refinement_config": status["refinement_config"],
            "quality_dimensions": agent.get_quality_dimensions(),
            "system_status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

def _preferences_to_message(preferences: TravelPreferences) -> str:
    """Convert TravelPreferences to natural language message"""
    travelers_desc = f"{len(preferences.travelers)} travelers"
    if preferences.travelers:
        types = [t.type.value for t in preferences.travelers]
        travelers_desc = f"{len(preferences.travelers)} travelers ({', '.join(types)})"
    
    message = f"""
    I need help planning a {preferences.trip_style.value} trip to {preferences.destination} 
    from {preferences.origin}. 
    
    Travel dates: {preferences.start_date.strftime('%Y-%m-%d')} to {preferences.end_date.strftime('%Y-%m-%d')}
    Budget: {preferences.budget.total} {preferences.budget.currency}
    Travelers: {travelers_desc}
    Interests: {', '.join(preferences.interests)}
    Goals: {', '.join(preferences.goals)}
    
    {preferences.additional_notes or ''}
    
    Please create a detailed travel plan with recommendations for flights, accommodation, 
    activities, and provide practical next steps.
    """
    
    return message.strip() 