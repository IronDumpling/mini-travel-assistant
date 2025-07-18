"""
Agent Management Endpoints - Agent configuration and status
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.base_agent import agent_manager

router = APIRouter()

class RefinementConfig(BaseModel):
    """Configuration for self-refinement"""
    enabled: bool = True
    quality_threshold: float = 0.75
    max_iterations: int = 3

class AgentConfigResponse(BaseModel):
    """Response for agent configuration"""
    message: str
    config: dict
    note: str

@router.post("/agent/configure", response_model=AgentConfigResponse)
async def configure_agent_refinement(config: RefinementConfig):
    """
    Configure the self-refinement settings for the travel agent.
    """
    try:
        # Get the registered travel agent
        agent = agent_manager.get_agent("travel_agent")
        if not agent:
            raise HTTPException(status_code=404, detail="Travel agent not found")
        
        # Configure refinement settings on the actual agent instance
        agent.configure_refinement(
            enabled=config.enabled,
            quality_threshold=config.quality_threshold,
            max_iterations=config.max_iterations
        )
        
        return AgentConfigResponse(
            message="Agent refinement configured successfully",
            config=config.model_dump(),
            note="Configuration applied to the active travel agent instance"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/agent/status")
async def get_agent_status():
    """
    Get the current status and capabilities of the travel agent.
    """
    try:
        # Get the registered travel agent
        agent = agent_manager.get_agent("travel_agent")
        if not agent:
            raise HTTPException(status_code=404, detail="Travel agent not found")
        
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
            "system_status": "operational",
            "agent_instance_id": id(agent),  # Add instance ID to verify same agent
            "conversation_history_length": len(agent.conversation_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/agent/capabilities")
async def get_agent_capabilities():
    """
    Get detailed information about agent capabilities and tools.
    """
    try:
        # Get the registered travel agent
        agent = agent_manager.get_agent("travel_agent")
        if not agent:
            raise HTTPException(status_code=404, detail="Travel agent not found")
        
        return {
            "agent_type": "TravelAgent",
            "base_capabilities": [
                "Natural language processing",
                "Travel planning",
                "Multi-tool coordination",
                "Self-refinement and quality improvement",
                "Context-aware recommendations"
            ],
            "available_tools": agent.get_available_tools(),
            "supported_languages": ["English"],  # TODO: Add more languages
            "quality_dimensions": agent.get_quality_dimensions(),
            "refinement_features": {
                "quality_assessment": True,
                "iterative_improvement": True,
                "confidence_scoring": True,
                "multi_dimensional_evaluation": True
            },
            "agent_instance_active": True,
            "tools_registered": len(agent.get_available_tools())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.post("/agent/reset")
async def reset_agent():
    """
    Reset the agent to default state.
    Useful for clearing any temporary configurations or state.
    """
    try:
        # Get the registered travel agent
        agent = agent_manager.get_agent("travel_agent")
        if not agent:
            raise HTTPException(status_code=404, detail="Travel agent not found")
        
        # Reset the agent state
        await agent.reset()
        
        return {
            "message": "Agent reset to default state successfully",
            "status": "success",
            "agent_instance_id": id(agent),
            "conversation_history_cleared": True,
            "status_reset": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent reset failed: {str(e)}")

@router.get("/agent/metrics")
async def get_agent_metrics():
    """
    Get performance metrics for the agent.
    TODO: Implement metrics collection and reporting.
    """
    try:
        # TODO: Implement metrics collection
        return {
            "metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": 0.0,
                "average_quality_score": 0.0,
                "refinement_usage_rate": 0.0
            },
            "note": "Metrics collection not yet implemented"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}") 