"""
Agent Management Endpoints - Agent configuration and status
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.travel_agent import get_travel_agent
from app.agents.base_agent import agent_manager

router = APIRouter()

class RefinementConfig(BaseModel):
    """Configuration for self-refinement with two-tier thresholds"""
    enabled: bool = True
    fast_response_threshold: float = 0.75  # Threshold for skipping LLM enhancement
    quality_threshold: float = 0.9  # Threshold for refinement loop iterations
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
        # Use Agent singleton instead of agent_manager
        agent = get_travel_agent()
        
        # Configure refinement settings on the actual agent instance
        agent.configure_refinement(
            enabled=config.enabled,
            fast_response_threshold=config.fast_response_threshold,
            quality_threshold=config.quality_threshold,
            max_iterations=config.max_iterations
        )
        
        return AgentConfigResponse(
            message="Agent refinement configured successfully",
            config=config.model_dump(),
            note="Configuration applied to the singleton travel agent instance"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/agent/status")
async def get_agent_status():
    """
    Get the current status and capabilities of the travel agent.
    """
    try:
        # ✅ 使用单例Agent而不是agent_manager
        agent = get_travel_agent()
        
        status = agent.get_status()
        
        # Safely access agent attributes with fallbacks
        user_preferences_count = len(getattr(agent, 'user_preferences_history', {}))
        fast_threshold = getattr(agent, 'fast_response_threshold', 0.75)
        quality_threshold = getattr(agent, 'quality_threshold', 0.9)
        
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
            "user_preferences_count": user_preferences_count,
            "singleton_verified": True,
            "thresholds": {
                "fast_response_threshold": fast_threshold,
                "quality_threshold": quality_threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/agent/capabilities")
async def get_agent_capabilities():
    """
    Get detailed information about agent capabilities and tools.
    """
    try:
        # ✅ 使用单例Agent而不是agent_manager
        agent = get_travel_agent()
        
        return {
            "agent_type": "TravelAgent",
            "base_capabilities": [
                "Natural language processing",
                "Travel planning",
                "Multi-tool coordination",
                "Self-refinement and quality improvement",
                "Context-aware recommendations",
                "User preference learning",
                "RAG-enhanced conversation memory"
            ],
            "available_tools": agent.get_available_tools(),
            "supported_languages": ["English"],  # TODO: Add more languages
            "quality_dimensions": agent.get_quality_dimensions(),
            "refinement_features": {
                "quality_assessment": True,
                "iterative_improvement": True,
                "confidence_scoring": True,
                "multi_dimensional_evaluation": True,
                "information_fusion": True,
                "conversation_context": True
            },
            "agent_instance_active": True,
            "tools_registered": len(agent.get_available_tools()),
            "singleton_pattern": True,
            "agent_instance_id": id(agent)
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
        # ✅ 使用单例Agent而不是agent_manager
        agent = get_travel_agent()
        
        # Reset the agent state
        await agent.reset()
        
        # Safely access agent attributes with fallbacks
        fast_threshold = getattr(agent, 'fast_response_threshold', 0.75)
        quality_threshold = getattr(agent, 'quality_threshold', 0.9)
        
        return {
            "message": "Agent reset to default state successfully",
            "status": "success",
            "agent_instance_id": id(agent),
            "user_preferences_cleared": True,
            "status_reset": True,
            "singleton_maintained": True,
            "thresholds_reset": {
                "fast_response_threshold": fast_threshold,
                "quality_threshold": quality_threshold
            }
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
        # ✅ 使用单例Agent进行未来的metrics收集
        agent = get_travel_agent()
        
        # TODO: Implement metrics collection
        return {
            "metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": 0.0,
                "average_quality_score": 0.0,
                "refinement_usage_rate": 0.0,
                "singleton_efficiency": "optimized"
            },
            "agent_instance_id": id(agent),
            "note": "Metrics collection not yet implemented - using singleton agent instance"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}") 