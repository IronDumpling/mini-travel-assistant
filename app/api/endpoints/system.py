"""
System Endpoints - Health checks and system status
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.tools.base_tool import tool_registry
from app.agents.base_agent import agent_manager
from datetime import datetime

router = APIRouter()

@router.get("/")
async def root():
    """Root path - system status overview"""
    try:
        # Get status of each component
        tool_status = tool_registry.get_registry_status()
        agent_status = agent_manager.get_system_status()
        
        return {
            "message": "Welcome to AI Travel Planning Agent v2.0",
            "status": "Running",
            "architecture": "Six-layer AI Agent architecture",
            "components": {
                "tools": {
                    "total": tool_status["total_tools"],
                    "categories": len(tool_status.get("categories", {})),
                    "active": len([tool for tool in tool_status.get("tools", {}).values() 
                                 if hasattr(tool, 'get') and tool.get("status") != "error"])
                },
                "agents": {
                    "total": agent_status["total_agents"],
                    "active": len([agent for agent in agent_status.get("agents", {}).values() 
                                 if hasattr(agent, 'get') and agent.get("status") != "stopped"])
                },
                "knowledge_base": "Loaded",
                "memory_system": "Active"
            },
            "capabilities": [
                "Intelligent travel planning",
                "Multi-tool coordination",
                "Retrieval-augmented generation",
                "Conversation memory management",
                "Self-learning and improvement"
            ]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.0.0"
    }

@router.get("/system/status")
async def system_status():
    """Detailed system status"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        return {
            "tools": tool_registry.get_registry_status(),
            "agents": agent_manager.get_system_status(),
            "memory": {
                "conversation_memory": "active",
                "session_manager": "active",
                "session_stats": session_manager.get_session_stats()
            },
            "system_info": {
                "version": "2.0.0",
                "architecture": "Six-layer AI Agent architecture",
                "uptime": "active"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        ) 