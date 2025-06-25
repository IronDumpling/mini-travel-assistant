"""
AI Travel Planning Agent - Main Entry File

Refactored based on the new architecture, integrating all core components:
- Core LLM and RAG knowledge layer  
- Tool/plugin layer  
- Context memory layer  
- Agent system  
- Monitoring and evaluation layer  
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

# Import core components
from app.core.knowledge_base import get_knowledge_base
from app.tools.base_tool import tool_registry
from app.agents.base_agent import agent_manager
from app.memory.conversation_memory import get_conversation_memory

# Import API routes
from app.api.endpoints import travel_plans

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Initialize on startup
    logger.info("🚀 Starting AI Travel Planning Agent")
    
    try:
        # 1. Initialize knowledge base
        logger.info("📚 Initializing knowledge base...")
        knowledge_base = await get_knowledge_base()
        logger.info(f"✅ Knowledge base initialized with {len(knowledge_base.knowledge_items)} items loaded")
        
        # 2. Initialize tool system
        logger.info("🔧 Initializing tool system...")
        # Tools are auto-registered via import
        registry_status = tool_registry.get_registry_status()
        logger.info(f"✅ Tool system initialized with {registry_status['total_tools']} tools registered")
        
        # 3. Initialize Agent system
        logger.info("🤖 Initializing Agent system...")
        # Agents are auto-registered via import
        agent_status = agent_manager.get_system_status()
        logger.info(f"✅ Agent system initialized with {agent_status['total_agents']} agents registered")
        
        # 4. Initialize memory system
        logger.info("🧠 Initializing memory system...")
        conversation_memory = get_conversation_memory()
        
        # Initialize session manager
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        logger.info("✅ Memory system initialized with session management")
        
        logger.info("🎉 AI Travel Planning Agent successfully started")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down AI Travel Planning Agent")


app = FastAPI(
    title="AI Travel Planning Agent",
    description="An intelligent travel planning system based on a new six-layer architecture, supporting RAG-based retrieval, multi-tool coordination, intelligent reasoning, and self-improvement",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
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
                    "active": len([tool for tool in tool_status.get("tools", []) 
                                 if tool.get("status") != "error"])
                },
                "agents": {
                    "total": agent_status["total_agents"],
                    "active": len([agent for agent in agent_status.get("agents", []) 
                                 if agent["status"] != "stopped"])
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/system/status")
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
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        )

@app.get("/sessions")
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

@app.post("/sessions")
async def create_session(title: str = None, description: str = None):
    """Create a new session"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        session_id = session_manager.create_session(title=title, description=description)
        session = session_manager.get_current_session()
        
        return {
            "session_id": session_id,
            "session": session.model_dump() if session else None,
            "message": "Session created successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to create session: {str(e)}"}
        )

@app.put("/sessions/{session_id}/switch")
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

# Include API routes
app.include_router(
    travel_plans.router, 
    prefix="/api/v1", 
    tags=["travel_plans"]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 