"""
AI Travel Planning Agent - Main Entry File

Refactored based on the new architecture, integrating all core components:
- Core LLM and RAG knowledge layer  
- Tool/plugin layer  
- Context memory layer  
- Agent system  
- Monitoring and evaluation layer  
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# 导入核心组件
from app.core.llm_service import get_llm_service
from app.core.rag_engine import get_rag_engine
from app.core.knowledge_base import get_knowledge_base
from app.tools.base_tool import tool_registry
from app.agents.base_agent import agent_manager
from app.memory.conversation_memory import get_conversation_memory

# 导入API路由
from app.api.endpoints import travel_plans

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info("✅ Memory system initialized")
        
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
                    "categories": tool_status["categories"]
                },
                "agents": {
                    "total": agent_status["total_agents"],
                    "active": sum(1 for agent in agent_status["agents"].values() 
                                  if agent["status"] != "stopped")
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
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "2.0.0"
    }


@app.get("/system/status")
async def system_status():
    """Detailed system status"""
    try:
        return {
            "tools": tool_registry.get_registry_status(),
            "agents": agent_manager.get_system_status(),
            "memory": {
                "conversation_sessions": len(get_conversation_memory().sessions)
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        )


# Include API routes
app.include_router(
    travel_plans.router, 
    prefix="/api/v1", 
    tags=["travel-plans"]
)

# TODO: 添加更多API路由
# app.include_router(knowledge_router, prefix="/api/v1", tags=["knowledge"])
# app.include_router(agents_router, prefix="/api/v1", tags=["agents"])
# app.include_router(tools_router, prefix="/api/v1", tags=["tools"])
# app.include_router(memory_router, prefix="/api/v1", tags=["memory"]) 