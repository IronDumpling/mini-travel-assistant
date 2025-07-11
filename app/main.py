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
import uvicorn
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import core components
from app.core.knowledge_base import get_knowledge_base
from app.tools.base_tool import tool_registry
from app.agents.base_agent import agent_manager
from app.memory.conversation_memory import get_conversation_memory

# Import tools to trigger registration
from app.tools import hotel_search, flight_search, attraction_search

# Import agents to trigger registration  
from app.agents import travel_agent

# Import API routes
from app.api.endpoints import (
    system_router,
    sessions_router,
    chat_router,
    plans_router,
    agent_router
)

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

# Include all API routes with organized structure
app.include_router(system_router, tags=["system"])
app.include_router(sessions_router, prefix="/api", tags=["sessions"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(plans_router, prefix="/api", tags=["plans"])
app.include_router(agent_router, prefix="/api", tags=["agent"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 