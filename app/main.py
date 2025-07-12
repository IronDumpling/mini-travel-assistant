"""
AI Travel Planning Agent - Main Entry File

Refactored based on the new architecture, integrating all core components:
- Core LLM and RAG knowledge layer  
- Tool/plugin layer  
- Context memory layer  
- Agent system  
- Monitoring and evaluation layer  
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

# Configure centralized logging
def setup_logging():
    """Configure centralized logging for the entire application"""
    
    # Remove default logger to avoid duplicate logs
    logger.remove()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Add console logging (colored and formatted)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file logging with rotation
    logger.add(
        sink="logs/app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        enqueue=True  # Makes logging thread-safe
    )
    
    # Add error-specific logging
    logger.add(
        sink="logs/errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        enqueue=True
    )
    
    # Add debug logging (optional, can be controlled by environment variable)
    debug_enabled = os.environ.get("DEBUG", "false").lower() == "true"
    if debug_enabled:
        logger.add(
            sink="logs/debug_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="3 days",
            compression="zip",
            enqueue=True
        )
    
    logger.info("📝 Centralized logging configured")
    logger.info(f"📂 Log files will be saved to: {logs_dir.absolute()}")
    logger.info(f"🐛 Debug logging: {'enabled' if debug_enabled else 'disabled'}")

# Set up logging first before any other imports
setup_logging()

def setup_environment_if_needed():
    """Check if required environment variables are set, and set them up if missing"""
    
    # Define required environment variables
    required_vars = {
        "LLM_PROVIDER": "deepseek",
        "LLM_MODEL": "deepseek-chat", 
        "LLM_API_KEY": "sk-d6f66ddb3a174cb3b57367e97207e1fe",
        "DEEPSEEK_API_KEY": "sk-d6f66ddb3a174cb3b57367e97207e1fe",
        "LLM_TEMPERATURE": "0.7",
        "LLM_MAX_TOKENS": "4000"
    }
    
    # Check if any required variables are missing
    missing_vars = []
    for key, default_value in required_vars.items():
        if not os.environ.get(key):
            missing_vars.append(key)
    
    if missing_vars:
        logger.info(f"🔧 Setting up missing environment variables: {', '.join(missing_vars)}")
        
        # Set environment variables for current session
        for key, value in required_vars.items():
            if not os.environ.get(key):
                os.environ[key] = value
        
        # Create or append to .env file
        env_file_path = ".env"
        env_exists = os.path.exists(env_file_path)
        
        # Read existing .env content to avoid duplicates
        existing_content = ""
        if env_exists:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Only write variables that aren't already in the file
        vars_to_write = {}
        for key, value in required_vars.items():
            if f"{key}=" not in existing_content:
                vars_to_write[key] = value
        
        if vars_to_write:
            mode = 'a' if env_exists else 'w'
            with open(env_file_path, mode, encoding='utf-8') as f:
                if env_exists:
                    f.write("\n")
                f.write("# DeepSeek API Configuration\n")
                for key, value in vars_to_write.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"✅ Added {len(vars_to_write)} environment variables to .env file")
        
        logger.info("📋 Environment Configuration:")
        logger.info(f"   LLM Provider: {os.environ.get('LLM_PROVIDER')}")
        logger.info(f"   LLM Model: {os.environ.get('LLM_MODEL')}")
        logger.info(f"   API Key: {os.environ.get('LLM_API_KEY')[:20]}...")
        logger.info(f"   Temperature: {os.environ.get('LLM_TEMPERATURE')}")
        logger.info(f"   Max Tokens: {os.environ.get('LLM_MAX_TOKENS')}")
    else:
        logger.info("✅ All required environment variables are already set")

# Set up environment variables before loading dotenv
setup_environment_if_needed()

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
        
        # 5. Initialize tool knowledge for RAG-based tool selection
        logger.info("🔍 Initializing tool knowledge for intelligent tool selection...")
        from app.tools.tool_executor import get_tool_executor
        tool_executor = get_tool_executor()
        await tool_executor.tool_selector.initialize_tool_knowledge()
        logger.info("✅ Tool knowledge initialized for RAG-based tool selection")
        
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