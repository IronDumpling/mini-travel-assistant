"""
API Endpoints Package

This package contains all API endpoint modules organized by functionality:
- system.py: System health checks and status
- sessions.py: Session management CRUD operations  
- chat.py: Conversational AI interface
- plans.py: Structured travel plan operations
- agent.py: Agent configuration and management
"""

# Import all routers for easy access
from .system import router as system_router
from .sessions import router as sessions_router
from .chat import router as chat_router
from .plans import router as plans_router
from .agent import router as agent_router

__all__ = [
    "system_router",
    "sessions_router", 
    "chat_router",
    "plans_router",
    "agent_router"
] 