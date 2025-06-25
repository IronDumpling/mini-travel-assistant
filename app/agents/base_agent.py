"""
Base Agent Module - Base agent class

TODO: Implement the following features
1. Agent lifecycle management
2. Multi-agent collaboration mechanism
3. Agent communication protocol
4. Agent status monitoring
5. Agent error recovery
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import asyncio
import uuid


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    STOPPED = "stopped"


class AgentMessage(BaseModel):
    """Agent message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    content: str
    message_type: str = "text"
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Agent response"""
    success: bool
    content: str
    actions_taken: List[str] = []
    next_steps: List[str] = []
    confidence: float = 1.0
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """Base agent class"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.conversation_history: List[AgentMessage] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Agent capabilities configuration
        self.capabilities: List[str] = self.get_capabilities()
        self.tools: List[str] = self.get_available_tools()
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process message (subclass must implement)"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities list (subclass must implement)"""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get available tools list (subclass must implement)"""
        pass
    
    async def think(self, context: Dict[str, Any]) -> str:
        """Thinking process"""
        # TODO: Implement thinking logic
        # 1. Analyze current context
        # 2. Make a plan
        # 3. Evaluate different choices
        # 4. Return thinking result
        self.status = AgentStatus.THINKING
        self.last_activity = datetime.now(timezone.utc)
        
        # Default implementation, subclasses can override
        return f"Agent {self.name} is thinking how to handle: {context}"
    
    async def act(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute action"""
        # TODO: Implement action execution logic
        self.status = AgentStatus.ACTING
        self.last_activity = datetime.utcnow()
        
        # Subclasses need to implement specific action execution logic
        return await self._execute_action(action, parameters)
    
    @abstractmethod
    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute specific action (subclass must implement)"""
        pass
    
    def add_message(self, message: AgentMessage):
        """Add message to conversation history"""
        self.conversation_history.append(message)
        # Keep history within reasonable range
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_conversation_context(self, last_n: int = 10) -> List[AgentMessage]:
        """Get recent conversation context"""
        return self.conversation_history[-last_n:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "conversation_length": len(self.conversation_history),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata
        }
    
    async def reset(self):
        """Reset agent status"""
        self.status = AgentStatus.IDLE
        self.conversation_history.clear()
        self.metadata.clear()
        self.last_activity = datetime.utcnow()


class AgentManager:
    """Agent manager"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.is_running = False
    
    def register_agent(self, agent: BaseAgent):
        """Register agent"""
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all agents"""
        return list(self.agents.keys())
    
    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send message to agent"""
        agent = self.get_agent(message.receiver)
        if not agent:
            return AgentResponse(
                success=False,
                content=f"Agent '{message.receiver}' not found"
            )
        
        agent.add_message(message)
        return await agent.process_message(message)
    
    async def broadcast_message(self, message: AgentMessage) -> Dict[str, AgentResponse]:
        """Broadcast message to all agents"""
        responses = {}
        for agent_name, agent in self.agents.items():
            if agent_name != message.sender:
                message.receiver = agent_name
                responses[agent_name] = await self.send_message(message)
        return responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "total_agents": len(self.agents),
            "agents": {
                name: agent.get_status() 
                for name, agent in self.agents.items()
            },
            "message_queue_size": len(self.message_queue),
            "is_running": self.is_running
        }


# Global agent manager
agent_manager = AgentManager() 