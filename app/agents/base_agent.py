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


class QualityAssessment(BaseModel):
    """Quality assessment result"""
    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: Dict[str, float] = {}
    improvement_suggestions: List[str] = []
    meets_threshold: bool = False
    assessment_details: Dict[str, Any] = {}


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
        
        # Self-refine configuration
        self.quality_threshold: float = 0.75
        self.max_refine_iterations: int = 3
        self.refine_enabled: bool = True
    
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
    
    async def process_with_refinement(self, message: AgentMessage) -> AgentResponse:
        """Process message with self-refinement loop"""
        if not self.refine_enabled:
            return await self.process_message(message)
        
        iteration = 0
        current_response = None
        refinement_history = []
        
        while iteration < self.max_refine_iterations:
            iteration += 1
            
            # Generate response
            if iteration == 1:
                current_response = await self.process_message(message)
            else:
                # Refine based on previous assessment
                current_response = await self._refine_response(
                    message, 
                    current_response, 
                    refinement_history[-1] if refinement_history else None
                )
            
            # Assess quality
            quality_assessment = await self._assess_response_quality(
                message, 
                current_response,
                iteration
            )
            
            refinement_history.append(quality_assessment)
            
            # Update response metadata with refinement info
            current_response.metadata.update({
                "refinement_iteration": iteration,
                "quality_score": quality_assessment.overall_score,
                "refinement_history": [
                    {
                        "iteration": i + 1,
                        "score": assess.overall_score,
                        "suggestions": assess.improvement_suggestions
                    }
                    for i, assess in enumerate(refinement_history)
                ]
            })
            
            # Check if quality threshold is met
            if quality_assessment.meets_threshold:
                current_response.metadata["refinement_status"] = "completed_early"
                break
            
            # If this is the last iteration, mark as completed
            if iteration >= self.max_refine_iterations:
                current_response.metadata["refinement_status"] = "max_iterations_reached"
                break
            
            current_response.metadata["refinement_status"] = "in_progress"
        
        return current_response
    
    async def _assess_response_quality(
        self, 
        original_message: AgentMessage, 
        response: AgentResponse,
        iteration: int
    ) -> QualityAssessment:
        """Assess the quality of a response"""
        # Get quality dimensions specific to this agent
        quality_dimensions = self.get_quality_dimensions()
        
        dimension_scores = {}
        improvement_suggestions = []
        
        for dimension, weight in quality_dimensions.items():
            score = await self._assess_dimension(
                dimension, 
                original_message, 
                response
            )
            dimension_scores[dimension] = score
            
            # Generate improvement suggestions for low-scoring dimensions
            if score < 0.6:
                suggestions = await self._generate_improvement_suggestions(
                    dimension, 
                    original_message, 
                    response, 
                    score
                )
                improvement_suggestions.extend(suggestions)
        
        # Calculate overall score (weighted average)
        total_weight = sum(quality_dimensions.values())
        overall_score = sum(
            score * quality_dimensions[dim] / total_weight 
            for dim, score in dimension_scores.items()
        )
        
        meets_threshold = overall_score >= self.quality_threshold
        
        return QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            improvement_suggestions=improvement_suggestions,
            meets_threshold=meets_threshold,
            assessment_details={
                "iteration": iteration,
                "threshold": self.quality_threshold,
                "quality_dimensions": quality_dimensions
            }
        )
    
    def get_quality_dimensions(self) -> Dict[str, float]:
        """Get quality assessment dimensions and their weights (subclasses can override)"""
        return {
            "relevance": 0.3,      # How relevant is the response to the user's request
            "completeness": 0.25,  # How complete is the response
            "accuracy": 0.25,      # How accurate is the information provided
            "actionability": 0.2   # How actionable/useful is the response
        }
    
    async def _assess_dimension(
        self, 
        dimension: str, 
        original_message: AgentMessage, 
        response: AgentResponse
    ) -> float:
        """Assess a specific quality dimension (subclasses should override)"""
        # Default implementation - subclasses should provide more sophisticated assessment
        if dimension == "relevance":
            return await self._assess_relevance(original_message, response)
        elif dimension == "completeness":
            return await self._assess_completeness(original_message, response)
        elif dimension == "accuracy":
            return await self._assess_accuracy(original_message, response)
        elif dimension == "actionability":
            return await self._assess_actionability(original_message, response)
        else:
            return 0.5  # Default neutral score
    
    async def _assess_relevance(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess response relevance"""
        # Basic implementation - check if response addresses the message
        if response.success and response.content:
            # Simple heuristic: longer responses that mention actions are more relevant
            base_score = 0.6
            if response.actions_taken:
                base_score += 0.2
            if len(response.content) > 100:
                base_score += 0.1
            if response.next_steps:
                base_score += 0.1
            return min(base_score, 1.0)
        return 0.3
    
    async def _assess_completeness(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess response completeness"""
        # Basic implementation - check if response has all components
        score = 0.5
        if response.actions_taken:
            score += 0.2
        if response.next_steps:
            score += 0.2
        if response.confidence > 0.7:
            score += 0.1
        return min(score, 1.0)
    
    async def _assess_accuracy(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess response accuracy"""
        # Basic implementation - use confidence as proxy for accuracy
        return response.confidence
    
    async def _assess_actionability(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess response actionability"""
        # Basic implementation - check if response provides actionable information
        score = 0.4
        if response.actions_taken:
            score += 0.3
        if response.next_steps:
            score += 0.3
        return min(score, 1.0)
    
    async def _generate_improvement_suggestions(
        self, 
        dimension: str, 
        original_message: AgentMessage, 
        response: AgentResponse,
        current_score: float
    ) -> List[str]:
        """Generate improvement suggestions for a specific dimension"""
        suggestions = []
        
        if dimension == "relevance" and current_score < 0.6:
            suggestions.append("Make the response more directly relevant to the user's specific request")
            if not response.actions_taken:
                suggestions.append("Include specific actions taken to address the user's needs")
        
        if dimension == "completeness" and current_score < 0.6:
            suggestions.append("Provide more comprehensive information")
            if not response.next_steps:
                suggestions.append("Include clear next steps for the user")
        
        if dimension == "accuracy" and current_score < 0.6:
            suggestions.append("Verify the accuracy of the information provided")
            suggestions.append("Use more reliable data sources")
        
        if dimension == "actionability" and current_score < 0.6:
            suggestions.append("Provide more specific and actionable recommendations")
            suggestions.append("Include concrete steps the user can take")
        
        return suggestions
    
    async def _refine_response(
        self, 
        original_message: AgentMessage, 
        current_response: AgentResponse,
        quality_assessment: Optional[QualityAssessment]
    ) -> AgentResponse:
        """Refine the response based on quality assessment (subclasses should override)"""
        # Default implementation - basic improvements
        if not quality_assessment:
            return current_response
        
        # Apply improvement suggestions
        improved_content = current_response.content
        
        # Add more detail if completeness is low
        if quality_assessment.dimension_scores.get("completeness", 1.0) < 0.6:
            improved_content += "\n\nAdditional details: I've analyzed your request and provided the most relevant information available."
        
        # Add next steps if actionability is low
        improved_next_steps = current_response.next_steps.copy()
        if quality_assessment.dimension_scores.get("actionability", 1.0) < 0.6:
            improved_next_steps.append("Please let me know if you need more specific guidance")
        
        # Increase confidence slightly after refinement
        improved_confidence = min(current_response.confidence + 0.1, 1.0)
        
        return AgentResponse(
            success=current_response.success,
            content=improved_content,
            actions_taken=current_response.actions_taken,
            next_steps=improved_next_steps,
            confidence=improved_confidence,
            metadata={
                **current_response.metadata,
                "refined": True,
                "improvement_applied": quality_assessment.improvement_suggestions
            }
        )

    async def think(self, context: Dict[str, Any]) -> str:
        """Thinking process"""
        # TODO: Implement thinking logic
        # 1. Analyze current context
        # 2. Make a plan
        # 3. Evaluate different choices
        # 4. Return thinking result
        self.status = AgentStatus.THINKING
        self.last_activity = datetime.utcnow()
        
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
            "refinement_config": {
                "enabled": self.refine_enabled,
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_refine_iterations
            },
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