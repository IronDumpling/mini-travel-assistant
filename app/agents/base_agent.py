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
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
import uuid
from app.core.logging_config import get_logger

logger = get_logger(__name__)


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)

        # Agent capabilities configuration
        self.capabilities: List[str] = self.get_capabilities()
        self.tools: List[str] = self.get_available_tools()

        # Self-refine configuration
        self.quality_threshold: float = 0.75
        self.max_refine_iterations: int = 3
        self.refine_enabled: bool = True

        # Error handling and state management
        self.error_count: int = 0
        self.max_errors: int = 5
        self.last_error: Optional[str] = None
        self.error_history: List[Dict[str, Any]] = []
        self.is_healthy: bool = True

        # Simple state management
        self.state_timeout_seconds: int = 60  # Max time in active states

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
        start_time = datetime.now(timezone.utc)

        while iteration < self.max_refine_iterations:
            iteration += 1

            # Generate response with timeout
            if iteration == 1:
                try:
                    current_response = await asyncio.wait_for(
                        self.process_message(message),
                        timeout=30.0,  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    self.record_error(
                        "Timeout during initial process_message",
                        {"message_id": message.id, "iteration": iteration},
                    )
                    current_response = AgentResponse(
                        success=False,
                        content="I'm taking too long to process your request. Please try again.",
                        confidence=0.0,
                    )
                    break
                except Exception as e:
                    self.record_error(
                        f"Exception during initial process_message: {str(e)}",
                        {"message_id": message.id, "iteration": iteration},
                    )
                    current_response = AgentResponse(
                        success=False,
                        content="An error occurred while processing your request.",
                        confidence=0.0,
                    )
                    break
            else:
                # Refine based on previous assessment
                if current_response is None:
                    self.record_error(
                        "No previous response available for refinement",
                        {"message_id": message.id, "iteration": iteration},
                    )
                    current_response = AgentResponse(
                        success=False,
                        content="Unable to refine response - no previous response available",
                        confidence=0.0,
                    )
                    break

                try:
                    current_response = await asyncio.wait_for(
                        self._refine_response(
                            message,
                            current_response,
                            refinement_history[-1] if refinement_history else None,
                        ),
                        timeout=15.0,  # 15 second timeout for refinement
                    )
                except asyncio.TimeoutError:
                    self.record_error(
                        "Timeout during response refinement",
                        {"message_id": message.id, "iteration": iteration},
                    )
                    # Continue with current response if refinement times out
                    current_response.metadata["refinement_timeout"] = True
                    break
                except Exception as e:
                    self.record_error(
                        f"Exception during response refinement: {str(e)}",
                        {"message_id": message.id, "iteration": iteration},
                    )
                    current_response.metadata["refinement_exception"] = str(e)
                    break

            # Assess quality with timeout and fallback
            try:
                quality_assessment = await asyncio.wait_for(
                    self._assess_response_quality(message, current_response, iteration),
                    timeout=10.0,  # 10 second timeout for quality assessment
                )
            except asyncio.TimeoutError:
                self.record_error(
                    "Timeout during quality assessment",
                    {"message_id": message.id, "iteration": iteration},
                )
                # Use fallback quality assessment
                quality_assessment = QualityAssessment(
                    overall_score=0.5,
                    dimension_scores={
                        "relevance": 0.5,
                        "completeness": 0.5,
                        "accuracy": 0.5,
                        "actionability": 0.5,
                    },
                    improvement_suggestions=[
                        "Quality assessment timed out, using fallback"
                    ],
                    meets_threshold=False,
                    assessment_details={"fallback": True, "iteration": iteration},
                )
            except Exception as e:
                self.record_error(
                    f"Exception during quality assessment: {str(e)}",
                    {"message_id": message.id, "iteration": iteration},
                )
                quality_assessment = QualityAssessment(
                    overall_score=0.0,
                    dimension_scores={},
                    improvement_suggestions=[
                        "Quality assessment failed due to exception"
                    ],
                    meets_threshold=False,
                    assessment_details={"exception": str(e), "iteration": iteration},
                )

            refinement_history.append(quality_assessment)

            # Update response metadata with refinement info
            current_response.metadata.update(
                {
                    "refinement_iteration": iteration,
                    "quality_score": quality_assessment.overall_score,
                    "refinement_history": [
                        {
                            "iteration": i + 1,
                            "score": assess.overall_score,
                            "suggestions": assess.improvement_suggestions,
                        }
                        for i, assess in enumerate(refinement_history)
                    ],
                    "total_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                }
            )

            # Check if quality threshold is met
            if quality_assessment.meets_threshold:
                current_response.metadata["refinement_status"] = "completed_early"
                break

            # Check for diminishing returns (stop if quality isn't improving)
            if iteration > 1 and len(refinement_history) >= 2:
                current_score = quality_assessment.overall_score
                previous_score = refinement_history[-2].overall_score
                if current_score <= previous_score + 0.05:  # No significant improvement
                    current_response.metadata["refinement_status"] = (
                        "diminishing_returns"
                    )
                    break

            # If this is the last iteration, mark as completed
            if iteration >= self.max_refine_iterations:
                current_response.metadata["refinement_status"] = (
                    "max_iterations_reached"
                )
                break

            current_response.metadata["refinement_status"] = "in_progress"

        # Ensure we always return a response
        if current_response is None:
            self.record_error(
                "Failed to process message (no response generated)",
                {"message_id": message.id},
            )
            current_response = AgentResponse(
                success=False, content="Failed to process message", confidence=0.0
            )

        return current_response

    async def _assess_response_quality(
        self, original_message: AgentMessage, response: AgentResponse, iteration: int
    ) -> QualityAssessment:
        """Assess the quality of a response"""
        # Get quality dimensions specific to this agent
        quality_dimensions = self.get_quality_dimensions()

        dimension_scores = {}
        improvement_suggestions = []

        for dimension, weight in quality_dimensions.items():
            score = await self._assess_dimension(dimension, original_message, response)
            dimension_scores[dimension] = score

            # Generate improvement suggestions for low-scoring dimensions
            if score < 0.6:
                suggestions = await self._generate_improvement_suggestions(
                    dimension, original_message, response, score
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
                "quality_dimensions": quality_dimensions,
            },
        )

    def get_quality_dimensions(self) -> Dict[str, float]:
        """Get quality assessment dimensions and their weights (subclasses can override)"""
        return {
            "relevance": 0.3,  # How relevant is the response to the user's request
            "completeness": 0.25,  # How complete is the response
            "accuracy": 0.25,  # How accurate is the information provided
            "actionability": 0.2,  # How actionable/useful is the response
        }

    async def _assess_dimension(
        self, dimension: str, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess a specific quality dimension using LLM-based assessment"""

        # Try to use LLM for quality assessment
        try:
            from app.core.llm_service import get_llm_service
            from app.core.prompt_manager import prompt_manager, PromptType

            llm_service = get_llm_service()

            if llm_service:
                # Use prompt manager for LLM-based quality assessment
                assessment_prompt = prompt_manager.get_prompt(
                    PromptType.QUALITY_ASSESSMENT,
                    original_message=original_message.content,
                    agent_response=response.content,
                    dimension=dimension,
                    actions_taken=response.actions_taken,
                    next_steps=response.next_steps,
                    confidence=response.confidence,
                )

                # Use structured completion for quality assessment
                schema = prompt_manager.get_schema(PromptType.QUALITY_ASSESSMENT)
                assessment_result = await llm_service.structured_completion(
                    messages=[{"role": "user", "content": assessment_prompt}],
                    response_schema=schema,
                    temperature=0.2,
                    max_tokens=300,
                )

                # Extract score for the specific dimension
                dimension_score = assessment_result.get("dimension_scores", {}).get(
                    dimension, 0.5
                )
                return float(dimension_score)

        except Exception as e:
            logger.warning(f"LLM-based quality assessment failed: {e}")
            # Fall back to heuristic assessment if LLM fails
            pass

        # Fallback to heuristic assessment
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

    async def _assess_relevance(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
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

    async def _assess_completeness(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
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

    async def _assess_accuracy(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess response accuracy"""
        # Basic implementation - use confidence as proxy for accuracy
        return response.confidence

    async def _assess_actionability(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
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
        current_score: float,
    ) -> List[str]:
        """Generate improvement suggestions using LLM-based analysis"""

        # Try to use LLM for improvement suggestions
        try:
            from app.core.llm_service import get_llm_service
            from app.core.prompt_manager import prompt_manager, PromptType

            llm_service = get_llm_service()

            if llm_service:
                # Use prompt manager for LLM-based quality assessment with suggestions
                assessment_prompt = prompt_manager.get_prompt(
                    PromptType.QUALITY_ASSESSMENT,
                    original_message=original_message.content,
                    agent_response=response.content,
                    dimension=dimension,
                    actions_taken=response.actions_taken,
                    next_steps=response.next_steps,
                    confidence=response.confidence,
                )

                # Use structured completion for quality assessment
                schema = prompt_manager.get_schema(PromptType.QUALITY_ASSESSMENT)
                assessment_result = await llm_service.structured_completion(
                    messages=[{"role": "user", "content": assessment_prompt}],
                    response_schema=schema,
                    temperature=0.3,
                    max_tokens=400,
                )

                # Extract improvement suggestions
                suggestions = assessment_result.get("improvement_suggestions", [])
                if suggestions:
                    return suggestions

        except Exception as e:
            logger.warning(f"LLM-based improvement suggestions failed: {e}")
            # Fall back to heuristic suggestions if LLM fails
            pass

        # Fallback to heuristic suggestions
        suggestions = []

        if dimension == "relevance" and current_score < 0.6:
            suggestions.append(
                "Make the response more directly relevant to the user's specific request"
            )
            if not response.actions_taken:
                suggestions.append(
                    "Include specific actions taken to address the user's needs"
                )

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
        quality_assessment: Optional[QualityAssessment],
    ) -> AgentResponse:
        """Refine the response based on quality assessment using LLM-based refinement"""

        if not quality_assessment:
            return current_response

        # Try to use LLM for response refinement
        try:
            from app.core.llm_service import get_llm_service
            from app.core.prompt_manager import prompt_manager, PromptType

            llm_service = get_llm_service()

            if llm_service:
                # Use prompt manager for LLM-based response refinement
                refinement_prompt = prompt_manager.get_prompt(
                    PromptType.RESPONSE_REFINEMENT,
                    original_response=current_response.content,
                    quality_assessment=quality_assessment.dict(),
                    improvement_areas=quality_assessment.improvement_suggestions,
                )

                # Use structured completion for response refinement
                schema = prompt_manager.get_schema(PromptType.RESPONSE_REFINEMENT)
                refinement_result = await llm_service.structured_completion(
                    messages=[{"role": "user", "content": refinement_prompt}],
                    response_schema=schema,
                    temperature=0.3,
                    max_tokens=800,
                )

                # Extract refined response components
                refined_content = refinement_result.get(
                    "refined_content", current_response.content
                )
                refined_actions = refinement_result.get(
                    "refined_actions", current_response.actions_taken
                )
                refined_next_steps = refinement_result.get(
                    "refined_next_steps", current_response.next_steps
                )
                confidence_boost = refinement_result.get("confidence_boost", 0.1)

                return AgentResponse(
                    success=current_response.success,
                    content=refined_content,
                    actions_taken=refined_actions,
                    next_steps=refined_next_steps,
                    confidence=min(current_response.confidence + confidence_boost, 1.0),
                    metadata={
                        **current_response.metadata,
                        "refined": True,
                        "refinement_method": "llm_based",
                        "improvement_applied": quality_assessment.improvement_suggestions,
                        "llm_refinement_applied": refinement_result.get(
                            "applied_improvements", []
                        ),
                    },
                )

        except Exception as e:
            logger.warning(f"LLM-based response refinement failed: {e}")
            # Fall back to heuristic refinement if LLM fails

        # Fallback to heuristic refinement
        improved_content = current_response.content

        # Add more detail if completeness is low
        if quality_assessment.dimension_scores.get("completeness", 1.0) < 0.6:
            improved_content += "\n\nAdditional details: I've analyzed your request and provided the most relevant information available."

        # Add next steps if actionability is low
        improved_next_steps = current_response.next_steps.copy()
        if quality_assessment.dimension_scores.get("actionability", 1.0) < 0.6:
            improved_next_steps.append(
                "Please let me know if you need more specific guidance"
            )

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
                "refinement_method": "heuristic_fallback",
                "improvement_applied": quality_assessment.improvement_suggestions,
            },
        )

    async def think(self, context: Dict[str, Any]) -> str:
        """Thinking process"""
        # TODO: Implement thinking logic
        # 1. Analyze current context
        # 2. Make a plan
        # 3. Evaluate different choices
        # 4. Return thinking result
        self.set_status(AgentStatus.THINKING)

        # Default implementation, subclasses can override
        result = f"Agent {self.name} is thinking how to handle: {context}"

        # Return to IDLE after thinking
        self.set_status(AgentStatus.IDLE)
        return result

    async def act(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute action"""
        # TODO: Implement action execution logic
        self.set_status(AgentStatus.ACTING)

        # Subclasses need to implement specific action execution logic
        result = await self._execute_action(action, parameters)

        # Return to IDLE after acting
        self.set_status(AgentStatus.IDLE)
        return result

    @abstractmethod
    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute specific action (subclass must implement)"""
        pass

    def add_message(self, message: AgentMessage):
        """Add message to conversation history"""

        # Skip adding if content is empty or None
        if not message.content or not message.content.strip():
            return

        # Validate message has required fields
        if not message.sender or not message.receiver:
            return

        self.conversation_history.append(message)
        # Keep history within reasonable range
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

    def get_conversation_context(self, last_n: int = 10) -> List[AgentMessage]:
        """Get recent conversation context"""
        # Ensure last_n is positive
        if last_n <= 0:
            return []

        # Ensure we have a conversation history
        if not self.conversation_history:
            return []

        # Ensure we don't exceed the list length
        actual_n = min(last_n, len(self.conversation_history))
        return self.conversation_history[-actual_n:]

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
                "max_iterations": self.max_refine_iterations,
            },
            "metadata": self.metadata,
            "health": self.get_health_status(),
        }

    async def reset(self):
        """Reset agent status"""
        self.status = AgentStatus.IDLE
        self.conversation_history.clear()
        self.metadata.clear()
        self.last_activity = datetime.now(timezone.utc)

        # Reset error state
        self.error_count = 0
        self.last_error = None
        self.error_history.clear()
        self.is_healthy = True

    def record_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Record an error and update agent state"""
        self.error_count += 1
        self.last_error = error
        self.last_activity = datetime.now(timezone.utc)

        error_record = {
            "timestamp": datetime.now(timezone.utc),
            "error": error,
            "context": context or {},
            "error_count": self.error_count,
        }
        self.error_history.append(error_record)

        # Keep error history manageable
        if len(self.error_history) > 20:
            self.error_history = self.error_history[-20:]

        # Mark as unhealthy if too many errors
        if self.error_count >= self.max_errors:
            self.is_healthy = False
            self.status = AgentStatus.ERROR

    def clear_errors(self):
        """Clear error state and mark as healthy"""
        self.error_count = 0
        self.last_error = None
        self.error_history.clear()
        self.is_healthy = True
        if self.status == AgentStatus.ERROR:
            self.status = AgentStatus.IDLE

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "is_healthy": self.is_healthy,
            "error_count": self.error_count,
            "max_errors": self.max_errors,
            "last_error": self.last_error,
            "error_history_length": len(self.error_history),
            "status": self.status,
        }

    def set_status(self, new_status: AgentStatus):
        """Set agent status"""
        self.status = new_status
        self.last_activity = datetime.now(timezone.utc)

    def force_idle(self):
        """Force agent to IDLE state"""
        if self.status != AgentStatus.IDLE:
            self.record_error(
                f"Forced from {self.status} to IDLE", {"forced_recovery": True}
            )
            self.set_status(AgentStatus.IDLE)


class AgentManager:
    """Agent manager"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.is_running = False
        self.error_count: int = 0
        self.last_error: Optional[str] = None

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

        # Validate message
        if not message.receiver or not message.content:
            return AgentResponse(
                success=False,
                content="I couldn't process your message. Please make sure you're sending to a valid agent with some content.",
                confidence=0.0,
            )

        # Get agent
        agent = self.get_agent(message.receiver)
        if not agent:
            return AgentResponse(
                success=False,
                content=f"I couldn't find the agent '{message.receiver}'. Available agents: {', '.join(self.list_agents())}",
                confidence=0.0,
            )

        # Add message to history (non-critical, continue if it fails)
        try:
            agent.add_message(message)
        except Exception:
            pass  # Silently continue - history is not critical

        # Process message with timeout
        try:
            response = await asyncio.wait_for(
                agent.process_message(message), timeout=60.0
            )
        except asyncio.TimeoutError:
            agent.record_error("Message processing timeout", {"message_id": message.id})
            return AgentResponse(
                success=False,
                content="I'm taking too long to respond. Please try again in a moment.",
                confidence=0.0,
            )
        except Exception as e:
            agent.record_error(
                f"Message processing error: {str(e)}", {"message_id": message.id}
            )
            return AgentResponse(
                success=False,
                content="I encountered an error while processing your message. Please try again.",
                confidence=0.0,
            )

        # Add response to history (non-critical)
        try:
            response_message = AgentMessage(
                sender=agent.name,
                receiver=message.sender,
                content=response.content,
                metadata={"response_to": message.id, "success": response.success},
            )
            agent.add_message(response_message)
        except Exception:
            pass  # Silently continue - history is not critical

        return response

    async def broadcast_message(
        self, message: AgentMessage
    ) -> Dict[str, AgentResponse]:
        """Broadcast message to all agents"""
        responses = {}
        for agent_name, agent in self.agents.items():
            if agent_name != message.sender:
                message.receiver = agent_name
                responses[agent_name] = await self.send_message(message)
        return responses

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        healthy_agents = [
            name for name, agent in self.agents.items() if agent.is_healthy
        ]
        unhealthy_agents = [
            name for name, agent in self.agents.items() if not agent.is_healthy
        ]

        return {
            "total_agents": len(self.agents),
            "healthy_agents": len(healthy_agents),
            "unhealthy_agents": len(unhealthy_agents),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()},
            "message_queue_size": len(self.message_queue),
            "is_running": self.is_running,
            "system_health": {
                "overall_healthy": len(unhealthy_agents) == 0,
                "error_count": self.error_count,
                "last_error": self.last_error,
            },
        }

    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agents"""
        return [name for name, agent in self.agents.items() if not agent.is_healthy]

    def clear_agent_errors(self, agent_name: str) -> bool:
        """Clear errors for a specific agent"""
        agent = self.get_agent(agent_name)
        if agent:
            agent.clear_errors()
            return True
        return False

    def force_all_agents_idle(self) -> List[str]:
        """Force all agents to IDLE state (simple recovery)"""
        forced_agents = []

        for name, agent in self.agents.items():
            if agent.status != AgentStatus.IDLE:
                agent.force_idle()
                forced_agents.append(name)

        return forced_agents


# Global agent manager
agent_manager = AgentManager()
