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
from app.core.prompt_manager import prompt_manager, PromptType
import asyncio

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
    
    # New structured plan fields for enhanced travel planning
    structured_plan: Optional[Dict[str, Any]] = None  # Structured plan data
    plan_events: List[Dict[str, Any]] = []  # Specific plan events
    plan_modifications: Optional[Dict[str, Any]] = None  # Plan modification suggestions


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
        self.process_timeout_seconds: int = 200
        self.refine_timeout_seconds: int = 200
        self.assess_timeout_seconds: int = 40

    @abstractmethod
    async def process_message(self, message: AgentMessage, skip_quality_check: bool = False) -> AgentResponse:
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

    async def process_with_refinement(self, message: AgentMessage, enable_refinement: Optional[bool] = None) -> AgentResponse:
        """Process message with self-refinement loop"""
        # Use parameter if provided, otherwise fall back to instance configuration
        should_refine = enable_refinement if enable_refinement is not None else self.refine_enabled
        
        if not should_refine:
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
                    # In refinement mode, skip internal quality check in process_message
                    # Let the refinement loop handle quality assessment with quality_threshold
                    current_response = await asyncio.wait_for(
                        self.process_message(message, skip_quality_check=True),
                        timeout=self.process_timeout_seconds,
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
                        timeout=self.refine_timeout_seconds,
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
                    timeout=self.assess_timeout_seconds,
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
                logger.warning(f"⚠️ Quality assessment timed out, using fallback score: 0.5")
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
                logger.error(f"❌ Quality assessment failed with exception: {str(e)}")

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

            # Check if quality threshold is met using scientific comparison
            if quality_assessment.meets_threshold:
                # ✅ Quality sufficient - complete refinement early
                current_response.metadata["refinement_status"] = "quality_threshold_met"
                current_response.metadata["early_completion_reason"] = f"Quality {quality_assessment.overall_score:.3f} >= threshold {self.quality_threshold}"
                logger.info(f"🎯 Refinement completed early: quality {quality_assessment.overall_score:.3f} >= {self.quality_threshold}")
                break

            # 📈 Check for diminishing returns (mathematical convergence analysis)
            if iteration > 1 and len(refinement_history) >= 2:
                current_score = quality_assessment.overall_score
                previous_score = refinement_history[-2].overall_score
                
                # Calculate improvement rate and statistical significance
                improvement = current_score - previous_score
                improvement_threshold = 0.05  # Minimum meaningful improvement
                
                # 🔧 Fix score plateau detection - be more lenient and require more data
                if iteration >= 4 and len(refinement_history) >= 4:  # Increased from 3 to 4
                    recent_scores = [h.overall_score for h in refinement_history[-4:]]  # Look at 4 scores instead of 3
                    score_variance = max(recent_scores) - min(recent_scores)
                    # Increased variance threshold from 0.01 to 0.03 to be less aggressive
                    if score_variance < 0.03:
                        current_response.metadata["refinement_status"] = "score_plateau"
                        current_response.metadata["plateau_reason"] = f"Score variance {score_variance:.3f} < 0.03 over {len(recent_scores)} iterations"
                        logger.info(f"🔄 Refinement stopped due to score plateau: variance {score_variance:.3f}")
                        break
                
                # 🚀 Modified diminishing returns: ensure at least 3 attempts before considering early termination
                if iteration >= 3 and improvement <= improvement_threshold:
                    # Calculate trend over multiple iterations for more stable decision
                    scores = [h.overall_score for h in refinement_history]
                    
                    # Check if we have at least 3 scores for trend analysis
                    if len(scores) >= 3:
                        # Calculate overall improvement from start
                        overall_improvement = scores[-1] - scores[0]
                        
                        # Calculate recent trend (last 2 improvements)
                        recent_trend = scores[-1] - scores[-2] if len(scores) >= 2 else 0
                        previous_trend = scores[-2] - scores[-3] if len(scores) >= 3 else 0
                        
                        # Only stop if we have consistent decline AND no overall progress
                        # This prevents premature stopping due to temporary dips
                        should_stop = False
                        
                        if improvement <= -0.1 and recent_trend <= -0.1 and previous_trend <= 0:
                            # Consistent significant decline over 2+ iterations
                            should_stop = True
                            stop_reason = f"consistent decline: {improvement:.3f}, trend: {recent_trend:.3f}"
                        elif overall_improvement <= 0 and improvement <= 0 and iteration >= 5:  # Increased from 4 to 5
                            # No overall progress after 5+ iterations (was 4+)
                            should_stop = True  
                            stop_reason = f"no overall progress after {iteration} iterations: {overall_improvement:.3f}"
                        elif improvement <= improvement_threshold and overall_improvement <= improvement_threshold and iteration >= 6:  # Increased from 5 to 6
                            # Very small improvements after many iterations
                            should_stop = True
                            stop_reason = f"minimal progress: improvement {improvement:.3f}, overall {overall_improvement:.3f}"
                        
                        if should_stop:
                            current_response.metadata["refinement_status"] = "diminishing_returns"
                            current_response.metadata["improvement_analysis"] = {
                                "recent_improvement": improvement,
                                "recent_trend": recent_trend,
                                "previous_trend": previous_trend,
                                "threshold": improvement_threshold,
                                "overall_improvement": overall_improvement,
                                "iteration": iteration,
                                "stop_reason": stop_reason
                            }
                            logger.info(f"📉 Refinement stopped after {iteration} iterations: {stop_reason}")
                            break

            # If this is the last iteration, mark as completed
            if iteration >= self.max_refine_iterations:
                current_response.metadata["refinement_status"] = "max_iterations_reached"
                current_response.metadata["max_iterations_reason"] = f"Reached maximum {self.max_refine_iterations} iterations"
                logger.info(f"🔚 Refinement completed: maximum iterations reached ({self.max_refine_iterations})")
                break

            current_response.metadata["refinement_status"] = "continuing"

        # Final refinement summary
        final_iterations = iteration
        total_refinement_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Count actual refinement loops (iterations beyond the first)
        actual_refinement_loops = max(0, final_iterations - 1)
        
        logger.info(f"🏁 Refinement completed: {actual_refinement_loops} loops, {total_refinement_time:.1f}s total")
        
        # Ensure the refinement loop count is correctly recorded
        if current_response:
            current_response.metadata["actual_refinement_loops"] = actual_refinement_loops
            current_response.metadata["total_iterations"] = final_iterations

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
        """
        Assess the quality of a response
        Uses fast heuristic assessment for performance optimization in refinement loops
        """
        # 🚀 Performance optimization: prioritize fast assessment in refinement mode
        if hasattr(self, '_fast_quality_assessment'):
            # Use fast heuristic assessment - this should be ~0.1s vs 10-20s for LLM
            try:
                overall_score = await self._fast_quality_assessment(original_message, response)
                
                # Generate basic dimension scores based on overall score with slight variance for realism
                quality_dimensions = self.get_quality_dimensions()
                dimension_scores = {}
                for dim, weight in quality_dimensions.items():
                    # Add small variance based on dimension type to make scores more realistic
                    variance = 0.05 if dim in ['accuracy', 'relevance'] else 0.1
                    dim_score = max(0.0, min(1.0, overall_score + (weight - 0.25) * variance))
                    dimension_scores[dim] = dim_score
                
                # 🚀 Skip improvement suggestions generation in refinement mode for speed
                # Only generate suggestions if quality is very low AND it's an early iteration
                improvement_suggestions = []
                if overall_score < 0.6 and iteration <= 2:
                    # Generate basic suggestions without LLM calls
                    improvement_suggestions = self._generate_fast_improvement_suggestions(
                        overall_score, response, iteration
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
                        "assessment_method": "fast_heuristic",
                        "quality_dimensions": list(quality_dimensions.keys()),
                        "assessment_time": "optimized_for_refinement"
                    },
                )
            except Exception as e:
                logger.warning(f"Fast quality assessment failed: {e}, falling back to basic assessment")
                # Fast fallback without LLM
                return self._create_basic_quality_assessment(response, iteration)
        
        # 🔧 Fallback for agents without fast assessment - but still optimize for refinement
        if iteration > 1:
            # In refinement iterations, use simplified assessment to save time
            return self._create_basic_quality_assessment(response, iteration)
        
        # Only use detailed LLM assessment for first iteration of agents without fast assessment
        return await self._detailed_quality_assessment(original_message, response, iteration)

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

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "capabilities": self.capabilities,
            "tools": self.tools,
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

    def _generate_fast_improvement_suggestions(
        self, overall_score: float, response: AgentResponse, iteration: int
    ) -> List[str]:
        """Generate improvement suggestions quickly without LLM calls"""
        suggestions = []
        
        if overall_score < 0.5:
            suggestions.append("Improve response completeness and detail")
            if not response.actions_taken:
                suggestions.append("Include specific actions taken")
            if not response.next_steps:
                suggestions.append("Provide clear next steps")
        elif overall_score < 0.7:
            if response.confidence < 0.8:
                suggestions.append("Increase response confidence with more specific information")
            if len(response.content) < 200:
                suggestions.append("Provide more comprehensive information")
        
        return suggestions
    
    def _create_basic_quality_assessment(
        self, response: AgentResponse, iteration: int
    ) -> QualityAssessment:
        """Create basic quality assessment without expensive LLM calls"""
        # Basic heuristic scoring based on response properties
        base_score = 0.6
        
        # Score based on response completeness
        if response.success:
            base_score += 0.1
        if response.actions_taken:
            base_score += 0.1
        if response.next_steps:
            base_score += 0.1
        if response.confidence > 0.7:
            base_score += 0.1
        if len(response.content) > 150:
            base_score += 0.05
        
        overall_score = min(base_score, 1.0)
        
        # Generate dimension scores
        quality_dimensions = self.get_quality_dimensions()
        dimension_scores = {dim: overall_score * 0.95 for dim in quality_dimensions}
        
        meets_threshold = overall_score >= self.quality_threshold
        
        return QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            improvement_suggestions=[],
            meets_threshold=meets_threshold,
            assessment_details={
                "iteration": iteration,
                "threshold": self.quality_threshold,
                "assessment_method": "basic_heuristic",
                "note": "Fast assessment for refinement optimization"
            },
        )
    
    async def _detailed_quality_assessment(
        self, original_message: AgentMessage, response: AgentResponse, iteration: int
    ) -> QualityAssessment:
        """Detailed quality assessment with LLM calls - used only when necessary"""
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
                "assessment_method": "detailed_llm",
                "quality_dimensions": quality_dimensions,
            },
        )


class AgentManager:
    """Agent manager"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.is_running = False
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.global_timeout_seconds: int = 800  # Timeout for agent message processing

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

        # Process message with timeout
        try:
            response = await asyncio.wait_for(
                agent.process_message(message), timeout=self.global_timeout_seconds
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
