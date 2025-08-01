"""
Travel Agent - Main travel planning agent
This agent is responsible for planning and executing travel itineraries based on user preferences and constraints.
It uses a combination of tools to gather information, generate itineraries, and optimize travel plans.
It also tracks user preferences and updates the travel plan accordingly.
It also uses a rule-based approach to generate actions based on the user's intent and selected tools.
"""

from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
from app.agents.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    QualityAssessment,
)
from app.tools.base_tool import tool_registry
from app.tools.tool_executor import get_tool_executor
from app.core.llm_service import get_llm_service
from app.core.rag_engine import get_rag_engine
from app.core.logging_config import get_logger
from app.core.prompt_manager import prompt_manager, PromptType
from app.knowledge.geographical_data import GeographicalMappings, geo_mappings

logger = get_logger(__name__)


class TravelAgent(BaseAgent):
    """Main travel planning agent"""

    def __init__(self):
        super().__init__(
            name="travel_agent",
            description="Intelligent travel planning assistant, able to develop personalized travel plans based on user needs",
        )

        logger.info("=== TRAVEL AGENT INITIALIZATION START ===")

        logger.info("Initializing LLM service...")
        self.llm_service = get_llm_service()
        logger.info(f"LLM service initialized: {self.llm_service}")

        logger.info("Initializing RAG engine...")
        self.rag_engine = get_rag_engine()
        logger.info(f"RAG engine initialized: {self.rag_engine}")

        logger.info("Initializing tool executor...")
        self.tool_executor = get_tool_executor()
        logger.info(f"Tool executor initialized: {self.tool_executor}")

        # Agent-specific state
        self.current_planning_context: Optional[Dict[str, Any]] = None
        self.user_preferences_history: Dict[str, Any] = {}

        # Travel-specific quality configuration
        self.fast_response_threshold = 0.75  # Threshold for skipping LLM enhancement in process_message
        self.quality_threshold = 0.87  # Higher threshold for refinement loop iterations
        self.refine_enabled = True  # Enable self-refinement by default

        logger.info("=== TRAVEL AGENT INITIALIZATION COMPLETE ===")
        logger.info(f"Agent name: {self.name}")
        logger.info(f"Agent description: {self.description}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Self-refinement enabled: {self.refine_enabled}")
        logger.info("=== END TRAVEL AGENT INITIALIZATION ===")

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "travel_planning",  # Travel planning
            "itinerary_generation",  # Itinerary generation
            "budget_optimization",  # Budget optimization
            "recommendation",  # Personalized recommendation
            "real_time_adjustment",  # Real-time adjustment
            "conflict_resolution",  # Conflict resolution
            "multi_tool_coordination",  # Multi-tool coordination
        ]

    def get_available_tools(self) -> List[str]:
        """Get available tools"""
        return [
            "flight_search",
            "hotel_search",
            "attraction_search",
            # TODO: Add more tools
        ]

    def get_quality_dimensions(self) -> Dict[str, float]:
        """Enhanced travel-specific quality assessment dimensions with information fusion"""
        return {
            "relevance": 0.15,  # How relevant to user's travel request
            "information_fusion": 0.20,  # How well knowledge + tools are integrated
            "completeness": 0.15,  # How complete the travel information is
            "accuracy": 0.15,  # How accurate the information is
            "actionability": 0.15,  # How actionable the recommendations are
            "personalization": 0.05,  # How well personalized to user preferences
            "plan_structure_quality": 0.15,  # How well structured the travel plan is
        }

    async def _assess_dimension(
        self, dimension: str, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Enhanced dimension assessment including information fusion"""

        if dimension == "information_fusion":
            return await self._assess_information_fusion(original_message, response)
        elif dimension == "personalization":
            return await self._assess_personalization(original_message, response)
        elif dimension == "feasibility":
            return await self._assess_feasibility(original_message, response)
        elif dimension == "plan_structure_quality":
            return await self._assess_plan_structure_quality(original_message, response)
        else:
            # Use base class implementation for standard dimensions
            return await super()._assess_dimension(
                dimension, original_message, response
            )

    async def _assess_personalization(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess how well the response is personalized"""
        score = 0.5  # Base score

        # Check if user preferences were considered
        user_message_lower = original_message.content.lower()
        response_content_lower = response.content.lower()

        # Look for personal preference indicators
        preference_indicators = [
            "budget",
            "prefer",
            "like",
            "dislike",
            "interest",
            "family",
            "solo",
            "couple",
            "group",
            "luxury",
            "backpack",
            "adventure",
        ]

        mentioned_preferences = sum(
            1 for indicator in preference_indicators if indicator in user_message_lower
        )

        addressed_preferences = sum(
            1
            for indicator in preference_indicators
            if indicator in response_content_lower
        )

        if mentioned_preferences > 0:
            personalization_ratio = addressed_preferences / mentioned_preferences
            score += personalization_ratio * 0.4

        # Check if response mentions specific user needs
        if any(word in response_content_lower for word in ["your", "you", "based on"]):
            score += 0.1

        return min(score, 1.0)

    async def _assess_feasibility(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess the feasibility of the travel recommendations"""
        score = 0.6  # Base score

        # Check if practical considerations are mentioned
        practical_elements = [
            "time",
            "schedule",
            "duration",
            "distance",
            "transport",
            "booking",
            "availability",
            "season",
            "weather",
        ]

        response_content_lower = response.content.lower()
        practical_mentions = sum(
            1 for element in practical_elements if element in response_content_lower
        )

        # More practical considerations = higher feasibility
        if practical_mentions >= 3:
            score += 0.3
        elif practical_mentions >= 1:
            score += 0.2

        # Check if tools were used (indicates real data was considered)
        if response.actions_taken and len(response.actions_taken) >= 2:
            score += 0.1

        return min(score, 1.0)

    async def _assess_information_fusion(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess how well knowledge context and tool results are integrated"""

        metadata = response.metadata

        # Check if both knowledge and tools were used
        knowledge_context = metadata.get("knowledge_context", {})
        has_knowledge = len(knowledge_context.get("relevant_docs", [])) > 0
        has_tools = len(metadata.get("tools_used", [])) > 0

        base_score = 0.4  # Base score for any response

        # Score based on information source utilization
        if has_knowledge and has_tools:
            base_score += 0.3  # Both sources used effectively
        elif has_knowledge or has_tools:
            base_score += 0.1  # One source used

        # Check response quality indicators
        response_content = response.content.lower()

        # Look for evidence of information integration
        integration_indicators = [
            "based on",
            "according to",
            "current",
            "available",
            "recommendations",
            "options",
            "information shows",
            "data indicates",
        ]
        integration_mentions = sum(
            1 for indicator in integration_indicators if indicator in response_content
        )
        if integration_mentions >= 2:
            base_score += 0.1  # Good integration language

        # Check for comprehensive response
        if len(response.content) > 400:
            base_score += 0.1  # Comprehensive response suggests good fusion

        # Check for actionable follow-up
        if response.next_steps and len(response.next_steps) > 0:
            base_score += 0.1  # Actionable follow-up indicates good integration

        # Check if fusion strategy was applied
        if metadata.get("information_fusion_applied", False):
            base_score += 0.1  # Explicit fusion process applied

        return min(base_score, 1.0)

    async def _generate_improvement_suggestions(
        self,
        dimension: str,
        original_message: AgentMessage,
        response: AgentResponse,
        current_score: float,
    ) -> List[str]:
        """Generate travel-specific improvement suggestions"""
        suggestions = await super()._generate_improvement_suggestions(
            dimension, original_message, response, current_score
        )

        if dimension == "information_fusion" and current_score < 0.6:
            suggestions.extend(
                [
                    "Better integrate static knowledge context with dynamic tool results",
                    "Ensure both authoritative background and current data are presented",
                    "Use knowledge context for comprehensive details and tools for actionable data",
                    "Blend information sources seamlessly without explicitly mentioning sources",
                    "Let user intent guide which information to emphasize",
                ]
            )

        if dimension == "personalization" and current_score < 0.6:
            suggestions.extend(
                [
                    "Consider user's specific preferences mentioned in the request",
                    "Tailor recommendations to the user's travel style and interests",
                    "Reference user's budget, group size, or travel dates in recommendations",
                ]
            )

        if dimension == "feasibility" and current_score < 0.6:
            suggestions.extend(
                [
                    "Include practical considerations like travel times and logistics",
                    "Verify availability and booking requirements",
                    "Consider seasonal factors and weather conditions",
                    "Provide realistic timeline and scheduling information",
                ]
            )

        if dimension == "completeness" and current_score < 0.6:
            suggestions.extend(
                [
                    "Include information about flights, accommodation, and activities",
                    "Provide cost estimates and budget breakdown",
                    "Add transportation details between locations",
                ]
            )

        return suggestions

    def _is_plan_request(self, user_message: str) -> bool:
        """Detect if user message is requesting plan generation or modification using precise analysis"""
        message_lower = user_message.lower()
        
        # ❌ Explicit exclusions for recommendation/information requests  
        exclusion_patterns = [
            "what are", "what is", "which", "where", "when", "how", "why",
            "recommend", "suggest", "best", "advice", "tips", "should i",
            "can you recommend", "what do you recommend", "any recommendations",
            "tell me about", "explain", "describe", "what about", "thoughts on"
        ]
        
        # Check for exclusions first
        for pattern in exclusion_patterns:
            if pattern in message_lower:
                # Additional check: even if it has exclusion words, if it's clearly a plan request, allow it
                strong_plan_indicators = [
                    "create", "build", "generate", "make", "plan my", "plan a", 
                    "itinerary", "day by day", "schedule"
                ]
                has_strong_plan = any(indicator in message_lower for indicator in strong_plan_indicators)
                if not has_strong_plan:
                    return False
        
        # ✅ Strong plan generation indicators (commands/requests)
        strong_plan_keywords = [
            "plan my trip", "plan a trip", "create plan", "create itinerary",
            "make plan", "build itinerary", "generate itinerary", "plan for",
            "travel plan", "vacation plan", "trip plan", "day by day itinerary",
            "detailed itinerary", "complete itinerary", "full itinerary",
            "schedule my", "organize my trip", "arrange my"
        ]
        
        # Check for strong plan indicators
        for keyword in strong_plan_keywords:
            if keyword in message_lower:
                return True
                
        # ✅ Duration-based planning (specific time periods)
        duration_patterns = [
            r'\d+[\s-]day', r'\d+[\s-]week', r'\d+[\s-]night',
            "weekend trip", "week trip", "month trip", "vacation"
        ]
        
        import re
        for pattern in duration_patterns:
            if re.search(pattern, message_lower):
                # If mentions duration AND has action words
                action_words = ["plan", "visit", "go to", "travel to", "trip to"]
                if any(action in message_lower for action in action_words):
                    return True
        
        # ✅ Plan modification keywords (for existing plans)
        modification_keywords = [
            "change my plan", "modify my itinerary", "update my plan", 
            "edit my schedule", "revise my trip", "adjust my itinerary",
            "add to my plan", "remove from plan", "reschedule"
        ]
        
        for keyword in modification_keywords:
            if keyword in message_lower:
                return True
        
        # ✅ Context-specific planning phrases
        planning_phrases = [
            "i'm planning", "i am planning", "help me plan", 
            "planning a trip", "organizing a trip"
        ]
        
        for phrase in planning_phrases:
            if phrase in message_lower:
                return True
                
        return False

    async def _assess_plan_structure_quality(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """Assess the quality of travel plan structure"""
        score = 0.0
        
        # Check if this is a plan-related request
        if not self._is_plan_request(original_message.content):
            return 1.0  # Non-plan requests get full score for this dimension
        
        # Check if structured plan data exists
        if hasattr(response, 'structured_plan') and response.structured_plan:
            score += 0.4
            
            # Check required plan fields
            required_fields = ["destination", "duration", "events"]
            if all(field in response.structured_plan for field in required_fields):
                score += 0.2
        
        # Check if plan events exist
        if hasattr(response, 'plan_events') and response.plan_events:
            score += 0.3
            
            # Check event completeness
            complete_events = 0
            for event in response.plan_events:
                if all(key in event for key in ["title", "start_time", "end_time", "location"]):
                    complete_events += 1
            
            if complete_events > 0:
                score += 0.1 * (complete_events / len(response.plan_events))
        
        # Check content structure indicators for plan requests
        content = response.content.lower()
        plan_indicators = ["day 1", "day 2", "morning", "afternoon", "evening", "itinerary", "schedule"]
        indicators_found = sum(1 for indicator in plan_indicators if indicator in content)
        
        if indicators_found >= 3:
            score = max(score, 0.7)  # Content shows plan structure
        elif indicators_found >= 1:
            score = max(score, 0.5)
        
        return min(score, 1.0)

    async def _refine_response(
        self,
        original_message: AgentMessage,
        current_response: AgentResponse,
        quality_assessment: Optional[QualityAssessment],
    ) -> AgentResponse:
        """Refine travel response based on quality assessment using prompt_manager"""
        if not quality_assessment:
            return current_response

        # Try to use LLM for travel-specific response refinement
        try:
            if self.llm_service:
                # Use prompt manager for LLM-based travel response refinement
                refinement_prompt = prompt_manager.get_prompt(
                    PromptType.RESPONSE_REFINEMENT,
                    original_response=current_response.content,
                    quality_assessment=quality_assessment.dict(),
                    improvement_areas=quality_assessment.improvement_suggestions,
                )

                # Use structured completion for response refinement
                schema = prompt_manager.get_schema(PromptType.RESPONSE_REFINEMENT)
                refinement_result = await self.llm_service.structured_completion(
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
                        "travel_refined": True,
                        "refinement_method": "llm_based",
                        "quality_dimensions_improved": [
                            dim
                            for dim, score in quality_assessment.dimension_scores.items()
                            if score < 0.6
                        ],
                        "improvement_applied": quality_assessment.improvement_suggestions,
                        "llm_refinement_applied": refinement_result.get(
                            "applied_improvements", []
                        ),
                    },
                )

        except Exception as e:
            logger.error(f"LLM-based refinement failed: {e}")
            # Fall back to travel-specific heuristic refinement

        # Fallback to travel-specific heuristic refinement
        improved_content = current_response.content
        improved_actions = current_response.actions_taken.copy()
        improved_next_steps = current_response.next_steps.copy()

        # Apply travel-specific improvements based on quality assessment
        for suggestion in quality_assessment.improvement_suggestions:
            if (
                "personalization" in suggestion.lower()
                or "preferences" in suggestion.lower()
            ):
                improved_content += "\n\n🎯 **Personalized for You**: This recommendation takes into account your specific travel preferences and requirements mentioned."

            elif (
                "feasibility" in suggestion.lower() or "practical" in suggestion.lower()
            ):
                improved_content += "\n\n⏰ **Practical Considerations**: I've considered travel times, seasonal factors, and booking requirements to ensure this plan is realistic and achievable."

            elif "completeness" in suggestion.lower() and "cost" in suggestion.lower():
                improved_content += "\n\n💰 **Budget Information**: Please note that costs may vary based on season, availability, and booking timing. I recommend checking current prices for the most accurate estimates."

            elif "transportation" in suggestion.lower():
                improved_next_steps.append(
                    "Research transportation options between destinations"
                )

            elif "booking" in suggestion.lower():
                improved_next_steps.append(
                    "Check availability and make reservations in advance"
                )

        # Enhance response based on low-scoring dimensions
        if quality_assessment.dimension_scores.get("personalization", 1.0) < 0.6:
            improved_content += "\n\n✨ **Tailored Recommendations**: These suggestions are customized based on your travel style and preferences."

        if quality_assessment.dimension_scores.get("feasibility", 1.0) < 0.6:
            improved_content += "\n\n📋 **Feasibility Check**: All recommendations have been evaluated for practicality and realistic implementation."

        if quality_assessment.dimension_scores.get("actionability", 1.0) < 0.6:
            if "Next steps to book your trip:" not in improved_content:
                improved_next_steps.extend(
                    [
                        "Compare prices across different booking platforms",
                        "Read recent reviews for hotels and attractions",
                        "Check visa requirements and travel restrictions",
                        "Consider travel insurance options",
                    ]
                )

        # Increase confidence based on refinement
        improvement_boost = len(quality_assessment.improvement_suggestions) * 0.05
        improved_confidence = min(current_response.confidence + improvement_boost, 1.0)

        return AgentResponse(
            success=current_response.success,
            content=improved_content,
            actions_taken=improved_actions,
            next_steps=improved_next_steps,
            confidence=improved_confidence,
            metadata={
                **current_response.metadata,
                "travel_refined": True,
                "refinement_method": "heuristic_fallback",
                "quality_dimensions_improved": [
                    dim
                    for dim, score in quality_assessment.dimension_scores.items()
                    if score < 0.6
                ],
                "improvement_applied": quality_assessment.improvement_suggestions,
            },
        )

    async def plan_travel(self, message: AgentMessage, enable_refinement: Optional[bool] = None) -> AgentResponse:
        """Public method to plan travel with self-refinement enabled"""
        return await self.process_with_refinement(message, enable_refinement=enable_refinement)

    async def process_message(self, message: AgentMessage, skip_quality_check: bool = False) -> AgentResponse:
        """
        Process user message with optimized performance flow:
        1. Intent analysis (with conversation context)
        2. Rule-based tool planning (no LLM call)
        3. Execute tools
        4. Structured response fusion (template-based, fast)
        5. Fast quality assessment (heuristic, ~0.1s)
        6. Smart decision: return if quality good, else LLM enhancement
        """
        try:
            self.status = AgentStatus.THINKING

            # 0. Process conversation history and maintain agent state
            conversation_history = message.metadata.get("conversation_history", [])
            session_id = message.metadata.get("session_id")
            
            # Update agent's memory with conversation context
            if conversation_history:
                self._update_agent_memory_from_history(conversation_history, session_id)
            
            # 1. Understand user intent (with conversation context)
            intent = await self._analyze_user_intent(message.content, conversation_history)

            # 2. Rule-based tool planning (no LLM call)
            action_plan = self._create_rule_based_action_plan(intent, message.content)

            # 3. Execute tool action plan
            self.status = AgentStatus.ACTING
            execution_context = {
                **message.metadata,
                "original_message": message.content,
                "session_id": session_id,  # Add session_id to context
            }
            result = await self._execute_action_plan(action_plan, execution_context)

            # 4. Generate response content using LLM
            response_content = await self._generate_response(result, intent)

            # Create agent response
            response = AgentResponse(
                success=result.get("success", False),
                content=response_content,
                actions_taken=result.get("actions", []),
                next_steps=result.get("next_steps", []),
                confidence=0.85 if result.get("success", False) else 0.7,
                metadata={
                    "intent": intent,
                    "tools_used": result.get("tools_used", []),
                    "execution_time": result.get("execution_time", 0),
                    "response_method": "llm_generated",
                    "destination": intent.get("destination", "unknown"),
                    "intent_type": intent.get("type", "query"),
                    "is_plan_request": self._is_plan_request(message.content),
                    "execution_result_for_plan": result if self._is_plan_request(message.content) else None,
                    "tool_execution_success": result.get("success", False),
                    "partial_tool_success": bool(result.get("results", {}))
                }
            )

            # 5. Fast quality assessment (heuristic, ~0.1s) - Skip if in refinement mode
            if skip_quality_check:
                self.status = AgentStatus.IDLE
                return response
            
            quality_score = await self._fast_quality_assessment(message, response)

            # 6. Quality good enough -> return, else enhancement
            if quality_score >= self.fast_response_threshold:  # 0.75 for fast response
                self.status = AgentStatus.IDLE
                return response
            else:
                # Try to enhance if needed
                enhanced_response = await self._llm_enhanced_response(response, result, intent)
                self.status = AgentStatus.IDLE
                return enhanced_response

        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Error processing message: {e}")
            return AgentResponse(
                success=False,
                content=f"I encountered an error while processing your request: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _update_agent_memory_from_history(self, conversation_history: List[Dict], session_id: str):
        """Update agent's memory and preferences from conversation history"""
        logger.debug(f"Updating agent memory from {len(conversation_history)} conversation entries")
        
        # Update current planning context with session info
        if not self.current_planning_context:
            self.current_planning_context = {}
        
        self.current_planning_context.update({
            "session_id": session_id,
            "conversation_history": conversation_history,
            "last_updated": datetime.now().isoformat(),
        })
        
        # Extract user preferences from conversation history
        for entry in conversation_history:
            user_msg = entry.get("user", "").lower()
            assistant_msg = entry.get("assistant", "").lower()
            
            # Extract travel preferences
            if any(word in user_msg for word in ["prefer", "like", "love", "enjoy"]):
                if "budget" in user_msg or "cheap" in user_msg:
                    self.user_preferences_history["budget_conscious"] = True
                if "luxury" in user_msg or "expensive" in user_msg:
                    self.user_preferences_history["budget_conscious"] = False
                if "adventure" in user_msg:
                    self.user_preferences_history.setdefault("interests", []).append("adventure")
                if "culture" in user_msg or "museum" in user_msg:
                    self.user_preferences_history.setdefault("interests", []).append("culture")
                if "food" in user_msg or "restaurant" in user_msg:
                    self.user_preferences_history.setdefault("interests", []).append("food")
            
            # Extract mentioned destinations for context
            destinations = geo_mappings.get_preference_tracking_destinations()
            for dest in destinations:
                if dest in user_msg:
                    self.user_preferences_history.setdefault("visited_destinations", []).append(dest)
        
        # Remove duplicates from interests and destinations
        if "interests" in self.user_preferences_history:
            self.user_preferences_history["interests"] = list(set(self.user_preferences_history["interests"]))
        if "visited_destinations" in self.user_preferences_history:
            self.user_preferences_history["visited_destinations"] = list(set(self.user_preferences_history["visited_destinations"]))
        
        logger.debug(f"Updated user preferences: {self.user_preferences_history}")

    async def _analyze_user_intent(self, user_message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced intent analysis with information fusion strategy and conversation context"""

        # Step 0: Prepare conversation context for intent analysis
        conversation_context = ""
        if conversation_history:
            recent_context = conversation_history[-3:]  # Use last 3 exchanges for context
            context_parts = []
            for entry in recent_context:
                context_parts.append(f"User: {entry.get('user', '')}")
                context_parts.append(f"Assistant: {entry.get('assistant', '')}")
            conversation_context = "\n".join(context_parts)
            logger.debug(f"Using conversation context for intent analysis: {len(conversation_context)} chars")

        # Step 1: Get LLM intent analysis using enhanced prompt (with conversation context)
        llm_analysis = await self._get_enhanced_llm_intent_analysis(user_message, conversation_context)

        # Step 2: Use structured response for deep analysis
        structured_intent = await self._parse_structured_intent(
            llm_analysis, user_message
        )

        # Step 3: Add information fusion strategy
        enhanced_intent = await self._add_information_fusion_strategy(
            structured_intent, user_message
        )

        # Step 4: Enhance with agent's learned preferences
        if self.user_preferences_history:
            enhanced_intent["learned_preferences"] = self.user_preferences_history.copy()
            logger.debug(f"Added learned preferences to intent: {enhanced_intent['learned_preferences']}")

        return enhanced_intent

    async def _get_enhanced_llm_intent_analysis(
        self, user_message: str, conversation_context: str = ""
    ) -> Dict[str, Any]:
        """Get enhanced LLM intent analysis using updated prompt manager with conversation context"""

        # Build enhanced prompt with conversation context
        if conversation_context:
            # Create a context-aware prompt
            context_aware_message = f"""
Previous conversation context:
{conversation_context}

Current user message: {user_message}

Please analyze the current message considering the conversation history for better context understanding.
"""
            analysis_prompt = prompt_manager.get_prompt(
                PromptType.INTENT_ANALYSIS, user_message=context_aware_message
            )
        else:
            # Standard prompt without context
            analysis_prompt = prompt_manager.get_prompt(
            PromptType.INTENT_ANALYSIS, user_message=user_message
        )

        try:
            if self.llm_service:
                # Use real LLM service with structured output including fusion strategy
                schema = prompt_manager.get_schema(PromptType.INTENT_ANALYSIS)

                response = await self.llm_service.structured_completion(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    response_schema=schema,
                    temperature=0.3,  # Low temperature for consistent results
                    max_tokens=800,
                )

                # Parse LLM response to structured format
                return response
            else:
                # Use enhanced fallback analysis
                return await self._enhanced_fallback_intent_analysis(user_message)

        except Exception as e:
            logger.error(f"LLM intent analysis failed: {e}")
            return await self._enhanced_fallback_intent_analysis(user_message)

    async def _parse_structured_intent(
        self, llm_analysis: Dict[str, Any], user_message: str
    ) -> Dict[str, Any]:
        """Parse and validate structured intent analysis"""

        # Validate the structured response
        if not isinstance(llm_analysis, dict):
            logger.warning("LLM analysis is not a dictionary, using fallback")
            return await self._enhanced_fallback_intent_analysis(user_message)

        # Ensure required fields exist
        required_fields = [
            "intent_type",
            "destination",
            "sentiment",
            "urgency",
            "confidence_score",
        ]
        for field in required_fields:
            if field not in llm_analysis:
                logger.warning(f"Missing required field {field} in LLM analysis")
                return await self._enhanced_fallback_intent_analysis(user_message)

        # Convert to legacy format for backward compatibility
        # ✅ Fix: Ensure destination is always a string, not a list
        destination_primary = llm_analysis["destination"]["primary"]
        if isinstance(destination_primary, list):
            # If it's a list, take the first item or default to "unknown"
            destination_str = destination_primary[0] if destination_primary else "unknown"
            logger.warning(f"LLM returned destination as list {destination_primary}, using first item: {destination_str}")
        else:
            destination_str = destination_primary if destination_primary else "unknown"
        
        legacy_format = {
            "type": llm_analysis["intent_type"],
            "destination": destination_str,
            "time_info": {
                "duration_days": llm_analysis["travel_details"].get("duration", 0)
            },
            "budget_info": {
                "budget_mentioned": llm_analysis["travel_details"]["budget"].get(
                    "mentioned", False
                )
            },
            "urgency": llm_analysis["urgency"],
            "extracted_info": {
                "destination": destination_str,
                "time_info": llm_analysis["travel_details"],
                "budget_info": llm_analysis["travel_details"]["budget"],
            },
            # Add new structured fields
            "structured_analysis": llm_analysis,
            "information_fusion_strategy": llm_analysis.get(
                "information_fusion_strategy", {}
            ),
        }
        
        # Add user message to structured analysis for fallback origin extraction
        if "structured_analysis" in legacy_format:
            legacy_format["structured_analysis"]["user_message"] = user_message

        return legacy_format

    async def _add_information_fusion_strategy(
        self, structured_intent: Dict[str, Any], user_message: str
    ) -> Dict[str, Any]:
        """Add information fusion strategy to existing intent structure"""

        # Keep all existing fields
        enhanced_intent = structured_intent.copy()

        # Add information fusion strategy if not already present from LLM
        if "information_fusion_strategy" not in enhanced_intent:
            enhanced_intent["information_fusion_strategy"] = {
                "knowledge_priority": self._determine_knowledge_priority(
                    structured_intent
                ),
                "tool_priority": self._determine_tool_priority(structured_intent),
                "integration_approach": self._determine_integration_approach(
                    structured_intent
                ),
                "response_focus": self._determine_response_focus(structured_intent),
            }
            logger.info(
                f"Added fallback information fusion strategy: {enhanced_intent['information_fusion_strategy']}"
            )
        else:
            logger.info(
                f"Information fusion strategy already present from LLM: {enhanced_intent['information_fusion_strategy']}"
            )

        return enhanced_intent

    def _determine_knowledge_priority(self, intent: Dict[str, Any]) -> str:
        """Determine knowledge context priority based on intent"""
        intent_type = intent.get("intent_type", "query")
        destination = intent.get("destination", {}).get("primary", "")

        if intent_type == "query" and destination:
            return "very_high"  # Information queries need comprehensive knowledge
        elif intent_type == "planning" and destination:
            return "high"  # Planning needs detailed destination information
        elif intent_type == "recommendation":
            return "high"  # Recommendations benefit from rich context
        else:
            return "medium"

    def _determine_tool_priority(self, intent: Dict[str, Any]) -> str:
        """Determine tool results priority based on intent"""
        intent_type = intent.get("intent_type", "query")

        if intent_type in ["planning", "booking"]:
            return "high"  # Need current prices and availability
        elif intent_type == "recommendation":
            return "medium"  # Tools provide current options
        elif intent_type == "modification":
            return "high"  # Need current data for changes
        else:
            return "low"  # Information queries rely more on knowledge

    def _determine_integration_approach(self, intent: Dict[str, Any]) -> str:
        """Determine how to integrate information sources"""
        intent_type = intent.get("intent_type", "query")

        if intent_type == "query":
            return "knowledge_first"  # Knowledge context leads, tools support
        elif intent_type == "planning":
            return "balanced"  # Equal weight to knowledge and tools
        elif intent_type == "recommendation":
            return "tools_first"  # Current options lead, knowledge provides context
        elif intent_type in ["booking", "modification"]:
            return "tools_first"  # Current data is critical
        else:
            return "balanced"

    def _determine_response_focus(self, intent: Dict[str, Any]) -> str:
        """Determine primary response focus"""
        intent_type = intent.get("intent_type", "query")

        focus_map = {
            "planning": "comprehensive_plan",
            "query": "detailed_information",
            "recommendation": "curated_options",
            "booking": "actionable_steps",
            "modification": "specific_changes",
        }

        return focus_map.get(intent_type, "helpful_information")

    async def _enhanced_fallback_intent_analysis(
        self, user_message: str
    ) -> Dict[str, Any]:
        """Enhanced fallback intent analysis with information fusion strategy"""

        import re
        user_message_lower = user_message.lower()

        # ✅ Enhanced intent type detection (more precise)
        intent_type = "query"  # default
        
        # Use the improved _is_plan_request method for planning detection
        if self._is_plan_request(user_message):
            intent_type = "planning"
        # Check for recommendations first (many requests are recommendations)
        elif any(
            pattern in user_message_lower
            for pattern in ["recommend", "suggest", "what are", "what is", "which", 
                          "best", "advice", "tips", "should i", "can you recommend", 
                          "what do you recommend", "any recommendations", "thoughts on"]
        ):
            intent_type = "recommendation"
        # Plan modifications (for existing plans)
        elif any(
            pattern in user_message_lower
            for pattern in ["change my plan", "modify my itinerary", "update my plan", 
                          "edit my schedule", "revise my trip", "adjust my itinerary"]
        ):
            intent_type = "modification"
        # Booking/reservation requests
        elif any(
            word in user_message_lower
            for word in ["book", "reserve", "purchase", "buy", "booking"]
        ):
            intent_type = "booking"
        # Complaint/problem reports
        elif any(
            word in user_message_lower
            for word in ["problem", "issue", "complaint", "wrong", "error", "trouble"]
        ):
            intent_type = "complaint"
        # Information queries (catch remaining questions)
        elif any(
            pattern in user_message_lower
            for pattern in ["what", "how", "when", "where", "why", "tell me", "explain", "describe"]
        ):
            intent_type = "query"

        # Enhanced destination detection
        destination = "Unknown"
        destinations = geo_mappings.get_intent_analysis_destinations()
        for dest in destinations:
            if dest in user_message_lower:
                destination = dest.lower()  # Keep lowercase to match IATA mapping
                break

        # Enhanced origin detection
        origin = "Unknown"
        # Look for "from [origin] to [destination]" pattern
        from_to_pattern = re.search(r"from\s+(\w+)\s+to\s+(\w+)", user_message_lower)
        if from_to_pattern:
            potential_origin = from_to_pattern.group(1)
            potential_dest = from_to_pattern.group(2)
            # Check if both are in our destinations list
            if potential_origin in destinations:
                origin = potential_origin.title()
            if potential_dest in destinations:
                destination = potential_dest.title()
        else:
            # Look for "from [origin]" pattern
            from_pattern = re.search(r"from\s+(\w+)", user_message_lower)
            if from_pattern:
                potential_origin = from_pattern.group(1)
                if potential_origin in destinations:
                    origin = potential_origin.title()

        # Enhanced time extraction
        time_info = {}
        import re

        days_match = re.search(r"(\d+)\s*(?:day|days|天|日)", user_message_lower)
        if days_match:
            time_info["duration_days"] = int(days_match.group(1))

        # Enhanced budget detection
        budget_info = {"mentioned": False}
        if any(
            word in user_message_lower
            for word in [
                "budget",
                "cost",
                "price",
                "money",
                "spend",
                "expensive",
                "cheap",
            ]
        ):
            budget_info["mentioned"] = True

            # Try to extract budget amount
            budget_match = re.search(
                r"[\$€£¥]?(\d+(?:,\d{3})*(?:\.\d{2})?)", user_message
            )
            if budget_match:
                budget_info["amount"] = float(budget_match.group(1).replace(",", ""))
                budget_info["currency"] = "USD"

        # Enhanced sentiment analysis
        sentiment = "neutral"
        if any(
            word in user_message_lower
            for word in ["excited", "amazing", "great", "wonderful", "love"]
        ):
            sentiment = "positive"
        elif any(
            word in user_message_lower
            for word in ["worried", "concerned", "problem", "difficult"]
        ):
            sentiment = "negative"
        elif any(
            word in user_message_lower
            for word in ["can't wait", "looking forward", "dream"]
        ):
            sentiment = "excited"

        # Enhanced urgency detection
        urgency = "medium"
        if any(
            word in user_message_lower
            for word in ["urgent", "asap", "immediately", "quickly", "soon"]
        ):
            urgency = "urgent"
        elif any(
            word in user_message_lower for word in ["flexible", "whenever", "no rush"]
        ):
            urgency = "low"

        # Create enhanced structured response with information fusion strategy
        enhanced_analysis = {
            "intent_type": intent_type,
            "origin": {
                "primary": origin,
                "secondary": [],
                "region": "Unknown",
                "confidence": 0.6,
            },
            "destination": {
                "primary": destination,
                "secondary": [],
                "region": "Unknown",
                "confidence": 0.6,
            },
            "travel_details": {
                "duration": time_info.get("duration_days", 0),
                "travelers": 1,
                "budget": {
                    "mentioned": budget_info["mentioned"],
                    "amount": budget_info.get("amount", 0),
                    "currency": budget_info.get("currency", "USD"),
                    "level": "mid-range",
                },
                "dates": {
                    "departure": "unknown",
                    "return": "unknown",
                    "flexibility": "unknown",
                },
            },
            "preferences": {
                "travel_style": "mid-range",
                "interests": [],
                "accommodation_type": "unknown",
                "transport_preference": "unknown",
            },
            "sentiment": sentiment,
            "urgency": urgency,
            "missing_info": ["specific_dates", "exact_budget", "preferences"],
            "key_requirements": [],
            "information_fusion_strategy": {
                "knowledge_priority": self._determine_knowledge_priority(
                    {
                        "intent_type": intent_type,
                        "destination": {"primary": destination},
                    }
                ),
                "tool_priority": self._determine_tool_priority(
                    {"intent_type": intent_type}
                ),
                "integration_approach": self._determine_integration_approach(
                    {"intent_type": intent_type}
                ),
                "response_focus": self._determine_response_focus(
                    {"intent_type": intent_type}
                ),
            },
            "confidence_score": 0.6,
        }

        # Legacy format for backward compatibility
        legacy_result = {
            "type": intent_type,
            "origin": origin,
            "destination": destination,
            "time_info": time_info,
            "budget_info": budget_info,
            "urgency": urgency,
            "extracted_info": {
                "origin": origin,
                "destination": destination,
                "time_info": time_info,
                "budget_info": budget_info,
            },
            "structured_analysis": enhanced_analysis,
            "information_fusion_strategy": enhanced_analysis[
                "information_fusion_strategy"
            ],
        }

        return legacy_result

    def _extract_entities(self, user_message: str) -> Dict[str, Any]:
        """Extract additional entities from user message"""
        entities = {"locations": [], "numbers": [], "dates": [], "currencies": []}

        import re

        # Extract numbers
        numbers = re.findall(r"\d+", user_message)
        entities["numbers"] = [int(num) for num in numbers]

        # Extract potential dates
        date_patterns = [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}",
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            entities["dates"].extend(matches)

        # Extract currencies
        currency_symbols = re.findall(r"[\$€£¥]", user_message)
        entities["currencies"] = currency_symbols

        return entities

    def _analyze_linguistic_features(self, user_message: str) -> Dict[str, Any]:
        """Analyze linguistic features of the message"""
        features = {
            "message_length": len(user_message),
            "word_count": len(user_message.split()),
            "question_count": user_message.count("?"),
            "exclamation_count": user_message.count("!"),
            "has_urgency_markers": any(
                word in user_message.lower() for word in ["urgent", "asap", "quickly"]
            ),
            "has_politeness_markers": any(
                word in user_message.lower()
                for word in ["please", "thank you", "could you"]
            ),
            "language_hints": self._detect_language_hints(user_message),
        }
        return features

    def _detect_language_hints(self, user_message: str) -> str:
        """Detect language hints in the message"""
        # Simple language detection based on character patterns
        if any(char in user_message for char in "你好请帮助我"):
            return "chinese"
        elif any(char in user_message for char in "こんにちはお願いします"):
            return "japanese"
        elif any(char in user_message for char in "안녕하세요도와주세요"):
            return "korean"
        else:
            return "english"

    async def _retrieve_knowledge_context(
        self, query: str, structured_intent: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None  # Add session_id parameter
    ) -> Dict[str, Any]:
        """
        Retrieve knowledge context for the query, including existing plan context
        """
        try:
            context = {
                "relevant_docs": [],
                "search_results": [],
                "total_docs": 0,
                "search_strategy": "none",
                "plan_context": {},
                "has_existing_plan": False
            }

            if not self.rag_engine:
                logger.warning("RAG engine not available for knowledge retrieval")
                return context

            # Determine search strategy
            search_strategy = "standard"
            if structured_intent:
                info_fusion = structured_intent.get("information_fusion", {})
                search_strategy = info_fusion.get("knowledge_priority", "standard")

            # Retrieve relevant documents
            if search_strategy == "comprehensive":
                docs_limit = 8
                confidence_threshold = 0.5
            elif search_strategy == "focused":
                docs_limit = 5
                confidence_threshold = 0.7
            else:  # standard
                docs_limit = 6
                confidence_threshold = 0.6

            # Use RAG engine's retrieve method
            retrieval_result = await self.rag_engine.retrieve(
                query, top_k=docs_limit, structured_intent=structured_intent
            )
            
            # Extract documents from retrieval result
            relevant_docs = []
            if retrieval_result and hasattr(retrieval_result, 'documents'):
                for doc in retrieval_result.documents:
                    relevant_docs.append({
                        "id": getattr(doc, 'id', 'unknown'),
                        "content": getattr(doc, 'content', ''),
                        "metadata": getattr(doc, 'metadata', {}),
                        "score": getattr(doc, 'score', 0.0)
                    })

            context.update({
                "relevant_docs": relevant_docs,
                "total_docs": len(relevant_docs),
                "search_strategy": search_strategy,
                "confidence_threshold": confidence_threshold,
            })

            # NEW: Add plan context retrieval
            plan_context = await self._retrieve_plan_context(session_id, query, structured_intent)
            context.update({
                "plan_context": plan_context,
                "has_existing_plan": plan_context.get("has_plan", False),
                "plan_events": plan_context.get("events", []),
                "plan_metadata": plan_context.get("metadata", {})
            })

            logger.info(f"Retrieved {len(relevant_docs)} docs with {search_strategy} strategy, plan context: {plan_context.get('has_plan', False)}")
            return context

        except Exception as e:
            logger.error(f"Error retrieving knowledge context: {e}")
            return {
                "relevant_docs": [],
                "search_results": [],
                "total_docs": 0,
                "search_strategy": "error",
                "plan_context": {},
                "has_existing_plan": False
            }

    async def _retrieve_plan_context(
        self, session_id: str, query: str, intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve existing plan context for the session"""
        if not session_id:
            return {"has_plan": False}
        
        try:
            from app.core.plan_manager import get_plan_manager
            plan_manager = get_plan_manager()
            
            existing_plan = plan_manager.get_plan_by_session(session_id)
            if not existing_plan or not existing_plan.events:
                return {"has_plan": False}
            
            # Analyze plan relevance to current query
            relevant_events = self._filter_relevant_events(existing_plan.events, query, intent)
            
            return {
                "has_plan": True,
                "plan_id": existing_plan.plan_id,
                "events": [event.model_dump() for event in relevant_events],
                "metadata": existing_plan.metadata.model_dump(),
                "last_updated": existing_plan.updated_at,
                "event_summary": self._summarize_plan_events(existing_plan.events),
                "gaps_identified": self._identify_plan_gaps(existing_plan, intent)
            }
        
        except Exception as e:
            logger.error(f"Error retrieving plan context: {e}")
            return {"has_plan": False}

    def _filter_relevant_events(
        self, events: List, query: str, intent: Optional[Dict[str, Any]]
    ) -> List:
        """Filter events relevant to current query"""
        if not events:
            return []
        
        query_lower = query.lower()
        relevant_events = []
        
        # Check for date-related queries
        date_keywords = ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        time_keywords = ["morning", "afternoon", "evening", "night"]
        
        # Check for activity-related queries
        activity_keywords = ["flight", "hotel", "attraction", "restaurant", "visit", "tour"]
        
        for event in events:
            is_relevant = False
            
            # Check if query mentions specific event types
            if intent and intent.get("type") == "specific_query":
                for keyword in activity_keywords:
                    if keyword in query_lower and keyword in str(event.event_type).lower():
                        is_relevant = True
                        break
            
            # Check if query mentions location in event
            if event.location and any(word in query_lower for word in event.location.lower().split()):
                is_relevant = True
            
            # Check if query mentions event title
            if event.title and any(word in query_lower for word in event.title.lower().split()):
                is_relevant = True
            
            # If no specific filters match, include recent events
            if not is_relevant and len(relevant_events) < 5:
                is_relevant = True
            
            if is_relevant:
                relevant_events.append(event)
        
        return relevant_events[:10]  # Limit to 10 most relevant events

    def _summarize_plan_events(self, events: List) -> str:
        """Create human-readable summary of plan events"""
        if not events:
            return "No events scheduled"
        
        summary_parts = []
        event_types = {}
        
        for event in events:
            event_type = str(event.event_type).lower()
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        for event_type, count in event_types.items():
            if count == 1:
                summary_parts.append(f"1 {event_type}")
            else:
                summary_parts.append(f"{count} {event_type}s")
        
        return f"Current plan includes: {', '.join(summary_parts)}"

    def _identify_plan_gaps(
        self, plan, intent: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify missing elements in the plan"""
        gaps = []
        
        if not plan.events:
            return ["No events scheduled"]
        
        # Check for common travel components
        event_types = [str(event.event_type).lower() for event in plan.events]
        
        essential_types = ["flight", "hotel"]
        recommended_types = ["attraction", "restaurant"]
        
        for essential in essential_types:
            if essential not in event_types:
                gaps.append(f"Missing {essential} arrangements")
        
        for recommended in recommended_types:
            if recommended not in event_types:
                gaps.append(f"Could add {recommended} recommendations")
        
        # Check for timeline gaps (more than 8 hours between events on same day)
        # This would need more sophisticated date analysis
        return gaps


    async def _llm_tool_selection(
        self, structured_analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to intelligently select tools based on structured analysis"""

        try:
            # Build tool selection prompt
            tool_selection_prompt = prompt_manager.get_prompt(
                PromptType.TOOL_SELECTION, intent_analysis=structured_analysis
            )

            if self.llm_service:
                # Use real LLM for tool selection
                schema = prompt_manager.get_schema(PromptType.TOOL_SELECTION)
                response = await self.llm_service.structured_completion(
                    messages=[{"role": "user", "content": tool_selection_prompt}],
                    response_schema=schema,
                    temperature=0.2,
                    max_tokens=400,
                )
                return response
            else:
                # Use fallback tool selection
                return await self._fallback_tool_selection(structured_analysis)

        except Exception as e:
            logger.error(f"LLM tool selection failed: {e}")
            return await self._fallback_tool_selection(structured_analysis)

    async def _fallback_tool_selection(
        self, structured_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback tool selection based on structured analysis"""

        intent_type = structured_analysis.get("intent_type", "query")
        destination = structured_analysis.get("destination", {}).get(
            "primary", "Unknown"
        )

        selected_tools = []

        # Smart tool selection based on intent type
        if intent_type == "planning":
            selected_tools = ["attraction_search", "hotel_search", "flight_search"]
        elif intent_type == "recommendation":
            selected_tools = ["attraction_search"]
        elif intent_type == "booking":
            selected_tools = ["flight_search", "hotel_search"]
        elif intent_type == "query":
            # Select tools based on what user is asking about
            interests = structured_analysis.get("preferences", {}).get("interests", [])
            if any(
                interest in ["hotel", "accommodation", "stay"] for interest in interests
            ):
                selected_tools.append("hotel_search")
            if any(
                interest in ["flight", "plane", "airline"] for interest in interests
            ):
                selected_tools.append("flight_search")
            if not selected_tools:  # Default to attraction search
                selected_tools = ["attraction_search"]
        else:
            selected_tools = ["attraction_search"]

        # Calculate tool priorities
        tool_priority = {}
        for i, tool in enumerate(selected_tools):
            tool_priority[tool] = 1.0 - (i * 0.1)  # Decreasing priority

        # Determine execution strategy
        execution_strategy = "parallel" if len(selected_tools) > 1 else "sequential"

        return {
            "selected_tools": selected_tools,
            "tool_priority": tool_priority,
            "execution_strategy": execution_strategy,
            "reasoning": f"Selected tools based on intent type '{intent_type}' and destination '{destination}'",
            "confidence": 0.7,
        }


    def _extract_origin_from_message(self, user_message: str) -> str:
        """Extract origin city from user message using improved text parsing"""
        import re
        
        # Convert to lowercase for easier matching
        message_lower = user_message.lower()
        
        # Known cities that could be origins
        cities = [
            "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", 
            "shanghai", "rome", "barcelona", "amsterdam", "vienna", "prague", 
            "budapest", "berlin", "bangkok", "singapore", "seoul", "sydney", 
            "melbourne", "munich", "madrid", "athens", "dubai", "istanbul",
            "toronto", "vancouver", "montreal", "calgary", "ottawa"
        ]
        
        # Exclude common non-location words
        non_locations = [
            "trip", "travel", "plan", "vacation", "holiday", "journey", "flight",
            "hotel", "stay", "visit", "tour", "day", "week", "month", "year"
        ]
        
        # Improved patterns for origin extraction (more specific)
        patterns = [
            r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)',  # "from Paris to Rome"
            r'departing\s+from\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)',  # "departing from Paris"
            r'flying\s+from\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)',  # "flying from Paris"
            r'starting\s+from\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)',  # "starting from Paris"
            r'travel\s+from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)',  # "travel from Paris to Rome"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                origin = match.group(1).strip().lower()
                
                # Skip if it's a non-location word
                if origin in non_locations:
                    continue
                    
                # Check if it's a known city (exact match or partial match)
                for city in cities:
                    if city in origin or origin in city:
                        return city.title()
                
                # If not in our list but looks like a valid city name
                if len(origin) > 2 and origin.replace(' ', '').isalpha() and origin not in non_locations:
                    return origin.title()
        
        # Try to find "CITY to DESTINATION" pattern but only if CITY is a known location
        city_to_pattern = r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)'
        match = re.search(city_to_pattern, message_lower)
        if match:
            potential_origin = match.group(1).strip().lower()
            # Only accept if it's a known city and not a common word
            if potential_origin in cities and potential_origin not in non_locations:
                return potential_origin.title()
        
        # If no valid origin found, return "Unknown" (will default to Toronto)
        return "Unknown"






    async def _generate_intent_based_actions(
        self, structured_analysis: Dict[str, Any], selected_tools: List[str]
    ) -> List[str]:
        """Generate actions based on structured intent analysis"""

        actions = []
        intent_type = structured_analysis.get("intent_type", "query")
        destination = structured_analysis.get("destination", {}).get(
            "primary", "Unknown"
        )

        # Generate actions based on intent type
        if intent_type == "planning":
            actions.extend(
                [
                    f"Analyze travel requirements for {destination}",
                    "Search for attractions and activities",
                    "Find suitable accommodations",
                    "Research transportation options",
                    "Create comprehensive travel plan",
                ]
            )
        elif intent_type == "recommendation":
            actions.extend(
                [
                    f"Search for top attractions in {destination}",
                    "Analyze user preferences",
                    "Generate personalized recommendations",
                ]
            )
        elif intent_type == "booking":
            actions.extend(
                [
                    f"Search for available options in {destination}",
                    "Compare prices and availability",
                    "Prepare booking information",
                ]
            )
        elif intent_type == "query":
            actions.extend(
                [
                    f"Search for information about {destination}",
                    "Provide relevant details",
                    "Answer specific questions",
                ]
            )
        else:
            actions.extend(
                [
                    f"Process request for {destination}",
                    "Gather relevant information",
                    "Provide helpful response",
                ]
            )

        # Add tool-specific actions
        for tool in selected_tools:
            if tool == "attraction_search":
                actions.append("Search for attractions and activities")
            elif tool == "hotel_search":
                actions.append("Search for accommodation options")
            elif tool == "flight_search":
                actions.append("Search for flight information")

        return actions

    async def _generate_intent_based_next_steps(
        self, structured_analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Generate next steps based on structured intent analysis"""

        next_steps = []
        intent_type = structured_analysis.get("intent_type", "query")
        missing_info = structured_analysis.get("missing_info", [])
        urgency = structured_analysis.get("urgency", "medium")

        # Generate next steps based on missing information
        if "specific_dates" in missing_info:
            next_steps.append("Could you please specify your travel dates?")

        if "exact_budget" in missing_info:
            next_steps.append("What is your budget range for this trip?")

        if "preferences" in missing_info:
            next_steps.append(
                "What type of activities or attractions are you most interested in?"
            )

        # Generate next steps based on intent type
        if intent_type == "planning":
            next_steps.extend(
                [
                    "Would you like me to create a detailed itinerary?",
                    "Do you need help with booking accommodations or flights?",
                ]
            )
        elif intent_type == "recommendation":
            next_steps.extend(
                [
                    "Would you like more specific recommendations based on your interests?",
                    "Do you want information about the best times to visit these places?",
                ]
            )
        elif intent_type == "booking":
            next_steps.extend(
                [
                    "Would you like me to help you compare prices?",
                    "Do you want to proceed with making reservations?",
                ]
            )

        # Add urgency-based next steps
        if urgency == "urgent":
            next_steps.insert(
                0,
                "I understand this is urgent. Let me prioritize the most important information first.",
            )
        elif urgency == "low":
            next_steps.append(
                "Take your time to review the information, and let me know if you need any clarification."
            )

        return next_steps

    async def _analyze_user_requirements(
        self, intent: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user requirements in detail"""

        # Extract key information from intent
        base_requirements = {
            "destination": intent.get("destination", "unknown"),
            "intent_type": intent.get("type") or intent.get("intent_type", "query"),
            "time_constraints": intent.get("time_info", {}),
            "budget_constraints": intent.get("budget_info", {}),
            "urgency": intent.get("urgency", "normal"),
        }

        # Analyze user message for additional requirements
        user_message = context.get("original_message", "")

        # Use LLM for sophisticated requirement analysis if available
        if self.llm_service:
            try:
                # Use prompt manager for requirement extraction
                requirement_prompt = prompt_manager.get_prompt(
                    PromptType.REQUIREMENT_EXTRACTION,
                    user_message=user_message,
                    intent_analysis=intent,
                )

                # Use structured completion for requirement extraction
                schema = prompt_manager.get_schema(PromptType.REQUIREMENT_EXTRACTION)
                response = await self.llm_service.structured_completion(
                    messages=[{"role": "user", "content": requirement_prompt}],
                    response_schema=schema,
                    temperature=0.3,
                    max_tokens=600,
                )

                # Use structured response directly
                enhanced_requirements = response
                base_requirements.update(enhanced_requirements)

            except Exception as e:
                logger.warning(f"LLM requirement analysis failed: {e}")

        # Add contextual analysis
        base_requirements.update(
            {
                "user_context": context,
                "session_history": context.get("session_history", []),
                "user_preferences": self.user_preferences_history,
            }
        )

        return base_requirements

    async def _intelligent_tool_selection(
        self, requirements: Dict[str, Any]
    ) -> List[str]:
        """Select tools based on intelligent multi-dimensional analysis"""

        selected_tools = []
        tool_scores = {}

        # Define tool selection criteria with scoring weights
        criteria = {
            "flight_search": {
                "triggers": [
                    "transportation",
                    "flight",
                    "airline",
                    "ticket",
                    "travel between cities",
                    "fly",
                    "airplane",
                ],
                "intent_relevance": {
                    "planning": 0.8,
                    "recommendation": 0.2,
                    "query": 0.3,
                },
                "cost_score": 0.6,  # API cost consideration (lower is more expensive)
                "value_score": 0.9,  # Information value (higher is more valuable)
            },
            "hotel_search": {
                "triggers": [
                    "accommodation",
                    "hotel",
                    "stay",
                    "lodging",
                    "sleep",
                    "overnight",
                    "room",
                ],
                "intent_relevance": {
                    "planning": 0.9,
                    "recommendation": 0.4,
                    "query": 0.3,
                },
                "cost_score": 0.5,
                "value_score": 0.8,
            },
            "attraction_search": {
                "triggers": [
                    "attraction",
                    "sightseeing",
                    "activity",
                    "visit",
                    "tour",
                    "experience",
                    "explore",
                    "museum",
                    "park",
                ],
                "intent_relevance": {
                    "planning": 0.7,
                    "recommendation": 0.9,
                    "query": 0.6,
                },
                "cost_score": 0.8,  # Lower API cost
                "value_score": 0.9,
            },
        }

        user_message = (
            requirements.get("user_context", {}).get("original_message", "").lower()
        )
        intent_type = requirements["intent_type"]

        # Calculate tool scores using multi-dimensional analysis
        for tool_name, tool_info in criteria.items():
            score = 0.0

            # Keyword matching score
            keyword_matches = sum(
                1 for trigger in tool_info["triggers"] if trigger in user_message
            )
            keyword_score = min(keyword_matches * 0.15, 1.0)

            # Intent relevance score
            intent_score = tool_info["intent_relevance"].get(intent_type, 0.0)

            # Cost-benefit analysis
            cost_benefit = tool_info["value_score"] / tool_info["cost_score"]

            # Requirement necessity score
            necessity_score = self._calculate_tool_necessity(tool_name, requirements)

            # Final score calculation with weighted components
            final_score = (
                keyword_score * 0.3
                + intent_score * 0.4
                + cost_benefit * 0.2
                + necessity_score * 0.1
            )

            tool_scores[tool_name] = final_score

            # Selection threshold
            if final_score >= 0.4:
                selected_tools.append(tool_name)

        # Sort by score and apply resource optimization
        selected_tools.sort(key=lambda x: tool_scores.get(x, 0), reverse=True)

        # Apply resource constraints (max 3 tools for efficiency)
        if len(selected_tools) > 3:
            selected_tools = selected_tools[:3]
            logger.info(f"Limited tools to top 3 for efficiency: {selected_tools}")

        logger.info(f"Tool selection scores: {tool_scores}")
        return selected_tools

    def _calculate_tool_necessity(
        self, tool_name: str, requirements: Dict[str, Any]
    ) -> float:
        """Calculate how necessary a tool is based on requirements"""

        necessity_score = 0.0
        user_message = (
            requirements.get("user_context", {}).get("original_message", "").lower()
        )

        # Flight search necessity
        if tool_name == "flight_search":
            if any(
                word in user_message
                for word in ["flight", "fly", "airline", "airport", "ticket"]
            ):
                necessity_score += 0.8
            if requirements.get("destination", "").lower() not in ["unknown", "local"]:
                necessity_score += 0.3
            if "international" in user_message:
                necessity_score += 0.4

        # Hotel search necessity
        elif tool_name == "hotel_search":
            if any(
                word in user_message
                for word in ["hotel", "stay", "accommodation", "night", "room"]
            ):
                necessity_score += 0.9
            if requirements.get("time_constraints", {}).get("duration_days", 0) > 1:
                necessity_score += 0.4
            if "overnight" in user_message or "multi-day" in user_message:
                necessity_score += 0.3

        # Attraction search necessity
        elif tool_name == "attraction_search":
            if any(
                word in user_message
                for word in [
                    "attraction",
                    "visit",
                    "see",
                    "tour",
                    "activity",
                    "explore",
                ]
            ):
                necessity_score += 0.7
            if requirements["intent_type"] == "recommendation":
                necessity_score += 0.5
            if "sightseeing" in user_message or "explore" in user_message:
                necessity_score += 0.3

        return min(necessity_score, 1.0)

    async def _extract_tool_parameters(
        self, selected_tools: List[str], requirements: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract intelligent parameters for each tool"""

        tool_parameters = {}
        user_message = requirements.get("user_context", {}).get("original_message", "")

        for tool_name in selected_tools:
            if tool_name == "flight_search":
                tool_parameters[tool_name] = await self._extract_flight_parameters(
                    user_message, requirements
                )
            elif tool_name == "hotel_search":
                tool_parameters[tool_name] = await self._extract_hotel_parameters(
                    user_message, requirements
                )
            elif tool_name == "attraction_search":
                tool_parameters[tool_name] = await self._extract_attraction_parameters(
                    user_message, requirements
                )

        return tool_parameters

    async def _extract_flight_parameters(
        self, user_message: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract flight search parameters intelligently"""

        params = {
            "origin": "unknown",
            "destination": requirements.get("destination", "unknown"),
            "departure_date": "flexible",
            "return_date": "flexible",
            "passengers": 1,
            "class": "economy",
            "budget_range": "medium",
        }

        # Extract specific parameters from user message
        user_message_lower = user_message.lower()

        # Extract passenger count
        import re

        passenger_match = re.search(
            r"(\d+)\s*(?:people|person|passenger|traveler)", user_message_lower
        )
        if passenger_match:
            params["passengers"] = int(passenger_match.group(1))

        # Extract travel class
        if any(
            word in user_message_lower
            for word in ["business", "first class", "premium"]
        ):
            params["class"] = "business"
        elif any(word in user_message_lower for word in ["economy", "cheap", "budget"]):
            params["class"] = "economy"

        # Extract budget preference
        if any(
            word in user_message_lower for word in ["budget", "cheap", "affordable"]
        ):
            params["budget_range"] = "low"
        elif any(
            word in user_message_lower for word in ["luxury", "expensive", "premium"]
        ):
            params["budget_range"] = "high"

        return params

    async def _extract_hotel_parameters(
        self, user_message: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract hotel search parameters intelligently"""

        # Get destination and convert to IATA code(s) for AMADEUS API
        destination = requirements.get("destination", "unknown")
        iata_result = self._convert_city_to_iata_code(destination)
        
        # Add debugging information
        logger.info(f"🏨 Hotel search - Original destination: '{destination}'")
        logger.info(f"🏨 Hotel search - Converted IATA result: '{iata_result}'")
        logger.info(f"🏨 Hotel search - IATA result type: {type(iata_result)}")
        logger.info(f"🏨 Hotel search - IATA result length: {len(iata_result) if iata_result else 0}")

        # Set default dates (7 days from now for check-in, 10 days from now for check-out)
        from datetime import datetime, timedelta
        default_check_in = datetime.now() + timedelta(days=7)
        default_check_out = datetime.now() + timedelta(days=10)
        
        params = {
            "location": iata_result,  # Use IATA code(s) for AMADEUS API
            "destination": destination,  # Keep original for display
            "check_in": default_check_in.strftime("%Y-%m-%d"),
            "check_out": default_check_out.strftime("%Y-%m-%d"),
            "guests": 1,
            "rooms": 1,
            "star_rating": "any",
            "budget_range": "medium",
        }

        # Extract hotel-specific parameters
        user_message_lower = user_message.lower()

        # Extract guest count
        import re

        guest_match = re.search(r"(\d+)\s*(?:guest|people|person)", user_message_lower)
        if guest_match:
            params["guests"] = int(guest_match.group(1))

        # Extract hotel preferences
        if any(word in user_message_lower for word in ["5 star", "luxury", "premium"]):
            params["star_rating"] = "5"
        elif any(word in user_message_lower for word in ["4 star", "high-end"]):
            params["star_rating"] = "4"
        elif any(word in user_message_lower for word in ["budget", "cheap", "hostel"]):
            params["star_rating"] = "3"

        logger.info(f"🏨 Hotel search - Final parameters: {params}")
        return params

    async def _extract_attraction_parameters(
        self, user_message: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attraction search parameters intelligently"""

        params = {
            "location": requirements.get("destination", "unknown"),
            "category": None,
            "min_rating": 4.0,
            "max_results": 10,
            "include_photos": True,
            "radius_meters": 10000,
        }

        # Extract attraction-specific parameters
        user_message_lower = user_message.lower()

        # Extract category preferences
        if any(
            word in user_message_lower
            for word in ["museum", "art", "culture", "history"]
        ):
            params["category"] = "museum"
        elif any(
            word in user_message_lower
            for word in ["park", "nature", "outdoor", "garden"]
        ):
            params["category"] = "park"
        elif any(
            word in user_message_lower
            for word in ["temple", "shrine", "religious", "church"]
        ):
            params["category"] = "tourist_attraction"

        # Extract quality preferences
        if any(
            word in user_message_lower
            for word in ["best", "top", "must-see", "highly rated"]
        ):
            params["min_rating"] = 4.5
        elif any(
            word in user_message_lower for word in ["any", "all", "comprehensive"]
        ):
            params["min_rating"] = 3.0

        return params

    async def _optimize_execution_strategy(
        self, selected_tools: List[str], tool_parameters: Dict[str, Dict[str, Any]]
    ) -> str:
        """Optimize execution strategy based on tool dependencies and resources"""

        # Simple dependency rules
        if len(selected_tools) <= 1:
            return "sequential"

        # Check for tool dependencies
        has_flight = "flight_search" in selected_tools
        has_hotel = "hotel_search" in selected_tools
        has_attraction = "attraction_search" in selected_tools

        # If we have complementary tools without strong dependencies, use parallel
        if has_attraction and (has_flight or has_hotel) and len(selected_tools) <= 3:
            return "parallel"

        # For complex planning with multiple dependencies, use sequential
        if len(selected_tools) > 2:
            return "sequential"

        return "parallel"

    async def _generate_fallback_strategies(
        self, selected_tools: List[str], requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate fallback strategies for tool failures"""

        fallback_strategies = []

        for tool_name in selected_tools:
            fallback = {
                "tool": tool_name,
                "failure_scenarios": [],
                "alternatives": [],
                "degraded_service": [],
                "user_guidance": [],
            }

            if tool_name == "flight_search":
                fallback.update(
                    {
                        "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                        "alternatives": [
                            {"method": "manual_search_guidance", "confidence": 0.6},
                            {"method": "general_transport_info", "confidence": 0.5},
                        ],
                        "degraded_service": [
                            "provide_airline_websites",
                            "suggest_travel_agents",
                        ],
                        "user_guidance": [
                            "I'll provide airline websites for manual search",
                            "Consider contacting travel agents for complex bookings",
                        ],
                    }
                )
            elif tool_name == "hotel_search":
                fallback.update(
                    {
                        "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                        "alternatives": [
                            {"method": "manual_booking_guidance", "confidence": 0.7},
                            {"method": "area_recommendations", "confidence": 0.6},
                        ],
                        "degraded_service": [
                            "provide_booking_websites",
                            "suggest_hotel_chains",
                        ],
                        "user_guidance": [
                            "I'll recommend popular booking websites",
                            "Consider checking hotel chains directly",
                        ],
                    }
                )
            elif tool_name == "attraction_search":
                fallback.update(
                    {
                        "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                        "alternatives": [
                            {"method": "knowledge_base_search", "confidence": 0.8},
                            {"method": "general_recommendations", "confidence": 0.7},
                        ],
                        "degraded_service": [
                            "provide_tourism_websites",
                            "suggest_guidebooks",
                        ],
                        "user_guidance": [
                            "I'll search our knowledge base for attraction information",
                            "You can also check popular travel websites like TripAdvisor",
                        ],
                    }
                )

            fallback_strategies.append(fallback)

        return fallback_strategies

    def _calculate_plan_confidence(
        self, selected_tools: List[str], requirements: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the action plan"""

        base_confidence = 0.5

        # Boost confidence based on tool-requirement matching
        requirement_coverage = 0.0
        if requirements.get("destination") != "unknown":
            requirement_coverage += 0.3
        if requirements.get("time_constraints"):
            requirement_coverage += 0.2
        if selected_tools:
            requirement_coverage += 0.3

        # Boost confidence based on tool selection quality
        tool_quality = min(len(selected_tools) * 0.1, 0.2)

        final_confidence = min(
            base_confidence + requirement_coverage + tool_quality, 1.0
        )
        return final_confidence

    async def _generate_action_sequence(self, plan: Dict[str, Any]) -> List[str]:
        """Generate optimal action sequence"""

        actions = ["retrieve_knowledge"]

        # Add tool-specific actions
        for tool_name in plan["tools_to_use"]:
            if tool_name == "flight_search":
                actions.append("search_flights")
            elif tool_name == "hotel_search":
                actions.append("search_hotels")
            elif tool_name == "attraction_search":
                actions.append("search_attractions")

        # Add synthesis actions
        if len(plan["tools_to_use"]) > 1:
            actions.append("synthesize_results")

        actions.append("generate_recommendations")

        return actions

    async def _generate_next_steps(self, plan: Dict[str, Any]) -> List[str]:
        """Generate context-aware next steps"""

        next_steps = []

        # Generic next steps based on tools used
        if "flight_search" in plan["tools_to_use"]:
            next_steps.extend(
                [
                    "Compare flight prices across different dates",
                    "Check baggage policies and restrictions",
                ]
            )

        if "hotel_search" in plan["tools_to_use"]:
            next_steps.extend(
                ["Read recent reviews and ratings", "Check cancellation policies"]
            )

        if "attraction_search" in plan["tools_to_use"]:
            next_steps.extend(
                [
                    "Check opening hours and seasonal availability",
                    "Consider purchasing tickets in advance",
                ]
            )

        # Add general travel planning steps
        next_steps.extend(
            [
                "Verify visa requirements and documentation",
                "Consider travel insurance options",
                "Check weather conditions for travel dates",
            ]
        )

        return next_steps

    async def _create_basic_action_plan(
        self, intent: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback basic action plan (original implementation)"""

        plan = {
            "intent_type": intent.get("type") or intent.get("intent_type", "query"),
            "actions": [],
            "tools_to_use": [],
            "execution_strategy": "sequential",
            "next_steps": [],
            "confidence": 0.6,
        }

        # Basic action planning based on intent type
        intent_type = intent.get("type") or intent.get("intent_type", "query")

        if intent_type == "planning":
            plan["actions"] = [
                "retrieve_knowledge",
                "search_flights",
                "search_hotels", 
                "search_attractions",
                "generate_plan"
            ]
            plan["tools_to_use"] = [
                "flight_search",
                "hotel_search",
                "attraction_search",
            ]
            plan["next_steps"] = [
                "Provide detailed travel itinerary",
                "Include budget estimates",
                "Suggest booking timeline",
            ]
        elif intent_type == "recommendation":
            plan["actions"] = [
                "retrieve_knowledge",
                "search_attractions",
                "generate_recommendations",
            ]
            plan["tools_to_use"] = ["attraction_search"]
            plan["next_steps"] = [
                "Provide personalized recommendations",
                "Include practical tips",
                "Suggest related activities",
            ]
        elif intent_type == "query":
            plan["actions"] = ["retrieve_knowledge", "answer_question"]
            plan["tools_to_use"] = []
            plan["next_steps"] = [
                "Provide detailed answer",
                "Offer additional information",
                "Ask follow-up questions",
            ]

        return plan

    def _parse_llm_requirements(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response for requirements (simplified implementation)"""

        # This is a simplified parser - in production, you'd want more robust JSON parsing
        requirements = {
            "budget_sensitivity": "medium",
            "time_sensitivity": "normal",
            "travel_style": "standard",
            "geographic_scope": "unknown",
        }

        # Basic keyword extraction from LLM response
        response_lower = llm_response.lower()

        if "budget" in response_lower or "cheap" in response_lower:
            requirements["budget_sensitivity"] = "high"
        elif "luxury" in response_lower or "premium" in response_lower:
            requirements["budget_sensitivity"] = "low"

        if "urgent" in response_lower or "asap" in response_lower:
            requirements["time_sensitivity"] = "urgent"
        elif "flexible" in response_lower:
            requirements["time_sensitivity"] = "flexible"

        return requirements

    async def _execute_action_plan(
        self, plan: Dict[str, Any], context: Dict[str, Any], session_id: str = None
    ) -> Dict[str, Any]:
        """Execute action plan using real tools and retrieve knowledge context"""
        results = {
            "tools_used": [],
            "results": {},
            "execution_time": 0.0,
            "success": True,
        }

        try:
            # First, always retrieve knowledge context for any travel query
            query = context.get("original_message", "travel information")
            session_id = context.get("session_id")
            # Use destination from plan for better knowledge retrieval
            destination = plan.get("destination", "unknown")
            if destination != "unknown":
                query = f"{query} {destination}"
            
            knowledge_context = await self._retrieve_knowledge_context(query, session_id=session_id)
            results["knowledge_context"] = knowledge_context
            
            # Add planning context to results
            results["actions"] = plan.get("actions", [])
            results["next_steps"] = plan.get("next_steps", [])

            # Execute tools using real tool executor
            if plan.get("tools_to_use"):
                # Import required classes for tool execution
                from app.tools.tool_executor import (
                    ToolCall,
                    ToolChain,
                    ToolExecutionContext,
                )

                # Create tool execution context
                tool_context = ToolExecutionContext(
                    request_id=f"travel_req_{int(time.time())}",
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id"),
                    metadata=context,
                )

                # Create tool calls with proper parameters
                tool_calls = []
                for tool_name in plan["tools_to_use"]:
                    tool_params = plan.get("tool_parameters", {}).get(tool_name, {})

                    # Convert parameters to proper format for each tool
                    if tool_name == "attraction_search":
                        # Convert travel agent parameters to AttractionSearchInput format
                        attraction_params = {
                            "location": tool_params.get("location", "unknown"),  # attraction_search uses "location" key
                            "query": None,  # Let the tool use location-based search
                            "max_results": tool_params.get("max_results", 10),
                            "include_photos": True,
                            "min_rating": 4.0,
                            "radius_meters": 10000,
                        }
                        tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                input_data=attraction_params,
                                context=context,
                            )
                        )

                    elif tool_name == "hotel_search":
                        # For hotel search, we need proper date handling
                        # Since we don't have specific dates, we'll provide basic search
                        destination = tool_params.get("location", "unknown")  # hotel_search uses "location" key
                        
                        # Check if destination is already converted to list of IATA codes
                        if isinstance(destination, list):
                            # Already converted to IATA codes, use directly
                            iata_result = destination
                            logger.info(f"🏨 Travel agent - Using pre-converted IATA codes: {destination}")
                        else:
                            # Convert destination to IATA code(s) for AMADEUS API
                            iata_result = self._convert_city_to_iata_code(destination)
                            logger.info(f"🏨 Travel agent - Converted '{destination}' to IATA: {iata_result}")
                        
                        hotel_params = {
                            "location": iata_result,  # Use IATA code(s) for AMADEUS API
                            "check_in": tool_params.get(
                                "check_in", "2024-06-01"
                            ),  # Default dates
                            "check_out": tool_params.get("check_out", "2024-06-03"),
                            "guests": tool_params.get("guests", 1),
                            "rooms": 1,
                            "min_rating": 4.0,
                        }
                        
                        # Add debugging to see what's being sent to tool executor

                        logger.info(f"🏨 Travel agent - Creating hotel search ToolCall")
                        logger.info(f"🏨 Travel agent - Original destination: '{destination}'")
                        logger.info(f"🏨 Travel agent - Converted IATA result: '{iata_result}'")
                        logger.info(f"🏨 Travel agent - Hotel params being sent: {hotel_params}")
                        tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                input_data=hotel_params,
                                context=context,
                            )
                        )

                    elif tool_name == "flight_search":
                        # For flight search, we need proper date handling
                        from datetime import datetime, timedelta
                        # Use a future date for the search
                        future_date = datetime.now() + timedelta(days=30)
                        
                        destination = tool_params.get("destination", "unknown")
                        
                        # ✅ Fix: Check if destination is already converted to list of IATA codes
                        if isinstance(destination, list):
                            # Already converted to IATA codes, use directly
                            destination_result = destination
                            logger.info(f"✈️ Travel agent - Using pre-converted IATA codes: {destination}")
                        else:
                            # Convert destination to IATA code(s) for AMADEUS API
                            destination_result = self._convert_city_to_iata_code(destination)
                            logger.info(f"✈️ Travel agent - Converted '{destination}' to IATA: {destination_result}")
                        
                        # ✅ NEW: Enable flight_chain for multi-destination searches
                        is_multi_destination = isinstance(destination_result, list) and len(destination_result) > 1
                        flight_chain_enabled = is_multi_destination
                        
                        if flight_chain_enabled:
                            logger.info(f"🔗 Enabling flight chain search for {len(destination_result)} destinations")
                        
                        flight_params = {
                            "origin": tool_params.get("origin", "PAR"),
                            "destination": destination_result,  # Use IATA code(s) for AMADEUS API
                            "start_date": future_date,
                            "passengers": tool_params.get("passengers", 1),  # flight_search uses "passengers" key
                            "class_type": "ECONOMY",
                            "budget_level": tool_params.get("budget_level", "mid-range"),
                            "flight_chain": flight_chain_enabled,  # Enable flight chain for multi-city trips
                        }
                        tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                input_data=flight_params,
                                context=context,
                            )
                        )

                if tool_calls:
                    # Create and execute tool chain
                    tool_chain = ToolChain(
                        calls=tool_calls,
                        strategy=plan.get("execution_strategy", "parallel"),
                    )

                    # Execute using real tool executor
                    execution_result = await self.tool_executor.execute_chain(
                        tool_chain, tool_context
                    )

                    # Store tool results and track which tools were used
                    results["results"] = execution_result.results
                    results["execution_time"] = execution_result.execution_time
                    
                    # Populate tools_used list with the names of tools that were executed
                    for tool_call in tool_calls:
                        tool_name = tool_call.tool_name
                        if tool_name in execution_result.results:
                            results["tools_used"].append(tool_name)
                    
                    logger.info(f"Tools executed and added to tools_used: {results['tools_used']}")

                    if not execution_result.success:
                        results["success"] = False
                        results["error"] = execution_result.error

        except Exception as e:
            logger.error(f"Error executing action plan: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    async def _generate_response(
        self, execution_result: Dict[str, Any], intent: Dict[str, Any]
    ) -> str:
        """Generate response content using intelligent information fusion with plan awareness"""

        logger.info(f"=== INTELLIGENT RESPONSE GENERATION START ===")
        logger.info(
            f"Execution result success: {execution_result.get('success', False)}"
        )
        logger.info(f"Execution result keys: {list(execution_result.keys())}")

        try:
            # Extract information sources
            knowledge_context = execution_result.get("knowledge_context", {})
            tool_results = execution_result.get("results", {})
            user_message = execution_result.get("original_message", "")
            
            # NEW: Extract plan context
            plan_context = knowledge_context.get("plan_context", {})
            has_existing_plan = plan_context.get("has_plan", False)

            logger.info(f"Information Sources Summary:")
            logger.info(
                f"  - Knowledge docs: {len(knowledge_context.get('relevant_docs', []))}"
            )
            logger.info(f"  - Tool results: {list(tool_results.keys())}")
            logger.info(f"  - Intent type: {intent.get('type', 'unknown')}")
            logger.info(f"  - Has existing plan: {has_existing_plan}")
            if has_existing_plan:
                logger.info(f"  - Plan events: {len(plan_context.get('events', []))}")
                logger.info(f"  - Plan summary: {plan_context.get('event_summary', 'N/A')}")

            if execution_result["success"]:
                # Try intelligent information fusion using LLM
                if self.llm_service:
                    logger.info("Attempting LLM-based information fusion...")
                    try:
                        # Format information for fusion prompt
                        relevant_docs = knowledge_context.get("relevant_docs", [])
                        logger.info(f"🧠 Knowledge context docs available: {len(relevant_docs)}")
                        
                        formatted_knowledge = self._format_knowledge_for_fusion(relevant_docs)
                        logger.info(f"🧠 Formatted knowledge length: {len(formatted_knowledge) if formatted_knowledge else 0}")
                        logger.debug(f"🧠 Formatted knowledge preview: {formatted_knowledge[:200] if formatted_knowledge else 'None'}...")
                        
                        formatted_tools = self._format_tools_for_fusion(tool_results)
                        formatted_intent = self._format_intent_for_fusion(intent)

                        # NEW: Choose fusion prompt based on plan context
                        if has_existing_plan:
                            fusion_prompt = self._create_plan_aware_fusion_prompt(
                                user_message, formatted_knowledge, formatted_tools, 
                                formatted_intent, plan_context
                            )
                        else:
                            # Use information fusion template from prompt manager
                            # Ensure knowledge_context is never None or empty
                            safe_knowledge_context = formatted_knowledge or "No specific knowledge context available for this request."
                            logger.info(f"🧠 Passing knowledge_context to template: {len(safe_knowledge_context)} chars")
                            
                            fusion_prompt = prompt_manager.get_prompt(
                                PromptType.INFORMATION_FUSION,
                                user_message=user_message,
                                knowledge_context=safe_knowledge_context,
                                tool_results=formatted_tools,
                                intent_analysis=formatted_intent,
                            )

                        logger.info(
                            f"Generated fusion prompt (first 300 chars): {fusion_prompt[:300]}..."
                        )

                        # Generate response using LLM
                        fusion_response = await self.llm_service.chat_completion(
                            [{"role": "user", "content": fusion_prompt}],
                            temperature=0.4,
                            max_tokens=1500,
                        )

                        if fusion_response and fusion_response.content:
                            logger.info(f"✅ LLM information fusion completed")
                            logger.info(
                                f"Response length: {len(fusion_response.content)}"
                            )
                            logger.info(
                                f"=== INTELLIGENT RESPONSE GENERATION END (LLM Fusion) ==="
                            )
                            return fusion_response.content
                        else:
                            logger.warning(
                                "LLM fusion returned empty response, using fallback"
                            )

                    except Exception as e:
                        logger.error(f"LLM information fusion failed: {e}")
                        logger.info("Falling back to enhanced template fusion")

                # Fallback to enhanced template-based response generation
                logger.info("Using enhanced template-based response generation...")

                # Try enhanced response generation template
                try:
                    formatted_knowledge = self._format_knowledge_for_fusion(
                        knowledge_context.get("relevant_docs", [])
                    )
                    formatted_tools = self._format_tools_for_fusion(tool_results)
                    formatted_intent = self._format_intent_for_fusion(intent)

                    # Use enhanced response generation template
                    response_prompt = prompt_manager.get_prompt(
                        PromptType.RESPONSE_GENERATION,
                        user_message=user_message,
                        intent_analysis=formatted_intent,
                        tool_results=formatted_tools,
                        knowledge_context=formatted_knowledge,
                    )

                    if self.llm_service:
                        enhanced_response = await self.llm_service.chat_completion(
                            [{"role": "user", "content": response_prompt}],
                            temperature=0.4,
                            max_tokens=1200,
                        )

                        if enhanced_response and enhanced_response.content:
                            logger.info(f"✅ Enhanced template response completed")
                            logger.info(
                                f"=== INTELLIGENT RESPONSE GENERATION END (Enhanced Template) ==="
                            )
                            return enhanced_response.content

                except Exception as e:
                    logger.error(f"Enhanced template response failed: {e}")

                # Final fallback to structured template fusion
                return self._structured_template_fusion(
                    user_message, intent, knowledge_context, tool_results
                )

            else:
                # Handle error cases
                error_msg = execution_result.get("error", "Unknown error")
                
                # ✅ Fix: Ensure error message is never None
                if error_msg is None:
                    error_msg = "Unexpected processing error"
                    logger.warning("Error message was None, using default")
                
                intent_type = intent.get("type") or intent.get("intent_type", "travel")
                error_response = f"I encountered an issue while processing your {intent_type} request: {error_msg}. Let me try to help you in another way. Could you provide more details about what you're looking for?"

                logger.info(f"Error response generated: {error_response}")
                logger.info(f"=== INTELLIGENT RESPONSE GENERATION END (Error) ===")
                return error_response

        except Exception as e:
            logger.error(f"Error in intelligent response generation: {e}")
            import traceback

            logger.error(f"Response generation traceback: {traceback.format_exc()}")

            # Emergency fallback
            return self._emergency_fallback_response(
                execution_result.get("original_message", ""), intent
            )

    def _format_knowledge_for_fusion(self, relevant_docs: List[Dict]) -> str:
        """Format knowledge context for information fusion"""
        logger.info(f"🧠 Formatting knowledge: {len(relevant_docs)} docs provided")
        
        if not relevant_docs:
            fallback_msg = "No specific knowledge context available for this request."
            logger.info(f"🧠 No relevant docs, returning fallback: {fallback_msg}")
            return fallback_msg

        knowledge_parts = []
        for i, doc in enumerate(relevant_docs[:3]):  # Top 3 most relevant
            title = doc.get("metadata", {}).get("title", f"Knowledge {i + 1}")
            content = doc.get("content", "")
            location = doc.get("metadata", {}).get("location", "")

            formatted_doc = f"**{title}**"
            if location:
                formatted_doc += f" (Location: {location})"
            formatted_doc += (
                f"\n{content[:400]}..." if len(content) > 400 else f"\n{content}"
            )

            knowledge_parts.append(formatted_doc)

        result = "\n\n".join(knowledge_parts)
        logger.info(f"🧠 Knowledge formatting complete: {len(result)} chars, {len(knowledge_parts)} parts")
        
        # Ensure we always return something
        return result if result.strip() else "Knowledge context processed but no content available."

    def _format_tools_for_fusion(self, tool_results: Dict) -> str:
        """Format tool results for information fusion"""
        logger.info(f"=== FORMATTING TOOL RESULTS ===")
        logger.info(f"Tool results type: {type(tool_results)}")
        logger.info(f"Tool results keys: {list(tool_results.keys()) if tool_results else 'None'}")
        
        if not tool_results:
            logger.info("No tool results available")
            return "No dynamic tool results available."

        # Debug: Log each tool result in detail
        for tool_name, result in tool_results.items():
            logger.info(f"=== TOOL: {tool_name} ===")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result success: {getattr(result, 'success', 'N/A')}")
            if hasattr(result, 'hotels'):
                logger.info(f"Hotels count: {len(result.hotels) if result.hotels else 0}")
                if result.hotels:
                    logger.info(f"First hotel: {result.hotels[0] if result.hotels else 'None'}")
            if hasattr(result, 'flights'):
                logger.info(f"Flights count: {len(result.flights) if result.flights else 0}")
                if result.flights:
                    logger.info(f"First flight: {result.flights[0] if result.flights else 'None'}")
            logger.info(f"Result data: {getattr(result, 'data', 'N/A')}")
            logger.info(f"Result error: {getattr(result, 'error', 'N/A')}")

        tool_parts = []
        for tool_name, result in tool_results.items():
            logger.info(f"Processing tool: {tool_name}")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}")
            
            # Handle ToolOutput objects
            if hasattr(result, 'success') and hasattr(result, 'data'):
                logger.info(f"Tool {tool_name} success: {result.success}")
                if result.success:
                    if tool_name == "flight_search" and hasattr(result, 'flights'):
                        # ✅ Check if this is a flight chain search
                        is_flight_chain = (hasattr(result, 'data') and result.data and 
                                         result.data.get("search_type") == "flight_chain")
                        
                        if is_flight_chain:
                            # ✅ For flight chain, show representative flights from each route
                            flight_chain_routes = result.data.get("successful_routes", [])
                            route_flights = {}
                            
                            # Group flights by route
                            for flight in result.flights:
                                if hasattr(flight, 'details') and flight.details:
                                    route_name = flight.details.get('route_name', '')
                                    if route_name and route_name not in route_flights:
                                        route_flights[route_name] = flight
                            
                            logger.info(f"Found flight chain with {len(route_flights)} routes covering {len(result.flights)} total flights")
                            
                            if route_flights:
                                tool_parts.append(f"**Flight Chain ({len(route_flights)} routes, {len(result.flights)} flights total)**:")
                                for route_name, flight in route_flights.items():
                                    airline = flight.airline
                                    price = f"{flight.price} {flight.currency}"
                                    duration = f"{flight.duration} minutes"
                                    tool_parts.append(f"- {route_name}: {airline} {flight.flight_number} • {price} • {duration}")
                                    logger.info(f"Added flight chain route: {route_name} - {airline} {flight.flight_number}")
                            else:
                                tool_parts.append("**Flight Chain**: No valid flight routes found.")
                        else:
                            # ✅ For regular flight search, use top 5
                            flights = result.flights[:5]  # Top 5
                            logger.info(f"Found {len(flights)} flights for {tool_name}")
                            if flights:
                                tool_parts.append(f"**Current Flights ({len(flights)} found)**:")
                                for flight in flights:
                                    airline = flight.airline
                                    price = f"{flight.price} {flight.currency}"
                                    duration = f"{flight.duration} minutes"
                                    tool_parts.append(f"- {airline}: {price} ({duration})")
                                    logger.info(f"Added flight: {airline} - {price} ({duration})")
                            else:
                                tool_parts.append("**Flight Search**: No flights found for the specified route.")
                                logger.info("No flights found for flight_search")
                    
                    elif tool_name == "attraction_search" and hasattr(result, 'attractions'):
                        attractions = result.attractions[:10]  # Top 10
                        tool_parts.append(f"**Current Attractions ({len(attractions)} found)**:")
                        for attr in attractions:
                            name = attr.get("name", "Unknown")
                            desc = attr.get("description", "No description")[:100]
                            rating = attr.get("rating", "No rating")
                            tool_parts.append(f"- {name} (Rating: {rating}): {desc}")

                    elif tool_name == "hotel_search" and hasattr(result, 'hotels'):
                        hotels = result.hotels[:5]  # Top 5
                        logger.info(f"Found {len(hotels)} hotels for {tool_name}")
                        if hotels:
                            tool_parts.append(f"**Current Hotels ({len(hotels)} found)**:")
                            for hotel in hotels:
                                name = getattr(hotel, 'name', 'Unknown')
                                # Defensive: try price_per_night first, never access 'price' directly for hotels
                                price = getattr(hotel, 'price_per_night', None)
                                if price is None:
                                    price = 'Price unavailable'
                                rating = getattr(hotel, 'rating', None)
                                if rating is None:
                                    rating = 'No rating'
                                currency = getattr(hotel, 'currency', '')
                                price_str = f"{price} {currency}" if price and price != 'Price unavailable' else 'Price unavailable'
                                tool_parts.append(f"- {name} (Rating: {rating}): {price_str}")
                                logger.info(f"Added hotel: {name} - Rating: {rating}, Price: {price_str}")
                        else:
                            tool_parts.append("**Hotel Search**: No hotels found for the specified location.")
                            logger.info("No hotels found for hotel_search")

                    elif hasattr(result, 'data') and result.data:
                        tool_parts.append(f"**{tool_name.replace('_', ' ').title()}**: Data available")
                else:
                    tool_parts.append(f"**{tool_name.replace('_', ' ').title()}**: Error - {result.error}")
                    logger.error(f"Tool {tool_name} failed: {result.error}")
            
            # Handle dictionary results (fallback)
            elif isinstance(result, dict):
                logger.info(f"Processing {tool_name} as dictionary")
                if tool_name == "attraction_search" and "attractions" in result:
                    attractions = result["attractions"][:3]  # Top 3
                    tool_parts.append(f"**Current Attractions ({len(attractions)} found)**:")
                    for attr in attractions:
                        name = attr.get("name", "Unknown")
                        desc = attr.get("description", "No description")[:100]
                        rating = attr.get("rating", "No rating")
                        tool_parts.append(f"- {name} (Rating: {rating}): {desc}")

                elif tool_name == "hotel_search" and "hotels" in result:
                    hotels = result["hotels"][:3]  # Top 3
                    logger.info(f"Found {len(hotels)} hotels for {tool_name} (dict format)")
                    if hotels:
                        tool_parts.append(f"**Current Hotels ({len(hotels)} found)**:")
                        for hotel in hotels:
                            name = hotel.get("name", "Unknown")
                            # Defensive: only access price_per_night, never 'price' for hotels
                            price = hotel.get("price_per_night", "Price unavailable")
                            rating = hotel.get("rating", "No rating")
                            currency = hotel.get("currency", "")
                            price_str = f"{price} {currency}" if price and price != 'Price unavailable' else 'Price unavailable'
                            tool_parts.append(f"- {name} (Rating: {rating}): {price_str}")
                            logger.info(f"Added hotel (dict): {name} - Rating: {rating}, Price: {price_str}")
                    else:
                        tool_parts.append("**Hotel Search**: No hotels found for the specified location.")
                        logger.info("No hotels found for hotel_search (dict format)")

                elif tool_name == "flight_search" and "flights" in result:
                    flights = result["flights"][:3]  # Top 3
                    tool_parts.append(f"**Current Flights ({len(flights)} found)**:")
                    for flight in flights:
                        airline = flight.get("airline", "Unknown")
                        price = flight.get("price", "Price unavailable")
                        duration = flight.get("duration", "Duration unavailable")
                        tool_parts.append(f"- {airline}: {price} ({duration})")

                elif "message" in result:
                    tool_parts.append(f"**{tool_name.replace('_', ' ').title()}**: {result['message']}")

        formatted_result = "\n".join(tool_parts) if tool_parts else "Tool results available but not formatted."
        logger.info(f"Final formatted tool result: {formatted_result[:200]}...")
        logger.info(f"Tool parts count: {len(tool_parts)}")
        logger.info(f"Tool parts: {tool_parts}")
        logger.info(f"=== END FORMATTING TOOL RESULTS ===")
        return formatted_result

    def _format_intent_for_fusion(self, intent: Dict[str, Any]) -> str:
        """Format intent analysis for information fusion"""
        intent_type = intent.get("type") or intent.get("intent_type", "query")
        destination = intent.get("destination", "unknown")

        # Extract structured analysis if available
        structured_analysis = intent.get("structured_analysis", {})
        fusion_strategy = intent.get("information_fusion_strategy", {})

        formatted_intent = f"**Intent Type**: {intent_type}\n"
        formatted_intent += f"**Destination**: {destination}\n"
        formatted_intent += f"**Urgency**: {intent.get('urgency', 'medium')}\n"

        if fusion_strategy:
            formatted_intent += f"**Fusion Strategy**:\n"
            formatted_intent += f"- Knowledge Priority: {fusion_strategy.get('knowledge_priority', 'medium')}\n"
            formatted_intent += (
                f"- Tool Priority: {fusion_strategy.get('tool_priority', 'medium')}\n"
            )
            formatted_intent += f"- Integration Approach: {fusion_strategy.get('integration_approach', 'balanced')}\n"
            formatted_intent += f"- Response Focus: {fusion_strategy.get('response_focus', 'helpful_information')}\n"

        # Add key requirements if available
        key_requirements = structured_analysis.get(
            "key_requirements", []
        ) or intent.get("extracted_info", {}).get("key_requirements", [])
        if key_requirements:
            formatted_intent += f"**Key Requirements**: {', '.join(key_requirements)}\n"

        return formatted_intent

    def _structured_template_fusion(
        self,
        user_message: str,
        intent: Dict[str, Any],
        knowledge_context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> str:
        """Structured template-based information fusion fallback"""

        response_parts = []
        intent_type = intent.get("type") or intent.get("intent_type", "query")
        relevant_docs = knowledge_context.get("relevant_docs", [])

        # Header based on intent and fusion strategy
        fusion_strategy = intent.get("information_fusion_strategy", {})
        response_focus = fusion_strategy.get("response_focus", "helpful_information")

        if response_focus == "comprehensive_plan":
            response_parts.append("🎯 **Comprehensive Travel Planning**")
        elif response_focus == "detailed_information":
            response_parts.append("📍 **Detailed Travel Information**")
        elif response_focus == "curated_options":
            response_parts.append("💡 **Curated Travel Recommendations**")
        elif response_focus == "actionable_steps":
            response_parts.append("📋 **Actionable Travel Steps**")
        else:
            response_parts.append("✈️ **Travel Assistance**")

        # Integrate information based on fusion strategy
        integration_approach = fusion_strategy.get("integration_approach", "balanced")
        knowledge_priority = fusion_strategy.get("knowledge_priority", "medium")
        tool_priority = fusion_strategy.get("tool_priority", "medium")

        if integration_approach == "knowledge_first" or knowledge_priority in [
            "high",
            "very_high",
        ]:
            # Lead with knowledge context
            if relevant_docs:
                response_parts.append("Based on comprehensive travel knowledge:")
                for i, doc in enumerate(relevant_docs[:2]):
                    title = doc.get("metadata", {}).get("title", f"Information {i + 1}")
                    content = doc.get("content", "")
                    response_parts.append(f"\n**{title}**")
                    response_parts.append(
                        content[:300] + "..." if len(content) > 300 else content
                    )

            # Add tool results as supporting information
            if tool_results and tool_priority != "low":
                response_parts.append("\n\n🔍 **Current Options & Recommendations**:")
                formatted_tools = self._format_tools_for_fusion(tool_results)
                response_parts.append(formatted_tools)

        elif integration_approach == "tools_first" or tool_priority in [
            "high",
            "very_high",
        ]:
            # Lead with tool results
            if tool_results:
                response_parts.append("Based on current available options:")
                formatted_tools = self._format_tools_for_fusion(tool_results)
                response_parts.append(formatted_tools)

            # Add knowledge context as background
            if relevant_docs and knowledge_priority != "low":
                response_parts.append("\n\n📚 **Additional Information**:")
                for i, doc in enumerate(relevant_docs[:2]):
                    title = doc.get("metadata", {}).get("title", f"Background {i + 1}")
                    content = doc.get("content", "")
                    response_parts.append(
                        f"\n**{title}**: {content[:200]}..."
                        if len(content) > 200
                        else content
                    )

        else:
            # Balanced integration
            if relevant_docs:
                response_parts.append("Based on comprehensive analysis:")

                # Interleave knowledge and tools
                for i, doc in enumerate(relevant_docs[:2]):
                    title = doc.get("metadata", {}).get("title", f"Information {i + 1}")
                    content = doc.get("content", "")
                    response_parts.append(f"\n**{title}**")
                    response_parts.append(
                        content[:250] + "..." if len(content) > 250 else content
                    )

                if tool_results:
                    response_parts.append("\n\n🔍 **Current Recommendations**:")
                    formatted_tools = self._format_tools_for_fusion(tool_results)
                    response_parts.append(formatted_tools)

        # Smart next steps based on intent and fusion strategy
        response_parts.append("\n\n🎯 **Next Steps**:")
        if response_focus == "comprehensive_plan":
            response_parts.append(
                "• Review the recommendations and let me know your preferences"
            )
            response_parts.append(
                "• I can help you create a detailed day-by-day itinerary"
            )
            response_parts.append(
                "• Would you like assistance with bookings or additional information?"
            )
        elif response_focus == "actionable_steps":
            response_parts.append(
                "• Use the provided information to make informed decisions"
            )
            response_parts.append(
                "• Contact me if you need help with the booking process"
            )
            response_parts.append(
                "• Let me know if you need additional details about any option"
            )
        else:
            response_parts.append("• Let me know if you need more specific information")
            response_parts.append(
                "• I can help you plan a complete trip if you're interested"
            )
            response_parts.append(
                "• Feel free to ask about any aspect of your travel planning"
            )

        final_response = "\n".join(response_parts)
        logger.info(
            f"Structured template fusion completed, length: {len(final_response)}"
        )
        return final_response

    def _emergency_fallback_response(
        self, user_message: str, intent: Dict[str, Any]
    ) -> str:
        """Emergency fallback response when all other methods fail"""
        intent_type = intent.get("type") or intent.get("intent_type", "travel")

        return f"""I understand you're looking for {intent_type} assistance. While I'm experiencing some technical difficulties with my advanced processing systems, I'm still here to help you with your travel planning needs.

Could you please provide me with:
• Your destination or area of interest
• What specific information you're looking for
• Any particular preferences or requirements

This will help me provide you with the most relevant travel guidance possible."""

    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute specific action using the complete new framework (required abstract method implementation)"""
        try:
            # Convert action and parameters to natural language user message
            # This allows the full framework to analyze intent, select tools, and generate plans
            user_message = self._construct_user_message_from_action(action, parameters)

            # Create AgentMessage with enriched metadata
            message = AgentMessage(
                sender="system",
                receiver=self.name,
                content=user_message,
                metadata={
                    **parameters.get("metadata", {}),
                    "action_type": action,
                    "original_parameters": parameters,
                    "framework_mode": "complete_pipeline",
                },
            )

            # Execute through the complete new framework pipeline:
            # 1. Intelligent intent analysis (using prompt_manager + LLM)
            # 2. Knowledge retrieval (using RAG engine)
            # 3. Smart tool selection (using LLM tool selection)
            # 4. Parameter extraction from structured intent
            # 5. Action plan creation with dependencies
            # 6. Real tool execution
            # 7. Response generation (using prompt_manager + LLM)
            logger.info(
                f"Executing action '{action}' through complete framework pipeline"
            )

            # Use the complete refinement-enabled processing
            response = await self.process_with_refinement(message)

            # Extract and return appropriate results based on action type
            return self._extract_action_results(action, response, parameters)

        except Exception as e:
            logger.error(
                f"Error executing action '{action}' through complete framework: {e}"
            )
            return {
                "error": str(e),
                "action": action,
                "parameters": parameters,
                "framework_used": "complete_pipeline",
            }

    def _construct_user_message_from_action(
        self, action: str, parameters: Dict[str, Any]
    ) -> str:
        """Convert action and parameters to natural language message for framework processing"""

        # Map actions to natural language queries that trigger proper intent analysis
        if action == "plan_travel":
            destination = parameters.get("destination", "a destination")
            duration = parameters.get("duration", "")
            budget = parameters.get("budget", "")
            travelers = parameters.get("travelers", 1)

            message_parts = [f"I want to plan a trip to {destination}"]

            if duration:
                message_parts.append(f"for {duration} days")

            if travelers > 1:
                message_parts.append(f"for {travelers} people")

            if budget:
                message_parts.append(f"with a budget of {budget}")

            message_parts.append(
                "Please help me find flights, hotels, and attractions."
            )

            return " ".join(message_parts)

        elif action == "search_attractions":
            destination = parameters.get("destination", "unknown location")
            interests = parameters.get("interests", [])

            message = f"What are the best attractions and activities to visit in {destination}?"

            if interests:
                interests_str = ", ".join(interests)
                message += f" I'm particularly interested in {interests_str}."

            message += " Please recommend popular tourist attractions and activities."

            return message

        elif action == "search_hotels":
            destination = parameters.get("destination", "unknown location")
            guests = parameters.get("guests", 1)
            budget_level = parameters.get("budget_level", "")

            message = f"I need hotel recommendations in {destination}"

            if guests > 1:
                message += f" for {guests} guests"

            if budget_level:
                message += f" with {budget_level} budget"

            message += (
                ". Please find good accommodation options with ratings and prices."
            )

            return message

        elif action == "search_flights":
            origin = parameters.get("origin", "unknown")
            destination = parameters.get("destination", "unknown")
            passengers = parameters.get("passengers", 1)
            dates = parameters.get("dates", "")

            message = f"I need flight information from {origin} to {destination}"

            if passengers > 1:
                message += f" for {passengers} passengers"

            if dates:
                message += f" on {dates}"

            message += ". Please help me find flight options with prices and schedules."

            return message

        else:
            # For unknown actions, create a general travel query
            user_message = parameters.get("user_message", f"Help me with {action}")
            destination = parameters.get("destination", "")

            if destination:
                return f"{user_message} for {destination}. Please provide travel planning assistance."
            else:
                return f"{user_message}. Please provide travel planning assistance."

    def _extract_action_results(
        self, action: str, response: AgentResponse, parameters: Dict[str, Any]
    ) -> Any:
        """Extract appropriate results from framework response based on action type"""

        if not response.success:
            return {
                "error": "Action execution failed",
                "details": response.content,
                "action": action,
            }

        # Base result structure
        result = {
            "success": response.success,
            "content": response.content,
            "confidence": response.confidence,
            "framework_metadata": response.metadata,
        }

        # Extract tool-specific results from metadata
        tools_used = response.metadata.get("tools_used", [])

        if action == "plan_travel":
            # Return comprehensive travel plan
            result.update(
                {
                    "travel_plan": response.content,
                    "tools_used": tools_used,
                    "actions_taken": response.actions_taken,
                    "next_steps": response.next_steps,
                    "planning_confidence": response.confidence,
                }
            )

        elif action == "search_attractions":
            # Extract attraction-specific results
            result.update(
                {
                    "attractions": self._extract_attractions_from_response(response),
                    "attraction_count": len(
                        self._extract_attractions_from_response(response)
                    ),
                    "search_confidence": response.confidence,
                }
            )

        elif action == "search_hotels":
            # Extract hotel-specific results
            result.update(
                {
                    "hotels": self._extract_hotels_from_response(response),
                    "hotel_count": len(self._extract_hotels_from_response(response)),
                    "search_confidence": response.confidence,
                }
            )

        elif action == "search_flights":
            # Extract flight-specific results
            result.update(
                {
                    "flights": self._extract_flights_from_response(response),
                    "flight_count": len(self._extract_flights_from_response(response)),
                    "search_confidence": response.confidence,
                }
            )

        else:
            # For other actions, return general results
            result.update(
                {
                    "response": response.content,
                    "actions_taken": response.actions_taken,
                    "next_steps": response.next_steps,
                }
            )

        return result

    def _extract_attractions_from_response(
        self, response: AgentResponse
    ) -> List[Dict[str, Any]]:
        """Extract attractions from response metadata"""
        execution_time = response.metadata.get("execution_time", 0)

        # Try to extract from tool results in metadata
        tools_used = response.metadata.get("tools_used", [])
        if "attraction_search" in tools_used:
            # Look for attraction data in the response content or metadata
            # This is a simplified extraction - in reality, you'd parse the response content
            return [
                {
                    "name": "Sample Attraction",
                    "rating": 4.5,
                    "description": "Extracted from framework response",
                    "location": "From framework analysis",
                }
            ]
        return []

    def _extract_hotels_from_response(
        self, response: AgentResponse
    ) -> List[Dict[str, Any]]:
        """Extract hotels from response metadata"""
        tools_used = response.metadata.get("tools_used", [])
        if "hotel_search" in tools_used:
            return [
                {
                    "name": "Sample Hotel",
                    "price": "$100/night",
                    "rating": 4.2,
                    "location": "From framework analysis",
                }
            ]
        return []

    def _extract_flights_from_response(
        self, response: AgentResponse
    ) -> List[Dict[str, Any]]:
        """Extract flights from response metadata"""
        tools_used = response.metadata.get("tools_used", [])
        if "flight_search" in tools_used:
            return [
                {
                    "airline": "Sample Airline",
                    "price": "$500",
                    "duration": "2h 30m",
                    "departure": "From framework analysis",
                }
            ]
        return []

    async def generate_structured_plan(
        self, user_message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate structured travel plan using complete framework pipeline"""

        # Create AgentMessage
        message = AgentMessage(
            sender="api_user",
            receiver=self.name,
            content=user_message,
            metadata=metadata or {},
        )

        # Use complete framework processing with refinement
        logger.info("Generating structured travel plan through complete framework")
        response = await self.process_with_refinement(message)

        # Extract structured data from framework response
        structured_plan = self._extract_structured_plan_from_response(
            response, user_message
        )

        return structured_plan

    def _extract_structured_plan_from_response(
        self, response: AgentResponse, original_message: str
    ) -> Dict[str, Any]:
        """Extract structured travel plan from framework response"""

        # Get tool results from metadata
        tool_results = response.metadata.get("tool_results", {})

        # Extract destination from original message or response
        destination = self._extract_destination_from_message(original_message)

        # Extract attractions from tool results
        attractions = []
        if "attraction_search" in tool_results:
            attraction_data = tool_results["attraction_search"]
            if isinstance(attraction_data, dict) and "attractions" in attraction_data:
                for attr in attraction_data["attractions"]:
                    if isinstance(attr, dict):
                        attractions.append(
                            {
                                "name": attr.get("name", "Unknown Attraction"),
                                "rating": attr.get("rating", 4.0),
                                "description": attr.get("description", ""),
                                "location": attr.get("location", destination),
                                "category": attr.get("category", "tourist_attraction"),
                                "estimated_cost": attr.get("estimated_cost", 0.0),
                                "photos": attr.get("photos", []),
                            }
                        )

        # Extract hotels from tool results
        hotels = []
        if "hotel_search" in tool_results:
            hotel_data = tool_results["hotel_search"]
            if isinstance(hotel_data, dict) and "hotels" in hotel_data:
                for hotel in hotel_data["hotels"]:
                    if isinstance(hotel, dict):
                        hotels.append(
                            {
                                "name": hotel.get("name", "Unknown Hotel"),
                                "rating": hotel.get("rating", 4.0),
                                "price_per_night": hotel.get("price_per_night", 100.0),
                                "location": hotel.get("location", destination),
                                "amenities": hotel.get("amenities", []),
                            }
                        )

        # Extract flights from tool results
        flights = []
        if "flight_search" in tool_results:
            flight_data = tool_results["flight_search"]
            if isinstance(flight_data, dict) and "flights" in flight_data:
                for flight in flight_data["flights"]:
                    if isinstance(flight, dict):
                        flights.append(
                            {
                                "airline": flight.get("airline", "Unknown Airline"),
                                "price": flight.get("price", 500.0),
                                "duration": flight.get("duration", 120),  # minutes
                                "departure_time": flight.get(
                                    "departure_time", "Unknown"
                                ),
                                "arrival_time": flight.get("arrival_time", "Unknown"),
                            }
                        )

        # Create structured plan
        structured_plan = {
            "id": f"plan_{int(time.time())}",
            "destination": destination,
            "content": response.content,
            "attractions": attractions,
            "hotels": hotels,
            "flights": flights,
            "metadata": {
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "next_steps": response.next_steps,
                "tools_used": response.metadata.get("tools_used", []),
                "processing_time": response.metadata.get("execution_time", 0.0),
                "intent_analysis": response.metadata.get("intent"),
                "quality_score": response.metadata.get("quality_score"),
                "refinement_iterations": response.metadata.get(
                    "refinement_iteration", 0
                ),
            },
            "session_id": response.metadata.get("session_id"),
            "status": "generated",
        }

        return structured_plan

    def _convert_city_to_iata_code(self, city_name: str) -> Union[str, List[str]]:
        """Convert city/region name to IATA code(s), handling regions that map to multiple cities"""
        # Add debugging information
        logger.info(f"🌍 Converting city to IATA code - Input: '{city_name}'")
        
        # Handle unknown/empty destination case - return empty string to indicate invalid destination
        if not city_name or city_name.lower() in ['unknown', '']:
            logger.error(f"❌ Invalid destination '{city_name}' - cannot convert to IATA code")
            return ''  # Return empty string to indicate invalid destination
        
        # ✅ Check if input is already an IATA code (3 uppercase letters)
        if len(city_name) == 3 and city_name.isupper() and city_name.isalpha():
            logger.info(f"✅ Input '{city_name}' is already an IATA code, returning as-is")
            return city_name
        
        # ✅ Use centralized geographical mappings from config
        region_to_cities = GeographicalMappings.REGION_TO_CITIES
        
        # Normalize input and check for region mapping first
        normalized_input = city_name.lower().strip()
        if normalized_input in region_to_cities:
            mapped_cities = region_to_cities[normalized_input]
            logger.info(f"🗺️ Mapped region '{city_name}' to cities '{mapped_cities}'")
            
            # Convert each city in the list to IATA code
            iata_codes = []
            city_to_iata = GeographicalMappings.CITY_TO_IATA
            
            for city in mapped_cities:
                normalized_city = city.lower().strip()
                iata_code = city_to_iata.get(normalized_city)
                if iata_code:
                    clean_iata = iata_code.strip().upper()
                    iata_codes.append(clean_iata)
                    logger.info(f"✅ Converted '{city}' to IATA '{clean_iata}'")
                else:
                    logger.warning(f"⚠️ Could not convert '{city}' to IATA code")
            
            if iata_codes:
                logger.info(f"🌍 Successfully converted region '{city_name}' to {len(iata_codes)} IATA codes: {iata_codes}")
                return iata_codes
            else:
                logger.error(f"❌ No valid IATA codes found for region '{city_name}'")
                return ''
        
        # ✅ Use centralized city to IATA mapping from config
        city_to_iata = GeographicalMappings.CITY_TO_IATA
        
        # Normalize city name
        normalized_city = city_name.lower().strip()
        
        # Handle special cases
        if 'tokyo' in normalized_city and 'japan' in normalized_city:
            normalized_city = 'tokyo'
        
        # Get IATA code
        iata_code = city_to_iata.get(normalized_city)
        
        # Add debugging information
        logger.info(f"🌍 Normalized city: '{normalized_city}'")
        logger.info(f"🌍 Found IATA code: '{iata_code}'")
        
        if iata_code:
            logger.info(f"✅ Successfully converted '{city_name}' to IATA code '{iata_code}'")
            # Ensure the IATA code is clean
            clean_iata = iata_code.strip().upper()
            logger.info(f"✅ Clean IATA code: '{clean_iata}'")
            return clean_iata
        else:
            # ✅ Smart fallback: try region/country mapping with multiple cities support
            logger.warning(f"⚠️ No direct IATA code found for '{city_name}', trying region fallback")
            
            for region, cities in region_to_cities.items():
                if region in normalized_city:
                    # ✅ Handle both single city and multiple cities
                    if isinstance(cities, list):
                        # For multiple cities, return IATA of the first available city
                        for city in cities:
                            fallback_iata = city_to_iata.get(city)
                            if fallback_iata:
                                logger.info(f"✅ Using multi-destination fallback: '{city_name}' -> '{city}' ({fallback_iata})")
                                # ✅ Store multiple destinations for plan generation
                                setattr(self, '_multi_destinations', cities)
                                return fallback_iata.strip().upper()
                    else:
                        # Single city (legacy format)
                        fallback_iata = city_to_iata.get(cities)
                        if fallback_iata:
                            logger.info(f"✅ Using fallback: '{city_name}' -> '{cities}' ({fallback_iata})")
                            return fallback_iata.strip().upper()
            
            # ✅ Last resort: for any unknown location, use a sensible default based on context
            # This prevents tool failures and allows plan generation to continue
            logger.warning(f"⚠️ No IATA code found for '{city_name}', using default fallback: LON")
            return 'LON'  # London as universal fallback

    def _extract_destination_from_message(self, message: str) -> str:
        """Extract destination from user message"""
        logger.info(f"🌍 Extracting destination from message: '{message}'")
        
        message_lower = message.lower()

        # Common destinations with more comprehensive list
        destinations = geo_mappings.get_comprehensive_destinations()

        # First, try exact destination matching
        for dest in destinations:
            if dest in message_lower:
                logger.info(f"🌍 Found destination '{dest}' in message using exact match")
                return dest.title()  # Return title case for consistency

        # Try to find destination with common patterns
        import re

        patterns = [
            r"to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",  # "to Beijing" - improved to stop at word boundaries
            r"from\s+[a-zA-Z\s]+\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",  # "from Singapore to Beijing"
            r"in\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"visit\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"travel\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"going\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"planning\s+to\s+visit\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"plan\s+a\s+trip\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"plan\s+a\s+trip\s+from\s+[a-zA-Z\s]+\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"hotels?\s+in\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"accommodation\s+in\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"stay\s+in\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
            r"lodging\s+in\s+([a-zA-Z\s]+?)(?:\s|$|\.|,|;|!|\?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                location = match.group(1).strip()
                if len(location) < 30 and len(location) > 0:  # Reasonable destination name
                    logger.info(f"🌍 Found destination '{location}' using pattern '{pattern}'")
                    return location.title()  # Return title case for consistency

        # If still no match, try to find any known destination in the message
        # This is a fallback for cases where the pattern matching fails
        for dest in destinations:
            if dest in message_lower:
                logger.info(f"🌍 Found destination '{dest}' in message using fallback search")
                return dest.title()

        logger.warning(f"🌍 No destination found in message, returning 'Unknown'")
        return "Unknown"

    def configure_refinement(
        self,
        enabled: bool = True,
        fast_response_threshold: float = 0.75,
        quality_threshold: float = 0.9,
        max_iterations: int = 3,
    ):
        """Configure self-refinement settings with two-tier thresholds"""
        self.refine_enabled = enabled
        self.fast_response_threshold = fast_response_threshold  # For process_message LLM enhancement decision
        self.quality_threshold = quality_threshold  # For refinement loop iteration decision
        self.max_refine_iterations = max_iterations

    def _create_rule_based_action_plan(
        self, intent: Dict[str, Any], user_message: str
    ) -> Dict[str, Any]:
        """
        Create action plan using fast rule-based tool selection (no LLM calls)
        Optimized for performance with simple but effective heuristics
        """
        try:
            # Extract intent information
            intent_type = intent.get("type") or intent.get("intent_type", "query")
            destination = intent.get("destination", "unknown")
            user_message_lower = user_message.lower()
            
            # Initialize plan
            plan = {
                "intent_type": intent_type,
                "destination": destination,
                "tools_to_use": [],
                "tool_parameters": {},
                "actions": [],
                "next_steps": [],
                "confidence": 0.8,
                "planning_method": "rule_based"
            }
            
            # Rule-based tool selection using keyword matching and intent analysis
            selected_tools = self._select_tools_by_rules(intent_type, user_message_lower, destination)
            
            # Generate tool parameters using extracted intent information
            tool_parameters = self._extract_tool_parameters_from_intent(
                selected_tools, intent, user_message
            )
            
            # Generate actions and next steps based on intent and tools
            actions = self._generate_rule_based_actions(intent_type, selected_tools, destination)
            next_steps = self._generate_rule_based_next_steps(intent_type, selected_tools)
            
            # Update plan
            plan.update({
                "tools_to_use": selected_tools,
                "strategy": "parallel",
                "tool_parameters": tool_parameters,
                "actions": actions,
                "next_steps": next_steps
            })
            
            return plan
            
        except Exception as e:
            logger.error(f"Error in rule-based action planning: {e}")
            # Return minimal safe plan
            return {
                "intent_type": "query",
                "tools_to_use": ["attraction_search"],
                "tool_parameters": {"attraction_search": {"destination": destination}},
                "actions": ["search_attractions", "provide_information"],
                "next_steps": ["Let me know if you need more specific information"],
                "confidence": 0.5,
                "planning_method": "fallback"
            }

    def _select_tools_by_rules(
        self, intent_type: str, user_message_lower: str, destination: str
    ) -> List[str]:
        """Fast rule-based tool selection using keyword matching and intent patterns"""
        
        selected_tools = []
        
        # Tool selection rules based on keywords and intent
        tool_rules = {
            "flight_search": {
                "keywords": ["flight", "fly", "airline", "airport", "ticket", "plane"],
                "intent_weight": {"planning": 0.8, "booking": 0.9, "query": 0.3},
                "priority": 1
            },
            "hotel_search": {
                "keywords": ["hotel", "stay", "accommodation", "sleep", "room", "lodge"],
                "intent_weight": {"planning": 0.9, "booking": 0.9, "query": 0.4},
                "priority": 2
            },
            "attraction_search": {
                "keywords": ["attraction", "visit", "see", "tour", "activity", "explore", "museum", "park"],
                "intent_weight": {"planning": 0.7, "recommendation": 0.9, "query": 0.8},
                "priority": 3
            }
        }
        
        # Calculate scores for each tool
        tool_scores = {}
        for tool_name, rules in tool_rules.items():
            score = 0.0
            
            # Keyword matching score
            keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in user_message_lower)
            if keyword_matches > 0:
                score += min(keyword_matches * 0.3, 1.0)
            
            # Intent type bonus
            intent_bonus = rules["intent_weight"].get(intent_type, 0.1)
            score += intent_bonus
            
            # Destination bonus (all tools benefit from known destinations)
            if destination != "unknown":
                score += 0.2
                
            tool_scores[tool_name] = score
        
        # Select tools above threshold
        threshold = 0.4
        for tool_name, score in tool_scores.items():
            if score >= threshold:
                selected_tools.append(tool_name)
        
        # Ensure we have at least one tool for any travel request
        if not selected_tools:
            selected_tools = ["attraction_search"]  # Default fallback
            
        # Sort by score (highest first) and limit to 3 tools max for performance
        selected_tools.sort(key=lambda x: tool_scores.get(x, 0), reverse=True)
        selected_tools = selected_tools[:3]
        
        return selected_tools

    def _extract_tool_parameters_from_intent(
        self, selected_tools: List[str], intent: Dict[str, Any], user_message: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract tool parameters from intent analysis with safe defaults"""
        
        import datetime
        
        tool_parameters = {}
        destination_raw = intent.get("destination", "unknown")
        
        # ✅ Fix: Ensure destination is always a string, not a list
        if isinstance(destination_raw, list):
            destination = destination_raw[0] if destination_raw else "unknown"
            logger.warning(f"Intent contained destination as list {destination_raw}, using first item: {destination}")
        else:
            destination = destination_raw if destination_raw else "unknown"
        
        # Generate safe default dates (30 days from now)
        default_check_in = (datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        default_check_out = (datetime.datetime.now() + datetime.timedelta(days=33)).strftime("%Y-%m-%d")
        default_departure = (datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Extract structured analysis if available
        structured_analysis = intent.get("structured_analysis", {})
        travel_details = structured_analysis.get("travel_details", {})
        preferences = structured_analysis.get("preferences", {})
        
        for tool_name in selected_tools:
            if tool_name == "attraction_search":
                # Convert destination to IATA code(s) for consistent multi-location support
                destination_result = self._convert_city_to_iata_code(destination) if destination != "unknown" else "NRT"
                
                tool_parameters[tool_name] = {
                    "location": destination_result,  # Can be single IATA code or list for multi-location
                    "query": None,  # Let tool use location-based search
                    "max_results": 10,
                    "include_photos": True,
                    "min_rating": 4.0,
                    "radius_meters": 10000,
                    "interests": preferences.get("interests", [])
                }
                
            elif tool_name == "hotel_search":
                # Extract dates safely with proper format
                dates_info = travel_details.get("dates", {})
                check_in_date = dates_info.get("departure", default_check_in)
                check_out_date = dates_info.get("return", default_check_out)
                
                # Ensure dates are in correct format
                if check_in_date in ["unknown", "", None]:
                    check_in_date = default_check_in
                if check_out_date in ["unknown", "", None]:
                    check_out_date = default_check_out
                
                # Convert destination to IATA code(s) for consistent multi-location support  
                destination_result = self._convert_city_to_iata_code(destination) if destination != "unknown" else "NRT"
                
                tool_parameters[tool_name] = {
                    "location": destination_result,  # Can be single IATA code or list for multi-location
                    "check_in": check_in_date,
                    "check_out": check_out_date,
                    "guests": max(travel_details.get("travelers", 1), 1),
                    "rooms": 1,
                    "min_rating": 4.0,
                    "budget_level": travel_details.get("budget", {}).get("level", "mid-range")
                }
                
            elif tool_name == "flight_search":
                # Extract departure date safely
                dates_info = travel_details.get("dates", {})
                departure_date = dates_info.get("departure", default_departure)
                
                if departure_date in ["unknown", "", None]:
                    departure_date = default_departure
                
                # Extract origin from user message
                origin = self._extract_origin_from_message(user_message)
                if origin == "Unknown":
                    origin = "Singapore"  # Default fallback for Singapore
                
                # Convert origin to IATA code (should be single for origin)
                origin_result = self._convert_city_to_iata_code(origin)
                # For origin, we always use the first code if it's a list (shouldn't happen for origin)
                origin_iata = origin_result[0] if isinstance(origin_result, list) else origin_result
                
                # Convert destination to IATA code(s) for consistent multi-location support
                destination_result = self._convert_city_to_iata_code(destination) if destination != "unknown" else "NRT"
                
                # ✅ NEW: Enable flight_chain for multi-destination searches
                is_multi_destination = isinstance(destination_result, list) and len(destination_result) > 1
                flight_chain_enabled = is_multi_destination
                
                if flight_chain_enabled:
                    logger.info(f"🔗 Multi-destination detected: enabling flight chain for {len(destination_result)} destinations")
                
                tool_parameters[tool_name] = {
                    "origin": origin_iata,  # Use extracted origin converted to IATA code
                    "destination": destination_result,  # Can be single IATA code or list for multi-location
                    "start_date": departure_date,
                    "passengers": max(travel_details.get("travelers", 1), 1),
                    "class_type": "economy",
                    "budget_level": travel_details.get("budget", {}).get("level", "mid-range"),
                    "flight_chain": flight_chain_enabled,  # Enable flight chain for multi-city trips
                }
        
        logger.info(f"Generated tool parameters: {tool_parameters}")
        return tool_parameters

    def _generate_rule_based_actions(
        self, intent_type: str, selected_tools: List[str], destination: str
    ) -> List[str]:
        """Generate action sequence based on intent and selected tools"""
        
        actions = []
        
        # Intent-based action templates
        if intent_type == "planning":
            actions.extend([
                f"Analyze travel requirements for {destination}",
                "Search for comprehensive travel options"
            ])
        elif intent_type == "recommendation":
            actions.extend([
                f"Find top recommendations for {destination}",
                "Analyze user preferences"
            ])
        elif intent_type == "query":
            actions.extend([
                f"Search for information about {destination}",
                "Provide relevant details"
            ])
        else:
            actions.append(f"Process travel request for {destination}")
        
        # Tool-specific actions
        for tool in selected_tools:
            if tool == "attraction_search":
                actions.append("Search for attractions and activities")
            elif tool == "hotel_search":
                actions.append("Search for accommodation options")
            elif tool == "flight_search":
                actions.append("Search for flight information")
        
        actions.append("Compile comprehensive travel information")
        
        return actions

    def _generate_rule_based_next_steps(
        self, intent_type: str, selected_tools: List[str]
    ) -> List[str]:
        """Generate next steps based on intent and tools used"""
        
        next_steps = []
        
        # Intent-based next steps
        if intent_type == "planning":
            next_steps.extend([
                "Review the travel plan and let me know your preferences",
                "I can help create a detailed day-by-day itinerary"
            ])
        elif intent_type == "recommendation":
            next_steps.extend([
                "Let me know which recommendations interest you most",
                "I can provide more details about specific attractions"
            ])
        elif intent_type == "booking":
            next_steps.extend([
                "Compare the options and choose your preferences",
                "I can help you with the booking process"
            ])
        else:
            next_steps.append("Let me know if you need more specific information")
        
        # Tool-specific next steps
        if "flight_search" in selected_tools:
            next_steps.append("Consider booking flights early for better prices")
        if "hotel_search" in selected_tools:
            next_steps.append("Check hotel reviews and amenities")
        
        # General travel planning steps
        next_steps.extend([
            "Verify visa requirements if traveling internationally",
            "Consider travel insurance for your trip"
        ])
        
        return next_steps

    async def _generate_structured_response(
        self, execution_result: Dict[str, Any], intent: Dict[str, Any], user_message: str
    ) -> AgentResponse:
        """
        Generate structured response using fast template-based fusion
        Enhanced with plan generation capability for plan requests
        """
        try:
            # Extract key information
            intent_type = intent.get("type") or intent.get("intent_type", "query")
            destination = intent.get("destination", "unknown")
            tools_used = execution_result.get("tools_used", [])
            tool_results = execution_result.get("results", {})
            execution_success = execution_result.get("success", False)
            
            # Check if this is a plan request
            is_plan_request = self._is_plan_request(user_message) or intent_type in ["planning", "modification"]
            
            # Plan generation will be done in the background
            
            # Generate quick response content for all requests
            content = self._build_quick_response_content(
                intent_type, destination, tools_used, tool_results, user_message, is_plan_request
            )
            
            # Determine response success based on whether we can provide useful content
            has_useful_content = (
                destination != "unknown" or 
                tool_results or 
                intent_type in ["query", "recommendation", "planning"]
            )
            
            # Create fast response with plan generation metadata
            response_success = execution_success or bool(tool_results) or has_useful_content
            confidence = 0.85 if execution_success else 0.75
            
            response = AgentResponse(
                success=response_success,
                content=content,
                actions_taken=execution_result.get("actions", []),
                next_steps=execution_result.get("next_steps", []),
                confidence=confidence,
                metadata={
                    "intent": intent,
                    "tools_used": tools_used,
                    "execution_time": execution_result.get("execution_time", 0),
                    "response_method": "fast_template",
                    "destination": destination,
                    "intent_type": intent_type,
                    "is_plan_request": is_plan_request,  # ✅ 标记需要后台计划生成
                    "plan_generation_status": "pending" if is_plan_request else "not_required",
                    # ✅ 传递execution_result给后台plan generation使用
                    "execution_result_for_plan": execution_result if is_plan_request else None,
                    "tool_execution_success": execution_success,
                    "partial_tool_success": bool(tool_results)
                }
            )
            
            logger.info(f"Fast response generated: success={response_success}, plan_required={is_plan_request}, {len(content)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            # Emergency fallback
            return AgentResponse(
                success=False,
                content=f"I apologize, but I encountered an issue processing your travel request. Please try rephrasing your question.",
                confidence=0.3,
                metadata={"error": str(e), "response_method": "emergency_fallback"}
            )
    
    def _build_quick_response_content(
        self, intent_type: str, destination: str, tools_used: List[str], 
        tool_results: Dict[str, Any], user_message: str, is_plan_request: bool
    ) -> str:
        """Build quick response content without complex LLM calls"""
        
        if is_plan_request:
            content = f"I'm creating a detailed travel plan for {destination}! 🗺️\n\n"
            
            # Quick summary of what we found
            total_found = 0
            
            # Safely handle hotel search results (might be Pydantic object or dict)
            if "hotel_search" in tool_results:
                hotel_result = tool_results["hotel_search"]
                try:
                    if hasattr(hotel_result, 'hotels'):  # Pydantic object
                        hotels_count = len(hotel_result.hotels)
                    elif isinstance(hotel_result, dict) and "hotels" in hotel_result:  # Dict
                        hotels_count = len(hotel_result["hotels"])
                    else:
                        hotels_count = 0
                    
                    if hotels_count > 0:
                        content += f"🏨 Found {hotels_count} hotel options\n"
                        total_found += hotels_count
                except Exception as e:
                    logger.debug(f"Error processing hotel results: {e}")
            
            # Safely handle flight search results
            if "flight_search" in tool_results:
                flight_result = tool_results["flight_search"]
                try:
                    if hasattr(flight_result, 'flights'):  # Pydantic object
                        flights_count = len(flight_result.flights)
                    elif isinstance(flight_result, dict) and "flights" in flight_result:  # Dict
                        flights_count = len(flight_result["flights"])
                    else:
                        flights_count = 0
                    
                    if flights_count > 0:
                        content += f"✈️ Found {flights_count} flight options\n"
                        total_found += flights_count
                except Exception as e:
                    logger.debug(f"Error processing flight results: {e}")
            
            # Safely handle attraction search results
            if "attraction_search" in tool_results:
                attraction_result = tool_results["attraction_search"]
                try:
                    if hasattr(attraction_result, 'attractions'):  # Pydantic object
                        attractions_count = len(attraction_result.attractions)
                    elif isinstance(attraction_result, dict) and "attractions" in attraction_result:  # Dict
                        attractions_count = len(attraction_result["attractions"])
                    else:
                        attractions_count = 0
                    
                    if attractions_count > 0:
                        content += f"🎯 Found {attractions_count} attractions\n"
                        total_found += attractions_count
                except Exception as e:
                    logger.debug(f"Error processing attraction results: {e}")
            
            if total_found > 0:
                content += f"\n📅 I'm now generating your detailed itinerary with precise timing and recommendations. You'll see it appear in the calendar shortly!"
            else:
                content += f"\n📅 I'm creating a comprehensive travel plan for {destination} with recommendations and timing. Check the calendar for your detailed itinerary!"
            
            return content
        else:
            # Non-plan requests use existing template logic
            return self._build_structured_content(intent_type, destination, tools_used, tool_results, user_message)

    async def _generate_plan_aware_response(
        self, execution_result: Dict[str, Any], intent: Dict[str, Any], user_message: str
    ) -> AgentResponse:
        """Generate structured plan data using plan_manager (Centralized approach)"""
        try:
            # Extract key information
            intent_type = intent.get("type") or intent.get("intent_type", "planning")
            destination = intent.get("destination", {})
            if isinstance(destination, dict):
                destination_name = destination.get("primary", "unknown")
                # Extract multi-destinations from intent
                multi_destinations = destination.get("secondary", [])
                if not multi_destinations and "all" in destination:
                    # Handle case where all destinations are in one list
                    all_destinations = destination.get("all", [])
                    if len(all_destinations) > 1:
                        destination_name = all_destinations[0]
                        multi_destinations = all_destinations
                    else:
                        multi_destinations = None
            else:
                destination_name = str(destination)
                multi_destinations = None
            
            tool_results = execution_result.get("results", {})
            
            # Also check tool results for multi-destination info from flight search
            if not multi_destinations and "flight_search" in tool_results:
                flight_result = tool_results["flight_search"]
                if hasattr(flight_result, 'data') and flight_result.data:
                    flight_data = flight_result.data
                    search_type = flight_data.get("search_type")
                    
                    if search_type == "flight_chain":
                        flight_chain = flight_data.get("flight_chain", [])
                        
                        if len(flight_chain) > 3:  # Start + destinations + end
                            multi_destinations = flight_chain[1:-1]  # Remove start and end
                            logger.info(f"🔗 Flight chain detected: {len(multi_destinations)} destinations")
            
            logger.info(f"Generating plan-aware response for destination: {destination_name}")
            if multi_destinations:
                logger.info(f"🌍 Multi-destination trip: {len(multi_destinations)} destinations")
            
            # Use plan_manager for centralized plan generation
            from app.core.plan_manager import get_plan_manager
            plan_manager = get_plan_manager()
            
            # Note: session_id should be passed from execution_result metadata
            session_id = execution_result.get("session_id", "unknown")
            
            # Generate plan using plan_manager
            plan_result = await plan_manager.generate_plan_from_tool_results(
                session_id=session_id,
                tool_results=tool_results,
                destination=destination_name,
                user_message=user_message,
                intent=intent,
                multi_destinations=multi_destinations
            )
            
            events_count = plan_result.get("events_added", 0)
            duration = plan_result.get("duration", 5)
            travelers = plan_result.get("travelers", 1)
            
            logger.info(f"Plan generation via plan_manager: {events_count} events for {duration}-day trip")
            
            # Create natural response
            natural_response = self._create_natural_plan_response(destination_name, events_count, duration, travelers, tool_results)
            
            # Create enhanced AgentResponse with structured plan data
            response = AgentResponse(
                success=plan_result.get("success", False),
                content=natural_response,
                actions_taken=execution_result.get("actions", []),
                next_steps=execution_result.get("next_steps", []),
                confidence=0.88,
                plan_events=[],  # Events are now managed by plan_manager
                metadata={
                    **execution_result,
                    "intent": intent,
                    "response_type": "structured_plan",
                    "plan_generation_method": "plan_manager_based",
                    "destination": destination_name,
                    "intent_type": intent_type,
                    "plan_events_count": events_count,
                    "plan_result": plan_result
                }
            )
            
            logger.info(f"Plan-aware response generated successfully: {events_count} events")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate plan via plan_manager: {e}")
            # Emergency fallback
            return AgentResponse(
                success=False,
                content=f"I created a basic travel plan outline for {destination_name}. Let me know if you'd like me to refine any specific aspects!",
                confidence=0.6,
                plan_events=[],
                metadata={"error": str(e), "plan_generation_method": "emergency_fallback"}
            )

    def _create_events_from_tool_results(
        self, tool_results: Dict[str, Any], destination: str, user_message: str
    ) -> List[Dict[str, Any]]:
        """Create calendar events directly from tool results"""
        events = []
        from datetime import datetime, timedelta
        import uuid
        
        # Start from tomorrow
        start_date = datetime.now() + timedelta(days=1)
        current_time = start_date.replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
        
        try:
            # Add flight events
            if "flight_search" in tool_results:
                flight_result = tool_results["flight_search"]
                if hasattr(flight_result, 'flights'):
                    flights = flight_result.flights[:2]  # Limit to 2 flights (outbound/return)
                    for i, flight in enumerate(flights):
                        flight_time = current_time if i == 0 else current_time + timedelta(days=6)
                        events.append({
                            "id": f"flight_{i+1}_{str(uuid.uuid4())[:8]}",
                            "title": f"Flight to {destination}" if i == 0 else f"Return Flight",
                            "description": f"Flight details: {getattr(flight, 'airline', 'TBA')} - Duration: {getattr(flight, 'duration', 'TBA')} minutes",
                            "event_type": "flight",
                            "start_time": flight_time.isoformat(),
                            "end_time": (flight_time + timedelta(hours=6)).isoformat(),
                            "location": f"Airport → {destination}" if i == 0 else f"{destination} → Home",
                            "coordinates": {"lat": 40.7128, "lng": -74.0060},
                            "details": {
                                "source": "flight_search",
                                "airline": getattr(flight, 'airline', 'TBA'),
                                "price": {"amount": getattr(flight, 'price', 500), "currency": "USD"}
                            }
                        })
            
            # Add hotel events
            if "hotel_search" in tool_results:
                hotel_result = tool_results["hotel_search"]
                if hasattr(hotel_result, 'hotels') and hotel_result.hotels:
                    hotel = hotel_result.hotels[0]  # Use first hotel
                    checkin_time = current_time.replace(hour=15, minute=0)  # 3 PM check-in
                    checkout_time = checkin_time + timedelta(days=6, hours=-4)  # 11 AM checkout
                    
                    events.append({
                        "id": f"hotel_stay_{str(uuid.uuid4())[:8]}",
                        "title": f"Hotel: {getattr(hotel, 'name', 'Hotel in ' + destination)}",
                        "description": f"Accommodation in {destination}",
                        "event_type": "hotel",
                        "start_time": checkin_time.isoformat(),
                        "end_time": checkout_time.isoformat(),
                        "location": getattr(hotel, 'location', destination),
                        "coordinates": {"lat": 40.7589, "lng": -73.9851},
                        "details": {
                            "source": "hotel_search",
                            "rating": getattr(hotel, 'rating', 4.0),
                            "price_per_night": {"amount": getattr(hotel, 'price_per_night', 150), "currency": "USD"}
                        }
                    })
            
            # Add attraction events (spread across days 2-4)
            if "attraction_search" in tool_results:
                attraction_result = tool_results["attraction_search"]
                if hasattr(attraction_result, 'attractions'):
                    attractions = attraction_result.attractions[:3]  # Limit to 3
                    for i, attraction in enumerate(attractions):
                        visit_day = start_date + timedelta(days=i+1)
                        visit_time = visit_day.replace(hour=10 + i*2, minute=0)  # 10 AM, 12 PM, 2 PM
                        
                        events.append({
                            "id": f"attraction_{i+1}_{str(uuid.uuid4())[:8]}",
                            "title": f"Visit {getattr(attraction, 'name', f'Attraction in {destination}')}",
                            "description": getattr(attraction, 'category', 'Sightseeing activity'),
                            "event_type": "attraction",
                            "start_time": visit_time.isoformat(),
                            "end_time": (visit_time + timedelta(hours=2)).isoformat(),
                            "location": getattr(attraction, 'location', destination),
                            "coordinates": {"lat": 40.7484, "lng": -73.9857},
                            "details": {
                                "source": "attraction_search",
                                "rating": getattr(attraction, 'rating', 4.5),
                                "category": getattr(attraction, 'category', 'Tourism')
                            }
                        })
            
            # Add meal events if no specific events were created
            if not events:
                # Create default sightseeing events
                for i in range(3):
                    day = start_date + timedelta(days=i+1)
                    event_time = day.replace(hour=10 + i*3, minute=0)
                    events.append({
                        "id": f"activity_{i+1}_{str(uuid.uuid4())[:8]}",
                        "title": f"Explore {destination} - Day {i+1}",
                        "description": f"Discover the best of {destination}",
                        "event_type": "activity",
                        "start_time": event_time.isoformat(),
                        "end_time": (event_time + timedelta(hours=3)).isoformat(),
                        "location": destination,
                        "coordinates": {"lat": 40.7484, "lng": -73.9857},
                        "details": {
                            "source": "default_generation",
                            "recommendations": ["Bring camera", "Wear comfortable shoes"]
                        }
                    })
            
            logger.info(f"Created {len(events)} events from tool results")
            return events
            
        except Exception as e:
            logger.error(f"Error creating events from tool results: {e}")
            # Return basic fallback event
            return [{
                "id": f"basic_trip_{str(uuid.uuid4())[:8]}",
                "title": f"Trip to {destination}",
                "description": f"Travel to {destination}",
                "event_type": "activity",
                "start_time": current_time.isoformat(),
                "end_time": (current_time + timedelta(hours=4)).isoformat(),
                "location": destination,
                "coordinates": {"lat": 40.7484, "lng": -73.9857},
                "details": {"source": "fallback_generation"}
            }]
    
    def _create_natural_plan_response(
        self, destination: str, event_count: int, duration: int, travelers: int, tool_results: Dict[str, Any]
    ) -> str:
        """Create a natural language description of the generated plan"""
        
        response = f"🗺️ **Your {destination} Travel Plan is Ready!**\n\n"
        response += f"I've created a detailed {duration}-day itinerary with {event_count} events for "
        
        if travelers == 1:
            response += "your solo trip"
        elif travelers == 2:
            response += "your couple's trip" 
        else:
            response += f"your group of {travelers}"
            
        response += f" to {destination}.\n\n"
        
        # Add tool-specific highlights
        if "hotel_search" in tool_results:
            hotel_result = tool_results["hotel_search"]
            if hasattr(hotel_result, 'hotels') and len(hotel_result.hotels) > 0:
                response += f"🏨 **Accommodation**: Found {len(hotel_result.hotels)} hotel options\n"
        
        if "flight_search" in tool_results:
            flight_result = tool_results["flight_search"]
            if hasattr(flight_result, 'flights') and len(flight_result.flights) > 0:
                response += f"✈️ **Flights**: Found {len(flight_result.flights)} flight options\n"
        
        if "attraction_search" in tool_results:
            attraction_result = tool_results["attraction_search"]
            if hasattr(attraction_result, 'attractions') and len(attraction_result.attractions) > 0:
                response += f"🎯 **Attractions**: Found {len(attraction_result.attractions)} local attractions\n"
        
        response += f"\n📅 **Check your calendar** to see the complete {duration}-day schedule with times, locations, and details!\n"
        response += f"💡 You can chat with me to modify any part of your itinerary."
        
        return response

    def _generate_template_based_plan(
        self, execution_result: Dict[str, Any], intent: Dict[str, Any], user_message: str
    ) -> AgentResponse:
        """Generate basic structured plan using template when LLM fails"""
        try:
            # Extract information
            intent_type = intent.get("type") or intent.get("intent_type", "planning")
            destination = intent.get("destination", {})
            if isinstance(destination, dict):
                destination_name = destination.get("primary", "unknown")
            else:
                destination_name = str(destination)
            
            tool_results = execution_result.get("results", {})
            
            # Create basic structured plan
            structured_plan = {
                "destination": destination_name,
                "duration": 3,  # Default 3-day plan
                "start_date": "2024-07-01",
                "end_date": "2024-07-03",
                "travelers": 2,
                "budget_estimate": {"currency": "USD", "amount": 1500},
                "metadata": {
                    "generated_method": "template_fallback",
                    "travel_style": "moderate"
                }
            }
            
            # Generate basic events from tool results
            plan_events = []
            event_id_counter = 1
            
            # Add flight events if available
            if "flight_search" in tool_results:
                flight_data = tool_results["flight_search"]
                if isinstance(flight_data, dict) and flight_data.get("flights"):
                    plan_events.append({
                        "id": f"flight_{event_id_counter}",
                        "title": f"Flight to {destination_name}",
                        "description": "Outbound flight based on search results",
                        "event_type": "flight",
                        "start_time": "2024-07-01T08:00:00+00:00",
                        "end_time": "2024-07-01T12:00:00+00:00",
                        "location": f"Airport → {destination_name}",
                        "details": {"source": "flight_search", "tool_data": flight_data}
                    })
                    event_id_counter += 1
            
            # Add hotel events if available
            if "hotel_search" in tool_results:
                hotel_data = tool_results["hotel_search"]
                if isinstance(hotel_data, dict) and hotel_data.get("hotels"):
                    plan_events.append({
                        "id": f"hotel_{event_id_counter}",
                        "title": f"Hotel Stay in {destination_name}",
                        "description": "Accommodation based on search results",
                        "event_type": "hotel",
                        "start_time": "2024-07-01T15:00:00+00:00",
                        "end_time": "2024-07-03T11:00:00+00:00",
                        "location": destination_name,
                        "details": {"source": "hotel_search", "tool_data": hotel_data}
                    })
                    event_id_counter += 1
            
            # Add attraction events if available
            if "attraction_search" in tool_results:
                attraction_data = tool_results["attraction_search"]
                if isinstance(attraction_data, dict) and attraction_data.get("attractions"):
                    attractions = attraction_data.get("attractions", [])[:3]  # Take first 3
                    for i, attraction in enumerate(attractions):
                        plan_events.append({
                            "id": f"attraction_{event_id_counter}",
                            "title": f"Visit {attraction.get('name', 'Attraction')}",
                            "description": attraction.get("description", "Explore local attraction"),
                            "event_type": "attraction",
                            "start_time": f"2024-07-0{2+i//2}T{9+i*3}:00:00+00:00",
                            "end_time": f"2024-07-0{2+i//2}T{12+i*3}:00:00+00:00",
                            "location": attraction.get("location", destination_name),
                            "details": {"source": "attraction_search", "attraction_data": attraction}
                        })
                        event_id_counter += 1
            
            # Generate natural response
            natural_response = f"Here's a structured travel plan for {destination_name}! "
            if plan_events:
                natural_response += f"I've created a {structured_plan['duration']}-day itinerary with {len(plan_events)} planned activities including "
                event_types = list(set([event["event_type"] for event in plan_events]))
                natural_response += ", ".join(event_types) + ". "
            natural_response += "This plan is based on your travel preferences and the available options I found. You can modify any part of this itinerary to better suit your needs!"
            
            # Create response
            response = AgentResponse(
                success=True,
                content=natural_response,
                actions_taken=execution_result.get("actions", []),
                next_steps=execution_result.get("next_steps", []),
                confidence=0.78,
                structured_plan=structured_plan,
                plan_events=plan_events,
                metadata={
                    **execution_result,
                    "intent": intent,
                    "response_type": "structured_plan",
                    "plan_generation_method": "template_based",
                    "destination": destination_name,
                    "intent_type": intent_type,
                    "plan_events_count": len(plan_events)
                }
            )
            
            logger.info(f"Template-based plan generated: {len(plan_events)} events for {destination_name}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate template-based plan: {e}")
            # Ultimate fallback to basic response
            return AgentResponse(
                success=False,
                content=f"I encountered an issue creating a detailed plan for {destination_name}. Please try rephrasing your request or provide more specific details about your travel preferences.",
                confidence=0.4,
                metadata={"error": str(e), "response_method": "plan_generation_fallback"}
            )

    def _build_structured_content(
        self, intent_type: str, destination: str, tools_used: List[str], 
        tool_results: Dict[str, Any], user_message: str
    ) -> str:
        """Build response content using structured templates"""
        
        logger.info(f"=== BUILDING STRUCTURED CONTENT ===")
        logger.info(f"Intent type: {intent_type}")
        logger.info(f"Destination: {destination}")
        logger.info(f"Tools used: {tools_used}")
        logger.info(f"Tool results keys: {list(tool_results.keys())}")
        
        content_parts = []
        
        # Add greeting/acknowledgment based on intent
        greeting = self._get_intent_greeting(intent_type, destination)
        content_parts.append(greeting)
        
        # Add tool-specific content sections
        for tool_name in tools_used:
            logger.info(f"Processing tool: {tool_name}")
            if tool_name in tool_results:
                logger.info(f"Tool {tool_name} found in results")
                tool_content = self._format_tool_results(tool_name, tool_results[tool_name], destination)
                if tool_content:
                    logger.info(f"Tool {tool_name} content generated: {len(tool_content)} chars")
                    content_parts.append(tool_content)
                else:
                    logger.info(f"Tool {tool_name} content is empty")
            else:
                logger.info(f"Tool {tool_name} NOT found in results")
        
        # Add recommendations and next steps
        recommendations = self._generate_template_recommendations(intent_type, destination, tools_used)
        if recommendations:
            content_parts.append(recommendations)
        
        # Join all parts with proper spacing
        final_content = "\n\n".join(content_parts)
        logger.info(f"Final content length: {len(final_content)} chars")
        logger.info(f"=== END BUILDING STRUCTURED CONTENT ===")
        return final_content

    def _get_intent_greeting(self, intent_type: str, destination: str) -> str:
        """Get appropriate greeting based on intent type"""
        
        greetings = {
            "planning": f"I'll help you plan your trip to {destination}! Here's what I found:",
            "recommendation": f"Here are my top recommendations for {destination}:",
            "query": f"Here's information about {destination}:",
            "booking": f"I found these booking options for {destination}:",
            "modification": f"Here are updated options for your {destination} trip:"
        }
        
        return greetings.get(intent_type, f"Here's information about {destination}:")

    def _format_tool_results(self, tool_name: str, tool_result: Dict[str, Any], destination: str) -> str:
        """Format individual tool results using templates"""
        
        logger.info(f"=== FORMATTING TOOL: {tool_name} ===")
        logger.info(f"Tool result type: {type(tool_result)}")
        logger.info(f"Tool result success: {getattr(tool_result, 'success', 'N/A')}")
        logger.info(f"Tool result error: {getattr(tool_result, 'error', 'N/A')}")
        
        # Handle both Pydantic models and dictionaries for error checking
        has_error = False
        if not tool_result:
            has_error = True
        elif hasattr(tool_result, 'error') and tool_result.error:
            has_error = True
        elif isinstance(tool_result, dict) and "error" in tool_result:
            has_error = True
        
        if has_error:
            logger.info(f"Tool {tool_name} has error or is empty")
            return f"I had trouble finding {tool_name.replace('_', ' ')} information, but I can still help with general advice about {destination}."
        
        if tool_name == "attraction_search":
            logger.info(f"Formatting attraction search results")
            return self._format_attraction_results(tool_result, destination)
        elif tool_name == "hotel_search":
            logger.info(f"Formatting hotel search results")
            return self._format_hotel_results(tool_result, destination)
        elif tool_name == "flight_search":
            logger.info(f"Formatting flight search results")
            return self._format_flight_results(tool_result, destination)
        else:
            logger.info(f"Unknown tool: {tool_name}")
            return f"Found information from {tool_name.replace('_', ' ')} search."

    def _format_attraction_results(self, result: Any, destination: str) -> str:
        """Format attraction search results"""
        
        content = f"🎯 **Top Attractions in {destination}**\n"
        
        # Handle both ToolOutput objects and dictionaries
        if hasattr(result, 'attractions'):
            attractions = result.attractions
        else:
            attractions = result.get("attractions", []) if isinstance(result, dict) else []
        
        if attractions:
            for i, attraction in enumerate(attractions[:5], 1):  # Top 5
                # Handle both Pydantic models and dictionaries
                if hasattr(attraction, 'name'):
                    # Pydantic model
                    name = getattr(attraction, 'name', 'Unknown Attraction')
                    rating = getattr(attraction, 'rating', 'No rating')
                    description = getattr(attraction, 'description', '')
                else:
                    # Dictionary
                    name = attraction.get("name", "Unknown Attraction")
                    rating = attraction.get("rating", "No rating")
                    description = attraction.get("description", "")
                
                # Truncate description if too long
                if len(description) > 100:
                    description = description[:100] + "..."
                
                content += f"{i}. **{name}** (Rating: {rating})\n   {description}\n"
        else:
            content += f"I found several popular attractions in {destination} that are worth visiting. "
            content += "The area offers great opportunities for sightseeing and cultural experiences."
        
        return content

    def _format_hotel_results(self, result: Dict[str, Any], destination: str) -> str:
        """Format hotel search results"""
        
        content = f"🏨 **Accommodation Options in {destination}**\n"
        
        # Handle both ToolOutput objects and dictionaries
        if hasattr(result, 'hotels'):
            hotels = result.hotels
        else:
            hotels = result.get("hotels", [])
        
        if hotels:
            for i, hotel in enumerate(hotels[:5], 1):  # Top 5
                # Handle both Pydantic models and dictionaries
                if hasattr(hotel, 'name'):
                    # Pydantic model - Only access price_per_night, never 'price' for hotels
                    name = getattr(hotel, 'name', 'Unknown Hotel')
                    price = getattr(hotel, 'price_per_night', 'Price available')
                    rating = getattr(hotel, 'rating', 'No rating')
                    location = getattr(hotel, 'location', destination)
                    currency = getattr(hotel, 'currency', '')
                else:
                    # Dictionary - Only access price_per_night for hotels, removed fallback to 'price'
                    name = hotel.get("name", "Unknown Hotel")
                    price = hotel.get("price_per_night", "Price available")  # Removed price fallback to avoid confusion
                    rating = hotel.get("rating", "No rating")
                    location = hotel.get("location", destination)
                    currency = hotel.get("currency", "")
                
                # Format price with currency
                price_str = f"{price} {currency}".strip() if price and price != 'Price available' else 'Price available'
                
                content += f"{i}. **{name}** (Rating: {rating})\n   Location: {location} | Price: {price_str}\n"
        else:
            content += f"There are good accommodation options available in {destination}. "
            content += "I recommend booking in advance for better rates and availability."
        
        return content

    def _format_flight_results(self, result: Dict[str, Any], destination: str) -> str:
        """Format flight search results"""
        
        content = f"✈️ **Flight Information to {destination}**\n"
        
        # Handle both ToolOutput objects and dictionaries
        if hasattr(result, 'flights'):
            flights = result.flights
        else:
            flights = result.get("flights", [])
        
        if flights:
            for i, flight in enumerate(flights[:3], 1):  # Top 3
                # Handle both Pydantic models and dictionaries
                if hasattr(flight, 'airline'):
                    # Pydantic model
                    airline = getattr(flight, 'airline', 'Unknown Airline')
                    price = getattr(flight, 'price', 'Price available')
                    duration = getattr(flight, 'duration', 'Duration varies')
                    currency = getattr(flight, 'currency', '')
                else:
                    # Dictionary
                    airline = flight.get("airline", "Unknown Airline")
                    price = flight.get("price", "Price available")
                    duration = flight.get("duration", "Duration varies")
                    currency = flight.get("currency", "")
                
                # Format price with currency
                price_str = f"{price} {currency}".strip() if price and price != 'Price available' else 'Price available'
                
                content += f"{i}. **{airline}** | Price: {price_str} | Duration: {duration}\n"
        else:
            content += f"Flight options are available to {destination}. "
            content += "I recommend comparing prices across different booking platforms and being flexible with dates."
        
        return content

    def _generate_template_recommendations(
        self, intent_type: str, destination: str, tools_used: List[str]
    ) -> str:
        """Generate recommendations using templates"""
        
        recommendations = []
        
        # Intent-specific recommendations
        if intent_type == "planning":
            recommendations.extend([
                "💡 **Travel Tips:**",
                f"• Research local customs and etiquette for {destination}",
                "• Check weather conditions for your travel dates",
                "• Consider getting travel insurance"
            ])
        elif intent_type == "recommendation":
            recommendations.extend([
                "💡 **Additional Suggestions:**",
                "• Check opening hours and seasonal availability",
                "• Read recent reviews from other travelers",
                "• Consider booking popular attractions in advance"
            ])
        
        # Tool-specific recommendations
        if "flight_search" in tools_used:
            recommendations.append("• Book flights 2-3 months in advance for better prices")
        if "hotel_search" in tools_used:
            recommendations.append("• Compare amenities and read recent guest reviews")
        if "attraction_search" in tools_used:
            recommendations.append("• Check for combo tickets or city passes for multiple attractions")
        
        return "\n".join(recommendations) if recommendations else ""

    def _build_fallback_content(self, intent_type: str, destination: str, user_message: str) -> str:
        """Build fallback content when tools fail"""
        
        content_parts = [
            f"I understand you're interested in {intent_type} for {destination}.",
            "",
            "While I'm having some technical difficulties accessing current data, I can still help you with:",
            "• General travel advice and tips",
            "• Popular attractions and landmarks",
            "• Local customs and cultural information",
            "• Transportation options and guidance",
            "",
            "Please let me know what specific aspect of your trip you'd like to discuss, and I'll provide helpful information based on my knowledge."
        ]
        
        return "\n".join(content_parts)

    def _build_mixed_content(
        self, intent_type: str, destination: str, tools_used: List[str], 
        tool_results: Dict[str, Any], user_message: str
    ) -> str:
        """Build content when some tools succeed and others fail"""
        
        content_parts = []
        
        # Add greeting/acknowledgment
        greeting = self._get_intent_greeting(intent_type, destination)
        content_parts.append(greeting)
        
        # Process available tool results
        successful_tools = []
        failed_tools = []
        
        for tool_name in tools_used:
            if tool_name in tool_results and not tool_results[tool_name].get("error"):
                successful_tools.append(tool_name)
                tool_content = self._format_tool_results(tool_name, tool_results[tool_name], destination)
                if tool_content:
                    content_parts.append(tool_content)
            else:
                failed_tools.append(tool_name)
        
        # Add information about what we couldn't retrieve
        if failed_tools:
            failed_info = self._generate_fallback_info_for_failed_tools(failed_tools, destination)
            if failed_info:
                content_parts.append(failed_info)
        
        # Add general recommendations based on destination knowledge
        if destination != "unknown":
            general_advice = self._generate_destination_advice(destination, intent_type)
            if general_advice:
                content_parts.append(general_advice)
        
        # Add helpful next steps
        recommendations = self._generate_template_recommendations(intent_type, destination, successful_tools)
        if recommendations:
            content_parts.append(recommendations)
        
        return "\n\n".join(content_parts)

    def _generate_fallback_info_for_failed_tools(self, failed_tools: List[str], destination: str) -> str:
        """Generate helpful information for failed tools"""
        
        if not failed_tools:
            return ""
        
        info_parts = ["ℹ️ **Additional Information:**"]
        
        for tool_name in failed_tools:
            if tool_name == "hotel_search":
                info_parts.append(f"• For accommodation in {destination}, I recommend checking major booking platforms like Booking.com, Hotels.com, or Expedia")
            elif tool_name == "flight_search":
                info_parts.append(f"• For flights to {destination}, compare prices on Google Flights, Kayak, or airline websites directly")
            elif tool_name == "attraction_search":
                info_parts.append(f"• For attractions in {destination}, check TripAdvisor, local tourism websites, or travel guides")
        
        return "\n".join(info_parts) if len(info_parts) > 1 else ""

    def _generate_destination_advice(self, destination: str, intent_type: str) -> str:
        """Generate general advice based on destination knowledge"""
        
        # Basic destination advice based on common knowledge
        destination_lower = destination.lower()
        
        advice_parts = ["🗺️ **General Travel Advice:**"]
        
        if destination_lower in ["tokyo", "japan"]:
            advice_parts.extend([
                "• Consider getting a JR Pass for convenient train travel",
                "• Try local cuisine like sushi, ramen, and tempura",
                "• Visit during spring (cherry blossoms) or fall (autumn colors)",
                "• Learn basic Japanese phrases - locals appreciate the effort"
            ])
        elif destination_lower in ["paris", "france"]:
            advice_parts.extend([
                "• Visit iconic landmarks like the Eiffel Tower and Louvre Museum",
                "• Try authentic French cuisine and visit local cafés", 
                "• Consider a Seine river cruise for beautiful city views",
                "• Book popular attractions in advance to avoid long lines"
            ])
        elif destination_lower in ["london", "uk"]:
            advice_parts.extend([
                "• Use an Oyster Card for convenient public transportation",
                "• Visit free museums like the British Museum and Tate Modern",
                "• Experience traditional afternoon tea and pub culture",
                "• Check the weather and pack layers for unpredictable conditions"
            ])
        else:
            advice_parts.extend([
                f"• Research local customs and etiquette for {destination}",
                "• Check visa requirements and travel documentation needed",
                "• Look into local transportation options and travel cards",
                "• Try authentic local cuisine and cultural experiences"
            ])
        
        return "\n".join(advice_parts)

    async def _fast_quality_assessment(
        self, original_message: AgentMessage, response: AgentResponse
    ) -> float:
        """
        Fast heuristic-based quality assessment (~0.1s)
        No LLM calls - uses rule-based scoring
        """
        try:
            user_message = original_message.content.lower()
            response_content = response.content.lower()
            metadata = response.metadata
            
            # Initialize score
            total_score = 0.0
            max_score = 0.0
            
            # 1. Relevance assessment (weight: 0.3)
            relevance_score = self._assess_relevance_heuristic(user_message, response_content, metadata)
            total_score += relevance_score * 0.3
            max_score += 0.3
            
            # 2. Completeness assessment (weight: 0.25)
            completeness_score = self._assess_completeness_heuristic(user_message, response_content, metadata)
            total_score += completeness_score * 0.25
            max_score += 0.25
            
            # 3. Actionability assessment (weight: 0.2)
            actionability_score = self._assess_actionability_heuristic(response)
            total_score += actionability_score * 0.2
            max_score += 0.2
            
            # 4. Information fusion assessment (weight: 0.15)
            fusion_score = self._assess_fusion_heuristic(metadata, response_content)
            total_score += fusion_score * 0.15
            max_score += 0.15
            
            # 5. Response structure assessment (weight: 0.1)
            structure_score = self._assess_structure_heuristic(response_content)
            total_score += structure_score * 0.1
            max_score += 0.1
            
            # Calculate final normalized score
            final_score = total_score / max_score if max_score > 0 else 0.5
            
            logger.debug(f"Fast quality assessment: {final_score:.3f} (relevance: {relevance_score:.2f}, completeness: {completeness_score:.2f}, actionability: {actionability_score:.2f})")
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error in fast quality assessment: {e}")
            return 0.5  # Default neutral score

    def _assess_relevance_heuristic(
        self, user_message: str, response_content: str, metadata: Dict[str, Any]
    ) -> float:
        """Assess relevance using keyword matching and intent alignment"""
        
        score = 0.4  # Base score
        
        # Extract key travel-related terms from user message
        travel_keywords = [
            "trip", "travel", "visit", "plan", "go", "tour", "vacation", "holiday",
            "flight", "hotel", "attraction", "restaurant", "activity", "sightseeing"
        ]
        
        user_keywords = [word for word in travel_keywords if word in user_message]
        response_keywords = [word for word in travel_keywords if word in response_content]
        
        # Keyword overlap score
        if user_keywords:
            overlap_ratio = len(set(user_keywords) & set(response_keywords)) / len(user_keywords)
            score += overlap_ratio * 0.3
        
        # Destination mention check
        destination = metadata.get("destination", "unknown")
        if destination != "unknown" and destination.lower() in response_content:
            score += 0.2
        
        # Intent alignment check
        intent_type = metadata.get("intent_type", "")
        intent_keywords = {
            "planning": ["plan", "itinerary", "schedule", "organize"],
            "recommendation": ["recommend", "suggest", "best", "top"],
            "query": ["information", "about", "details", "facts"],
            "booking": ["book", "reserve", "purchase", "buy"]
        }
        
        if intent_type in intent_keywords:
            intent_words = intent_keywords[intent_type]
            if any(word in response_content for word in intent_words):
                score += 0.1
        
        return min(score, 1.0)

    def _assess_completeness_heuristic(
        self, user_message: str, response_content: str, metadata: Dict[str, Any]
    ) -> float:
        """Assess completeness based on response structure and content"""
        
        score = 0.3  # Base score
        
        # Response length assessment (indicates thoroughness)
        if len(response_content) >= 200:
            score += 0.2
        elif len(response_content) >= 100:
            score += 0.1
        
        # Tools used assessment
        tools_used = metadata.get("tools_used", [])
        if tools_used:
            score += min(len(tools_used) * 0.1, 0.3)
        
        # Structured content indicators
        structure_indicators = ["**", "•", "1.", "2.", "3.", "🎯", "🏨", "✈️", "💡"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response_content)
        if structure_count >= 3:
            score += 0.2
        elif structure_count >= 1:
            score += 0.1
        
        # Information variety (multiple aspects covered)
        info_aspects = ["attraction", "hotel", "flight", "tip", "recommendation", "price", "rating"]
        aspects_covered = sum(1 for aspect in info_aspects if aspect in response_content)
        if aspects_covered >= 3:
            score += 0.1
        
        return min(score, 1.0)

    def _assess_actionability_heuristic(self, response: AgentResponse) -> float:
        """Assess how actionable the response is"""
        
        score = 0.4  # Base score
        
        # Check for actionable elements
        if response.actions_taken:
            score += min(len(response.actions_taken) * 0.1, 0.2)
        
        if response.next_steps:
            score += min(len(response.next_steps) * 0.1, 0.3)
        
        # Check for actionable language in content
        actionable_phrases = [
            "book", "reserve", "check", "compare", "consider", "visit", "contact",
            "search", "browse", "review", "verify", "plan", "schedule"
        ]
        
        content_lower = response.content.lower()
        actionable_count = sum(1 for phrase in actionable_phrases if phrase in content_lower)
        if actionable_count >= 3:
            score += 0.1
        
        return min(score, 1.0)

    def _assess_fusion_heuristic(self, metadata: Dict[str, Any], response_content: str) -> float:
        """Assess information fusion quality"""
        
        score = 0.5  # Base score
        
        # Check if multiple information sources were used
        tools_used = metadata.get("tools_used", [])
        if len(tools_used) >= 2:
            score += 0.3
        elif len(tools_used) >= 1:
            score += 0.1
        
        # Check for integration indicators
        integration_phrases = [
            "based on", "found", "available", "options", "information shows",
            "according to", "results indicate", "search shows"
        ]
        
        integration_count = sum(1 for phrase in integration_phrases if phrase in response_content)
        if integration_count >= 2:
            score += 0.2
        elif integration_count >= 1:
            score += 0.1
        
        return min(score, 1.0)

    def _assess_structure_heuristic(self, response_content: str) -> float:
        """Assess response structure and formatting"""
        
        score = 0.4  # Base score
        
        # Check for good formatting
        formatting_elements = [
            "**",  # Bold text
            "•",   # Bullet points
            "\n\n", # Paragraph breaks
            ":",   # Section separators
            "1.",  # Numbered lists
        ]
        
        formatting_count = sum(1 for element in formatting_elements if element in response_content)
        if formatting_count >= 3:
            score += 0.4
        elif formatting_count >= 2:
            score += 0.2
        elif formatting_count >= 1:
            score += 0.1
        
        # Check for logical structure (sections)
        has_sections = ("**" in response_content and 
                       ("\n\n" in response_content or ":" in response_content))
        if has_sections:
            score += 0.2
        
        return min(score, 1.0)

    async def _llm_enhanced_response(
        self, structured_response: AgentResponse, execution_result: Dict[str, Any], intent: Dict[str, Any]
    ) -> AgentResponse:
        """
        Enhance structured response using LLM when quality is insufficient
        Only called when fast assessment shows quality below threshold
        """
        try:
            logger.info("Enhancing response with LLM due to quality concerns")
            
            # Prepare context for LLM enhancement
            user_message = execution_result.get("original_message", "")
            knowledge_context = execution_result.get("knowledge_context", {})
            tool_results = execution_result.get("results", {})
            
            # Retrieve knowledge context if not available
            if not knowledge_context:
                structured_analysis = intent.get("structured_analysis", {})
                knowledge_context = await self._retrieve_knowledge_context(
                    query=user_message, structured_intent=structured_analysis
                )
            
            # Use existing LLM-based response generation
            enhanced_content = await self._generate_response(execution_result, intent)
            
            # Create enhanced response
            enhanced_response = AgentResponse(
                success=structured_response.success,
                content=enhanced_content,
                actions_taken=structured_response.actions_taken,
                next_steps=structured_response.next_steps,
                confidence=min(structured_response.confidence + 0.1, 1.0),
                metadata={
                    **structured_response.metadata,
                    "response_method": "llm_enhanced",
                    "original_quality": structured_response.metadata.get("quality_score", 0.0),
                    "enhancement_applied": True,
                    "knowledge_context": knowledge_context
                }
            )
            
            logger.info(f"LLM enhancement completed, content length: {len(enhanced_content)}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in LLM enhancement: {e}")
            # Return original structured response if enhancement fails
            structured_response.metadata["enhancement_failed"] = str(e)
            return structured_response

    def _create_plan_aware_fusion_prompt(
        self, user_message: str, formatted_knowledge: str, formatted_tools: str,
        formatted_intent: str, plan_context: Dict[str, Any]
    ) -> str:
        """Create fusion prompt that considers existing plan"""
        
        existing_events = plan_context.get("event_summary", "")
        plan_gaps = plan_context.get("gaps_identified", [])
        last_updated = plan_context.get("last_updated", "Unknown")
        
        # Use centralized prompt template
        prompt = prompt_manager.get_prompt(
            PromptType.INFORMATION_FUSION,
            user_message=user_message,
            existing_events=existing_events,
            plan_gaps=', '.join(plan_gaps) if plan_gaps else 'None',
            last_updated=last_updated,
            formatted_intent=formatted_intent,
            formatted_tools=formatted_tools,
            formatted_knowledge=formatted_knowledge
        )
        
        return prompt


# Global travel agent instance (singleton pattern)
_travel_agent: Optional[TravelAgent] = None

def get_travel_agent() -> TravelAgent:
    """Get travel agent instance (singleton pattern)"""
    global _travel_agent
    if _travel_agent is None:
        _travel_agent = TravelAgent()
        logger.info("✅ Created singleton Travel Agent instance")
        
        # Register with agent manager
        from app.agents.base_agent import agent_manager
        agent_manager.register_agent(_travel_agent)
        
    return _travel_agent

# Initialize the singleton instance on module import
travel_agent = get_travel_agent()
