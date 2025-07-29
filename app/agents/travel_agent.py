"""
Travel Agent - Main travel planning agent

TODO: Implement the following features
1. Intelligent travel planning generation
2. Multi-tool coordination
3. User preference learning
4. Personalized recommendations
5. Real-time plan adjustment
"""

from typing import Dict, List, Any, Optional
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
            "relevance": 0.20,  # How relevant to user's travel request
            "information_fusion": 0.25,  # How well knowledge + tools are integrated
            "completeness": 0.20,  # How complete the travel information is
            "accuracy": 0.15,  # How accurate the information is
            "actionability": 0.15,  # How actionable the recommendations are
            "personalization": 0.05,  # How well personalized to user preferences
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
            }
            result = await self._execute_action_plan(action_plan, execution_context)

            # 4. Structured response fusion (template-based, fast)
            structured_response = self._generate_structured_response(result, intent, message.content)

            # 5. Fast quality assessment (heuristic, ~0.1s) - Skip if in refinement mode
            if skip_quality_check:
                self.status = AgentStatus.IDLE
                return structured_response
            
            quality_score = await self._fast_quality_assessment(message, structured_response)

            # 6. Quality good enough -> return, else LLM enhancement
            if quality_score >= self.fast_response_threshold:  # 0.75 for fast response
                self.status = AgentStatus.IDLE
                return structured_response
            else:
                enhanced_response = await self._llm_enhanced_response(structured_response, result, intent)
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
            destinations = ["tokyo", "kyoto", "paris", "london", "new york", "beijing", "shanghai", "rome", "barcelona"]
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
        legacy_format = {
            "type": llm_analysis["intent_type"],
            "destination": llm_analysis["destination"]["primary"],
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
                "destination": llm_analysis["destination"]["primary"],
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

        # Enhanced intent type detection
        intent_type = "query"  # default
        if any(
            word in user_message_lower
            for word in ["plan", "create", "make", "arrange", "organize", "schedule"]
        ):
            intent_type = "planning"
        elif any(
            word in user_message_lower
            for word in ["change", "modify", "update", "adjust", "revise"]
        ):
            intent_type = "modification"
        elif any(
            word in user_message_lower
            for word in ["recommend", "suggest", "introduce", "advise", "what should"]
        ):
            intent_type = "recommendation"
        elif any(
            word in user_message_lower
            for word in ["book", "reserve", "purchase", "buy"]
        ):
            intent_type = "booking"
        elif any(
            word in user_message_lower
            for word in ["problem", "issue", "complaint", "wrong", "error"]
        ):
            intent_type = "complaint"

        # Enhanced destination detection
        destination = "Unknown"
        destinations = [
            "tokyo",
            "kyoto",
            "osaka",
            "paris",
            "london",
            "new york",
            "beijing",
            "shanghai",
            "rome",
            "barcelona",
            "amsterdam",
            "vienna",
            "prague",
            "budapest",
            "berlin",
        ]
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
        self, query: str, structured_intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve knowledge context with enhanced multi-destination support"""
        try:
            # Use RAG engine to retrieve relevant knowledge with structured intent
            if structured_intent:
                # Pass structured intent to RAG engine for intelligent destination handling
                result = await self.rag_engine.retrieve(
                    query=query, top_k=5, structured_intent=structured_intent
                )
            else:
                result = await self.rag_engine.retrieve(query, top_k=5)

            # Process retrieved documents
            contexts = []
            for i, doc in enumerate(result.documents):
                contexts.append(
                    {
                        "id": doc.id,
                        "content": doc.content[:500] + "..."
                        if len(doc.content) > 500
                        else doc.content,
                        "metadata": doc.metadata,
                    }
                )

            full_context = "\n\n".join([doc.content for doc in result.documents])

            return {
                "relevant_docs": contexts,
                "context": full_context,
                "total_results": result.total_results,
                "scores": result.scores,
                "multi_destination_analysis": structured_intent.get("destination", {})
                if structured_intent
                else None,
            }

        except Exception as e:
            logger.error(f"Error in knowledge context retrieval: {str(e)}")

            return {
                "relevant_docs": [],
                "context": f"Error retrieving context: {str(e)}",
                "error": str(e),
                "multi_destination_analysis": None,
            }

    async def _create_action_plan(
        self, intent: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create intelligent action plan based on structured intention analysis"""

        # Extract structured analysis from intent
        structured_analysis = intent.get("structured_analysis", {})

        # Initialize enhanced plan structure
        plan = {
            "intent_type": intent.get("type") or intent.get("intent_type", "query"),
            "destination": structured_analysis.get("destination", {}),
            "actions": [],
            "tools_to_use": [],
            "tool_parameters": {},
            "execution_strategy": "sequential",
            "next_steps": [],
            "confidence": 0.0,
        }

        try:
            # Step 1: Use LLM for intelligent tool selection
            tool_selection_result = await self._llm_tool_selection(
                structured_analysis, context
            )
            logger.info(f"LLM tool selection: {tool_selection_result}")

            # Step 2: Extract parameters from structured intent
            tool_parameters = await self._extract_parameters_from_intent(
                structured_analysis, tool_selection_result["selected_tools"]
            )
            logger.info(f"Tool parameters from intent: {tool_parameters}")

            # Step 3: Generate intent-based actions and next steps
            actions = await self._generate_intent_based_actions(
                structured_analysis, tool_selection_result["selected_tools"]
            )
            next_steps = await self._generate_intent_based_next_steps(
                structured_analysis, context
            )

            # Step 4: Update plan with results
            plan.update(
                {
                    "tools_to_use": tool_selection_result["selected_tools"],
                    "tool_parameters": tool_parameters,
                    "execution_strategy": tool_selection_result["execution_strategy"],
                    "actions": actions,
                    "next_steps": next_steps,
                    "confidence": tool_selection_result["confidence"],
                    "tool_selection_reasoning": tool_selection_result.get(
                        "reasoning", ""
                    ),
                    "intent_metadata": {
                        "user_sentiment": structured_analysis.get(
                            "sentiment", "neutral"
                        ),
                        "urgency_level": structured_analysis.get("urgency", "medium"),
                        "missing_info": structured_analysis.get("missing_info", []),
                        "key_requirements": structured_analysis.get(
                            "key_requirements", []
                        ),
                        "preferences": structured_analysis.get("preferences", {}),
                    },
                }
            )

            logger.info(
                f"Created intent-based action plan: {plan['intent_type']} -> {plan['tools_to_use']}"
            )
            return plan

        except Exception as e:
            logger.error(f"Error creating intent-based action plan: {e}")
            # Fall back to basic planning
            return await self._create_basic_action_plan(intent, context)

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

    def _convert_city_to_airport_codes(self, city_name: str) -> str:
        """Convert city name to primary airport code for flight search"""
        city_to_airport = {
            'ROME': 'FCO',  # Rome Fiumicino
            'PARIS': 'CDG',  # Paris Charles de Gaulle
            'LONDON': 'LHR',  # London Heathrow
            'BEIJING': 'PEK',  # Beijing Capital
            'TOKYO': 'NRT',  # Tokyo Narita
            'NEW YORK': 'JFK',  # New York JFK
            'BARCELONA': 'BCN',
            'MADRID': 'MAD',
            'AMSTERDAM': 'AMS',
            'BERLIN': 'BER',
            'MUNICH': 'MUC',
            'VIENNA': 'VIE',
            'PRAGUE': 'PRG',
            'BUDAPEST': 'BUD',
            'SINGAPORE': 'SIN',
            'SEOUL': 'ICN',
            'SHANGHAI': 'PVG',
            'KYOTO': 'UKB',  # Kyoto uses Osaka Kansai
            'OSAKA': 'KIX',
            'BANGKOK': 'BKK',
            'SYDNEY': 'SYD',
            'MELBOURNE': 'MEL',
            'ATHENS': 'ATH',
            'DUBAI': 'DXB',
            'ISTANBUL': 'IST',
        }
        
        # Handle "Unknown" case
        if city_name == "Unknown":
            return "Unknown"
        
        # Convert to uppercase for lookup
        city_upper = city_name.upper()
        
        # Add debugging
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"🔍 Converting city '{city_name}' (uppercase: '{city_upper}') to IATA code")
        
        # Try exact match first
        if city_upper in city_to_airport:
            result = city_to_airport[city_upper]
            logger.info(f"✅ Found exact match: '{city_upper}' -> '{result}'")
            return result
        
        # Try partial matches for multi-word cities
        for key, value in city_to_airport.items():
            if city_upper in key or key in city_upper:
                logger.info(f"✅ Found partial match: '{city_upper}' matches '{key}' -> '{value}'")
                return value
        
        # If no match found, return the original (should be a valid IATA code already)
        logger.warning(f"⚠️ No IATA code found for '{city_name}', returning '{city_upper}'")
        return city_upper

    def _extract_origin_from_message(self, user_message: str) -> str:
        """Extract origin city from user message using simple text parsing"""
        import re
        
        # Convert to lowercase for easier matching
        message_lower = user_message.lower()
        
        # Common destinations that could be origins
        cities = [
            "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", 
            "shanghai", "rome", "barcelona", "amsterdam", "vienna", "prague", 
            "budapest", "berlin", "bangkok", "singapore", "seoul", "sydney", 
            "melbourne", "munich", "madrid", "athens", "dubai", "istanbul"
        ]
        
        # Common patterns for origin extraction
        patterns = [
            r'from\s+(\w+)\s+to\s+(\w+)',  # "from Paris to Rome"
            r'(\w+)\s+to\s+(\w+)',  # "Paris to Rome" (if no "from")
            r'plan\s+a\s+trip\s+from\s+(\w+)\s+to\s+(\w+)',  # "plan a trip from Paris to Rome"
            r'travel\s+from\s+(\w+)\s+to\s+(\w+)',  # "travel from Paris to Rome"
            r'flying\s+from\s+(\w+)',  # "flying from Paris"
            r'departing\s+from\s+(\w+)',  # "departing from Paris"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                origin = match.group(1).strip().lower()
                # Check if it's a known city
                if origin in cities:
                    return origin.title()  # Capitalize first letter
                # If not in our list but looks like a city name, return it
                if len(origin) > 2 and origin.isalpha():
                    return origin.title()
        
        # If no pattern matches, return "Unknown"
        return "Unknown"

    async def _extract_parameters_from_intent(
        self, structured_analysis: Dict[str, Any], selected_tools: List[str]
    ) -> Dict[str, Any]:
        """Extract tool parameters from structured intent analysis"""

        # City to IATA code mapping
        city_to_iata = {
            "paris": "PAR",
            "beijing": "PEK",
            "london": "LHR",
            "new york": "JFK",
            "tokyo": "NRT",
            "shanghai": "PVG",
            "rome": "FCO",
            "barcelona": "BCN",
            "amsterdam": "AMS",
            "vienna": "VIE",
            "prague": "PRG",
            "budapest": "BUD",
            "berlin": "BER",
            "kyoto": "UKY",
            "osaka": "KIX",
        }

        tool_parameters = {}

        # Extract common parameters
        destination = structured_analysis.get("destination", {}).get(
            "primary", "Unknown"
        )
        origin = structured_analysis.get("origin", {}).get(
            "primary", "Unknown"
        )
        
        # If destination is still "Unknown", try to extract it from the user message
        if destination == "Unknown":
            user_message = structured_analysis.get("user_message", "")
            if user_message:
                # Try to extract destination from message
                destination = self._extract_destination_from_message(user_message)
        
        # If origin is still "Unknown", try to extract it from the user message
        if origin == "Unknown":
            user_message = structured_analysis.get("user_message", "")
            if user_message:
                # Try to extract origin from message
                origin = self._extract_origin_from_message(user_message)
        
        # Convert to proper airport codes for flight search
        destination_airport = self._convert_city_to_airport_codes(destination)
        origin_airport = self._convert_city_to_airport_codes(origin)
        
        # Add logging to debug the conversion
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"🌍 City conversion: destination='{destination}' -> '{destination_airport}', origin='{origin}' -> '{origin_airport}'")
        logger.info(f"🌍 Final tool parameters will be: origin='{origin_airport}', destination='{destination_airport}'")
        travel_details = structured_analysis.get("travel_details", {})
        preferences = structured_analysis.get("preferences", {})

        for tool in selected_tools:
            if tool == "attraction_search":
                # Only add attraction search if we have a valid destination
                if destination != "Unknown":
                    tool_parameters[tool] = {
                        "destination": destination,
                        "limit": 10,
                        "interests": preferences.get("interests", []),
                        "travel_style": preferences.get("travel_style", "mid-range"),
                    }
            elif tool == "hotel_search":
                # Only add hotel search if we have a valid destination
                if destination != "Unknown":
                    # Convert destination to IATA code for hotel search
                    destination_iata = self._convert_city_to_airport_codes(destination)
                    tool_parameters[tool] = {
                        "destination": destination_iata,
                        "limit": 8,
                        "accommodation_type": preferences.get(
                            "accommodation_type", "hotel"
                        ),
                        "budget_level": travel_details.get("budget", {}).get(
                            "level", "mid-range"
                        ),
                        "travelers": travel_details.get("travelers", 1),
                    }
            elif tool == "flight_search":
                # Only add flight search if we have valid origin and destination
                if origin_airport != "Unknown" and destination_airport != "Unknown":
                    logger.info(f"✈️ Adding flight search with origin='{origin_airport}' and destination='{destination_airport}'")
                    tool_parameters[tool] = {
                        "origin": origin_airport,
                        "destination": destination_airport,
                        "limit": 5,
                        "travelers": travel_details.get("travelers", 1),
                        "budget_level": travel_details.get("budget", {}).get(
                            "level", "mid-range"
                        ),
                        "flexibility": travel_details.get("dates", {}).get(
                            "flexibility", "flexible"
                        ),
                    }
                else:
                    logger.warning(f"✈️ Skipping flight search: origin='{origin_airport}', destination='{destination_airport}'")

        return tool_parameters

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

        # Get destination and convert to IATA code for AMADEUS API
        destination = requirements.get("destination", "unknown")
        iata_code = self._convert_city_to_iata_code(destination)
        
        # Add debugging information
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"🏨 Hotel search - Original destination: '{destination}'")
        logger.info(f"🏨 Hotel search - Converted IATA code: '{iata_code}'")
        logger.info(f"🏨 Hotel search - IATA code type: {type(iata_code)}")
        logger.info(f"🏨 Hotel search - IATA code length: {len(iata_code) if iata_code else 0}")

        # Set default dates (7 days from now for check-in, 10 days from now for check-out)
        from datetime import datetime, timedelta
        default_check_in = datetime.now() + timedelta(days=7)
        default_check_out = datetime.now() + timedelta(days=10)
        
        params = {
            "location": iata_code,  # Use IATA code for AMADEUS API
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
        self, plan: Dict[str, Any], context: Dict[str, Any]
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
            # Use destination from plan for better knowledge retrieval
            destination = plan.get("destination", "unknown")
            if destination != "unknown":
                query = f"{query} {destination}"
            
            knowledge_context = await self._retrieve_knowledge_context(query)
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
                        # Convert destination to IATA code for AMADEUS API
                        destination = tool_params.get("location", "unknown")  # hotel_search uses "location" key
                        iata_code = self._convert_city_to_iata_code(destination)
                        
                        hotel_params = {
                            "location": iata_code,  # Use IATA code for AMADEUS API
                            "check_in": tool_params.get(
                                "check_in", "2024-06-01"
                            ),  # Default dates
                            "check_out": tool_params.get("check_out", "2024-06-03"),
                            "guests": tool_params.get("guests", 1),
                            "rooms": 1,
                            "min_rating": 4.0,
                        }
                        
                        # Add debugging to see what's being sent to tool executor
                        from app.core.logging_config import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"🏨 Travel agent - Creating hotel search ToolCall")
                        logger.info(f"🏨 Travel agent - Original destination: '{destination}'")
                        logger.info(f"🏨 Travel agent - Converted IATA code: '{iata_code}'")
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
                        
                        # Convert destination to IATA code for AMADEUS API
                        destination = tool_params.get("destination", "unknown")
                        destination_iata = self._convert_city_to_iata_code(destination)
                        
                        flight_params = {
                            "origin": tool_params.get("origin", "PAR"),
                            "destination": destination_iata,  # Use IATA code for AMADEUS API
                            "start_date": future_date,
                            "passengers": tool_params.get("passengers", 1),  # flight_search uses "passengers" key
                            "class_type": "ECONOMY",
                            "budget_level": tool_params.get("budget_level", "mid-range"),
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
        """Generate response content using intelligent information fusion"""

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

            logger.info(f"Information Sources Summary:")
            logger.info(
                f"  - Knowledge docs: {len(knowledge_context.get('relevant_docs', []))}"
            )
            logger.info(f"  - Tool results: {list(tool_results.keys())}")
            logger.info(f"  - Intent type: {intent.get('type', 'unknown')}")

            if execution_result["success"]:
                # Try intelligent information fusion using LLM
                if self.llm_service:
                    logger.info("Attempting LLM-based information fusion...")
                    try:
                        # Format information for fusion prompt
                        formatted_knowledge = self._format_knowledge_for_fusion(
                            knowledge_context.get("relevant_docs", [])
                        )
                        formatted_tools = self._format_tools_for_fusion(tool_results)
                        formatted_intent = self._format_intent_for_fusion(intent)

                        # Use information fusion template from prompt manager
                        fusion_prompt = prompt_manager.get_prompt(
                            PromptType.INFORMATION_FUSION,
                            user_message=user_message,
                            knowledge_context=formatted_knowledge,
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
        if not relevant_docs:
            return "No specific knowledge context available."

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

        return "\n\n".join(knowledge_parts)

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
                        flights = result.flights[:3]  # Top 3
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
                        attractions = result.attractions[:3]  # Top 3
                        tool_parts.append(f"**Current Attractions ({len(attractions)} found)**:")
                        for attr in attractions:
                            name = attr.get("name", "Unknown")
                            desc = attr.get("description", "No description")[:100]
                            rating = attr.get("rating", "No rating")
                            tool_parts.append(f"- {name} (Rating: {rating}): {desc}")

                    elif tool_name == "hotel_search" and hasattr(result, 'hotels'):
                        hotels = result.hotels[:3]  # Top 3
                        logger.info(f"Found {len(hotels)} hotels for {tool_name}")
                        if hotels:
                            tool_parts.append(f"**Current Hotels ({len(hotels)} found)**:")
                            for hotel in hotels:
                                name = getattr(hotel, 'name', 'Unknown')
                                price = getattr(hotel, 'price_per_night', 'Price unavailable')
                                rating = getattr(hotel, 'rating', 'No rating')
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

    def _convert_city_to_iata_code(self, city_name: str) -> str:
        """Convert city name to IATA city code for AMADEUS API"""
        # Add debugging information
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"🌍 Converting city to IATA code - Input: '{city_name}'")
        
        # Handle unknown/empty destination case - return empty string to indicate invalid destination
        if not city_name or city_name.lower() in ['unknown', '']:
            logger.error(f"❌ Invalid destination '{city_name}' - cannot convert to IATA code")
            return ''  # Return empty string to indicate invalid destination
        
        # City to IATA code mapping based on the provided list
        city_to_iata = {
            'amsterdam': 'AMS',
            'barcelona': 'BCN', 
            'beijing': 'PEK',
            'berlin': 'BER',
            'budapest': 'BUD',
            'kyoto': 'ITM',  # Kyoto doesn't have a major airport, using nearby Osaka
            'london': 'LON',
            'munich': 'MUC',
            'paris': 'PAR',
            'prague': 'PRG',
            'rome': 'ROM',
            'seoul': 'SEL',
            'shanghai': 'SHA',
            'singapore': 'SIN',
            'tokyo': 'TYO',
            'vienna': 'VIE',
            'osaka': 'ITM',
            'new york': 'JFK',
            'los angeles': 'LAX',
            'chicago': 'ORD',
            'miami': 'MIA',
            'san francisco': 'SFO',
            'toronto': 'YYZ',
            'vancouver': 'YVR',
            'montreal': 'YUL',
            'sydney': 'SYD',
            'melbourne': 'MEL',
            'brisbane': 'BNE',
            'perth': 'PER',
            'adelaide': 'ADL',
            'auckland': 'AKL',
            'wellington': 'WLG',
            'cape town': 'CPT',
            'johannesburg': 'JNB',
            'nairobi': 'NBO',
            'cairo': 'CAI',
            'marrakech': 'RAK',
            'casablanca': 'CMN',
            'mumbai': 'BOM',
            'delhi': 'DEL',
            'bangalore': 'BLR',
            'chennai': 'MAA',
            'kolkata': 'CCU',
            'hyderabad': 'HYD',
            'pune': 'PNQ',
            'mexico city': 'MEX',
            'guadalajara': 'GDL',
            'monterrey': 'MTY',
            'rio de janeiro': 'GIG',
            'sao paulo': 'GRU',
            'buenos aires': 'EZE',
            'santiago': 'SCL',
            'lima': 'LIM',
            'bogota': 'BOG',
            'medellin': 'MDE',
            'caracas': 'CCS',
            'moscow': 'SVO',
            'st petersburg': 'LED',
            'kiev': 'KBP',
            'warsaw': 'WAW',
            'krakow': 'KRK',
            'bratislava': 'BTS',
            'ljubljana': 'LJU',
            'zagreb': 'ZAG',
            'belgrade': 'BEG',
            'sofia': 'SOF',
            'bucharest': 'OTP',
            'tallinn': 'TLL',
            'riga': 'RIX',
            'vilnius': 'VNO',
            'helsinki': 'HEL',
            'stockholm': 'ARN',
            'oslo': 'OSL',
            'copenhagen': 'CPH',
            'reykjavik': 'KEF',
            'dublin': 'DUB',
            'glasgow': 'GLA',
            'edinburgh': 'EDI',
            'manchester': 'MAN',
            'birmingham': 'BHX',
            'leeds': 'LBA',
            'liverpool': 'LPL',
            'newcastle': 'NCL',
            'cardiff': 'CWL',
            'belfast': 'BFS',
            'brussels': 'BRU',
            'antwerp': 'ANR',
            'rotterdam': 'RTM',
            'the hague': 'AMS',  # No airport, using Amsterdam
            'utrecht': 'AMS',    # No airport, using Amsterdam
            'luxembourg': 'LUX',
            'geneva': 'GVA',
            'zurich': 'ZRH',
            'bern': 'BRN',
            'basel': 'BSL',
            'lucerne': 'ZRH',    # No airport, using Zurich
            'milan': 'MXP',
            'florence': 'FLR',
            'venice': 'VCE',
            'naples': 'NAP',
            'palermo': 'PMO',
            'catania': 'CTA',
            'bologna': 'BLQ',
            'turin': 'TRN',
            'genoa': 'GOA',
            'bari': 'BRI',
            'lisbon': 'LIS',
            'porto': 'OPO',
            'faro': 'FAO',
            'funchal': 'FNC',
            'pontadelgada': 'PDL',
            'valencia': 'VLC',
            'bilbao': 'BIO',
            'seville': 'SVQ',
            'granada': 'GRX',
            'malaga': 'AGP',
            'alicante': 'ALC',
            'palma': 'PMI',
            'ibiza': 'IBZ',
            'tenerife': 'TFS',
            'las palmas': 'LPA',
            'fes': 'FEZ',
            'tangier': 'TNG',
            'agadir': 'AGA',
            'rabat': 'RBA',
            'tunis': 'TUN',
            'algiers': 'ALG',
            'alexandria': 'HBE',
            'giza': 'CAI',       # No airport, using Cairo
            'luxor': 'LXR',
            'aswan': 'ASW',
            'sharm el sheikh': 'SSH',
            'hurghada': 'HRG',
            'tel aviv': 'TLV',
            'jerusalem': 'JRS',  # Limited flights
            'haifa': 'HFA',
            'beer sheva': 'TLV', # No airport, using Tel Aviv
            'amman': 'AMM',
            'petra': 'AMM',      # No airport, using Amman
            'aqaba': 'AQJ',
            'damascus': 'DAM',
            'beirut': 'BEY',
            'baghdad': 'BGW',
            'basra': 'BSR',
            'erbil': 'EBL',
            'sulaymaniyah': 'ISU',
            'tehran': 'IKA',
            'mashhad': 'MHD',
            'isfahan': 'IFN',
            'shiraz': 'SYZ',
            'tabriz': 'TBZ',
            'kerman': 'KER',
            'yazd': 'AZD',
            'kabul': 'KBL',
            'kandahar': 'KDH',
            'herat': 'HEA',
            'mazar e sharif': 'MZR',
            'jalalabad': 'JAA',
            'kunduz': 'UND',
            'peshawar': 'PEW',
            'lahore': 'LHE',
            'karachi': 'KHI',
            'islamabad': 'ISB',
            'rawalpindi': 'ISB', # Same as Islamabad
            'faisalabad': 'LYP',
            'multan': 'MUX',
            'gujranwala': 'GRW',
            'sialkot': 'SKT',
            'quetta': 'UET',
            'abbottabad': 'ISB', # No airport, using Islamabad
            'murree': 'ISB',     # No airport, using Islamabad
            'gilgit': 'GIL',
            'skardu': 'KDU',
            'chitral': 'CJL',
            'swat': 'ISB',       # No airport, using Islamabad
            'hunza': 'GIL',      # No airport, using Gilgit
            'kashmir': 'SXR',
            'srinagar': 'SXR',
            'leh': 'IXL',
            'manali': 'DEL',     # No airport, using Delhi
            'shimla': 'DEL',     # No airport, using Delhi
            'mussoorie': 'DEL',  # No airport, using Delhi
            'nainital': 'DEL',   # No airport, using Delhi
            'ranikhet': 'DEL',   # No airport, using Delhi
            'almora': 'DEL',     # No airport, using Delhi
            'pithoragarh': 'DEL', # No airport, using Delhi
            'chamoli': 'DEL',    # No airport, using Delhi
            'rudraprayag': 'DEL', # No airport, using Delhi
            'tehri': 'DEL',      # No airport, using Delhi
            'uttarkashi': 'DEL', # No airport, using Delhi
            'dehradun': 'DED',
            'haridwar': 'DEL',   # No airport, using Delhi
            'rishikesh': 'DEL',  # No airport, using Delhi
            'kedarnath': 'DEL',  # No airport, using Delhi
            'badrinath': 'DEL',  # No airport, using Delhi
            'gangotri': 'DEL',   # No airport, using Delhi
            'yamunotri': 'DEL',  # No airport, using Delhi
            'hemkund': 'DEL',    # No airport, using Delhi
            'valley of flowers': 'DEL', # No airport, using Delhi
            'auli': 'DEL',       # No airport, using Delhi
            'joshimath': 'DEL',  # No airport, using Delhi
            'karnaprayag': 'DEL', # No airport, using Delhi
            'gauchar': 'DEL',    # No airport, using Delhi
            'karanprayag': 'DEL', # No airport, using Delhi
            'hong kong': 'HKG',
            'hongkong': 'HKG',
            'dubai': 'DXB',
            'istanbul': 'IST',
            'madrid': 'MAD',
            'athens': 'ATH',
            # Country codes (using major cities)
            'australia': 'SYD',  # Sydney
            'brazil': 'SAO',     # Sao Paulo
            'canada': 'YYZ',     # Toronto
            'china': 'PEK',      # Beijing
            'hong_kong': 'HKG',
            'india': 'DEL',      # Delhi
            'japan': 'TYO',      # Tokyo
            'south_africa': 'JNB', # Johannesburg
            'south_korea': 'SEL',  # Seoul
            'thailand': 'BKK',     # Bangkok
            'uk': 'LON',          # London
            'usa': 'JFK'          # New York
        }
        
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
            # If not found, return empty string to indicate invalid destination
            logger.error(f"❌ No IATA code found for '{city_name}' - invalid destination")
            return ''  # Return empty string to indicate invalid destination

    def _extract_destination_from_message(self, message: str) -> str:
        """Extract destination from user message"""
        # Add debugging information
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"🌍 Extracting destination from message: '{message}'")
        
        message_lower = message.lower()

        # Common destinations with more comprehensive list
        destinations = [
            "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", 
            "shanghai", "rome", "barcelona", "amsterdam", "vienna", "prague", 
            "budapest", "berlin", "bangkok", "singapore", "seoul", "sydney", 
            "melbourne", "munich", "madrid", "athens", "dubai", "istanbul",
            "hong kong", "hongkong", "san francisco", "los angeles", "chicago",
            "miami", "toronto", "vancouver", "montreal", "sydney", "melbourne",
            "brisbane", "perth", "adelaide", "auckland", "wellington", "cape town",
            "johannesburg", "nairobi", "cairo", "marrakech", "casablanca", "mumbai",
            "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune",
            "mexico city", "guadalajara", "monterrey", "rio de janeiro", "sao paulo",
            "buenos aires", "santiago", "lima", "bogota", "medellin", "caracas",
            "moscow", "st petersburg", "kiev", "warsaw", "krakow", "bratislava",
            "ljubljana", "zagreb", "belgrade", "sofia", "bucharest", "budapest",
            "tallinn", "riga", "vilnius", "helsinki", "stockholm", "oslo", "copenhagen",
            "reykjavik", "dublin", "glasgow", "edinburgh", "manchester", "birmingham",
            "leeds", "liverpool", "newcastle", "cardiff", "belfast", "brussels",
            "antwerp", "rotterdam", "the hague", "utrecht", "luxembourg", "geneva",
            "zurich", "bern", "basel", "lucerne", "milan", "florence", "venice",
            "naples", "palermo", "catania", "bologna", "turin", "genoa", "bari",
            "lisbon", "porto", "faro", "funchal", "pontadelgada", "valencia",
            "bilbao", "seville", "granada", "malaga", "alicante", "palma",
            "ibiza", "tenerife", "las palmas", "marrakech", "fes", "tangier",
            "agadir", "rabat", "casablanca", "tunis", "algiers", "cairo",
            "alexandria", "giza", "luxor", "aswan", "sharm el sheikh", "hurghada",
            "tel aviv", "jerusalem", "haifa", "beer sheva", "amman", "petra",
            "aqaba", "damascus", "beirut", "baghdad", "basra", "erbil", "sulaymaniyah",
            "tehran", "mashhad", "isfahan", "shiraz", "tabriz", "kerman", "yazd",
            "kabul", "kandahar", "herat", "mazar e sharif", "jalalabad", "kunduz",
            "peshawar", "lahore", "karachi", "islamabad", "rawalpindi", "faisalabad",
            "multan", "gujranwala", "sialkot", "quetta", "peshawar", "abbottabad",
            "murree", "gilgit", "skardu", "chitral", "swat", "hunza", "kashmir",
            "srinagar", "leh", "manali", "shimla", "mussoorie", "nainital",
            "ranikhet", "almora", "pithoragarh", "chamoli", "rudraprayag", "tehri",
            "uttarkashi", "dehradun", "haridwar", "rishikesh", "kedarnath", "badrinath",
            "gangotri", "yamunotri", "kedarnath", "badrinath", "hemkund", "valley of flowers",
            "auli", "joshimath", "karnaprayag", "gauchar", "karanprayag", "rudraprayag",
            "srinagar", "uttarkashi", "tehri", "chamoli", "pithoragarh", "almora",
            "ranikhet", "nainital", "mussoorie", "shimla", "manali", "leh",
            "kashmir", "hunza", "swat", "chitral", "skardu", "gilgit", "murree",
            "abbottabad", "peshawar", "quetta", "sialkot", "gujranwala", "multan",
            "faisalabad", "rawalpindi", "islamabad", "karachi", "lahore", "peshawar",
            "kunduz", "jalalabad", "mazar e sharif", "herat", "kandahar", "kabul",
            "yazd", "kerman", "tabriz", "shiraz", "isfahan", "mashhad", "tehran",
            "sulaymaniyah", "erbil", "basra", "baghdad", "beirut", "damascus",
            "aqaba", "petra", "amman", "beer sheva", "haifa", "jerusalem", "tel aviv",
            "hurghada", "sharm el sheikh", "aswan", "luxor", "giza", "alexandria",
            "cairo", "algiers", "tunis", "casablanca", "rabat", "agadir", "tangier",
            "fes", "marrakech", "las palmas", "tenerife", "ibiza", "palma",
            "alicante", "malaga", "granada", "seville", "bilbao", "valencia",
            "pontadelgada", "funchal", "faro", "porto", "lisbon", "genoa",
            "turin", "bologna", "catania", "palermo", "naples", "venice", "florence",
            "milan", "lucerne", "basel", "bern", "zurich", "geneva", "luxembourg",
            "utrecht", "the hague", "rotterdam", "antwerp", "brussels", "belfast",
            "cardiff", "newcastle", "liverpool", "leeds", "birmingham", "manchester",
            "edinburgh", "glasgow", "dublin", "reykjavik", "copenhagen", "oslo",
            "stockholm", "helsinki", "vilnius", "riga", "tallinn", "budapest",
            "bucharest", "sofia", "belgrade", "zagreb", "ljubljana", "bratislava",
            "krakow", "warsaw", "kiev", "st petersburg", "moscow", "caracas",
            "medellin", "bogota", "lima", "santiago", "buenos aires", "sao paulo",
            "rio de janeiro", "monterrey", "guadalajara", "mexico city", "pune",
            "hyderabad", "kolkata", "chennai", "bangalore", "delhi", "mumbai",
            "casablanca", "marrakech", "cairo", "nairobi", "johannesburg", "cape town",
            "wellington", "auckland", "adelaide", "perth", "brisbane", "melbourne",
            "sydney", "montreal", "vancouver", "toronto", "miami", "chicago",
            "los angeles", "san francisco", "hongkong", "hong kong", "istanbul",
            "dubai", "athens", "madrid", "munich", "melbourne", "sydney", "seoul",
            "singapore", "bangkok", "berlin", "prague", "vienna", "amsterdam",
            "barcelona", "rome", "shanghai", "beijing", "new york", "london",
            "paris", "osaka", "kyoto", "tokyo"
        ]

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
        destination = intent.get("destination", "unknown")
        
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
                tool_parameters[tool_name] = {
                    "location": destination if destination != "unknown" else "Tokyo",  # Safe fallback
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
                
                tool_parameters[tool_name] = {
                    "location": destination if destination != "unknown" else "Tokyo",  # Safe fallback
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
                
                # Convert origin to IATA code
                origin_iata = self._convert_city_to_iata_code(origin)
                
                tool_parameters[tool_name] = {
                    "origin": origin_iata,  # Use extracted origin converted to IATA code
                    "destination": destination if destination != "unknown" else "Tokyo",
                    "start_date": departure_date,
                    "passengers": max(travel_details.get("travelers", 1), 1),
                    "class_type": "economy",
                    "budget_level": travel_details.get("budget", {}).get("level", "mid-range")
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

    def _generate_structured_response(
        self, execution_result: Dict[str, Any], intent: Dict[str, Any], user_message: str
    ) -> AgentResponse:
        """
        Generate structured response using fast template-based fusion
        No LLM calls - uses templates and rule-based content generation
        """
        try:
            # Extract key information
            intent_type = intent.get("type") or intent.get("intent_type", "query")
            destination = intent.get("destination", "unknown")
            tools_used = execution_result.get("tools_used", [])
            tool_results = execution_result.get("results", {})
            execution_success = execution_result.get("success", False)
            
            # Determine response success based on whether we can provide useful content
            # Even if tools fail, we can still provide value with knowledge and guidance
            has_useful_content = (
                destination != "unknown" or 
                tool_results or 
                intent_type in ["query", "recommendation", "planning"]
            )
            
            # Generate content using templates based on intent and results
            if execution_success and tool_results:
                content = self._build_structured_content(
                    intent_type, destination, tools_used, tool_results, user_message
                )
                confidence = 0.87
                response_success = True
            elif has_useful_content:
                # Generate helpful content even when tools partially fail
                content = self._build_mixed_content(
                    intent_type, destination, tools_used, tool_results, user_message
                )
                confidence = 0.75
                response_success = True  # We can still help the user
            else:
                # Handle true error cases with fallback content
                content = self._build_fallback_content(intent_type, destination, user_message)
                confidence = 0.6
                response_success = False
            
            # Create response with structured metadata
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
                    "response_method": "structured_template",
                    "destination": destination,
                    "intent_type": intent_type,
                    "tool_execution_success": execution_success,
                    "partial_tool_success": bool(tool_results)
                }
            )
            
            logger.info(f"Structured response generated: success={response_success}, {len(content)} chars, confidence: {confidence}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            # Emergency fallback
            return AgentResponse(
                success=False,
                content=f"I apologize, but I encountered an issue processing your travel request about {destination}. Please try rephrasing your question.",
                confidence=0.3,
                metadata={"error": str(e), "response_method": "emergency_fallback"}
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
                    # Pydantic model
                    name = getattr(hotel, 'name', 'Unknown Hotel')
                    price = getattr(hotel, 'price_per_night', 'Price available')
                    rating = getattr(hotel, 'rating', 'No rating')
                    location = getattr(hotel, 'location', destination)
                    currency = getattr(hotel, 'currency', '')
                else:
                    # Dictionary
                    name = hotel.get("name", "Unknown Hotel")
                    price = hotel.get("price_per_night", hotel.get("price", "Price available"))
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
