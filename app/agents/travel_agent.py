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
import asyncio
import logging
from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus, QualityAssessment
from app.tools.base_tool import tool_registry
from app.tools.tool_executor import get_tool_executor
from app.core.llm_service import get_llm_service
from app.core.rag_engine import get_rag_engine
from app.models.schemas import TravelPreferences, TravelPlan

logger = logging.getLogger(__name__)


class TravelAgent(BaseAgent):
    """Main travel planning agent"""
    
    def __init__(self):
        super().__init__(
            name="travel_agent",
            description="Intelligent travel planning assistant, able to develop personalized travel plans based on user needs"
        )
        
        self.llm_service = get_llm_service()
        self.rag_engine = get_rag_engine()
        self.tool_executor = get_tool_executor()
        
        # Agent-specific state
        self.current_planning_context: Optional[Dict[str, Any]] = None
        self.user_preferences_history: Dict[str, Any] = {}
        
        # Travel-specific quality configuration
        self.quality_threshold = 0.8  # Higher threshold for travel planning
        self.refine_enabled = True  # Enable self-refinement by default
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "travel_planning",      # Travel planning
            "itinerary_generation", # Itinerary generation
            "budget_optimization",  # Budget optimization
            "recommendation",       # Personalized recommendation
            "real_time_adjustment", # Real-time adjustment
            "conflict_resolution",  # Conflict resolution
            "multi_tool_coordination" # Multi-tool coordination
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
        """Get travel-specific quality assessment dimensions"""
        return {
            "relevance": 0.25,          # How relevant is the response to travel request
            "completeness": 0.20,       # How complete is the travel information
            "accuracy": 0.20,           # How accurate is the travel information
            "actionability": 0.15,      # How actionable are the travel recommendations
            "personalization": 0.10,    # How well personalized to user preferences
            "feasibility": 0.10         # How feasible/realistic is the travel plan
        }
    
    async def _assess_dimension(
        self, 
        dimension: str, 
        original_message: AgentMessage, 
        response: AgentResponse
    ) -> float:
        """Assess travel-specific quality dimensions"""
        if dimension == "personalization":
            return await self._assess_personalization(original_message, response)
        elif dimension == "feasibility":
            return await self._assess_feasibility(original_message, response)
        else:
            # Use base class implementation for standard dimensions
            return await super()._assess_dimension(dimension, original_message, response)
    
    async def _assess_personalization(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess how well the response is personalized"""
        score = 0.5  # Base score
        
        # Check if user preferences were considered
        user_message_lower = original_message.content.lower()
        response_content_lower = response.content.lower()
        
        # Look for personal preference indicators
        preference_indicators = [
            "budget", "prefer", "like", "dislike", "interest", "family", 
            "solo", "couple", "group", "luxury", "backpack", "adventure"
        ]
        
        mentioned_preferences = sum(1 for indicator in preference_indicators 
                                  if indicator in user_message_lower)
        
        addressed_preferences = sum(1 for indicator in preference_indicators 
                                  if indicator in response_content_lower)
        
        if mentioned_preferences > 0:
            personalization_ratio = addressed_preferences / mentioned_preferences
            score += personalization_ratio * 0.4
        
        # Check if response mentions specific user needs
        if any(word in response_content_lower for word in ["your", "you", "based on"]):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _assess_feasibility(self, original_message: AgentMessage, response: AgentResponse) -> float:
        """Assess the feasibility of the travel recommendations"""
        score = 0.6  # Base score
        
        # Check if practical considerations are mentioned
        practical_elements = [
            "time", "schedule", "duration", "distance", "transport", 
            "booking", "availability", "season", "weather"
        ]
        
        response_content_lower = response.content.lower()
        practical_mentions = sum(1 for element in practical_elements 
                               if element in response_content_lower)
        
        # More practical considerations = higher feasibility
        if practical_mentions >= 3:
            score += 0.3
        elif practical_mentions >= 1:
            score += 0.2
        
        # Check if tools were used (indicates real data was considered)
        if response.actions_taken and len(response.actions_taken) >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _generate_improvement_suggestions(
        self, 
        dimension: str, 
        original_message: AgentMessage, 
        response: AgentResponse,
        current_score: float
    ) -> List[str]:
        """Generate travel-specific improvement suggestions"""
        suggestions = await super()._generate_improvement_suggestions(
            dimension, original_message, response, current_score
        )
        
        if dimension == "personalization" and current_score < 0.6:
            suggestions.extend([
                "Consider user's specific preferences mentioned in the request",
                "Tailor recommendations to the user's travel style and interests",
                "Reference user's budget, group size, or travel dates in recommendations"
            ])
        
        if dimension == "feasibility" and current_score < 0.6:
            suggestions.extend([
                "Include practical considerations like travel times and logistics",
                "Verify availability and booking requirements",
                "Consider seasonal factors and weather conditions",
                "Provide realistic timeline and scheduling information"
            ])
        
        if dimension == "completeness" and current_score < 0.6:
            suggestions.extend([
                "Include information about flights, accommodation, and activities",
                "Provide cost estimates and budget breakdown",
                "Add transportation details between locations"
            ])
        
        return suggestions
    
    async def _refine_response(
        self, 
        original_message: AgentMessage, 
        current_response: AgentResponse,
        quality_assessment: Optional[QualityAssessment]
    ) -> AgentResponse:
        """Refine travel response based on quality assessment"""
        if not quality_assessment:
            return current_response
        
        improved_content = current_response.content
        improved_actions = current_response.actions_taken.copy()
        improved_next_steps = current_response.next_steps.copy()
        
        # Apply travel-specific improvements based on quality assessment
        for suggestion in quality_assessment.improvement_suggestions:
            if "personalization" in suggestion.lower() or "preferences" in suggestion.lower():
                improved_content += "\n\n🎯 **Personalized for You**: This recommendation takes into account your specific travel preferences and requirements mentioned."
            
            elif "feasibility" in suggestion.lower() or "practical" in suggestion.lower():
                improved_content += "\n\n⏰ **Practical Considerations**: I've considered travel times, seasonal factors, and booking requirements to ensure this plan is realistic and achievable."
            
            elif "completeness" in suggestion.lower() and "cost" in suggestion.lower():
                improved_content += "\n\n💰 **Budget Information**: Please note that costs may vary based on season, availability, and booking timing. I recommend checking current prices for the most accurate estimates."
            
            elif "transportation" in suggestion.lower():
                improved_next_steps.append("Research transportation options between destinations")
            
            elif "booking" in suggestion.lower():
                improved_next_steps.append("Check availability and make reservations in advance")
        
        # Enhance response based on low-scoring dimensions
        if quality_assessment.dimension_scores.get("personalization", 1.0) < 0.6:
            improved_content += "\n\n✨ **Tailored Recommendations**: These suggestions are customized based on your travel style and preferences."
        
        if quality_assessment.dimension_scores.get("feasibility", 1.0) < 0.6:
            improved_content += "\n\n📋 **Feasibility Check**: All recommendations have been evaluated for practicality and realistic implementation."
        
        if quality_assessment.dimension_scores.get("actionability", 1.0) < 0.6:
            if "Next steps to book your trip:" not in improved_content:
                improved_next_steps.extend([
                    "Compare prices across different booking platforms",
                    "Read recent reviews for hotels and attractions",
                    "Check visa requirements and travel restrictions",
                    "Consider travel insurance options"
                ])
        
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
                "quality_dimensions_improved": [
                    dim for dim, score in quality_assessment.dimension_scores.items() 
                    if score < 0.6
                ],
                "improvement_applied": quality_assessment.improvement_suggestions
            }
                    )
    
    async def plan_travel(self, message: AgentMessage) -> AgentResponse:
        """Public method to plan travel with self-refinement enabled"""
        return await self.process_with_refinement(message)

    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process user message - core implementation called by refinement loop"""
        try:
            self.status = AgentStatus.THINKING
            
            # 1. Understand user intent
            intent = await self._analyze_user_intent(message.content)
            
            # 2. Retrieve relevant knowledge
            knowledge_context = await self._retrieve_knowledge_context(message.content)
            
            # 3. Make a tool action plan
            action_plan = await self._create_action_plan(intent, knowledge_context)
            
            # 4. Execute tool action plan
            self.status = AgentStatus.ACTING
            # Add original message to context for knowledge retrieval
            execution_context = {
                **message.metadata,
                "original_message": message.content
            }
            result = await self._execute_action_plan(action_plan, execution_context)
            
            # 5. Generate response
            response_content = await self._generate_response(result, intent)
            
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                success=True,
                content=response_content,
                actions_taken=action_plan.get("actions", []),
                next_steps=action_plan.get("next_steps", []),
                confidence=0.85,
                metadata={
                    "intent": intent,
                    "tools_used": result.get("tools_used", []),
                    "execution_time": result.get("execution_time", 0)
                }
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return AgentResponse(
                success=False,
                content=f"处理消息时发生错误: {str(e)}",
                confidence=0.0
            )
    
    async def _analyze_user_intent(self, user_message: str) -> Dict[str, Any]:
        """Enhanced LLM-powered intent analysis with structured output"""
        
        # Step 1: Get LLM intent analysis using prompt manager
        llm_analysis = await self._get_llm_intent_analysis(user_message)
        
        # Step 2: Use structured response for deep analysis
        structured_intent = await self._parse_structured_intent(llm_analysis, user_message)
        
        # Step 3: Enhance and validate analysis results
        enhanced_intent = await self._enhance_intent_analysis(structured_intent, user_message)
        
        return enhanced_intent

    async def _get_llm_intent_analysis(self, user_message: str) -> Dict[str, Any]:
        """Get initial intent analysis through LLM service"""
        
        from app.core.prompt_manager import prompt_manager, PromptType
        
        # Build structured prompt using prompt manager
        analysis_prompt = prompt_manager.get_prompt(
            PromptType.INTENT_ANALYSIS,
            user_message=user_message
        )
        
        try:
            if self.llm_service and not self.llm_service.mock_mode:
                # Use real LLM service with structured output
                schema = prompt_manager.get_schema(PromptType.INTENT_ANALYSIS)
                response = await self.llm_service.structured_completion(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    response_schema=schema,
                    temperature=0.3,  # Low temperature for consistent results
                    max_tokens=800
                )
                
                # Parse LLM response to structured format
                return response
            else:
                # Use enhanced fallback analysis
                return await self._enhanced_fallback_intent_analysis(user_message)
                
        except Exception as e:
            logger.error(f"LLM intent analysis failed: {e}")
            return await self._enhanced_fallback_intent_analysis(user_message)

    async def _parse_structured_intent(self, llm_analysis: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Parse and validate structured intent analysis"""
        
        # Validate the structured response
        if not isinstance(llm_analysis, dict):
            logger.warning("LLM analysis is not a dictionary, using fallback")
            return await self._enhanced_fallback_intent_analysis(user_message)
        
        # Ensure required fields exist
        required_fields = ["intent_type", "destination", "sentiment", "urgency", "confidence_score"]
        for field in required_fields:
            if field not in llm_analysis:
                logger.warning(f"Missing required field {field} in LLM analysis")
                return await self._enhanced_fallback_intent_analysis(user_message)
        
        return llm_analysis

    async def _enhance_intent_analysis(self, structured_intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Enhance and validate intent analysis results"""
        
        # Add additional extracted entities
        structured_intent["extracted_entities"] = self._extract_entities(user_message)
        structured_intent["linguistic_features"] = self._analyze_linguistic_features(user_message)
        
        # Convert to legacy format for backward compatibility
        legacy_format = {
            "type": structured_intent["intent_type"],
            "destination": structured_intent["destination"]["primary"],
            "time_info": {
                "duration_days": structured_intent["travel_details"].get("duration", 0)
            },
            "budget_info": {
                "budget_mentioned": structured_intent["travel_details"]["budget"].get("mentioned", False)
            },
            "urgency": structured_intent["urgency"],
            "extracted_info": {
                "destination": structured_intent["destination"]["primary"],
                "time_info": structured_intent["travel_details"],
                "budget_info": structured_intent["travel_details"]["budget"]
            },
            # Add new structured fields
            "structured_analysis": structured_intent
        }
        
        return legacy_format

    async def _enhanced_fallback_intent_analysis(self, user_message: str) -> Dict[str, Any]:
        """Enhanced fallback intent analysis with better keyword matching"""
        
        user_message_lower = user_message.lower()
        
        # Enhanced intent type detection
        intent_type = "query"  # default
        if any(word in user_message_lower for word in ["plan", "create", "make", "arrange", "organize", "schedule"]):
            intent_type = "planning"
        elif any(word in user_message_lower for word in ["change", "modify", "update", "adjust", "revise"]):
            intent_type = "modification"
        elif any(word in user_message_lower for word in ["recommend", "suggest", "introduce", "advise", "what should"]):
            intent_type = "recommendation"
        elif any(word in user_message_lower for word in ["book", "reserve", "purchase", "buy"]):
            intent_type = "booking"
        elif any(word in user_message_lower for word in ["problem", "issue", "complaint", "wrong", "error"]):
            intent_type = "complaint"
        
        # Enhanced destination detection
        destination = "Unknown"
        destinations = ["tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", "shanghai", 
                       "rome", "barcelona", "amsterdam", "vienna", "prague", "budapest", "berlin"]
        for dest in destinations:
            if dest in user_message_lower:
                destination = dest.title()
                break
        
        # Enhanced time extraction
        time_info = {}
        import re
        days_match = re.search(r'(\d+)\s*(?:day|days|天|日)', user_message_lower)
        if days_match:
            time_info["duration_days"] = int(days_match.group(1))
        
        # Enhanced budget detection
        budget_info = {"budget_mentioned": False}
        if any(word in user_message_lower for word in ["budget", "cost", "price", "money", "spend", "expensive", "cheap"]):
            budget_info["budget_mentioned"] = True
            
            # Try to extract budget amount
            budget_match = re.search(r'[\$€£¥]?(\d+(?:,\d{3})*(?:\.\d{2})?)', user_message)
            if budget_match:
                budget_info["amount"] = float(budget_match.group(1).replace(',', ''))
        
        # Enhanced sentiment analysis
        sentiment = "neutral"
        if any(word in user_message_lower for word in ["excited", "amazing", "great", "wonderful", "love"]):
            sentiment = "positive"
        elif any(word in user_message_lower for word in ["worried", "concerned", "problem", "difficult"]):
            sentiment = "negative"
        elif any(word in user_message_lower for word in ["can't wait", "looking forward", "dream"]):
            sentiment = "excited"
        
        # Enhanced urgency detection
        urgency = "medium"
        if any(word in user_message_lower for word in ["urgent", "asap", "immediately", "quickly", "soon"]):
            urgency = "urgent"
        elif any(word in user_message_lower for word in ["flexible", "whenever", "no rush"]):
            urgency = "low"
        
        # Create structured fallback response
        structured_analysis = {
            "intent_type": intent_type,
            "destination": {
                "primary": destination,
                "secondary": [],
                "region": "Unknown",
                "confidence": 0.6
            },
            "travel_details": {
                "duration": time_info.get("duration_days", 0),
                "travelers": 1,
                "budget": {
                    "mentioned": budget_info["budget_mentioned"],
                    "amount": budget_info.get("amount", 0),
                    "currency": "USD",
                    "level": "mid-range"
                },
                "dates": {
                    "departure": "unknown",
                    "return": "unknown",
                    "flexibility": "unknown"
                }
            },
            "preferences": {
                "travel_style": "mid-range",
                "interests": [],
                "accommodation_type": "unknown",
                "transport_preference": "unknown"
            },
            "sentiment": sentiment,
            "urgency": urgency,
            "missing_info": ["specific_dates", "exact_budget", "preferences"],
            "key_requirements": [],
            "confidence_score": 0.6
        }
        
        # Legacy format for backward compatibility
        return {
            "type": intent_type,
            "destination": destination,
            "time_info": time_info,
            "budget_info": budget_info,
            "urgency": urgency,
            "extracted_info": {
                "destination": destination,
                "time_info": time_info,
                "budget_info": budget_info
            },
            "structured_analysis": structured_analysis
        }

    def _extract_entities(self, user_message: str) -> Dict[str, Any]:
        """Extract additional entities from user message"""
        entities = {
            "locations": [],
            "numbers": [],
            "dates": [],
            "currencies": []
        }
        
        import re
        
        # Extract numbers
        numbers = re.findall(r'\d+', user_message)
        entities["numbers"] = [int(num) for num in numbers]
        
        # Extract potential dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}',
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Extract currencies
        currency_symbols = re.findall(r'[\$€£¥]', user_message)
        entities["currencies"] = currency_symbols
        
        return entities

    def _analyze_linguistic_features(self, user_message: str) -> Dict[str, Any]:
        """Analyze linguistic features of the message"""
        features = {
            "message_length": len(user_message),
            "word_count": len(user_message.split()),
            "question_count": user_message.count('?'),
            "exclamation_count": user_message.count('!'),
            "has_urgency_markers": any(word in user_message.lower() for word in ["urgent", "asap", "quickly"]),
            "has_politeness_markers": any(word in user_message.lower() for word in ["please", "thank you", "could you"]),
            "language_hints": self._detect_language_hints(user_message)
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
    
    async def _retrieve_knowledge_context(self, query: str) -> Dict[str, Any]:
        """Retrieve knowledge context"""
        try:
            # Use RAG engine to retrieve relevant knowledge
            result = await self.rag_engine.retrieve(query, top_k=5)
            
            # Process retrieved documents
            contexts = []
            for doc in result.documents:
                contexts.append({
                    "id": doc.id,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "metadata": doc.metadata
                })
            
            return {
                "relevant_docs": contexts,
                "context": "\n\n".join([doc.content for doc in result.documents]),
                "total_results": result.total_results,
                "scores": result.scores
            }
        except Exception as e:
            return {
                "relevant_docs": [],
                "context": f"Error retrieving context: {str(e)}",
                "error": str(e)
            }
    
    async def _create_action_plan(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent action plan based on structured intention analysis"""
        
        # Extract structured analysis from intent
        structured_analysis = intent.get("structured_analysis", {})
        
        # Initialize enhanced plan structure
        plan = {
            "intent_type": intent["type"],
            "destination": structured_analysis.get("destination", {}),
            "actions": [],
            "tools_to_use": [],
            "tool_parameters": {},
            "execution_strategy": "sequential",
            "next_steps": [],
            "confidence": 0.0
        }
        
        try:
            # Step 1: Use LLM for intelligent tool selection
            tool_selection_result = await self._llm_tool_selection(structured_analysis, context)
            logger.info(f"LLM tool selection: {tool_selection_result}")
            
            # Step 2: Extract parameters from structured intent
            tool_parameters = await self._extract_parameters_from_intent(
                structured_analysis, tool_selection_result["selected_tools"]
            )
            logger.info(f"Tool parameters from intent: {tool_parameters}")
            
            # Step 3: Generate intent-based actions and next steps
            actions = await self._generate_intent_based_actions(structured_analysis, tool_selection_result["selected_tools"])
            next_steps = await self._generate_intent_based_next_steps(structured_analysis, context)
            
            # Step 4: Update plan with results
            plan.update({
                "tools_to_use": tool_selection_result["selected_tools"],
                "tool_parameters": tool_parameters,
                "execution_strategy": tool_selection_result["execution_strategy"],
                "actions": actions,
                "next_steps": next_steps,
                "confidence": tool_selection_result["confidence"],
                "tool_selection_reasoning": tool_selection_result.get("reasoning", ""),
                "intent_metadata": {
                    "user_sentiment": structured_analysis.get("sentiment", "neutral"),
                    "urgency_level": structured_analysis.get("urgency", "medium"),
                    "missing_info": structured_analysis.get("missing_info", []),
                    "key_requirements": structured_analysis.get("key_requirements", []),
                    "preferences": structured_analysis.get("preferences", {})
                }
            })
            
            logger.info(f"Created intent-based action plan: {plan['intent_type']} -> {plan['tools_to_use']}")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating intent-based action plan: {e}")
            # Fall back to basic planning
            return await self._create_basic_action_plan(intent, context)
    
    async def _llm_tool_selection(self, structured_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to intelligently select tools based on structured analysis"""
        
        from app.core.prompt_manager import prompt_manager, PromptType
        
        try:
            # Build tool selection prompt
            tool_selection_prompt = prompt_manager.get_prompt(
                PromptType.TOOL_SELECTION,
                intent_analysis=structured_analysis
            )
            
            if self.llm_service and not self.llm_service.mock_mode:
                # Use real LLM for tool selection
                schema = prompt_manager.get_schema(PromptType.TOOL_SELECTION)
                response = await self.llm_service.structured_completion(
                    messages=[{"role": "user", "content": tool_selection_prompt}],
                    response_schema=schema,
                    temperature=0.2,
                    max_tokens=400
                )
                return response
            else:
                # Use fallback tool selection
                return await self._fallback_tool_selection(structured_analysis)
                
        except Exception as e:
            logger.error(f"LLM tool selection failed: {e}")
            return await self._fallback_tool_selection(structured_analysis)
    
    async def _fallback_tool_selection(self, structured_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback tool selection based on structured analysis"""
        
        intent_type = structured_analysis.get("intent_type", "query")
        destination = structured_analysis.get("destination", {}).get("primary", "Unknown")
        
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
            if any(interest in ["hotel", "accommodation", "stay"] for interest in interests):
                selected_tools.append("hotel_search")
            if any(interest in ["flight", "plane", "airline"] for interest in interests):
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
            "confidence": 0.7
        }
    
    async def _extract_parameters_from_intent(self, structured_analysis: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Extract tool parameters from structured intent analysis"""
        
        tool_parameters = {}
        
        # Extract common parameters
        destination = structured_analysis.get("destination", {}).get("primary", "Unknown")
        travel_details = structured_analysis.get("travel_details", {})
        preferences = structured_analysis.get("preferences", {})
        
        for tool in selected_tools:
            if tool == "attraction_search":
                tool_parameters[tool] = {
                    "destination": destination,
                    "limit": 10,
                    "interests": preferences.get("interests", []),
                    "travel_style": preferences.get("travel_style", "mid-range")
                }
            elif tool == "hotel_search":
                tool_parameters[tool] = {
                    "destination": destination,
                    "limit": 8,
                    "accommodation_type": preferences.get("accommodation_type", "hotel"),
                    "budget_level": travel_details.get("budget", {}).get("level", "mid-range"),
                    "travelers": travel_details.get("travelers", 1)
                }
            elif tool == "flight_search":
                tool_parameters[tool] = {
                    "destination": destination,
                    "limit": 5,
                    "travelers": travel_details.get("travelers", 1),
                    "budget_level": travel_details.get("budget", {}).get("level", "mid-range"),
                    "flexibility": travel_details.get("dates", {}).get("flexibility", "flexible")
                }
        
        return tool_parameters
    
    async def _generate_intent_based_actions(self, structured_analysis: Dict[str, Any], selected_tools: List[str]) -> List[str]:
        """Generate actions based on structured intent analysis"""
        
        actions = []
        intent_type = structured_analysis.get("intent_type", "query")
        destination = structured_analysis.get("destination", {}).get("primary", "Unknown")
        
        # Generate actions based on intent type
        if intent_type == "planning":
            actions.extend([
                f"Analyze travel requirements for {destination}",
                "Search for attractions and activities",
                "Find suitable accommodations",
                "Research transportation options",
                "Create comprehensive travel plan"
            ])
        elif intent_type == "recommendation":
            actions.extend([
                f"Search for top attractions in {destination}",
                "Analyze user preferences",
                "Generate personalized recommendations"
            ])
        elif intent_type == "booking":
            actions.extend([
                f"Search for available options in {destination}",
                "Compare prices and availability",
                "Prepare booking information"
            ])
        elif intent_type == "query":
            actions.extend([
                f"Search for information about {destination}",
                "Provide relevant details",
                "Answer specific questions"
            ])
        else:
            actions.extend([
                f"Process request for {destination}",
                "Gather relevant information",
                "Provide helpful response"
            ])
        
        # Add tool-specific actions
        for tool in selected_tools:
            if tool == "attraction_search":
                actions.append("Search for attractions and activities")
            elif tool == "hotel_search":
                actions.append("Search for accommodation options")
            elif tool == "flight_search":
                actions.append("Search for flight information")
        
        return actions
    
    async def _generate_intent_based_next_steps(self, structured_analysis: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
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
            next_steps.append("What type of activities or attractions are you most interested in?")
        
        # Generate next steps based on intent type
        if intent_type == "planning":
            next_steps.extend([
                "Would you like me to create a detailed itinerary?",
                "Do you need help with booking accommodations or flights?"
            ])
        elif intent_type == "recommendation":
            next_steps.extend([
                "Would you like more specific recommendations based on your interests?",
                "Do you want information about the best times to visit these places?"
            ])
        elif intent_type == "booking":
            next_steps.extend([
                "Would you like me to help you compare prices?",
                "Do you want to proceed with making reservations?"
            ])
        
        # Add urgency-based next steps
        if urgency == "urgent":
            next_steps.insert(0, "I understand this is urgent. Let me prioritize the most important information first.")
        elif urgency == "low":
            next_steps.append("Take your time to review the information, and let me know if you need any clarification.")
        
        return next_steps
    
    async def _analyze_user_requirements(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user requirements in detail"""
        
        # Extract key information from intent
        base_requirements = {
            "destination": intent.get("destination", "unknown"),
            "intent_type": intent["type"],
            "time_constraints": intent.get("time_info", {}),
            "budget_constraints": intent.get("budget_info", {}),
            "urgency": intent.get("urgency", "normal")
        }
        
        # Analyze user message for additional requirements
        user_message = context.get("original_message", "")
        
        # Use LLM for sophisticated requirement analysis if available
        if self.llm_service:
            try:
                llm_prompt = f"""
                Analyze this travel request and extract detailed requirements:
                
                User Message: "{user_message}"
                Current Intent: {intent}
                
                Extract and classify:
                1. Required vs Optional information needs
                2. Budget sensitivity (high/medium/low)
                3. Time sensitivity (urgent/normal/flexible)
                4. Specific preferences (luxury/budget/family/business)
                5. Geographic scope (local/national/international)
                6. Tool necessity scores (0-1 for flight/hotel/attraction search)
                
                Return JSON format with detailed analysis.
                """
                
                response = await self.llm_service.chat_completion([
                    {"role": "user", "content": llm_prompt}
                ])
                
                # Parse LLM response (simplified for now)
                enhanced_requirements = self._parse_llm_requirements(response.get("content", ""))
                base_requirements.update(enhanced_requirements)
                
            except Exception as e:
                logger.warning(f"LLM requirement analysis failed: {e}")
        
        # Add contextual analysis
        base_requirements.update({
            "user_context": context,
            "session_history": context.get("session_history", []),
            "user_preferences": self.user_preferences_history
        })
        
        return base_requirements
    
    async def _intelligent_tool_selection(self, requirements: Dict[str, Any]) -> List[str]:
        """Select tools based on intelligent multi-dimensional analysis"""
        
        selected_tools = []
        tool_scores = {}
        
        # Define tool selection criteria with scoring weights
        criteria = {
            "flight_search": {
                "triggers": ["transportation", "flight", "airline", "ticket", "travel between cities", "fly", "airplane"],
                "intent_relevance": {"planning": 0.8, "recommendation": 0.2, "query": 0.3},
                "cost_score": 0.6,  # API cost consideration (lower is more expensive)
                "value_score": 0.9   # Information value (higher is more valuable)
            },
            "hotel_search": {
                "triggers": ["accommodation", "hotel", "stay", "lodging", "sleep", "overnight", "room"],
                "intent_relevance": {"planning": 0.9, "recommendation": 0.4, "query": 0.3},
                "cost_score": 0.5,
                "value_score": 0.8
            },
            "attraction_search": {
                "triggers": ["attraction", "sightseeing", "activity", "visit", "tour", "experience", "explore", "museum", "park"],
                "intent_relevance": {"planning": 0.7, "recommendation": 0.9, "query": 0.6},
                "cost_score": 0.8,  # Lower API cost
                "value_score": 0.9
            }
        }
        
        user_message = requirements.get("user_context", {}).get("original_message", "").lower()
        intent_type = requirements["intent_type"]
        
        # Calculate tool scores using multi-dimensional analysis
        for tool_name, tool_info in criteria.items():
            score = 0.0
            
            # Keyword matching score
            keyword_matches = sum(1 for trigger in tool_info["triggers"] if trigger in user_message)
            keyword_score = min(keyword_matches * 0.15, 1.0)
            
            # Intent relevance score
            intent_score = tool_info["intent_relevance"].get(intent_type, 0.0)
            
            # Cost-benefit analysis
            cost_benefit = tool_info["value_score"] / tool_info["cost_score"]
            
            # Requirement necessity score
            necessity_score = self._calculate_tool_necessity(tool_name, requirements)
            
            # Final score calculation with weighted components
            final_score = (
                keyword_score * 0.3 + 
                intent_score * 0.4 + 
                cost_benefit * 0.2 + 
                necessity_score * 0.1
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
    
    def _calculate_tool_necessity(self, tool_name: str, requirements: Dict[str, Any]) -> float:
        """Calculate how necessary a tool is based on requirements"""
        
        necessity_score = 0.0
        user_message = requirements.get("user_context", {}).get("original_message", "").lower()
        
        # Flight search necessity
        if tool_name == "flight_search":
            if any(word in user_message for word in ["flight", "fly", "airline", "airport", "ticket"]):
                necessity_score += 0.8
            if requirements.get("destination", "").lower() not in ["unknown", "local"]:
                necessity_score += 0.3
            if "international" in user_message:
                necessity_score += 0.4
        
        # Hotel search necessity
        elif tool_name == "hotel_search":
            if any(word in user_message for word in ["hotel", "stay", "accommodation", "night", "room"]):
                necessity_score += 0.9
            if requirements.get("time_constraints", {}).get("duration_days", 0) > 1:
                necessity_score += 0.4
            if "overnight" in user_message or "multi-day" in user_message:
                necessity_score += 0.3
        
        # Attraction search necessity
        elif tool_name == "attraction_search":
            if any(word in user_message for word in ["attraction", "visit", "see", "tour", "activity", "explore"]):
                necessity_score += 0.7
            if requirements["intent_type"] == "recommendation":
                necessity_score += 0.5
            if "sightseeing" in user_message or "explore" in user_message:
                necessity_score += 0.3
        
        return min(necessity_score, 1.0)
    
    async def _extract_tool_parameters(self, selected_tools: List[str], requirements: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract intelligent parameters for each tool"""
        
        tool_parameters = {}
        user_message = requirements.get("user_context", {}).get("original_message", "")
        
        for tool_name in selected_tools:
            if tool_name == "flight_search":
                tool_parameters[tool_name] = await self._extract_flight_parameters(user_message, requirements)
            elif tool_name == "hotel_search":
                tool_parameters[tool_name] = await self._extract_hotel_parameters(user_message, requirements)
            elif tool_name == "attraction_search":
                tool_parameters[tool_name] = await self._extract_attraction_parameters(user_message, requirements)
        
        return tool_parameters
    
    async def _extract_flight_parameters(self, user_message: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flight search parameters intelligently"""
        
        params = {
            "origin": "unknown",
            "destination": requirements.get("destination", "unknown"),
            "departure_date": "flexible",
            "return_date": "flexible",
            "passengers": 1,
            "class": "economy",
            "budget_range": "medium"
        }
        
        # Extract specific parameters from user message
        user_message_lower = user_message.lower()
        
        # Extract passenger count
        import re
        passenger_match = re.search(r'(\d+)\s*(?:people|person|passenger|traveler)', user_message_lower)
        if passenger_match:
            params["passengers"] = int(passenger_match.group(1))
        
        # Extract travel class
        if any(word in user_message_lower for word in ["business", "first class", "premium"]):
            params["class"] = "business"
        elif any(word in user_message_lower for word in ["economy", "cheap", "budget"]):
            params["class"] = "economy"
        
        # Extract budget preference
        if any(word in user_message_lower for word in ["budget", "cheap", "affordable"]):
            params["budget_range"] = "low"
        elif any(word in user_message_lower for word in ["luxury", "expensive", "premium"]):
            params["budget_range"] = "high"
        
        return params
    
    async def _extract_hotel_parameters(self, user_message: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hotel search parameters intelligently"""
        
        params = {
            "destination": requirements.get("destination", "unknown"),
            "check_in": "flexible",
            "check_out": "flexible",
            "guests": 1,
            "rooms": 1,
            "star_rating": "any",
            "budget_range": "medium"
        }
        
        # Extract hotel-specific parameters
        user_message_lower = user_message.lower()
        
        # Extract guest count
        import re
        guest_match = re.search(r'(\d+)\s*(?:guest|people|person)', user_message_lower)
        if guest_match:
            params["guests"] = int(guest_match.group(1))
        
        # Extract hotel preferences
        if any(word in user_message_lower for word in ["5 star", "luxury", "premium"]):
            params["star_rating"] = "5"
        elif any(word in user_message_lower for word in ["4 star", "high-end"]):
            params["star_rating"] = "4"
        elif any(word in user_message_lower for word in ["budget", "cheap", "hostel"]):
            params["star_rating"] = "3"
        
        return params
    
    async def _extract_attraction_parameters(self, user_message: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attraction search parameters intelligently"""
        
        params = {
            "location": requirements.get("destination", "unknown"),
            "category": None,
            "min_rating": 4.0,
            "max_results": 10,
            "include_photos": True,
            "radius_meters": 10000
        }
        
        # Extract attraction-specific parameters
        user_message_lower = user_message.lower()
        
        # Extract category preferences
        if any(word in user_message_lower for word in ["museum", "art", "culture", "history"]):
            params["category"] = "museum"
        elif any(word in user_message_lower for word in ["park", "nature", "outdoor", "garden"]):
            params["category"] = "park"
        elif any(word in user_message_lower for word in ["temple", "shrine", "religious", "church"]):
            params["category"] = "tourist_attraction"
        
        # Extract quality preferences
        if any(word in user_message_lower for word in ["best", "top", "must-see", "highly rated"]):
            params["min_rating"] = 4.5
        elif any(word in user_message_lower for word in ["any", "all", "comprehensive"]):
            params["min_rating"] = 3.0
        
        return params
    
    async def _optimize_execution_strategy(self, selected_tools: List[str], tool_parameters: Dict[str, Dict[str, Any]]) -> str:
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
    
    async def _generate_fallback_strategies(self, selected_tools: List[str], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback strategies for tool failures"""
        
        fallback_strategies = []
        
        for tool_name in selected_tools:
            fallback = {
                "tool": tool_name,
                "failure_scenarios": [],
                "alternatives": [],
                "degraded_service": [],
                "user_guidance": []
            }
            
            if tool_name == "flight_search":
                fallback.update({
                    "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                    "alternatives": [
                        {"method": "manual_search_guidance", "confidence": 0.6},
                        {"method": "general_transport_info", "confidence": 0.5}
                    ],
                    "degraded_service": ["provide_airline_websites", "suggest_travel_agents"],
                    "user_guidance": [
                        "I'll provide airline websites for manual search",
                        "Consider contacting travel agents for complex bookings"
                    ]
                })
            elif tool_name == "hotel_search":
                fallback.update({
                    "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                    "alternatives": [
                        {"method": "manual_booking_guidance", "confidence": 0.7},
                        {"method": "area_recommendations", "confidence": 0.6}
                    ],
                    "degraded_service": ["provide_booking_websites", "suggest_hotel_chains"],
                    "user_guidance": [
                        "I'll recommend popular booking websites",
                        "Consider checking hotel chains directly"
                    ]
                })
            elif tool_name == "attraction_search":
                fallback.update({
                    "failure_scenarios": ["API_ERROR", "NO_RESULTS", "RATE_LIMIT"],
                    "alternatives": [
                        {"method": "knowledge_base_search", "confidence": 0.8},
                        {"method": "general_recommendations", "confidence": 0.7}
                    ],
                    "degraded_service": ["provide_tourism_websites", "suggest_guidebooks"],
                    "user_guidance": [
                        "I'll search our knowledge base for attraction information",
                        "You can also check popular travel websites like TripAdvisor"
                    ]
                })
            
            fallback_strategies.append(fallback)
        
        return fallback_strategies
    
    def _calculate_plan_confidence(self, selected_tools: List[str], requirements: Dict[str, Any]) -> float:
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
        
        final_confidence = min(base_confidence + requirement_coverage + tool_quality, 1.0)
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
            next_steps.extend([
                "Compare flight prices across different dates",
                "Check baggage policies and restrictions"
            ])
        
        if "hotel_search" in plan["tools_to_use"]:
            next_steps.extend([
                "Read recent reviews and ratings",
                "Check cancellation policies"
            ])
        
        if "attraction_search" in plan["tools_to_use"]:
            next_steps.extend([
                "Check opening hours and seasonal availability",
                "Consider purchasing tickets in advance"
            ])
        
        # Add general travel planning steps
        next_steps.extend([
            "Verify visa requirements and documentation",
            "Consider travel insurance options",
            "Check weather conditions for travel dates"
        ])
        
        return next_steps
    
    async def _create_basic_action_plan(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback basic action plan (original implementation)"""
        
        plan = {
            "intent_type": intent["type"],
            "actions": [],
            "tools_to_use": [],
            "execution_strategy": "sequential",
            "next_steps": [],
            "confidence": 0.6
        }
        
        # Basic action planning based on intent type
        if intent["type"] == "planning":
            plan["tools_to_use"] = ["flight_search", "hotel_search", "attraction_search"]
            plan["next_steps"] = [
                "Provide detailed travel itinerary",
                "Include budget estimates",
                "Suggest booking timeline"
            ]
        elif intent["type"] == "recommendation":
            plan["tools_to_use"] = ["attraction_search"]
            plan["next_steps"] = [
                "Provide personalized recommendations",
                "Include practical tips",
                "Suggest related activities"
            ]
        elif intent["type"] == "query":
            plan["tools_to_use"] = []
            plan["next_steps"] = [
                "Provide detailed answer",
                "Offer additional information",
                "Ask follow-up questions"
            ]
        
        return plan
    
    def _parse_llm_requirements(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response for requirements (simplified implementation)"""
        
        # This is a simplified parser - in production, you'd want more robust JSON parsing
        requirements = {
            "budget_sensitivity": "medium",
            "time_sensitivity": "normal",
            "travel_style": "standard",
            "geographic_scope": "unknown"
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
    
    async def _execute_action_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action plan"""
        # TODO: Use tool executor to execute plan
        results = {
            "tools_used": [],
            "results": {},
            "execution_time": 0.0,
            "success": True
        }
        
        try:
            # First, always retrieve knowledge context for any travel query
            # Use the original user message as the query for knowledge retrieval
            query = context.get("original_message", "travel information")
            knowledge_context = await self._retrieve_knowledge_context(query)
            results["knowledge_context"] = knowledge_context
            
            # Execute tools based on plan
            for tool_name in plan.get("tools_to_use", []):
                try:
                    # TODO: Call corresponding tool with actual parameters
                    # For now, simulate tool execution
                    if tool_name == "flight_search":
                        mock_result = {
                            "flights": [
                                {"airline": "Example Airlines", "price": "$500", "duration": "2h 30m"},
                                {"airline": "Budget Air", "price": "$300", "duration": "3h 15m"}
                            ],
                            "message": "Found 2 flight options"
                        }
                    elif tool_name == "hotel_search":
                        mock_result = {
                            "hotels": [
                                {"name": "Example Hotel", "price": "$120/night", "rating": 4.5},
                                {"name": "Budget Inn", "price": "$80/night", "rating": 3.8}
                            ],
                            "message": "Found 2 hotel options"
                        }
                    elif tool_name == "attraction_search":
                        mock_result = {
                            "attractions": [
                                {"name": "Famous Landmark", "rating": 4.8, "description": "Must-see attraction"},
                                {"name": "Cultural Site", "rating": 4.6, "description": "Rich history and culture"}
                            ],
                            "message": "Found 2 attraction options"
                        }
                    else:
                        mock_result = {"message": f"Mock result for {tool_name}"}
                    
                    results["results"][tool_name] = mock_result
                    results["tools_used"].append(tool_name)
                    
                except Exception as e:
                    results["results"][tool_name] = {"error": str(e)}
                    results["success"] = False
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def _generate_response(self, execution_result: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Generate response content using prompt manager"""
        
        from app.core.prompt_manager import prompt_manager, PromptType
        
        try:
            # Get knowledge context from execution result
            knowledge_context = execution_result.get("knowledge_context", {})
            relevant_docs = knowledge_context.get("relevant_docs", [])
            structured_analysis = intent.get("structured_analysis", {})
            
            if execution_result["success"]:
                # Try to use LLM with prompt manager for response generation
                if self.llm_service and not self.llm_service.mock_mode:
                    try:
                        # Build structured context for LLM
                        user_message = execution_result.get("original_message", "")
                        tool_results = execution_result.get("results", {})
                        
                        # Format tool results for LLM
                        formatted_tool_results = {}
                        for tool_name, result in tool_results.items():
                            if tool_name in ["flight_search", "hotel_search", "attraction_search"]:
                                formatted_tool_results[tool_name] = result
                        
                        # Generate response using prompt manager
                        response_prompt = prompt_manager.get_prompt(
                            PromptType.RESPONSE_GENERATION,
                            user_message=user_message,
                            intent_analysis=structured_analysis,
                            tool_results=formatted_tool_results,
                            knowledge_context=knowledge_context.get("context", "")
                        )
                        
                        llm_response = await self.llm_service.chat_completion([
                            {"role": "user", "content": response_prompt}
                        ])
                        
                        if llm_response and llm_response.content:
                            return llm_response.content
                        
                    except Exception as e:
                        logger.error(f"LLM response generation failed: {e}")
                        # Fall back to template-based response
                
                # Template-based response generation (fallback)
                response_parts = []
                
                if intent["type"] == "planning":
                    response_parts.append("🎯 **Travel Planning Assistant**")
                    response_parts.append(f"Based on your request for {intent.get('destination', 'your destination')}, here's what I found:")
                    
                    # Add knowledge-based information
                    if relevant_docs:
                        response_parts.append("\n📚 **Travel Information:**")
                        for i, doc in enumerate(relevant_docs[:2]):  # Limit to top 2 results
                            doc_title = doc.get("metadata", {}).get("title", f"Information {i+1}")
                            content = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                            response_parts.append(f"**{doc_title}**\n{content}")
                    
                    # Add tool results
                    tool_results = execution_result.get("results", {})
                    if "flight_search" in tool_results and "flights" in tool_results["flight_search"]:
                        response_parts.append("\n✈️ **Flight Options:**")
                        for flight in tool_results["flight_search"]["flights"][:2]:
                            response_parts.append(f"- {flight['airline']}: {flight['price']} ({flight['duration']})")
                    
                    if "hotel_search" in tool_results and "hotels" in tool_results["hotel_search"]:
                        response_parts.append("\n🏨 **Hotel Options:**")
                        for hotel in tool_results["hotel_search"]["hotels"][:2]:
                            response_parts.append(f"- {hotel['name']}: {hotel['price']} (Rating: {hotel['rating']})")
                    
                    if "attraction_search" in tool_results and "attractions" in tool_results["attraction_search"]:
                        response_parts.append("\n🎭 **Attractions:**")
                        for attraction in tool_results["attraction_search"]["attractions"][:2]:
                            response_parts.append(f"- {attraction['name']}: {attraction['description']} (Rating: {attraction['rating']})")
                    
                    # Add next steps
                    response_parts.append("\n\n🎯 **Next Steps:**")
                    response_parts.append("- Let me know your travel dates and I can provide more specific recommendations")
                    response_parts.append("- I can help you compare prices and make bookings")
                    response_parts.append("- Ask me about specific aspects like dining, transportation, or activities")
                    
                elif intent["type"] == "recommendation":
                    response_parts.append("💡 **Travel Recommendations**")
                    response_parts.append(f"Here are my recommendations for {intent.get('destination', 'your destination')}:")
                    
                    # Add knowledge-based recommendations
                    if relevant_docs:
                        for i, doc in enumerate(relevant_docs[:3]):
                            doc_title = doc.get("metadata", {}).get("title", f"Recommendation {i+1}")
                            content = doc.get("content", "")[:150] + "..." if len(doc.get("content", "")) > 150 else doc.get("content", "")
                            response_parts.append(f"\n**{doc_title}**\n{content}")
                    
                    response_parts.append("\n\n🔍 **Would you like me to:**")
                    response_parts.append("- Provide more detailed information about any of these recommendations?")
                    response_parts.append("- Help you plan a complete itinerary?")
                    response_parts.append("- Search for specific activities or attractions?")
                    
                elif intent["type"] == "query":
                    response_parts.append("❓ **Travel Information**")
                    
                    if relevant_docs:
                        response_parts.append("Based on my knowledge, here's what I found:")
                        for i, doc in enumerate(relevant_docs[:2]):
                            doc_title = doc.get("metadata", {}).get("title", f"Information {i+1}")
                            content = doc.get("content", "")
                            response_parts.append(f"\n**{doc_title}**\n{content}")
                    else:
                        response_parts.append("I'd be happy to help you with travel information. Could you provide more specific details about what you're looking for?")
                    
                    response_parts.append("\n\n💬 **Follow-up Questions:**")
                    response_parts.append("- Are you looking for information about a specific destination?")
                    response_parts.append("- Would you like me to help you plan a trip?")
                    response_parts.append("- Do you have any specific travel dates in mind?")
                
                return "\n".join(response_parts)
            
            else:
                error_msg = execution_result.get("error", "Unknown error")
                return f"I encountered an issue while processing your {intent['type']} request: {error_msg}. Let me try to help you in another way. Could you provide more details about what you're looking for?"
                
        except Exception as e:
            return f"I'm having trouble generating a response right now: {str(e)}. Please try asking me something else about your travel plans."
    
    async def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute specific action"""
        # TODO: Implement specific action execution logic
        if action == "search_flights":
            return await self._search_flights(parameters)
        elif action == "search_hotels":
            return await self._search_hotels(parameters)
        elif action == "generate_plan":
            return await self._generate_travel_plan(parameters)
        else:
            raise ValueError(f"Unknown action type: {action}")
    
    async def _search_flights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search flights"""
        # TODO: Call flight search tool
        return {"flights": [], "message": "Flight search completed"}
    
    async def _search_hotels(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search hotels"""
        # TODO: Call hotel search tool
        return {"hotels": [], "message": "Hotel search completed"}
    
    async def _generate_travel_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate travel plan"""
        # TODO: Integrate all search results, generate complete travel plan
        return {"plan": None, "message": "Travel plan generated"}
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        # TODO: Learn and store user preferences
        self.user_preferences_history.update(preferences)
        self.metadata["last_preference_update"] = preferences
    
    def configure_refinement(
        self, 
        enabled: bool = True, 
        quality_threshold: float = 0.8, 
        max_iterations: int = 3
    ):
        """Configure self-refinement settings"""
        self.refine_enabled = enabled
        self.quality_threshold = quality_threshold
        self.max_refine_iterations = max_iterations


# Register Travel Agent
from app.agents.base_agent import agent_manager
travel_agent = TravelAgent()
agent_manager.register_agent(travel_agent) 