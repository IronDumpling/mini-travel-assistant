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
from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus, QualityAssessment
from app.tools.base_tool import tool_registry
from app.tools.tool_executor import get_tool_executor
from app.core.llm_service import get_llm_service
from app.core.rag_engine import get_rag_engine
from app.models.schemas import TravelPreferences, TravelPlan


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
            
            # 3. Make an action plan
            action_plan = await self._create_action_plan(intent, knowledge_context)
            
            # 4. Execute action plan
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
        """Analyze user intent"""
        # TODO: Use LLM to analyze user intent
        # 1. Identify intent type (planning, query, modification, complaint, etc.)
        # 2. Extract key information (destination, time, budget, etc.)
        # 3. Identify user sentiment and preferences
        
        # For now, implement basic keyword-based intent analysis
        user_message_lower = user_message.lower()
        
        # Determine intent type
        intent_type = "query"  # default
        if any(word in user_message_lower for word in ["帮我", "规划", "安排", "制定", "plan", "create", "make"]):
            intent_type = "planning"
        elif any(word in user_message_lower for word in ["修改", "更改", "调整", "change", "modify", "update"]):
            intent_type = "modification"
        elif any(word in user_message_lower for word in ["推荐", "建议", "介绍", "recommend", "suggest", "introduce"]):
            intent_type = "recommendation"
        
        # Extract destination information
        destination = "Unknown"
        destinations = ["东京", "京都", "大阪", "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", "shanghai"]
        for dest in destinations:
            if dest in user_message_lower:
                destination = dest
                break
        
        # Extract time information
        time_info = {}
        if any(word in user_message_lower for word in ["天", "日", "day", "days"]):
            # Try to extract number of days
            import re
            days_match = re.search(r'(\d+)天|(\d+)日|(\d+)\s*day', user_message_lower)
            if days_match:
                days = int(days_match.group(1) or days_match.group(2) or days_match.group(3))
                time_info["duration_days"] = days
        
        # Extract budget information
        budget_info = {}
        if any(word in user_message_lower for word in ["预算", "budget", "cost", "price", "钱", "费用"]):
            budget_info["budget_mentioned"] = True
        
        # TODO: Call LLM service for more sophisticated analysis
        # prompt = f"""
        # Analyze the intent and key information of the following user message:
        # User message: {user_message}
        # 
        # Please identify:
        # 1. Intent type (planning, query, modification, complaint, etc.)
        # 2. Travel-related information (destination, time, budget, number of people, etc.)
        # 3. User preferences and requirements
        # 4. Urgency level
        # 
        # Return the result in JSON format.
        # """
        # response = await self.llm_service.chat_completion([
        #     {"role": "user", "content": prompt}
        # ])
        
        return {
            "type": intent_type,
            "destination": destination,
            "time_info": time_info,
            "budget_info": budget_info,
            "urgency": "normal",
            "extracted_info": {
                "destination": destination,
                "time_info": time_info,
                "budget_info": budget_info
            }
        }
    
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
        """Create action plan"""
        # TODO: Use LLM to create more sophisticated action plan
        # 1. Determine tools to execute
        # 2. Plan tool execution order
        # 3. Set execution parameters
        
        plan = {
            "intent_type": intent["type"],
            "actions": [],
            "tools_to_use": [],
            "execution_strategy": "sequential",
            "next_steps": []
        }
        
        # Basic action planning based on intent type
        if intent["type"] == "planning":
            plan["actions"] = [
                "retrieve_knowledge",
                "search_flights",
                "search_hotels", 
                "search_attractions",
                "generate_itinerary"
            ]
            plan["tools_to_use"] = [
                "flight_search",
                "hotel_search",
                "attraction_search"
            ]
            plan["next_steps"] = [
                "Provide detailed travel itinerary",
                "Include budget estimates",
                "Suggest booking timeline"
            ]
        elif intent["type"] == "recommendation":
            plan["actions"] = [
                "retrieve_knowledge",
                "search_attractions",
                "generate_recommendations"
            ]
            plan["tools_to_use"] = [
                "attraction_search"
            ]
            plan["next_steps"] = [
                "Provide personalized recommendations",
                "Include practical tips",
                "Suggest related activities"
            ]
        elif intent["type"] == "query":
            plan["actions"] = [
                "retrieve_knowledge",
                "answer_question"
            ]
            plan["tools_to_use"] = []
            plan["next_steps"] = [
                "Provide detailed answer",
                "Offer additional information",
                "Ask follow-up questions"
            ]
        
        # TODO: Use LLM for more sophisticated planning
        # llm_prompt = f"""
        # Based on the user intent: {intent}
        # And context: {context}
        # Create a detailed action plan including:
        # 1. Sequence of actions to take
        # 2. Tools to use
        # 3. Parameters for each tool
        # 4. Expected outcomes
        # """
        
        return plan
    
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
        """Generate response content"""
        try:
            # Get knowledge context from execution result
            knowledge_context = execution_result.get("knowledge_context", {})
            relevant_docs = knowledge_context.get("relevant_docs", [])
            
            if execution_result["success"]:
                # Create response based on intent and results
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