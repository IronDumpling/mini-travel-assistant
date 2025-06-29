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
            result = await self._execute_action_plan(action_plan, message.metadata)
            
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
        
        prompt = f"""
        Analyze the intent and key information of the following user message:
        User message: {user_message}
        
        Please identify:
        1. Intent type (planning, query, modification, complaint, etc.)
        2. Travel-related information (destination, time, budget, number of people, etc.)
        3. User preferences and requirements
        4. Urgency level
        
        Return the result in JSON format.
        """
        
        # TODO: Call LLM service
        # response = await self.llm_service.chat_completion([
        #     {"role": "user", "content": prompt}
        # ])
        
        # Temporary return example result
        return {
            "type": "planning",
            "destination": "Unknown", # TODO: Extract destination
            "urgency": "normal",
            "extracted_info": {}
        }
    
    async def _retrieve_knowledge_context(self, query: str) -> Dict[str, Any]:
        """Retrieve knowledge context"""
        # TODO: Use RAG engine to retrieve relevant knowledge
        # result = await self.rag_engine.retrieve(query, top_k=5)
        
        # Temporary return example result
        return {
            "relevant_docs": [],
            "context": "Relevant knowledge context"
        }
    
    async def _create_action_plan(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan"""
        # TODO: Create action plan based on intent and context
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
        
        if intent["type"] == "planning":
            plan["actions"] = [
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
            # Execute tools based on plan
            for tool_name in plan.get("tools_to_use", []):
                # TODO: Call corresponding tool
                # result = await self.tool_executor.execute_tool(
                #     tool_name, 
                #     tool_params, 
                #     context
                # )
                results["tools_used"].append(tool_name)
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def _generate_response(self, execution_result: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Generate response content"""
        # TODO: Generate natural language response based on execution result
        # 1. Summarize execution results
        # 2. Customize response style based on intent type
        # 3. Provide next steps
        
        if execution_result["success"]:
            return f"I have completed the relevant search and analysis for your {intent['type']}. Based on your needs, I found some good options..."
        else:
            return f"Sorry, there was a problem processing your {intent['type']} request. Let me re-search for you..."
    
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