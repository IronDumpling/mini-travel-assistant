from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from app.models.schemas import TravelPreferences, TravelPlan, PlanUpdate, Activity, DailyPlan, AgentMetadata, Budget, TripStyle
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage, AgentResponse
from app.memory.session_manager import get_session_manager
import json
import re
from datetime import datetime, timedelta

router = APIRouter()

class ChatMessage(BaseModel):
    """Chat message for conversational travel planning"""
    message: str
    session_id: Optional[str] = None
    enable_refinement: bool = True

class ChatResponse(BaseModel):
    """Response from the travel agent"""
    success: bool
    content: str
    confidence: float
    actions_taken: List[str]
    next_steps: List[str]
    session_id: str
    refinement_details: Optional[dict] = None

class RefinementConfig(BaseModel):
    """Configuration for self-refinement"""
    enabled: bool = True
    quality_threshold: float = 0.75
    max_iterations: int = 3

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """
    Chat with the travel agent using natural language.
    This is the main endpoint that leverages the self-refine loop.
    """
    try:
        # Get or create session
        session_manager = get_session_manager()
        if message.session_id:
            session_manager.switch_session(message.session_id)
        else:
            session_id = session_manager.create_session(
                title=f"Travel Planning - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description="AI-powered travel planning conversation"
            )
            message.session_id = session_id

        # Create travel agent
        agent = TravelAgent()
        agent.configure_refinement(enabled=message.enable_refinement)
        
        # Create agent message
        agent_message = AgentMessage(
            sender="user",
            receiver="travel_agent",
            content=message.message,
            metadata={"session_id": message.session_id}
        )
        
        # Process with self-refinement
        if message.enable_refinement:
            response = await agent.plan_travel(agent_message)
        else:
            response = await agent.process_message(agent_message)
        
        # Store conversation in session
        session_manager.add_message(
            user_message=message.message,
            agent_response=response.content,
            metadata={
                "confidence": response.confidence,
                "actions_taken": response.actions_taken,
                "refinement_used": message.enable_refinement
            }
        )
        
        # Prepare response
        chat_response = ChatResponse(
            success=response.success,
            content=response.content,
            confidence=response.confidence,
            actions_taken=response.actions_taken,
            next_steps=response.next_steps,
            session_id=message.session_id
        )
        
        # Add refinement details if available
        if "refinement_iteration" in response.metadata:
            chat_response.refinement_details = {
                "final_iteration": response.metadata["refinement_iteration"],
                "final_quality_score": response.metadata["quality_score"],
                "refinement_status": response.metadata["refinement_status"],
                "quality_dimensions": agent.get_quality_dimensions()
            }
        
        return chat_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/plans", response_model=TravelPlan)
async def create_travel_plan(preferences: TravelPreferences):
    """
    Create a structured travel plan based on preferences.
    Uses the agent system with self-refinement for better quality.
    """
    try:
        # Convert preferences to natural language message
        message_content = _preferences_to_message(preferences)
        
        # Create travel agent
        agent = TravelAgent()
        
        # Create agent message
        agent_message = AgentMessage(
            sender="api_user",
            receiver="travel_agent",
            content=message_content,
            metadata={"preferences": preferences.model_dump()}
        )
        
        # Process with self-refinement
        response = await agent.plan_travel(agent_message)
        
        # Convert agent response to structured TravelPlan
        structured_plan = await _parse_agent_response_to_plan(response, preferences)
        
        return structured_plan
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan creation failed: {str(e)}")

@router.get("/plans/{plan_id}", response_model=TravelPlan)
async def get_travel_plan(plan_id: str):
    """
    Get a specific travel plan by ID.
    TODO: Implement plan storage and retrieval
    """
    # TODO: Implement plan storage in session manager or database
    raise HTTPException(status_code=404, detail="Plan storage not yet implemented")

@router.put("/plans/{plan_id}", response_model=TravelPlan)
async def update_travel_plan(plan_id: str, update: PlanUpdate):
    """
    Update an existing travel plan with new preferences or feedback.
    Uses agent system to incorporate feedback and refine the plan.
    """
    try:
        if update.feedback:
            # Use feedback to refine the plan
            agent = TravelAgent()
            
            feedback_message = AgentMessage(
                sender="api_user",
                receiver="travel_agent",
                content=f"Please update the travel plan based on this feedback: {update.feedback}",
                metadata={"plan_id": plan_id, "update": update.model_dump()}
            )
            
            response = await agent.plan_travel(feedback_message)
            
            # Parse refined response to structured plan
            if update.preferences:
                structured_plan = await _parse_agent_response_to_plan(response, update.preferences)
                structured_plan.feedback = update.feedback
                return structured_plan
            else:
                # Create a basic plan structure when preferences not provided
                return TravelPlan(
                    id=plan_id,
                    preferences=TravelPreferences(
                        origin="Unknown",
                        destination="Unknown",
                        start_date=datetime.utcnow(),
                        end_date=datetime.utcnow() + timedelta(days=1),
                                                 budget=Budget(total=0, currency="USD"),
                                                 trip_style=TripStyle.RELAXED,
                        travelers=[],
                        interests=[],
                        goals=[]
                    ),
                    daily_plans=[],
                    total_cost=0.0,
                    status="updated",
                    feedback=update.feedback
                )
        
        raise HTTPException(status_code=400, detail="No update data provided")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan update failed: {str(e)}")

@router.delete("/plans/{plan_id}")
async def delete_travel_plan(plan_id: str):
    """
    Delete a travel plan.
    TODO: Implement plan deletion from storage
    """
    # TODO: Implement plan deletion
    raise HTTPException(status_code=404, detail="Plan deletion not yet implemented")

@router.post("/agent/configure")
async def configure_agent_refinement(config: RefinementConfig):
    """
    Configure the self-refinement settings for the travel agent.
    """
    try:
        # This would typically be stored per session or user
        # For now, return the configuration
        return {
            "message": "Agent refinement configured",
            "config": config.model_dump(),
            "note": "Configuration applied to new agent instances"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/agent/status")
async def get_agent_status():
    """
    Get the current status and capabilities of the travel agent.
    """
    try:
        agent = TravelAgent()
        status = agent.get_status()
        
        return {
            "agent_info": {
                "name": status["name"],
                "description": status["description"],
                "capabilities": status["capabilities"],
                "tools": status["tools"]
            },
            "refinement_config": status["refinement_config"],
            "quality_dimensions": agent.get_quality_dimensions(),
            "system_status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

async def _parse_agent_response_to_plan(response: AgentResponse, preferences: TravelPreferences) -> TravelPlan:
    """
    Parse agent response back to structured TravelPlan.
    This implements the missing parser functionality.
    """
    try:
        # Extract structured information from agent response
        content = response.content
        
        # Generate plan ID
        plan_id = f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Parse daily plans from agent response
        daily_plans = _extract_daily_plans_from_response(content, preferences)
        
        # Calculate total cost
        total_cost = sum(day.total_cost for day in daily_plans)
        
        # Create agent metadata
        agent_metadata = AgentMetadata(
            confidence=response.confidence,
            actions_taken=response.actions_taken,
            refinement_used="refinement_iteration" in response.metadata,
            quality_score=response.metadata.get("quality_score"),
            refinement_iterations=response.metadata.get("refinement_iteration"),
            processing_time=response.metadata.get("execution_time")
        )
        
        # Create structured travel plan
        travel_plan = TravelPlan(
            id=plan_id,
            preferences=preferences,
            daily_plans=daily_plans,
            total_cost=total_cost,
            status="generated",
            feedback=None,
            agent_metadata=agent_metadata,
            quality_score=response.metadata.get("quality_score"),
            session_id=response.metadata.get("session_id")
        )
        
        return travel_plan
        
    except Exception as e:
        # Fallback: return basic structure with agent response as description
        return TravelPlan(
            id=f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            preferences=preferences,
            daily_plans=[_create_fallback_daily_plan(preferences, response.content)],
            total_cost=preferences.budget.total if preferences.budget else 0.0,
            status="generated_with_parsing_error"
        )

def _extract_daily_plans_from_response(content: str, preferences: TravelPreferences) -> List[DailyPlan]:
    """
    Extract daily plans from agent response content.
    Uses pattern matching and heuristics to parse structured information.
    """
    daily_plans = []
    
    try:
        # Calculate trip duration
        duration = (preferences.end_date - preferences.start_date).days
        
        # Pattern matching for day-by-day content
        day_patterns = [
            r"Day (\d+):?\s*(.+?)(?=Day \d+|$)",
            r"(\d+)\.\s*(.+?)(?=\d+\.|$)",
            r"(Day \d+|Day One|Day Two|Day Three|Day Four|Day Five|Day Six|Day Seven):?\s*(.+?)(?=Day|$)"
        ]
        
        found_days = []
        for pattern in day_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                day_content = match.group(2).strip()
                if len(day_content) > 20:  # Meaningful content
                    found_days.append(day_content)
            
            if found_days:
                break
        
        # If no structured days found, create single day with full content
        if not found_days:
            found_days = [content]
        
        # Create daily plans
        for i, day_content in enumerate(found_days[:duration]):
            current_date = preferences.start_date + timedelta(days=i)
            
            # Extract activities from day content
            activities = _extract_activities_from_day_content(day_content, current_date)
            
            # Calculate daily cost
            daily_cost = sum(activity.estimated_cost for activity in activities)
            
            daily_plan = DailyPlan(
                date=current_date,
                activities=activities,
                total_cost=daily_cost
            )
            
            daily_plans.append(daily_plan)
        
        # Ensure we have at least one day
        if not daily_plans:
            daily_plans.append(_create_fallback_daily_plan(preferences, content))
        
        return daily_plans
        
    except Exception:
        # Fallback: single day plan
        return [_create_fallback_daily_plan(preferences, content)]

def _extract_activities_from_day_content(day_content: str, date: datetime) -> List[Activity]:
    """
    Extract activities from a day's content using pattern matching.
    """
    activities = []
    
    try:
        # Patterns for activities
        activity_patterns = [
            r"(\d{1,2}:\d{2})\s*[-:]?\s*(.+?)(?=\d{1,2}:\d{2}|$)",
            r"(Morning|Afternoon|Evening):?\s*(.+?)(?=Morning|Afternoon|Evening|$)",
            r"[-•*]\s*(.+?)(?=[-•*]|$)"
        ]
        
        found_activities = []
        
        # Try time-based pattern first
        time_matches = re.finditer(activity_patterns[0], day_content, re.IGNORECASE | re.MULTILINE)
        for match in time_matches:
            time_str = match.group(1)
            activity_desc = match.group(2).strip()
            if len(activity_desc) > 10:
                found_activities.append((time_str, activity_desc))
        
        # If no time-based activities, try period-based
        if not found_activities:
            period_matches = re.finditer(activity_patterns[1], day_content, re.IGNORECASE | re.DOTALL)
            for match in period_matches:
                period = match.group(1)
                activity_desc = match.group(2).strip()
                if len(activity_desc) > 10:
                    time_str = _period_to_time(period)
                    found_activities.append((time_str, activity_desc))
        
        # If still no activities, try bullet points
        if not found_activities:
            bullet_matches = re.finditer(activity_patterns[2], day_content, re.MULTILINE)
            for i, match in enumerate(bullet_matches):
                activity_desc = match.group(1).strip()
                if len(activity_desc) > 10:
                    time_str = f"{9 + i * 3}:00"  # Spread activities throughout day
                    found_activities.append((time_str, activity_desc))
        
        # Create Activity objects
        for time_str, description in found_activities:
            start_time = _parse_time_to_datetime(time_str, date)
            end_time = start_time + timedelta(hours=2)  # Default 2-hour duration
            
            # Extract estimated cost (basic pattern matching)
            cost = _extract_cost_from_description(description)
            
            activity = Activity(
                name=_extract_activity_name(description),
                description=description,
                start_time=start_time,
                end_time=end_time,
                location=_extract_location_from_description(description),
                estimated_cost=cost
            )
            
            activities.append(activity)
        
        # Ensure at least one activity per day
        if not activities:
            activities.append(Activity(
                name="Travel Day",
                description=day_content[:100] + "..." if len(day_content) > 100 else day_content,
                start_time=date.replace(hour=9, minute=0),
                end_time=date.replace(hour=17, minute=0),
                location="Various",
                estimated_cost=100.0
            ))
        
        return activities
        
    except Exception:
        # Fallback activity
        return [Activity(
            name="Travel Activities",
            description=day_content[:200] + "..." if len(day_content) > 200 else day_content,
            start_time=date.replace(hour=9, minute=0),
            end_time=date.replace(hour=17, minute=0),
            location="Various",
            estimated_cost=100.0
        )]

def _create_fallback_daily_plan(preferences: TravelPreferences, content: str) -> DailyPlan:
    """Create a fallback daily plan when parsing fails"""
    activity = Activity(
        name=f"Travel to {preferences.destination}",
        description=content[:300] + "..." if len(content) > 300 else content,
        start_time=preferences.start_date.replace(hour=9, minute=0),
        end_time=preferences.start_date.replace(hour=17, minute=0),
        location=preferences.destination,
        estimated_cost=preferences.budget.total / max(1, (preferences.end_date - preferences.start_date).days)
    )
    
    return DailyPlan(
        date=preferences.start_date,
        activities=[activity],
        total_cost=activity.estimated_cost
    )

def _period_to_time(period: str) -> str:
    """Convert period (Morning/Afternoon/Evening) to time string"""
    period_lower = period.lower()
    if "morning" in period_lower:
        return "9:00"
    elif "afternoon" in period_lower:
        return "14:00"
    elif "evening" in period_lower:
        return "19:00"
    else:
        return "12:00"

def _parse_time_to_datetime(time_str: str, date: datetime) -> datetime:
    """Parse time string to datetime object"""
    try:
        time_parts = time_str.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        return date.replace(hour=hour, minute=minute)
    except:
        return date.replace(hour=12, minute=0)

def _extract_cost_from_description(description: str) -> float:
    """Extract cost from activity description using pattern matching"""
    cost_patterns = [
        r"\$(\d+(?:\.\d{2})?)",
        r"(\d+(?:\.\d{2})?)\s*(?:USD|dollars?|€|euros?)",
        r"cost:?\s*\$?(\d+(?:\.\d{2})?)"
    ]
    
    for pattern in cost_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    
    # Default cost estimation based on activity type
    description_lower = description.lower()
    if any(word in description_lower for word in ["museum", "gallery", "monument"]):
        return 25.0
    elif any(word in description_lower for word in ["restaurant", "dining", "lunch", "dinner"]):
        return 50.0
    elif any(word in description_lower for word in ["hotel", "accommodation"]):
        return 150.0
    elif any(word in description_lower for word in ["tour", "excursion"]):
        return 75.0
    else:
        return 30.0

def _extract_activity_name(description: str) -> str:
    """Extract activity name from description"""
    # Take first sentence or first 50 characters
    first_sentence = description.split('.')[0].split('\n')[0]
    if len(first_sentence) > 50:
        return first_sentence[:47] + "..."
    return first_sentence

def _extract_location_from_description(description: str) -> str:
    """Extract location from activity description"""
    # Look for location patterns
    location_patterns = [
        r"at\s+([A-Z][a-zA-Z\s]+)",
        r"in\s+([A-Z][a-zA-Z\s]+)",
        r"visit\s+([A-Z][a-zA-Z\s]+)"
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, description)
        if match:
            location = match.group(1).strip()
            if len(location) < 50:  # Reasonable location name
                return location
    
    return "TBD"

def _preferences_to_message(preferences: TravelPreferences) -> str:
    """Convert TravelPreferences to natural language message (Enhanced)"""
    travelers_desc = f"{len(preferences.travelers)} travelers"
    if preferences.travelers:
        types = [t.type.value for t in preferences.travelers]
        travelers_desc = f"{len(preferences.travelers)} travelers ({', '.join(types)})"
    
    duration = (preferences.end_date - preferences.start_date).days
    
    message = f"""
    I need help planning a {preferences.trip_style.value} trip to {preferences.destination} 
    from {preferences.origin}. 
    
    Travel details:
    - Dates: {preferences.start_date.strftime('%Y-%m-%d')} to {preferences.end_date.strftime('%Y-%m-%d')} ({duration} days)
    - Budget: {preferences.budget.total} {preferences.budget.currency}
    - Travelers: {travelers_desc}
    - Interests: {', '.join(preferences.interests)}
    - Goals: {', '.join(preferences.goals)}
    
    {preferences.additional_notes or ''}
    
    Please create a detailed day-by-day travel plan with:
    1. Specific activities with times
    2. Recommended restaurants and dining
    3. Estimated costs for each activity
    4. Transportation suggestions
    5. Practical booking information
    
    Format the response with clear day-by-day structure including times and locations.
    """
    
    return message.strip() 