"""
Travel Plans Endpoints - Simplified API using new framework

Streamlined travel plan endpoints that leverage the new framework's
structured outputs instead of complex parsing.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict, Any
from app.api.schemas import TravelRequest, TravelPlan, TravelPlanUpdate, TravelPlanResponse, FrameworkMetadata
from app.agents.travel_agent import TravelAgent
from datetime import datetime
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("/plans", response_model=TravelPlanResponse)
async def create_travel_plan(request: TravelRequest):
    """
    Create a travel plan using the new framework.
    Uses the agent's structured output generation.
    """
    try:
        # Convert request to natural language message
        message_content = _request_to_message(request)
        
        # Create travel agent
        agent = TravelAgent()
        
        # Generate structured plan using the complete framework
        structured_plan_data = await agent.generate_structured_plan(
            user_message=message_content,
            metadata={"request": request.model_dump()}
        )
        
        # Convert to API schema
        plan = _create_travel_plan_from_structured_data(structured_plan_data, request)
        
        return TravelPlanResponse(
            success=True,
            plan=plan,
            message="Travel plan created successfully"
        )
        
    except Exception as e:
        logger.error(f"Travel plan creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan creation failed: {str(e)}"
        )

@router.get("/plans/{plan_id}", response_model=TravelPlanResponse)
async def get_travel_plan(plan_id: str):
    """
    Get a specific travel plan by ID.
    TODO: Implement plan storage and retrieval
    """
    # TODO: Implement plan storage in session manager or database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Plan storage not yet implemented"
    )

@router.put("/plans/{plan_id}", response_model=TravelPlanResponse)
async def update_travel_plan(plan_id: str, update: TravelPlanUpdate):
    """
    Update an existing travel plan with feedback.
    Uses the new framework to incorporate feedback and refine the plan.
    """
    try:
        if update.feedback:
            # Create feedback message
            feedback_message = f"Please update the travel plan based on this feedback: {update.feedback}"
            
            # If updated request is provided, incorporate it
            if update.updated_request:
                original_message = _request_to_message(update.updated_request)
                feedback_message = f"{original_message}\n\nAdditional feedback: {update.feedback}"
            
            # Create travel agent and generate updated plan
            agent = TravelAgent()
            
            structured_plan_data = await agent.generate_structured_plan(
                user_message=feedback_message,
                metadata={"plan_id": plan_id, "update": update.model_dump()}
            )
            
            # Convert to API schema
            plan = _create_travel_plan_from_structured_data(
                structured_plan_data, 
                update.updated_request or _create_default_request()
            )
            plan.id = plan_id  # Keep the original plan ID
            
            return TravelPlanResponse(
                success=True,
                plan=plan,
                message="Travel plan updated successfully"
            )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update data provided"
        )
        
    except Exception as e:
        logger.error(f"Travel plan update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan update failed: {str(e)}"
        )

@router.delete("/plans/{plan_id}")
async def delete_travel_plan(plan_id: str):
    """
    Delete a travel plan.
    TODO: Implement plan deletion from storage
    """
    # TODO: Implement plan deletion
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Plan deletion not yet implemented"
    )

@router.get("/plans")
async def list_travel_plans(limit: int = 10, offset: int = 0):
    """
    List all travel plans with pagination.
    TODO: Implement plan listing from storage
    """
    # TODO: Implement plan listing
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Plan listing not yet implemented"
    )

# Helper functions - Simplified for new framework

def _request_to_message(request: TravelRequest) -> str:
    """Convert TravelRequest to natural language message"""
    message_parts = [
        f"I want to plan a {request.trip_style.value} trip to {request.destination}"
    ]
    
    if request.origin:
        message_parts.append(f"from {request.origin}")
    
    if request.duration_days:
        message_parts.append(f"for {request.duration_days} days")
    
    if request.travelers > 1:
        message_parts.append(f"for {request.travelers} people")
    
    if request.budget:
        message_parts.append(f"with a budget of {request.budget} {request.budget_currency}")
    
    if request.interests:
        message_parts.append(f"with interests in {', '.join(request.interests)}")
    
    if request.additional_requirements:
        message_parts.append(f"Additional requirements: {request.additional_requirements}")
    
    message_parts.append("Please help me find flights, hotels, and attractions with detailed recommendations.")
    
    return ". ".join(message_parts)

def _create_travel_plan_from_structured_data(structured_data: Dict[str, Any], request: TravelRequest) -> TravelPlan:
    """Create TravelPlan from agent's structured output"""
    from app.api.schemas import Attraction, Hotel, Flight, FrameworkMetadata
    
    # Convert structured data to schema objects
    attractions = [Attraction(**attr) for attr in structured_data.get("attractions", [])]
    hotels = [Hotel(**hotel) for hotel in structured_data.get("hotels", [])]
    flights = [Flight(**flight) for flight in structured_data.get("flights", [])]
    
    # Create framework metadata
    metadata_dict = structured_data.get("metadata", {})
    framework_metadata = FrameworkMetadata(
        confidence=metadata_dict.get("confidence", 0.8),
        actions_taken=metadata_dict.get("actions_taken", []),
        next_steps=metadata_dict.get("next_steps", []),
        tools_used=metadata_dict.get("tools_used", []),
        processing_time=metadata_dict.get("processing_time", 0.0),
        intent_analysis=metadata_dict.get("intent_analysis"),
        quality_score=metadata_dict.get("quality_score"),
        refinement_iterations=metadata_dict.get("refinement_iterations", 0)
    )
    
    # Create travel plan
    travel_plan = TravelPlan(
        id=structured_data.get("id", f"plan_{int(datetime.utcnow().timestamp())}"),
        request=request,
        content=structured_data.get("content", ""),
        attractions=attractions,
        hotels=hotels,
        flights=flights,
        metadata=framework_metadata,
        session_id=structured_data.get("session_id"),
        status=structured_data.get("status", "generated")
    )
    
    return travel_plan

def _create_default_request() -> TravelRequest:
    """Create a default TravelRequest for fallback scenarios"""
    return TravelRequest(
        destination="Unknown",
        origin="Unknown",
        duration_days=3,
        travelers=1,
        budget=1000.0,
        budget_currency="USD",
        interests=["general"],
        additional_requirements="Basic travel planning"
    ) 