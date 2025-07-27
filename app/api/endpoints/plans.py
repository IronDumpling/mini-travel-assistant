"""
Travel Plans Endpoints - Simplified API using new framework

Streamlined travel plan endpoints that leverage the new framework's
structured outputs instead of complex parsing.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict, Any
from app.api.schemas import (
    TravelRequest, TravelPlan, TravelPlanUpdate, TravelPlanResponse,
    SessionTravelPlan, CalendarEvent, PlanResponse, PlanUpdateRequest
)
from app.agents.travel_agent import get_travel_agent
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
        
        # Use Agent singleton instead of creating new instance
        agent = get_travel_agent()
        
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
            
            # ✅ 使用单例Agent，而不是每次创建新实例
            agent = get_travel_agent()
            
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

# ===== Session Plan Endpoints =====

@router.get("/session-plans/{session_id}", response_model=PlanResponse)
async def get_session_plan(session_id: str):
    """Get travel plan for a specific session"""
    try:
        from app.core.plan_manager import get_plan_manager
        plan_manager = get_plan_manager()
        
        plan = plan_manager.get_plan_by_session(session_id)
        if not plan:
            return PlanResponse(
                success=False,
                message="No plan found for this session",
                events_count=0
            )
        
        return PlanResponse(
            success=True,
            plan=plan,
            message="Plan retrieved successfully",
            events_count=len(plan.events)
        )
        
    except Exception as e:
        logger.error(f"Failed to get plan for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve plan: {str(e)}"
        )

@router.get("/plans", response_model=List[SessionTravelPlan])
async def get_plans_by_session(session_id: str):
    """Get plans by session ID (for frontend compatibility)"""
    try:
        from app.core.plan_manager import get_plan_manager
        plan_manager = get_plan_manager()
        
        plan = plan_manager.get_plan_by_session(session_id)
        if not plan:
            return []
        
        return [plan]
        
    except Exception as e:
        logger.error(f"Failed to get plans for session {session_id}: {e}")
        return []

@router.put("/session-plans/{session_id}", response_model=PlanResponse)
async def update_session_plan(session_id: str, update_request: PlanUpdateRequest):
    """Update travel plan for a session"""
    try:
        from app.core.plan_manager import get_plan_manager
        plan_manager = get_plan_manager()
        
        plan = plan_manager.get_plan_by_session(session_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for this session"
            )
        
        # Apply updates
        updated = False
        
        # Add new events
        for event in update_request.events_to_add:
            success = plan_manager.add_event(session_id, event)
            if success:
                updated = True
        
        # Update existing events
        for event in update_request.events_to_update:
            success = plan_manager.update_event(session_id, event)
            if success:
                updated = True
        
        # Remove events
        for event_id in update_request.events_to_remove:
            success = plan_manager.remove_event(session_id, event_id)
            if success:
                updated = True
        
        # Update metadata if provided
        if update_request.metadata_updates:
            for key, value in update_request.metadata_updates.model_dump().items():
                if value is not None and hasattr(plan.metadata, key):
                    setattr(plan.metadata, key, value)
                    updated = True
        
        if updated:
            # Get updated plan
            updated_plan = plan_manager.get_plan_by_session(session_id)
            return PlanResponse(
                success=True,
                plan=updated_plan,
                message="Plan updated successfully",
                events_count=len(updated_plan.events) if updated_plan else 0
            )
        else:
            return PlanResponse(
                success=False,
                plan=plan,
                message="No updates applied",
                events_count=len(plan.events)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plan for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update plan: {str(e)}"
        )

@router.delete("/session-plans/{session_id}")
async def delete_session_plan(session_id: str):
    """Delete travel plan for a session"""
    try:
        from app.core.plan_manager import get_plan_manager
        plan_manager = get_plan_manager()
        
        plan = plan_manager.get_plan_by_session(session_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for this session"
            )
        
        success = plan_manager.delete_plan(plan.plan_id)
        if success:
            return {
                "message": f"Plan deleted successfully for session {session_id}",
                "session_id": session_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete plan"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete plan for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete plan: {str(e)}"
        )