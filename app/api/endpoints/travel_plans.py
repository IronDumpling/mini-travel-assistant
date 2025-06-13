from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.schemas import TravelPreferences, TravelPlan, PlanUpdate
from app.services.travel_planner import TravelPlannerService

router = APIRouter()
travel_planner = TravelPlannerService()

@router.post("/plans", response_model=TravelPlan)
async def create_travel_plan(preferences: TravelPreferences):
    """
    Create a new travel plan based on user preferences.
    """
    try:
        plan = await travel_planner.generate_plan(preferences)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plans/{plan_id}", response_model=TravelPlan)
async def get_travel_plan(plan_id: str):
    """
    Get a specific travel plan by ID.
    """
    # In a real implementation, fetch from database
    raise HTTPException(status_code=404, detail="Plan not found")

@router.put("/plans/{plan_id}", response_model=TravelPlan)
async def update_travel_plan(plan_id: str, update: PlanUpdate):
    """
    Update an existing travel plan with new preferences or feedback.
    """
    # In a real implementation, update in database and regenerate if needed
    raise HTTPException(status_code=404, detail="Plan not found")

@router.delete("/plans/{plan_id}")
async def delete_travel_plan(plan_id: str):
    """
    Delete a travel plan.
    """
    # In a real implementation, delete from database
    raise HTTPException(status_code=404, detail="Plan not found") 