from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TripStyle(str, Enum):
    RELAXED = "relaxed"
    ADVENTURE = "adventure"
    CULTURAL = "cultural"
    LUXURY = "luxury"
    BUDGET = "budget"

class TravelerType(str, Enum):
    ADULT = "adult"
    CHILD = "child"
    SENIOR = "senior"

class Traveler(BaseModel):
    age: int
    type: TravelerType

class Budget(BaseModel):
    total: float
    currency: str = "USD"

class TravelPreferences(BaseModel):
    origin: str
    destination: str
    start_date: datetime
    end_date: datetime
    budget: Budget
    trip_style: TripStyle
    travelers: List[Traveler]
    interests: List[str]
    goals: List[str]
    additional_notes: Optional[str] = None

class Activity(BaseModel):
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str
    estimated_cost: float
    booking_info: Optional[Dict[str, Any]] = None

class DailyPlan(BaseModel):
    date: datetime
    activities: List[Activity]
    total_cost: float

class TravelPlan(BaseModel):
    id: str
    preferences: TravelPreferences
    daily_plans: List[DailyPlan]
    total_cost: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "draft"  # draft, confirmed, completed
    feedback: Optional[str] = None

class PlanUpdate(BaseModel):
    preferences: Optional[TravelPreferences] = None
    feedback: Optional[str] = None 