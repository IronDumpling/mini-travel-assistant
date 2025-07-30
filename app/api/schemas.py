"""
API Layer Schemas - Streamlined data models for API contracts

These schemas serve as the API contract layer, working with the new framework's
structured outputs rather than requiring complex parsing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class TripStyle(str, Enum):
    """Trip style enumeration"""
    RELAXED = "relaxed"
    ADVENTURE = "adventure"
    CULTURAL = "cultural"
    LUXURY = "luxury"
    BUDGET = "budget"


class TravelRequest(BaseModel):
    """Simplified travel request model"""
    destination: str
    origin: Optional[str] = None
    duration_days: Optional[int] = None
    travelers: int = 1
    budget: Optional[float] = None
    budget_currency: str = "USD"
    trip_style: TripStyle = TripStyle.RELAXED
    interests: List[str] = []
    additional_requirements: Optional[str] = None


class Attraction(BaseModel):
    """Attraction model aligned with tool output"""
    name: str
    rating: float
    description: str
    location: str
    category: str
    estimated_cost: float = 0.0
    photos: List[str] = []


class Hotel(BaseModel):
    """Hotel model aligned with tool output"""
    name: str
    rating: float
    price_per_night: float
    location: str
    amenities: List[str] = []


class Flight(BaseModel):
    """Flight model aligned with tool output"""
    airline: str
    price: float
    duration: int  # minutes
    departure_time: str
    arrival_time: str


class FrameworkMetadata(BaseModel):
    """Metadata from the agent framework"""
    confidence: float
    actions_taken: List[str] = []
    next_steps: List[str] = []
    tools_used: List[str] = []
    processing_time: float = 0.0
    intent_analysis: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    refinement_iterations: Optional[int] = None


class TravelPlan(BaseModel):
    """Modern travel plan model using framework's structured output"""
    id: str
    request: TravelRequest
    
    # Rich content from agent
    content: str
    
    # Structured data from tools
    attractions: List[Attraction] = []
    hotels: List[Hotel] = []
    flights: List[Flight] = []
    
    # Framework metadata
    metadata: FrameworkMetadata
    
    # API metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    status: str = "generated"


class TravelPlanUpdate(BaseModel):
    """Travel plan update request"""
    feedback: Optional[str] = None
    updated_request: Optional[TravelRequest] = None


class TravelPlanResponse(BaseModel):
    """API response for travel plans"""
    success: bool
    plan: Optional[TravelPlan] = None
    error: Optional[str] = None
    message: Optional[str] = None 

# ===== Plan-related Models =====

class CalendarEventType(str, Enum):
    """Types of calendar events"""
    FLIGHT = "flight"
    HOTEL = "hotel"
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    TRANSPORTATION = "transportation"
    ACTIVITY = "activity"
    MEETING = "meeting"
    FREE_TIME = "free_time"

class CalendarEvent(BaseModel):
    """Individual calendar event"""
    id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    event_type: CalendarEventType = Field(..., description="Type of event")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    location: Optional[str] = Field(None, description="Event location")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    confidence: float = Field(0.8, description="Confidence score for this event")
    source: str = Field("agent", description="Source of this event (agent, user, etc.)")

class TravelPlanMetadata(BaseModel):
    """Metadata for travel plan"""
    destination: Optional[str] = None
    duration_days: Optional[int] = None
    travelers: int = 1
    budget: Optional[float] = None
    budget_currency: str = "USD"
    interests: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 0.8
    completion_status: str = "draft"  # draft, partial, complete

class SessionTravelPlan(BaseModel):
    """Travel plan for a session"""
    plan_id: str = Field(..., description="Unique plan identifier")
    session_id: str = Field(..., description="Associated session ID")
    events: List[CalendarEvent] = Field(default_factory=list, description="Calendar events")
    metadata: TravelPlanMetadata = Field(default_factory=TravelPlanMetadata, description="Plan metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PlanUpdateRequest(BaseModel):
    """Request to update a plan"""
    session_id: str
    events_to_add: List[CalendarEvent] = Field(default_factory=list)
    events_to_update: List[CalendarEvent] = Field(default_factory=list)
    events_to_remove: List[str] = Field(default_factory=list)  # Event IDs
    metadata_updates: Optional[TravelPlanMetadata] = None

class PlanResponse(BaseModel):
    """Response for plan operations"""
    success: bool
    plan: Optional[SessionTravelPlan] = None
    message: str
    events_count: int = 0 