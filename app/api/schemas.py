"""
API Layer Schemas - Streamlined data models for API contracts

These schemas serve as the API contract layer, working with the new framework's
structured outputs rather than requiring complex parsing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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