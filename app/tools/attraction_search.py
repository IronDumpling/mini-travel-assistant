"""
Attraction Search Tool - Attraction Search Tool

Refactored based on the new architecture, integrating the tool base class and standardized interfaces
TODO: Improve the following features
1. Multi-platform attraction data aggregation
2. Real-time availability and booking integration
3. User review sentiment analysis
4. Personalized recommendations based on interests
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata

class Attraction(BaseModel):
    name: str
    description: str
    location: str
    category: str
    rating: float
    price_range: str
    opening_hours: Dict[str, str]
    price: Optional[float] = None
    currency: Optional[str] = None
    booking_url: Optional[str] = None
    image_url: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None

class AttractionSearchInput(ToolInput):
    """Attraction search input"""
    location: str
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    date: Optional[datetime] = None
    interests: List[str] = []
    radius_km: Optional[float] = None

class AttractionSearchOutput(ToolOutput):
    """Attraction search output"""
    attractions: List[Attraction] = []

class AttractionSearchTool(BaseTool):
    """Attraction search tool class"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="attraction_search",
            description="Search for tourist attractions and activities with filtering and availability checking",
            category="entertainment",
            tags=["attraction", "tourism", "activity", "sightseeing", "entertainment"],
            timeout=30
        )
        super().__init__(metadata)
        
        self.api_key = "YOUR_API_KEY"  # TODO: Read from configuration
        self.base_url = "https://api.attraction-search.com/v1"  # TODO: Support multiple API sources

    async def _execute(self, input_data: AttractionSearchInput, context: ToolExecutionContext) -> AttractionSearchOutput:
        """Execute attraction search"""
        try:
            attractions = await self._search_attractions(
                input_data.location,
                input_data.category,
                input_data.max_price,
                input_data.date
            )
            
            # Filter attractions based on search criteria
            filtered_attractions = self._filter_attractions(attractions, input_data)
            
            return AttractionSearchOutput(
                success=True,
                attractions=filtered_attractions,
                data={
                    "total_results": len(attractions),
                    "filtered_results": len(filtered_attractions),
                    "search_location": input_data.location,
                    "search_category": input_data.category
                }
            )
        except Exception as e:
            return AttractionSearchOutput(
                success=False,
                error=str(e),
                attractions=[]
            )

    async def _search_attractions(
        self,
        location: str,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        date: Optional[datetime] = None
    ) -> List[Attraction]:
        """Search for attractions"""
        async with aiohttp.ClientSession() as session:
            params = {
                "location": location
            }
            
            if category:
                params["category"] = category
            if max_price:
                params["max_price"] = max_price
            if date:
                params["date"] = date.strftime("%Y-%m-%d")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_attraction(attraction) for attraction in data["attractions"]]
                else:
                    # TODO: Better error handling
                    return []

    def _filter_attractions(self, attractions: List[Attraction], criteria: AttractionSearchInput) -> List[Attraction]:
        """Filter attractions based on search criteria"""
        filtered = attractions
        
        if criteria.min_rating:
            filtered = [a for a in filtered if a.rating >= criteria.min_rating]
        
        if criteria.max_price and criteria.max_price > 0:
            filtered = [a for a in filtered if a.price is None or a.price <= criteria.max_price]
        
        if criteria.interests:
            # TODO: Implement intelligent interest matching
            # For now, simple keyword matching in description and category
            filtered = [
                a for a in filtered 
                if any(interest.lower() in a.description.lower() or 
                      interest.lower() in a.category.lower() 
                      for interest in criteria.interests)
            ]
        
        # TODO: Add location-based filtering if radius_km is specified
        # TODO: Add more sophisticated filtering logic
        
        return filtered

    def _parse_attraction(self, attraction_data: dict) -> Attraction:
        """Parse attraction data"""
        return Attraction(
            name=attraction_data["name"],
            description=attraction_data["description"],
            location=attraction_data["location"],
            category=attraction_data["category"],
            rating=float(attraction_data["rating"]),
            price_range=attraction_data["price_range"],
            opening_hours=attraction_data["opening_hours"],
            price=attraction_data.get("price"),
            currency=attraction_data.get("currency"),
            booking_url=attraction_data.get("booking_url"),
            image_url=attraction_data.get("image_url"),
            contact_info=attraction_data.get("contact_info")
        )

    async def check_availability(
        self,
        attraction_name: str,
        date: datetime,
        context: Optional[ToolExecutionContext] = None
    ) -> Dict[str, Any]:
        """Check attraction availability for a specific date"""
        async with aiohttp.ClientSession() as session:
            params = {
                "name": attraction_name,
                "date": date.strftime("%Y-%m-%d")
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.get(
                f"{self.base_url}/availability",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "is_available": False,
                        "error": f"Failed to check availability: {response.status}"
                    }

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema"""
        return AttractionSearchInput.model_json_schema()
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema"""
        return AttractionSearchOutput.model_json_schema()

# Register the tool
from app.tools.base_tool import tool_registry
attraction_search_tool = AttractionSearchTool()
tool_registry.register(attraction_search_tool) 