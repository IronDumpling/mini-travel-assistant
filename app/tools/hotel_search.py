"""
Hotel Search Tool - Hotel Search Tool

Refactored based on the new architecture, integrating the tool base class
TODO: Improve the following features
1. Multi-platform hotel price comparison
2. User review analysis
3. Location optimization recommendations
4. Personalized recommendations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
from pydantic import BaseModel
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class Hotel(BaseModel):
    name: str
    location: str
    price_per_night: float
    currency: str
    rating: float
    amenities: List[str]

class HotelSearchInput(ToolInput):
    """Hotel search input"""
    location: str
    check_in: datetime
    check_out: datetime
    guests: int = 1
    rooms: int = 1
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    required_amenities: List[str] = []

class HotelSearchOutput(ToolOutput):
    """Hotel search output"""
    hotels: List[Hotel] = []

class HotelSearchTool(BaseTool):
    """Hotel search tool class"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="hotel_search",
            description="Search for hotel accommodation information, with multiple filtering conditions",
            category="accommodation",
            tags=["hotel", "accommodation", "search", "booking"],
            timeout=30
        )
        super().__init__(metadata)
        
        self.api_key = "YOUR_API_KEY"  # TODO: Read from configuration
        self.base_url = "https://api.hotel-search.com/v1"  # TODO: Support multiple API sources

    async def _execute(self, input_data: HotelSearchInput, context: ToolExecutionContext) -> HotelSearchOutput:
        """Execute hotel search"""
        try:
            logger.info(f"Searching hotels in {input_data.location} for {input_data.guests} guests, check-in: {input_data.check_in.strftime('%Y-%m-%d')}")
            
            hotels = await self._search_hotels(
                input_data.location,
                input_data.check_in,
                input_data.check_out,
                input_data.guests
            )
            
            # Filter hotels based on search criteria
            filtered_hotels = self._filter_hotels(hotels, input_data)
            
            logger.info(f"Hotel search completed: {len(hotels)} total, {len(filtered_hotels)} after filtering")
            
            return HotelSearchOutput(
                success=True,
                hotels=filtered_hotels,
                data={
                    "total_results": len(hotels),
                    "filtered_results": len(filtered_hotels)
                }
            )
        except Exception as e:
            logger.error(f"Hotel search failed for {input_data.location}: {e}")
            return HotelSearchOutput(
                success=False,
                error=str(e),
                hotels=[]
            )
    
    async def _search_hotels(
        self,
        location: str,
        check_in: datetime,
        check_out: datetime,
        guests: int = 1
    ) -> List[Hotel]:
        """Search for hotels"""
        async with aiohttp.ClientSession() as session:
            params = {
                "location": location,
                "check_in": check_in.strftime("%Y-%m-%d"),
                "check_out": check_out.strftime("%Y-%m-%d"),
                "guests": guests
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.debug(f"Calling hotel search API with params: {params}")
            
            async with session.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    hotels_count = len(data.get("hotels", []))
                    logger.debug(f"Hotel API returned {hotels_count} results")
                    return [self._parse_hotel(hotel) for hotel in data["hotels"]]
                else:
                    logger.warning(f"Hotel search API failed with status {response.status}")
                    return []
    
    def _filter_hotels(self, hotels: List[Hotel], criteria: HotelSearchInput) -> List[Hotel]:
        """Filter hotels based on search criteria"""
        filtered = hotels
        
        if criteria.min_rating:
            filtered = [h for h in filtered if h.rating >= criteria.min_rating]
        
        if criteria.max_price:
            filtered = [h for h in filtered if h.price_per_night <= criteria.max_price]
        
        if criteria.required_amenities:
            filtered = [
                h for h in filtered 
                if all(amenity in h.amenities for amenity in criteria.required_amenities)
            ]
        
        return filtered

    def _parse_hotel(self, hotel_data: dict) -> Hotel:
        """Parse hotel data"""
        return Hotel(
            name=hotel_data["name"],
            location=hotel_data["location"],
            price_per_night=float(hotel_data["price_per_night"]),
            currency=hotel_data["currency"],
            rating=float(hotel_data["rating"]),
            amenities=hotel_data["amenities"]
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema"""
        return HotelSearchInput.model_json_schema()
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema"""
        return HotelSearchOutput.model_json_schema()

# Register the tool
from app.tools.base_tool import tool_registry
hotel_search_tool = HotelSearchTool()
tool_registry.register(hotel_search_tool) 