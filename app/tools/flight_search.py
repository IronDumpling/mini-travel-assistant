"""
Flight Search Tool - Flight Search Utility

Refactored based on the new architecture, integrating the tool base class and standardized interfaces  
TODO: Improve the following features  
1. Integration with multiple flight search APIs (Amadeus, Skyscanner, etc.)  
2. Intelligent price comparison and recommendations  
3. Flight change and cancellation notifications  
4. User preference memory  
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata

class Flight(BaseModel):
    airline: str
    flight_number: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    currency: str
    stops: int
    duration: int  # in minutes

class FlightSearchInput(ToolInput):
    """Flight search input"""
    origin: str
    destination: str
    start_date: datetime
    end_date: Optional[datetime] = None
    passengers: int = 1
    class_type: str = "economy"  # economy, business, first
    max_price: Optional[float] = None
    preferred_airlines: List[str] = []

class FlightSearchOutput(ToolOutput):
    """Flight search output"""
    flights: List[Flight] = []

class FlightSearchTool(BaseTool):
    """Flight search tool class"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="flight_search",
            description="Search for available flight information, with multiple filtering conditions",
            category="transportation",
            tags=["flight", "ticket", "search", "travel"],
            timeout=30
        )
        super().__init__(metadata)
        
        self.api_key = "YOUR_API_KEY"  # TODO: Read from configuration
        self.base_url = "https://api.flight-search.com/v1"  # TODO: Support multiple API sources

    async def _execute(self, input_data: FlightSearchInput, context: ToolExecutionContext) -> FlightSearchOutput:
        """Execute flight search"""
        try:
            flights = await self._search_flights(
                input_data.origin,
                input_data.destination,
                input_data.start_date,
                input_data.end_date,
                input_data.passengers
            )
            
            # TODO: Filter and sort flights based on user preferences
            filtered_flights = self._filter_flights(flights, input_data)
            
            return FlightSearchOutput(
                success=True,
                flights=filtered_flights,
                data={
                    "total_results": len(flights),
                    "filtered_results": len(filtered_flights)
                }
            )
        except Exception as e:
            return FlightSearchOutput(
                success=False,
                error=str(e),
                flights=[]
            )
    
    async def _search_flights(
        self,
        origin: str,
        destination: str,
        start_date: datetime,
        end_date: Optional[datetime],
        passengers: int = 1
    ) -> List[Flight]:
        """Search for flights"""
        async with aiohttp.ClientSession() as session:
            params = {
                "origin": origin,
                "destination": destination,
                "departure_date": start_date.strftime("%Y-%m-%d"),
                "passengers": passengers
            }
            
            if end_date:
                params["return_date"] = end_date.strftime("%Y-%m-%d")
            
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
                    return [self._parse_flight(flight) for flight in data["flights"]]
                else:
                    # TODO: Better error handling
                    return []
    
    def _filter_flights(self, flights: List[Flight], criteria: FlightSearchInput) -> List[Flight]:
        """Filter flights based on search criteria"""
        # TODO: Implement intelligent filtering logic
        filtered = flights
        
        if criteria.max_price:
            filtered = [f for f in filtered if f.price <= criteria.max_price]
        
        if criteria.preferred_airlines:
            filtered = [f for f in filtered if f.airline in criteria.preferred_airlines]
        
        # TODO: Add more filtering conditions
        return filtered
    
    def _parse_flight(self, flight_data: dict) -> Flight:
        """Parse flight data"""
        return Flight(
            airline=flight_data["airline"],
            flight_number=flight_data["flight_number"],
            departure_time=datetime.fromisoformat(flight_data["departure_time"]),
            arrival_time=datetime.fromisoformat(flight_data["arrival_time"]),
            price=float(flight_data["price"]),
            currency=flight_data["currency"],
            stops=int(flight_data["stops"]),
            duration=int(flight_data["duration"])
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema"""
        return FlightSearchInput.model_json_schema()
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema"""
        return FlightSearchOutput.model_json_schema()

# Register the tool
from app.tools.base_tool import tool_registry
flight_search_tool = FlightSearchTool()
tool_registry.register(flight_search_tool) 