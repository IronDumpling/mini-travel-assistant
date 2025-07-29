"""
Flight Search Tool - Flight Search Utility

Refactored based on the new architecture, integrating the tool base class and standardized interfaces  
TODO: Improve the following features  
1. Integration with multiple flight search APIs (Amadeus, Skyscanner, etc.)  
2. Intelligent price comparison and recommendations  
3. Flight change and cancellation notifications  
4. User preference memory  
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class Flight(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    stops: Optional[int] = None
    duration: Optional[int] = None  # in minutes
    # For inspiration search
    origin: Optional[str] = None
    destination: Optional[str] = None
    return_date: Optional[datetime] = None

class FlightSearchInput(ToolInput):
    """Flight search input"""
    origin: str = Field(description="IATA code or city of departure")
    destination: Optional[str] = Field(default=None, description="IATA code or city of arrival (optional for inspiration search)")
    start_date: Optional[datetime] = Field(default=None, description="Departure date (YYYY-MM-DD)")
    end_date: Optional[datetime] = Field(default=None, description="Return date (YYYY-MM-DD), if round trip")
    passengers: int = Field(default=1, description="Number of passengers")
    class_type: str = Field(default="economy", description="Travel class: economy, business, first")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter")
    preferred_airlines: List[str] = Field(default_factory=list, description="Preferred airlines (IATA codes)")
    inspiration_search: bool = Field(default=False, description="If true, use the Flight Inspiration Search API")

class FlightSearchOutput(ToolOutput):
    """Flight search output"""
    flights: List[Flight] = []

class FlightSearchTool(BaseTool):
    """Amadeus API based flight search tool"""

    def __init__(self):
        metadata = ToolMetadata(
            name="flight_search",
            description="Search for available flight information using Amadeus API with multiple filtering conditions",
            category="transportation",
            tags=["flight", "ticket", "search", "travel", "amadeus"],
            timeout=30
        )
        super().__init__(metadata)
        self.api_key = None
        self.api_secret = None
        self.access_token = None
        self.base_url = "https://test.api.amadeus.com/v2"  # Amadeus Self-Service API base
        self.auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"

    def _ensure_api_key(self) -> None:
        """Ensure API key and secret are available, checking environment if not already loaded."""
        if not self.api_key or not self.api_secret:
            self.api_key = os.getenv("FLIGHT_SEARCH_API_KEY")
            self.api_secret = os.getenv("FLIGHT_SEARCH_API_SECRET")
            if not self.api_key or not self.api_secret:
                raise ValueError("FLIGHT_SEARCH_API_KEY and FLIGHT_SEARCH_API_SECRET environment variables are required")

    async def _get_access_token(self) -> str:
        """Obtain OAuth2 access token from Amadeus API."""
        # Always get a fresh token to avoid expiration issues
        self._ensure_api_key()
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            async with session.post(self.auth_url, data=data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    return self.access_token
                else:
                    error_text = await response.text()
                    raise Exception(f"Amadeus Auth error: {response.status} - {error_text}")

    async def _execute(self, input_data: FlightSearchInput, context: ToolExecutionContext) -> FlightSearchOutput:
        """Execute flight search using Amadeus API or Inspiration Search API"""
        try:
            token = await self._get_access_token()
            if input_data.inspiration_search:
                flights = await self._search_flight_destinations(input_data, token)
                return FlightSearchOutput(
                    success=True,
                    flights=flights,
                    data={
                        "total_results": len(flights),
                        "search_type": "inspiration",
                        "search_parameters": input_data.model_dump(),
                    }
                )
            else:
                flights = await self._search_flights_amadeus(input_data, token)
                filtered_flights = self._filter_flights(flights, input_data)
                return FlightSearchOutput(
                    success=True,
                    flights=filtered_flights,
                    data={
                        "total_results": len(flights),
                        "filtered_results": len(filtered_flights),
                        "search_type": "offers",
                        "search_parameters": input_data.model_dump(),
                    }
                )
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            return FlightSearchOutput(
                success=False,
                error=str(e),
                flights=[]
            )

    async def _search_flights_amadeus(self, input_data: FlightSearchInput, token: str) -> List[Flight]:
        """Search for flights using Amadeus API."""
        async with aiohttp.ClientSession() as session:
            params = {
                "originLocationCode": input_data.origin,
                "adults": input_data.passengers,
                "travelClass": input_data.class_type.upper(),  # Fix: must be uppercase for Amadeus
                "currencyCode": "USD",
                "max": 20
            }
            if input_data.destination:
                params["destinationLocationCode"] = input_data.destination
            if input_data.start_date is not None:
                params["departureDate"] = input_data.start_date.strftime("%Y-%m-%d")
            if input_data.end_date is not None:
                params["returnDate"] = input_data.end_date.strftime("%Y-%m-%d")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            logger.debug(f"Calling flight search API with params: {params}")
            
            async with session.get(
                f"{self.base_url}/shopping/flight-offers",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_flight_amadeus(offer) for offer in data.get("data", [])]
                elif response.status == 401:
                    # Token expired, clear it and retry once
                    self.access_token = None
                    new_token = await self._get_access_token()
                    headers["Authorization"] = f"Bearer {new_token}"
                    async with session.get(
                        f"{self.base_url}/shopping/flight-offers",
                        params=params,
                        headers=headers
                    ) as retry_response:
                        if retry_response.status == 200:
                            data = await retry_response.json()
                            return [self._parse_flight_amadeus(offer) for offer in data.get("data", [])]
                        else:
                            error_text = await retry_response.text()
                            raise Exception(f"Amadeus Flight Search error: {retry_response.status} - {error_text}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Amadeus Flight Search error: {response.status} - {error_text}")

    async def _search_flight_destinations(self, input_data: FlightSearchInput, token: str) -> List[Flight]:
        """Search for flight inspiration destinations using Amadeus API."""
        async with aiohttp.ClientSession() as session:
            params = {
                "origin": input_data.origin,
            }
            if input_data.max_price is not None:
                params["maxPrice"] = str(input_data.max_price)
            if input_data.start_date is not None:
                params["departureDate"] = input_data.start_date.strftime("%Y-%m-%d")
            if input_data.end_date is not None:
                params["returnDate"] = input_data.end_date.strftime("%Y-%m-%d")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            async with session.get(
                "https://test.api.amadeus.com/v1/shopping/flight-destinations",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = []
                    for item in data.get("data", []):
                        flights.append(Flight(
                            origin=item.get("origin"),
                            destination=item.get("destination"),
                            departure_time=datetime.fromisoformat(item["departureDate"]) if "departureDate" in item else None,
                            return_date=datetime.fromisoformat(item["returnDate"]) if "returnDate" in item else None,
                            price=float(item["price"]["total"]) if "price" in item and "total" in item["price"] else None,
                            currency="EUR"  # Amadeus returns EUR by default for this endpoint
                        ))
                    return flights
                elif response.status == 401:
                    # Token expired, clear it and retry once
                    self.access_token = None
                    new_token = await self._get_access_token()
                    headers["Authorization"] = f"Bearer {new_token}"
                    async with session.get(
                        "https://test.api.amadeus.com/v1/shopping/flight-destinations",
                        params=params,
                        headers=headers
                    ) as retry_response:
                        if retry_response.status == 200:
                            data = await retry_response.json()
                            flights = []
                            for item in data.get("data", []):
                                flights.append(Flight(
                                    origin=item.get("origin"),
                                    destination=item.get("destination"),
                                    departure_time=datetime.fromisoformat(item["departureDate"]) if "departureDate" in item else None,
                                    return_date=datetime.fromisoformat(item["returnDate"]) if "returnDate" in item else None,
                                    price=float(item["price"]["total"]) if "price" in item and "total" in item["price"] else None,
                                    currency="EUR"  # Amadeus returns EUR by default for this endpoint
                                ))
                            return flights
                        else:
                            error_text = await retry_response.text()
                            raise Exception(f"Amadeus Inspiration Search error: {retry_response.status} - {error_text}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Amadeus Inspiration Search error: {response.status} - {error_text}")

    def _parse_flight_amadeus(self, offer: dict) -> Flight:
        """Parse Amadeus flight offer data into Flight model."""
        itinerary = offer["itineraries"][0]["segments"][0]
        price = float(offer["price"]["total"])
        return Flight(
            airline=itinerary["carrierCode"],
            flight_number=itinerary["number"],
            departure_time=datetime.fromisoformat(itinerary["departure"]["at"]),
            arrival_time=datetime.fromisoformat(itinerary["arrival"]["at"]),
            price=price,
            currency=offer["price"]["currency"],
            stops=len(offer["itineraries"][0]["segments"]) - 1,
            duration=self._parse_duration(itinerary["duration"])
        )

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string (e.g., 'PT2H30M') to minutes."""
        import re
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration_str)
        hours = int(match.group(1)) if match and match.group(1) else 0
        minutes = int(match.group(2)) if match and match.group(2) else 0
        return hours * 60 + minutes

    def _filter_flights(self, flights: List[Flight], criteria: FlightSearchInput) -> List[Flight]:
        filtered = flights
        if criteria.max_price:
            filtered = [f for f in filtered if f.price is not None and f.price <= criteria.max_price]
        if criteria.preferred_airlines:
            filtered = [f for f in filtered if f.airline in criteria.preferred_airlines]
        return filtered

    def get_input_schema(self) -> Dict[str, Any]:
        return FlightSearchInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        return FlightSearchOutput.model_json_schema()

# Register the tool
from app.tools.base_tool import tool_registry
flight_search_tool = FlightSearchTool()
tool_registry.register(flight_search_tool) 