"""
Flight Search Tool - Flight Search Utility

Refactored based on the new architecture, integrating the tool base class and standardized interfaces  
Integrated with Amadeus API for real flight search functionality
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
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
    # For multi-destination search tracking
    search_destination: Optional[str] = None
    # For storing additional flight details and metadata
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class FlightSearchInput(ToolInput):
    """Flight search input"""
    origin: str = Field(description="IATA code or city of departure")
    destination: Optional[Union[str, List[str]]] = Field(default=None, description="IATA code/city of arrival or list of destinations for multi-destination search")
    start_date: Optional[datetime] = Field(default=None, description="Departure date (YYYY-MM-DD)")
    end_date: Optional[datetime] = Field(default=None, description="Return date (YYYY-MM-DD), if round trip")
    passengers: int = Field(default=1, description="Number of passengers")
    class_type: str = Field(default="economy", description="Travel class: economy, business, first")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter")
    preferred_airlines: List[str] = Field(default_factory=list, description="Preferred airlines (IATA codes)")
    inspiration_search: bool = Field(default=False, description="If true, use the Flight Inspiration Search API")
    flight_chain: bool = Field(default=False, description="If true, search flight chain: origin → dest1 → dest2 → ... → origin")

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
            timeout=60  # Increased to 60 seconds for multi-destination searches
        )
        super().__init__(metadata)
        
        # Amadeus API credentials - read from environment or use defaults
        self.api_key = os.getenv("AMADEUS_API_KEY", "GWY1ifYiXAfeF5alfhaOCVBc0DmKElZL")
        self.api_secret = os.getenv("AMADEUS_API_SECRET", "S03PNzgrbfMNOVz7")
        self.base_url = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com/v2")  # Updated to v2
        self.auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"  # Auth still uses v1
        self.access_token = None

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
            # ✅ Enhanced: Support multiple destinations by iterating through list
            if isinstance(input_data.destination, list):
                return await self._execute_multi_destination_search(input_data, context)
            
            # Handle origin as list (take first item as primary origin)
            if isinstance(input_data.origin, list):
                input_data.origin = input_data.origin[0] if input_data.origin else "YYZ"
                logger.warning(f"Flight search received origin as list, using first item: {input_data.origin}")
            
            # Set default origin to Toronto (YYZ) if not provided or invalid
            if not input_data.origin or input_data.origin.lower() in ['unknown', '']:
                input_data.origin = "YYZ"
                logger.info(f"No valid origin provided, using default departure location: Toronto (YYZ)")
            
            if not input_data.inspiration_search and (not input_data.destination or input_data.destination.lower() in ['unknown', '']):
                return FlightSearchOutput(
                    success=False,
                    error="Invalid destination: destination is required for flight offers search and cannot be 'unknown'",
                    flights=[]
                )
            
            # Check for O/D overlap - origin and destination cannot be the same
            if not input_data.inspiration_search and input_data.origin and input_data.destination:
                if input_data.origin.upper() == input_data.destination.upper():
                    logger.warning(f"⚠️ Origin and destination are the same ('{input_data.origin}') - O/D overlap detected")
                    return FlightSearchOutput(
                        success=False,
                        error=f"Invalid flight search: origin and destination cannot be the same ('{input_data.origin}')",
                        flights=[]
                    )
            
            # Validate that origin and destination are 3-letter codes or valid city names
            if len(input_data.origin) != 3 or not input_data.origin.isalpha():
                logger.warning(f"Origin '{input_data.origin}' may not be a valid IATA code")
            
            if input_data.destination and (len(input_data.destination) != 3 or not input_data.destination.isalpha()):
                logger.warning(f"Destination '{input_data.destination}' may not be a valid IATA code")
            
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

    async def _execute_multi_destination_search(
        self, input_data: FlightSearchInput, context: ToolExecutionContext
    ) -> FlightSearchOutput:
        """Execute flight search for multiple destinations - supports both multi-destination and flight chain modes"""
        
        destinations = input_data.destination
        if not destinations:
            return FlightSearchOutput(
                success=False,
                error="No valid destinations provided in list",
                flights=[]
            )
        
        # Handle origin as list (use first item)
        if isinstance(input_data.origin, list):
            origin = input_data.origin[0] if input_data.origin else "YYZ"
        else:
            origin = input_data.origin or "YYZ"
        
        # ✅ NEW: Check if this is a flight chain search
        if input_data.flight_chain:
            logger.info(f"🔗 Executing FLIGHT CHAIN search: {origin} → {' → '.join(destinations)} → {origin}")
            return await self._execute_flight_chain_search(input_data, context, origin, destinations)
        else:
            logger.info(f"🌍 Executing MULTI-DESTINATION search from {origin} to {len(destinations)} destinations: {destinations}")
            return await self._execute_traditional_multi_destination_search(input_data, context, origin, destinations)

    async def _execute_flight_chain_search(
        self, input_data: FlightSearchInput, context: ToolExecutionContext, origin: str, destinations: List[str]
    ) -> FlightSearchOutput:
        """Execute flight chain search: Origin → A → B → C → ... → Origin"""
        
        # Build complete flight chain: [Origin, A, B, C, ..., Origin]
        flight_chain = [origin] + destinations + [origin]
        logger.info(f"✈️ Flight chain: {' → '.join(flight_chain)}")
        
        all_flights = []
        successful_routes = []
        failed_routes = []
        
        # Search each consecutive pair in the chain
        for i in range(len(flight_chain) - 1):
            route_origin = flight_chain[i]
            route_destination = flight_chain[i + 1]
            route_name = f"{route_origin} → {route_destination}"
            
            # Skip if origin and destination are the same
            if route_origin.upper() == route_destination.upper():
                failed_routes.append(route_name)
                logger.warning(f"⚠️ Skipping route {route_name} - same origin and destination")
                continue
            
            try:
                # Create flight search for this route segment
                route_input = FlightSearchInput(
                    origin=route_origin,
                    destination=route_destination,
                    start_date=input_data.start_date,
                    end_date=input_data.end_date,
                    passengers=input_data.passengers,
                    class_type=input_data.class_type,
                    max_price=input_data.max_price,
                    inspiration_search=input_data.inspiration_search,
                    flight_chain=False  # Prevent recursion
                )
                
                logger.info(f"✈️ Searching route {i+1}/{len(flight_chain)-1}: {route_name}")
                
                # Execute search for this route with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_single_destination_search(route_input, context),
                        timeout=15.0  # Reduced timeout per route
                    )
                except asyncio.TimeoutError:
                    failed_routes.append(route_name)
                    logger.warning(f"⏰ Flight search for route {route_name} timed out after 15 seconds")
                    continue
                
                if result.success and result.flights:
                    # Tag flights with route information
                    for flight in result.flights:
                        flight.search_destination = route_destination
                        # Add route sequence info to flight details
                        flight.details.update({
                            "route_sequence": i + 1,
                            "total_routes": len(flight_chain) - 1,
                            "route_name": route_name,
                            "flight_chain": True
                        })
                        logger.debug(f"🏷️ Tagged flight {flight.flight_number} for route {route_name} (sequence {i+1})")
                    
                    all_flights.extend(result.flights)
                    successful_routes.append(route_name)
                    logger.info(f"✅ Found {len(result.flights)} flights for route {route_name}")
                else:
                    failed_routes.append(route_name)
                    logger.warning(f"⚠️ No flights found for route {route_name}: {result.error}")
                    
            except Exception as e:
                failed_routes.append(route_name)
                logger.error(f"❌ Error searching flights for route {route_name}: {e}")
        
        # Prepare flight chain result
        total_results = len(all_flights)
        search_summary = f"Flight chain search: {len(successful_routes)}/{len(flight_chain)-1} routes successful"
        
        if failed_routes:
            search_summary += f" (failed routes: {failed_routes})"
        
        if total_results == 0:
            return FlightSearchOutput(
                success=False,
                error=f"No flights found for any route in the flight chain {' → '.join(flight_chain)}",
                flights=[]
            )
        
        # Sort by route sequence, then by price
        all_flights.sort(key=lambda x: (
            x.details.get("route_sequence", 999) if hasattr(x, 'details') and x.details else 999,
            x.price or float('inf')
        ))
        
        logger.info(f"🎯 Flight chain search completed: {total_results} total flights for {len(successful_routes)} routes")
        
        return FlightSearchOutput(
            success=True,
            flights=all_flights,
            data={
                "search_type": "flight_chain",
                "flight_chain": flight_chain,
                "total_routes": len(flight_chain) - 1,
                "successful_routes": successful_routes,
                "failed_routes": failed_routes,
                "results_per_route": {route: len([f for f in all_flights if f.details and f.details.get("route_name") == route]) for route in successful_routes},
                "search_summary": search_summary,
                "origin": origin
            }
        )

    async def _execute_traditional_multi_destination_search(
        self, input_data: FlightSearchInput, context: ToolExecutionContext, origin: str, destinations: List[str]
    ) -> FlightSearchOutput:
        """Execute traditional multi-destination search: Origin → A, Origin → B, Origin → C"""
        
        all_flights = []
        successful_destinations = []
        failed_destinations = []
        
        # Execute search for each destination with individual timeout handling
        for destination in destinations:
            if not destination or destination.lower() in ['unknown', '']:
                failed_destinations.append(destination or "unknown")
                continue
                
            # Skip if origin and destination are the same
            if origin.upper() == destination.upper():
                failed_destinations.append(destination)
                logger.warning(f"⚠️ Skipping destination {destination} - same as origin {origin}")
                continue
                
            try:
                # Create a copy of input_data with single destination
                single_destination_input = FlightSearchInput(
                    origin=origin,
                    destination=destination,
                    start_date=input_data.start_date,
                    end_date=input_data.end_date,
                    passengers=input_data.passengers,
                    class_type=input_data.class_type,
                    max_price=input_data.max_price,
                    inspiration_search=input_data.inspiration_search,
                    flight_chain=False
                )
                
                logger.info(f"✈️ Searching flights from {origin} to {destination}")
                
                # Execute single destination search with individual timeout (20 seconds per destination)
                try:
                    result = await asyncio.wait_for(
                        self._execute_single_destination_search(single_destination_input, context),
                        timeout=20.0
                    )
                except asyncio.TimeoutError:
                    failed_destinations.append(destination)
                    logger.warning(f"⏰ Flight search to {destination} timed out after 20 seconds, continuing with other destinations")
                    continue
                
                if result.success and result.flights:
                    # Add destination context to each flight
                    for flight in result.flights:
                        flight.search_destination = destination
                        logger.debug(f"🏷️ Tagged flight {flight.flight_number} with search_destination: {destination}")
                    all_flights.extend(result.flights)
                    successful_destinations.append(destination)
                    logger.info(f"✅ Found {len(result.flights)} flights to {destination}, tagged with search context")
                else:
                    failed_destinations.append(destination)
                    logger.warning(f"⚠️ No flights found to {destination}: {result.error}")
                    
            except Exception as e:
                failed_destinations.append(destination)
                logger.error(f"❌ Error searching flights to {destination}: {e}")
        
        # Prepare result summary
        total_results = len(all_flights)
        search_summary = f"Searched flights from {origin} to {len(destinations)} destinations: {successful_destinations}"
        
        if failed_destinations:
            search_summary += f" (failed: {failed_destinations})"
        
        if total_results == 0:
            return FlightSearchOutput(
                success=False,
                error=f"No flights found to any of the {len(destinations)} destinations from {origin}",
                flights=[]
            )
        
        # Sort by price and departure time
        all_flights.sort(key=lambda x: (x.price or float('inf'), x.departure_time or datetime(1900, 1, 1)))
        
        logger.info(f"🎯 Traditional multi-destination flight search completed: {total_results} total flights to {len(successful_destinations)} destinations")
        
        return FlightSearchOutput(
            success=True,
            flights=all_flights,
            data={
                "search_type": "multi_destination",
                "searched_destinations": len(destinations),
                "successful_destinations": successful_destinations,
                "failed_destinations": failed_destinations,
                "results_per_destination": {dest: len([f for f in all_flights if f.search_destination == dest]) for dest in successful_destinations},
                "search_summary": search_summary,
                "origin": origin
            }
        )

    async def _execute_single_destination_search(
        self, input_data: FlightSearchInput, context: ToolExecutionContext
    ) -> FlightSearchOutput:
        """Execute flight search for a single destination (extracted from main logic)"""
        
        # Check for O/D overlap - origin and destination cannot be the same
        if not input_data.inspiration_search and input_data.origin and input_data.destination:
            if input_data.origin.upper() == input_data.destination.upper():
                logger.warning(f"⚠️ Origin and destination are the same ('{input_data.origin}') - O/D overlap detected")
                return FlightSearchOutput(
                    success=False,
                    error=f"Invalid flight search: origin and destination cannot be the same ('{input_data.origin}')",
                    flights=[]
                )
        
        # Validate that origin and destination are 3-letter codes or valid city names
        if input_data.origin and (len(input_data.origin) != 3 or not input_data.origin.isalpha()):
            logger.warning(f"Origin '{input_data.origin}' may not be a valid IATA code")
        
        if input_data.destination and (len(input_data.destination) != 3 or not input_data.destination.isalpha()):
            logger.warning(f"Destination '{input_data.destination}' may not be a valid IATA code")
        
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