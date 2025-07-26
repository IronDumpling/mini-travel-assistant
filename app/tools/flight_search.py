"""
Flight Search Tool - Flight Search Utility

Refactored based on the new architecture, integrating the tool base class and standardized interfaces  
Integrated with Amadeus API for real flight search functionality
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp
import json
from pydantic import BaseModel
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
    origin: str
    destination: str
    cabin_class: str

class FlightSearchInput(ToolInput):
    """Flight search input"""
    origin: str
    destination: str
    start_date: datetime
    end_date: Optional[datetime] = None
    passengers: int = 1
    class_type: str = "ECONOMY"  # ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    max_price: Optional[float] = None
    preferred_airlines: List[str] = []

class FlightSearchOutput(ToolOutput):
    """Flight search output"""
    flights: List[Flight] = []

class FlightSearchTool(BaseTool):
    """Flight search tool class using Amadeus API"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="flight_search",
            description="Search for available flight information using Amadeus API, with multiple filtering conditions",
            category="transportation",
            tags=["flight", "ticket", "search", "travel", "amadeus"],
            timeout=60
        )
        super().__init__(metadata)
        
        # Amadeus API credentials - read from environment or use defaults
        import os
        self.api_key = os.getenv("AMADEUS_API_KEY", "Kh68GhxPfbkDhFWFgBBc1ASBS6ptw0Ck")
        self.api_secret = os.getenv("AMADEUS_API_SECRET", "XhzA3QEJ7ZWvADos")
        self.base_url = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com/v2")  # Updated to v2
        self.auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"  # Auth still uses v1
        self.access_token = None
        self.token_expires_at = None

    async def _get_access_token(self) -> str:
        """Get OAuth2 access token from Amadeus"""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            print(f"Using cached token: {self.access_token[:20]}...")
            return self.access_token
        
        print("Getting new access token from Amadeus...")
        async with aiohttp.ClientSession() as session:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            print(f"Auth URL: {self.auth_url}")
            print(f"Auth data: {auth_data}")
            
            async with session.post(
                self.auth_url,
                data=auth_data,
                headers=headers
            ) as response:
                print(f"Token response status: {response.status}")
                response_text = await response.text()
                print(f"Token response: {response_text[:200]}...")
                
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    # Token expires in 30 minutes, set expiry 5 minutes earlier for safety
                    self.token_expires_at = datetime.now() + timedelta(minutes=25)
                    print(f"Token obtained successfully: {self.access_token[:20]}...")
                    return self.access_token
                else:
                    raise Exception(f"Failed to get access token: {response.status} - {response_text}")

    async def _execute(self, input_data: FlightSearchInput, context: ToolExecutionContext) -> FlightSearchOutput:
        """Execute flight search"""
        try:
            # Get access token
            access_token = await self._get_access_token()
            if not access_token:
                # Fallback to provided token for testing
                access_token = "F33nvpLougZGcAV871S19ree1mlo"
                print("Using fallback access token for testing")
            
            # Search for flights
            flights = await self._search_flights(
                input_data.origin,
                input_data.destination,
                input_data.start_date,
                input_data.end_date,
                input_data.passengers,
                input_data.class_type,
                access_token
            )
            
            # Filter flights based on user preferences
            filtered_flights = self._filter_flights(flights, input_data)
            
            return FlightSearchOutput(
                success=True,
                flights=filtered_flights,
                data={
                    "total_results": len(flights),
                    "filtered_results": len(filtered_flights),
                    "search_params": {
                        "origin": input_data.origin,
                        "destination": input_data.destination,
                        "departure_date": input_data.start_date.strftime("%Y-%m-%d"),
                        "return_date": input_data.end_date.strftime("%Y-%m-%d") if input_data.end_date else None,
                        "passengers": input_data.passengers,
                        "class_type": input_data.class_type
                    }
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
        passengers: int = 1,
        class_type: str = "ECONOMY",
        access_token: str = None
    ) -> List[Flight]:
        """Search for flights using Amadeus Flight Offers Search v2 API"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/vnd.amadeus+json"  # v2 API requires this content type
            }
            
            # Create a mapping of common city names to airport codes
            city_to_airport = {
                'ROME': ['FCO', 'CIA'],  # Rome Fiumicino and Ciampino
                'PARIS': ['CDG', 'ORY'],  # Paris Charles de Gaulle and Orly
                'LONDON': ['LHR', 'LGW', 'STN'],  # London Heathrow, Gatwick, Stansted
                'BEIJING': ['PEK', 'PKX'],  # Beijing Capital and Daxing
                'TOKYO': ['NRT', 'HND'],  # Tokyo Narita and Haneda
                'NEW YORK': ['JFK', 'EWR', 'LGA'],  # New York airports
                'BARCELONA': ['BCN'],
                'MADRID': ['MAD'],
                'AMSTERDAM': ['AMS'],
                'BERLIN': ['BER'],
                'MUNICH': ['MUC'],
                'VIENNA': ['VIE'],
                'PRAGUE': ['PRG'],
                'BUDAPEST': ['BUD'],
                'SINGAPORE': ['SIN'],
                'SEOUL': ['ICN', 'GMP'],
                'SHANGHAI': ['PVG', 'SHA'],
                'KYOTO': ['UKB'],  # Kyoto uses Osaka Kansai
            }
            
            # Get possible airport codes for origin and destination
            origin_upper = origin.upper()
            destination_upper = destination.upper()
            
            possible_origin_airports = city_to_airport.get(origin_upper, [origin_upper])
            possible_dest_airports = city_to_airport.get(destination_upper, [destination_upper])
            
            print(f"Searching flights from {origin_upper} to {destination_upper}")
            print(f"Possible origin airports: {possible_origin_airports}")
            print(f"Possible destination airports: {possible_dest_airports}")
            
            all_flights = []
            
            # Search for flights from each possible origin airport to each possible destination
            for origin_airport in possible_origin_airports:
                for dest_airport in possible_dest_airports:
                    # Build search parameters for Flight Offers Search v2 API
                    params = {
                        "originLocationCode": origin_airport,
                        "destinationLocationCode": dest_airport,
                        "departureDate": start_date.strftime("%Y-%m-%d"),
                        "adults": passengers,
                        "currencyCode": "USD",
                        "max": 5,  # Limit to 5 results
                        "travelClass": class_type
                    }
                    
                    # Add return date if provided
                    if end_date:
                        params["returnDate"] = end_date.strftime("%Y-%m-%d")
                    
                    # Use the Flight Offers Search v2 endpoint
                    url = f"{self.base_url}/shopping/flight-offers"
                    
                    print(f"Making request to: {url}")
                    print(f"Parameters: {params}")
                    print(f"Headers: {headers}")
                    
                    async with session.get(url, params=params, headers=headers) as response:
                        print(f"Response status: {response.status}")
                        response_text = await response.text()
                        print(f"Response body: {response_text[:500]}...")  # Print first 500 chars
                        
                        if response.status == 200:
                            data = await response.json()
                            flight_offers = data.get("data", [])
                            print(f"Found {len(flight_offers)} flight offers")
                            
                            # Parse each flight offer
                            for offer in flight_offers:
                                try:
                                    flight = self._parse_amadeus_flight(offer, origin, destination)
                                    if flight:
                                        all_flights.append(flight)
                                except Exception as e:
                                    print(f"Error parsing flight offer: {e}")
                                    continue
                        else:
                            print(f"Flight search failed for {origin_airport} to {dest_airport}: {response.status} - {response_text}")
            
            print(f"Total flights found: {len(all_flights)}")
            return all_flights
    
    def _parse_destination_to_flight(self, destination_data: dict, origin: str, destination: str, start_date: datetime) -> Flight:
        """Parse destination data to flight format"""
        # Extract price information
        price_data = destination_data.get("price", {})
        price = float(price_data.get("total", 0))
        currency = "USD"  # Default currency
        
        # Get departure and return dates
        departure_date = destination_data.get("departureDate", start_date.strftime("%Y-%m-%d"))
        return_date = destination_data.get("returnDate")
        
        # Create mock flight data since destinations API doesn't provide detailed flight info
        departure_time = datetime.fromisoformat(f"{departure_date}T10:00:00+00:00")
        arrival_time = datetime.fromisoformat(f"{departure_date}T18:00:00+00:00")
        
        return Flight(
            airline="Multiple Airlines",  # Destinations API doesn't provide specific airline
            flight_number="Various",
            departure_time=departure_time,
            arrival_time=arrival_time,
            price=price,
            currency=currency,
            stops=1,  # Default to 1 stop for international flights
            duration=480,  # 8 hours default duration
            origin=origin,
            destination=destination,
            cabin_class="ECONOMY"
        )
    
    def _parse_amadeus_flight(self, flight_data: dict, origin: str, destination: str) -> Flight:
        """Parse Amadeus flight data (for flight offers API)"""
        # Extract itinerary information
        itineraries = flight_data.get("itineraries", [])
        if not itineraries:
            raise ValueError("No itinerary data found")
        
        # Get first itinerary (outbound)
        outbound = itineraries[0]
        segments = outbound.get("segments", [])
        
        if not segments:
            raise ValueError("No segments found in itinerary")
        
        # Calculate total duration and stops
        total_duration = outbound.get("duration", "PT0H0M")
        duration_minutes = self._parse_duration(total_duration)
        stops = len(segments) - 1
        
        # Get first and last segment for departure/arrival times
        first_segment = segments[0]
        last_segment = segments[-1]
        
        # Parse departure and arrival times
        departure_time = datetime.fromisoformat(first_segment["departure"]["at"].replace("Z", "+00:00"))
        arrival_time = datetime.fromisoformat(last_segment["arrival"]["at"].replace("Z", "+00:00"))
        
        # Get airline information from first segment
        airline = first_segment["carrierCode"]
        flight_number = first_segment["number"]
        
        # Get pricing information
        pricing = flight_data.get("pricingOptions", {})
        price = float(flight_data.get("price", {}).get("total", 0))
        currency = flight_data.get("price", {}).get("currency", "USD")
        
        # Get cabin class
        cabin_class = flight_data.get("travelerPricings", [{}])[0].get("fareDetailsBySegment", [{}])[0].get("cabin", "ECONOMY")
        
        return Flight(
            airline=airline,
            flight_number=flight_number,
            departure_time=departure_time,
            arrival_time=arrival_time,
            price=price,
            currency=currency,
            stops=stops,
            duration=duration_minutes,
            origin=origin,
            destination=destination,
            cabin_class=cabin_class
        )
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string to minutes"""
        # Format: PT2H30M (2 hours 30 minutes)
        hours = 0
        minutes = 0
        
        if "H" in duration_str:
            hours = int(duration_str.split("H")[0].split("T")[1])
        if "M" in duration_str:
            minutes = int(duration_str.split("M")[0].split("H")[-1] if "H" in duration_str else duration_str.split("T")[1])
        
        return hours * 60 + minutes
    
    def _filter_flights(self, flights: List[Flight], criteria: FlightSearchInput) -> List[Flight]:
        """Filter flights based on search criteria"""
        filtered = flights
        
        if criteria.max_price:
            filtered = [f for f in filtered if f.price <= criteria.max_price]
        
        if criteria.preferred_airlines:
            filtered = [f for f in filtered if f.airline in criteria.preferred_airlines]
        
        # Sort by price (lowest first)
        filtered.sort(key=lambda x: x.price)
        
        return filtered
    
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