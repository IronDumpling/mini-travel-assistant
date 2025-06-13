from typing import List, Optional
from datetime import datetime
import aiohttp
from pydantic import BaseModel

class Flight(BaseModel):
    airline: str
    flight_number: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    currency: str
    stops: int
    duration: int  # in minutes

class FlightSearch:
    def __init__(self):
        self.api_key = "YOUR_API_KEY"  # Replace with actual API key
        self.base_url = "https://api.flight-search.com/v1"  # Replace with actual API endpoint

    async def search(
        self,
        origin: str,
        destination: str,
        start_date: datetime,
        end_date: datetime,
        passengers: int = 1
    ) -> List[Flight]:
        """
        Search for available flights using the flight search API.
        """
        async with aiohttp.ClientSession() as session:
            params = {
                "origin": origin,
                "destination": destination,
                "departure_date": start_date.strftime("%Y-%m-%d"),
                "return_date": end_date.strftime("%Y-%m-%d"),
                "passengers": passengers
            }
            
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
                    # In a real implementation, handle errors appropriately
                    return []

    def _parse_flight(self, flight_data: dict) -> Flight:
        """
        Parse raw flight data into a Flight object.
        """
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