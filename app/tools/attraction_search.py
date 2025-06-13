from typing import List, Optional
from datetime import datetime
import aiohttp
from pydantic import BaseModel

class AttractionAvailability(BaseModel):
    is_available: bool
    price: Optional[float] = None
    currency: Optional[str] = None
    opening_hours: Optional[dict] = None
    booking_url: Optional[str] = None

class Attraction(BaseModel):
    name: str
    description: str
    location: str
    category: str
    rating: float
    price_range: str
    opening_hours: dict

class AttractionSearch:
    def __init__(self):
        self.api_key = "YOUR_API_KEY"  # Replace with actual API key
        self.base_url = "https://api.attraction-search.com/v1"  # Replace with actual API endpoint

    async def search(
        self,
        location: str,
        category: Optional[str] = None,
        max_price: Optional[float] = None
    ) -> List[Attraction]:
        """
        Search for attractions in a specific location.
        """
        async with aiohttp.ClientSession() as session:
            params = {
                "location": location,
                "category": category,
                "max_price": max_price
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
                    return [self._parse_attraction(attraction) for attraction in data["attractions"]]
                else:
                    return []

    async def check_availability(
        self,
        attraction_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> AttractionAvailability:
        """
        Check if an attraction is available at specific times.
        """
        async with aiohttp.ClientSession() as session:
            params = {
                "name": attraction_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
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
                    data = await response.json()
                    return AttractionAvailability(**data)
                else:
                    return AttractionAvailability(is_available=False)

    def _parse_attraction(self, attraction_data: dict) -> Attraction:
        """
        Parse raw attraction data into an Attraction object.
        """
        return Attraction(
            name=attraction_data["name"],
            description=attraction_data["description"],
            location=attraction_data["location"],
            category=attraction_data["category"],
            rating=float(attraction_data["rating"]),
            price_range=attraction_data["price_range"],
            opening_hours=attraction_data["opening_hours"]
        ) 