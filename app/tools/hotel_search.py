from typing import List, Optional
from datetime import datetime
import aiohttp
from pydantic import BaseModel

class Hotel(BaseModel):
    name: str
    location: str
    price_per_night: float
    currency: str
    rating: float
    amenities: List[str]

class HotelSearch:
    def __init__(self):
        self.api_key = "YOUR_API_KEY"  # Replace with actual API key
        self.base_url = "https://api.hotel-search.com/v1"  # Replace with actual API endpoint

    async def search(
        self,
        location: str,
        check_in: datetime,
        check_out: datetime,
        guests: int = 1
    ) -> List[Hotel]:
        """
        Search for available hotels using the hotel search API.
        """
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
            
            async with session.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_hotel(hotel) for hotel in data["hotels"]]
                else:
                    return []

    def _parse_hotel(self, hotel_data: dict) -> Hotel:
        """
        Parse raw hotel data into a Hotel object.
        """
        return Hotel(
            name=hotel_data["name"],
            location=hotel_data["location"],
            price_per_night=float(hotel_data["price_per_night"]),
            currency=hotel_data["currency"],
            rating=float(hotel_data["rating"]),
            amenities=hotel_data["amenities"]
        ) 