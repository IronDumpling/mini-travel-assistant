"""
Hotel Search Tool - Hotel Search Tool

Refactored based on the new architecture, integrating the tool base class
TODO: Improve the following features
1. Multi-platform hotel price comparison
2. User review analysis
3. Location optimization recommendations
4. Personalized recommendations
"""

import os
from amadeus import Client, ResponseError
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata

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
    total_results: int = 0
    search_location: str = ""

class HotelSearchTool(BaseTool):
    """Hotel search tool class using Amadeus API"""

    def __init__(self):
        metadata = ToolMetadata(
            name="hotel_search",
            description="Search for hotel accommodation information, with multiple filtering conditions (Amadeus API)",
            category="accommodation",
            tags=["hotel", "accommodation", "search", "booking", "amadeus"],
            timeout=30
        )
        super().__init__(metadata)
        self.amadeus = None

    def _ensure_api_client(self):
        if not self.amadeus:
            api_key = os.getenv("HOTEL_SEARCH_API_KEY")
            api_secret = os.getenv("HOTEL_SEARCH_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("HOTEL_SEARCH_API_KEY and HOTEL_SEARCH_API_SECRET must be set in the environment.")
            self.amadeus = Client(
                client_id=api_key,
                client_secret=api_secret
            )

    async def _execute(self, input_data: HotelSearchInput, context: ToolExecutionContext) -> HotelSearchOutput:
        self._ensure_api_client()
        try:
            hotels = await self._search_hotels(input_data)
            filtered_hotels = self._filter_hotels(hotels, input_data)
            return HotelSearchOutput(
                success=True,
                hotels=filtered_hotels,
                total_results=len(filtered_hotels),
                search_location=input_data.location,
                data={
                    "original_results": len(hotels),
                    "filtered_results": len(filtered_hotels),
                    "search_parameters": input_data.model_dump(),
                },
            )
        except Exception as e:
            return HotelSearchOutput(
                success=False,
                error=str(e),
                hotels=[],
                total_results=0,
                search_location=input_data.location,
            )

    async def _search_hotels(self, input_data: HotelSearchInput) -> List[Hotel]:
        self._ensure_api_client()
        # Amadeus API is synchronous, so run in thread executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_hotels_sync, input_data)

    def _search_hotels_sync(self, input_data: HotelSearchInput) -> List[Hotel]:
        self._ensure_api_client()
        # Geocode location to get city code (Amadeus requires city code, e.g. PAR for Paris)
        city_code = self._get_city_code(input_data.location)
        if not city_code:
            return []
        try:
            response = self.amadeus.shopping.hotel_offers.get(
                cityCode=city_code,
                checkInDate=input_data.check_in.strftime("%Y-%m-%d"),
                checkOutDate=input_data.check_out.strftime("%Y-%m-%d"),
                adults=input_data.guests,
                roomQuantity=input_data.rooms,
                # TODO: Add more filters as needed
            )
            hotels = []
            for hotel_offer in response.data:
                hotel_info = hotel_offer.get("hotel", {})
                name = hotel_info.get("name", "Unknown")
                location = hotel_info.get("address", {}).get("lines", [""])[0]
                rating = float(hotel_info.get("rating", 0.0))
                amenities = hotel_info.get("amenities", [])
                # Parse price from first available offer
                price_per_night = 0.0
                currency = "USD"
                if hotel_offer.get("offers"):
                    offer = hotel_offer["offers"][0]
                    price_per_night = float(offer["price"]["total"])/((input_data.check_out - input_data.check_in).days or 1)
                    currency = offer["price"]["currency"]
                hotels.append(Hotel(
                    name=name,
                    location=location,
                    price_per_night=price_per_night,
                    currency=currency,
                    rating=rating,
                    amenities=amenities
                ))
            return hotels
        except ResponseError as e:
            # TODO: Better error handling/logging
            return []

    def _get_city_code(self, location: str) -> Optional[str]:
        self._ensure_api_client()
        # Use Amadeus location API to get city code
        try:
            response = self.amadeus.reference_data.locations.get(
                keyword=location,
                subType="CITY"
            )
            if response.data and len(response.data) > 0:
                return response.data[0]["iataCode"]
        except Exception:
            pass
        return None

    def _filter_hotels(self, hotels: List[Hotel], criteria: HotelSearchInput) -> List[Hotel]:
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

    def get_input_schema(self) -> Dict[str, Any]:
        return HotelSearchInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        return HotelSearchOutput.model_json_schema()

from app.tools.base_tool import tool_registry
hotel_search_tool = HotelSearchTool()
tool_registry.register(hotel_search_tool) 