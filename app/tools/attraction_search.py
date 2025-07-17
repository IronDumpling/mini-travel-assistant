"""
Attraction Search Tool - Google Places API (New) Integration

A comprehensive tool for searching tourist attractions and activities using Google Places API (New).
Features include text search, nearby search, place details, and photo retrieval.
"""

import os
from typing import List, Optional, Dict, Any
import aiohttp
import json
from pydantic import BaseModel, Field
from app.tools.base_tool import (
    BaseTool,
    ToolInput,
    ToolOutput,
    ToolExecutionContext,
    ToolMetadata,
)


class Attraction(BaseModel):
    """Attraction data model"""

    name: str
    description: str
    location: str
    category: str
    rating: float
    price_level: Optional[int] = None  # 0-4 scale from Google Places
    opening_hours: Optional[Dict[str, str]] = None
    place_id: Optional[str] = None
    formatted_address: Optional[str] = None
    phone_number: Optional[str] = None
    website: Optional[str] = None
    photo_urls: List[str] = []
    reviews_count: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AttractionSearchInput(ToolInput):
    """Attraction search input parameters"""

    location: str = Field(description="Location to search for attractions")
    query: Optional[str] = Field(
        default=None, description="Search query for specific attractions"
    )
    category: Optional[str] = Field(
        default=None,
        description="Attraction category (e.g., 'tourist_attraction', 'museum', 'park')",
    )
    radius_meters: Optional[int] = Field(
        default=5000, description="Search radius in meters (max 50000)"
    )
    min_rating: Optional[float] = Field(
        default=None, description="Minimum rating filter (0-5)"
    )
    max_results: Optional[int] = Field(
        default=10, description="Maximum number of results"
    )
    include_photos: Optional[bool] = Field(
        default=True, description="Include photo URLs in results"
    )
    price_levels: Optional[List[int]] = Field(
        default=None, description="Price level filter (0-4)"
    )


class AttractionSearchOutput(ToolOutput):
    """Attraction search output"""

    attractions: List[Attraction] = []
    total_results: int = 0
    search_location: str = ""
    next_page_token: Optional[str] = None


class AttractionSearchTool(BaseTool):
    """Google Places API (New) based attraction search tool"""

    def __init__(self):
        metadata = ToolMetadata(
            name="attraction_search",
            description="Search for tourist attractions using Google Places API (New) with advanced filtering",
            category="entertainment",
            tags=[
                "attraction",
                "tourism",
                "activity",
                "sightseeing",
                "entertainment",
                "google_places",
            ],
            timeout=30,
        )
        super().__init__(metadata)

        # API key will be checked when the tool is actually used
        self.api_key = None

        # Google Places API (New) endpoints
        self.base_url = "https://places.googleapis.com/v1"
        self.places_search_url = f"{self.base_url}/places:searchText"
        self.nearby_search_url = f"{self.base_url}/places:searchNearby"
        self.place_details_url = f"{self.base_url}/places"
        self.place_photos_url = f"{self.base_url}/places"

    def _ensure_api_key(self) -> None:
        """Ensure API key is available, checking environment if not already loaded."""
        if not self.api_key:
            self.api_key = os.getenv("ATTRACTION_SEARCH_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "ATTRACTION_SEARCH_API_KEY environment variable is required"
                )

    async def _execute(
        self, input_data: AttractionSearchInput, context: ToolExecutionContext
    ) -> AttractionSearchOutput:
        """Execute attraction search using Google Places API (New)"""
        try:
            # Ensure API key is available before proceeding
            self._ensure_api_key()

            # Determine search strategy based on input
            if input_data.query:
                # Use text search for specific queries
                attractions = await self._text_search(input_data)
            else:
                # Use nearby search for location-based discovery
                attractions = await self._nearby_search(input_data)

            # Filter attractions based on criteria
            filtered_attractions = self._filter_attractions(attractions, input_data)

            # Limit results
            if input_data.max_results:
                filtered_attractions = filtered_attractions[: input_data.max_results]

            return AttractionSearchOutput(
                success=True,
                attractions=filtered_attractions,
                total_results=len(filtered_attractions),
                search_location=input_data.location,
                data={
                    "search_type": "text" if input_data.query else "nearby",
                    "original_results": len(attractions),
                    "filtered_results": len(filtered_attractions),
                    "search_parameters": input_data.model_dump(),
                },
            )

        except Exception as e:
            return AttractionSearchOutput(
                success=False,
                error=str(e),
                attractions=[],
                total_results=0,
                search_location=input_data.location,
            )

    async def _text_search(self, input_data: AttractionSearchInput) -> List[Attraction]:
        """Perform text search using Google Places API (New)"""
        async with aiohttp.ClientSession() as session:
            # Build search query
            search_query = (
                f"{input_data.query} in {input_data.location}"
                if input_data.query
                else input_data.location
            )

            request_body = {
                "textQuery": search_query,
                "maxResultCount": min(
                    input_data.max_results or 20, 20
                ),  # Max 20 for text search
                "locationBias": {
                    "rectangle": {
                        "low": {"latitude": -90, "longitude": -180},
                        "high": {"latitude": 90, "longitude": 180},
                    }
                },
            }

            # Add category filter if specified
            if input_data.category:
                request_body["includedType"] = input_data.category

            # Add price level filter if specified
            if input_data.price_levels:
                request_body["priceLevel"] = input_data.price_levels

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri,places.photos",
            }

            async with session.post(
                self.places_search_url, json=request_body, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        await self._parse_place(place, input_data.include_photos)
                        for place in data.get("places", [])
                    ]
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Google Places API error: {response.status} - {error_text}"
                    )

    async def _nearby_search(
        self, input_data: AttractionSearchInput
    ) -> List[Attraction]:
        """Perform nearby search using Google Places API (New)"""
        async with aiohttp.ClientSession() as session:
            # For nearby search, we need to geocode the location first
            location_coords = await self._geocode_location(input_data.location)

            request_body = {
                "maxResultCount": min(input_data.max_results or 20, 20),
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": location_coords["lat"],
                            "longitude": location_coords["lng"],
                        },
                        "radius": min(
                            input_data.radius_meters or 5000, 50000
                        ),  # Max 50km
                    }
                },
            }

            # Add category filter if specified
            if input_data.category:
                request_body["includedTypes"] = [input_data.category]
            else:
                # Default to tourist attractions if no category specified
                request_body["includedTypes"] = [
                    "tourist_attraction",
                    "museum",
                    "park",
                    "zoo",
                    "aquarium",
                    "amusement_park",
                ]

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri,places.photos",
            }

            async with session.post(
                self.nearby_search_url, json=request_body, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        await self._parse_place(place, input_data.include_photos)
                        for place in data.get("places", [])
                    ]
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Google Places API error: {response.status} - {error_text}"
                    )

    async def _geocode_location(self, location: str) -> Dict[str, float]:
        """Geocode a location string to coordinates using Google Geocoding API"""
        async with aiohttp.ClientSession() as session:
            params = {"address": location, "key": self.api_key}

            async with session.get(
                "https://maps.googleapis.com/maps/api/geocode/json", params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["results"]:
                        return data["results"][0]["geometry"]["location"]
                    else:
                        # Default to a reasonable location if geocoding fails
                        return {"lat": 40.7128, "lng": -74.0060}  # New York City
                else:
                    raise Exception(f"Geocoding error: {response.status}")

    async def _parse_place(
        self, place_data: dict, include_photos: bool = True
    ) -> Attraction:
        """Parse Google Places API response into Attraction object"""
        # Extract basic information
        name = place_data.get("displayName", {}).get("text", "Unknown")
        formatted_address = place_data.get("formattedAddress", "")
        rating = place_data.get("rating", 0.0)
        user_rating_count = place_data.get("userRatingCount", 0)
        price_level = place_data.get("priceLevel")
        types = place_data.get("types", [])
        location = place_data.get("location", {})
        phone = place_data.get("internationalPhoneNumber")
        website = place_data.get("websiteUri")

        # Extract opening hours
        opening_hours = None
        if "regularOpeningHours" in place_data:
            opening_hours = {}
            for period in place_data["regularOpeningHours"].get("periods", []):
                day = period.get("open", {}).get("day", 0)
                day_name = [
                    "Sunday",
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ][day]
                open_time = period.get("open", {}).get("time", "")
                close_time = period.get("close", {}).get("time", "")
                if open_time and close_time:
                    opening_hours[day_name] = f"{open_time}-{close_time}"

        # Extract photos
        photo_urls = []
        if include_photos and "photos" in place_data:
            photo_urls = await self._get_photo_urls(
                place_data["photos"][:3]
            )  # Limit to 3 photos

        # Determine category and description
        category = self._determine_category(types)
        description = self._generate_description(
            name, category, formatted_address, rating, user_rating_count
        )

        return Attraction(
            name=name,
            description=description,
            location=formatted_address,
            category=category,
            rating=rating,
            price_level=price_level,
            opening_hours=opening_hours,
            place_id=place_data.get("id"),
            formatted_address=formatted_address,
            phone_number=phone,
            website=website,
            photo_urls=photo_urls,
            reviews_count=user_rating_count,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
        )

    async def _get_photo_urls(self, photos: List[dict]) -> List[str]:
        """Get photo URLs from Google Places photos"""
        photo_urls = []
        for photo in photos:
            if "name" in photo:
                # Construct photo URL using the photo reference
                photo_url = f"{self.base_url}/{photo['name']}/media?maxHeightPx=400&maxWidthPx=400&key={self.api_key}"
                photo_urls.append(photo_url)
        return photo_urls

    def _determine_category(self, types: List[str]) -> str:
        """Determine attraction category from Google Places types"""
        category_mapping = {
            "tourist_attraction": "Tourist Attraction",
            "museum": "Museum",
            "park": "Park",
            "zoo": "Zoo",
            "aquarium": "Aquarium",
            "amusement_park": "Amusement Park",
            "art_gallery": "Art Gallery",
            "church": "Religious Site",
            "hindu_temple": "Religious Site",
            "mosque": "Religious Site",
            "synagogue": "Religious Site",
            "shopping_mall": "Shopping",
            "night_club": "Entertainment",
            "casino": "Entertainment",
            "stadium": "Sports & Recreation",
            "gym": "Sports & Recreation",
        }

        for place_type in types:
            if place_type in category_mapping:
                return category_mapping[place_type]

        return "Attraction"

    def _generate_description(
        self, name: str, category: str, address: str, rating: float, review_count: int
    ) -> str:
        """Generate attraction description from available data"""
        description_parts = [f"{name} is a {category.lower()}"]

        if address:
            description_parts.append(f"located at {address}")

        if rating > 0:
            description_parts.append(f"with a {rating}/5 rating")
            if review_count > 0:
                description_parts.append(f"based on {review_count} reviews")

        return " ".join(description_parts) + "."

    def _filter_attractions(
        self, attractions: List[Attraction], criteria: AttractionSearchInput
    ) -> List[Attraction]:
        """Filter attractions based on search criteria"""
        filtered = attractions

        # Filter by minimum rating
        if criteria.min_rating:
            filtered = [a for a in filtered if a.rating >= criteria.min_rating]

        # Filter by price levels
        if criteria.price_levels:
            filtered = [a for a in filtered if a.price_level in criteria.price_levels]

        # Sort by rating (descending)
        filtered.sort(key=lambda x: x.rating, reverse=True)

        return filtered

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for the tool"""
        return AttractionSearchInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema for the tool"""
        return AttractionSearchOutput.model_json_schema()


# Register the tool
from app.tools.base_tool import tool_registry

attraction_search_tool = AttractionSearchTool()
tool_registry.register(attraction_search_tool)
