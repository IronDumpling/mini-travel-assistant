"""
Attraction Search Tool - Google Places API (New) Integration

A comprehensive tool for searching tourist attractions and activities using Google Places API (New).
Features include text search, nearby search, place details, and photo retrieval.
"""

import os
from typing import List, Optional, Dict, Any, Union
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
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Attraction(BaseModel):
    """Attraction data model"""

    name: str
    description: str
    location: str
    category: str
    rating: float
    price_level: Optional[int] = None  # 0-4 scale from Google Places
    price_level_description: Optional[str] = None  # Human-readable price level
    opening_hours: Optional[Dict[str, str]] = None
    place_id: Optional[str] = None
    formatted_address: Optional[str] = None
    phone_number: Optional[str] = None
    website: Optional[str] = None
    photo_urls: List[str] = []
    reviews_count: Optional[int] = None
    review_summary: Optional[str] = None  # Summary of recent reviews
    individual_reviews: List[Dict[str, Any]] = []  # Individual user reviews
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Additional cost and review fields
    cost_range: Optional[str] = None  # Cost range (e.g., "$10-20")
    popular_times: Optional[Dict[str, str]] = None  # Busy times information
    accessibility_info: Optional[str] = None  # Accessibility information


class AttractionSearchInput(ToolInput):
    """Attraction search input parameters"""

    location: Union[str, List[str]] = Field(description="Location(s) to search for attractions - can be single location or list")
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
            # ✅ Enhanced: Support multiple locations by iterating through list
            if isinstance(input_data.location, list):
                return await self._execute_multi_location_search(input_data, context)
            
            # Validate input parameters for single location
            if not input_data.location or input_data.location.lower() in [
                "unknown",
                "",
            ]:
                return AttractionSearchOutput(
                    success=False,
                    error="Invalid location: location is required and cannot be 'unknown'",
                    attractions=[],
                    total_results=0,
                    search_location=input_data.location or "unknown",
                )

            logger.info(
                f"Searching attractions in {input_data.location}, query: {input_data.query or 'nearby search'}"
            )

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

            logger.info(
                f"Attraction search completed: {len(attractions)} total, {len(filtered_attractions)} after filtering"
            )

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
            logger.error(f"Attraction search failed for {input_data.location}: {e}")
            return AttractionSearchOutput(
                success=False,
                error=str(e),
                attractions=[],
                total_results=0,
                search_location=input_data.location,
            )

    async def _execute_multi_location_search(
        self, input_data: AttractionSearchInput, context: ToolExecutionContext
    ) -> AttractionSearchOutput:
        """Execute attraction search for multiple locations and merge results"""
        
        locations = input_data.location
        if not locations:
            return AttractionSearchOutput(
                success=False,
                error="No valid locations provided in list",
                attractions=[],
                total_results=0,
                search_location="empty_list"
            )
        
        logger.info(f"🌍 Executing multi-location attraction search for {len(locations)} locations: {locations}")
        
        all_attractions = []
        successful_locations = []
        failed_locations = []
        
        # Execute search for each location
        for location in locations:
            if not location or location.lower() in ['unknown', '']:
                failed_locations.append(location or "unknown")
                continue
                
            try:
                # Create a copy of input_data with single location
                single_location_input = AttractionSearchInput(
                    location=location,
                    query=input_data.query,
                    category=input_data.category,
                    radius_meters=input_data.radius_meters,
                    min_rating=input_data.min_rating,
                    max_results=input_data.max_results,
                    include_photos=input_data.include_photos,
                    price_levels=input_data.price_levels
                )
                
                logger.info(f"🎡 Searching attractions in: {location}")
                
                # Execute single location search by calling the main logic
                result = await self._execute_single_location_search(single_location_input, context)
                
                if result.success and result.attractions:
                    # Add location context to each attraction
                    for attraction in result.attractions:
                        attraction.search_location = location
                    all_attractions.extend(result.attractions)
                    successful_locations.append(location)
                    logger.info(f"✅ Found {len(result.attractions)} attractions in {location}")
                else:
                    failed_locations.append(location)
                    logger.warning(f"⚠️ No attractions found in {location}: {result.error}")
                    
            except Exception as e:
                failed_locations.append(location)
                logger.error(f"❌ Error searching attractions in {location}: {e}")
        
        # Prepare result summary
        total_results = len(all_attractions)
        search_summary = f"Searched {len(locations)} locations: {successful_locations}"
        
        if failed_locations:
            search_summary += f" (failed: {failed_locations})"
        
        if total_results == 0:
            return AttractionSearchOutput(
                success=False,
                error=f"No attractions found in any of the {len(locations)} locations",
                attractions=[],
                total_results=0,
                search_location=search_summary
            )
        
        # Sort by rating and limit results if needed
        all_attractions.sort(key=lambda x: x.rating or 0, reverse=True)
        
        # Apply global limit across all locations
        max_total_results = input_data.max_results * len(successful_locations) if input_data.max_results else len(all_attractions)
        limited_attractions = all_attractions[:max_total_results]
        
        logger.info(f"🎯 Multi-location search completed: {total_results} total attractions from {len(successful_locations)} locations")
        
        return AttractionSearchOutput(
            success=True,
            attractions=limited_attractions,
            total_results=total_results,
            search_location=search_summary,
            data={
                "searched_locations": len(locations),
                "successful_locations": successful_locations,
                "failed_locations": failed_locations,
                "results_per_location": {loc: len([a for a in limited_attractions if hasattr(a, 'search_location') and a.search_location == loc]) for loc in successful_locations}
            }
        )

    async def _execute_single_location_search(
        self, input_data: AttractionSearchInput, context: ToolExecutionContext
    ) -> AttractionSearchOutput:
        """Execute attraction search for a single location (extracted from main logic)"""
        logger.info(f"Searching attractions in {input_data.location}, query: {input_data.query or 'nearby search'}")
        
        # Ensure API key is available before proceeding
        self._ensure_api_key()

        # Determine search strategy based on input
        if input_data.query:
            # Use text search for specific queries
            attractions = await self._text_search(input_data)
        else:
            # Use nearby search for location-based discovery
            attractions = await self._nearby_search(input_data)

        if not attractions:
            logger.warning(f"No attractions found for location: {input_data.location}")
            return AttractionSearchOutput(
                success=True,
                attractions=[],
                total_results=0,
                search_location=input_data.location,
                data={"message": "No attractions found matching criteria"}
            )

        # Apply filters and sorting
        filtered_attractions = self._filter_and_sort_attractions(
            attractions, input_data
        )

        logger.info(
            f"Found {len(filtered_attractions)} attractions in {input_data.location}"
        )

        return AttractionSearchOutput(
            success=True,
            attractions=filtered_attractions,
            total_results=len(filtered_attractions),
            search_location=input_data.location,
            data={
                "search_method": "text_search" if input_data.query else "nearby_search",
                "total_found": len(attractions),
                "after_filtering": len(filtered_attractions),
            }
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
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri,places.photos,places.accessibilityOptions,places.editorialSummary,places.reviews",
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
                            input_data.radius_meters or 30000, 30000
                        ),  # Max 30km
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
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri,places.photos,places.accessibilityOptions,places.editorialSummary,places.reviews",
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
        price_level = self._parse_price_level(place_data.get("priceLevel"))
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

        # Extract additional cost and review information
        price_level_description = self._get_price_level_description(price_level)
        review_summary = self._extract_review_summary(place_data)
        individual_reviews = self._extract_individual_reviews(place_data)
        popular_times = self._extract_popular_times(place_data)
        accessibility_info = self._extract_accessibility_info(place_data)
        cost_range = self._get_cost_range(price_level)

        # Generate description
        description = self._generate_description(
            name,
            category,
            formatted_address,
            rating,
            user_rating_count,
            price_level_description,
        )

        return Attraction(
            name=name,
            description=description,
            location=formatted_address,
            category=category,
            rating=rating,
            price_level=price_level,
            price_level_description=price_level_description,
            opening_hours=opening_hours,
            place_id=place_data.get("id"),
            formatted_address=formatted_address,
            phone_number=phone,
            website=website,
            photo_urls=photo_urls,
            reviews_count=user_rating_count,
            review_summary=review_summary,
            individual_reviews=individual_reviews,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
            cost_range=cost_range,
            popular_times=popular_times,
            accessibility_info=accessibility_info,
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
        self,
        name: str,
        category: str,
        address: str,
        rating: float,
        review_count: int,
        price_level_description: Optional[str] = None,
    ) -> str:
        """Generate attraction description from available data"""
        description_parts = [f"{name} is a {category.lower()}"]

        if address:
            description_parts.append(f"located at {address}")

        if rating > 0:
            description_parts.append(f"with a {rating}/5 rating")
            if review_count > 0:
                description_parts.append(f"based on {review_count} reviews")

        if price_level_description:
            description_parts.append(f"({price_level_description})")

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

    def _parse_price_level(self, price_level_raw: any) -> Optional[int]:
        """Parse price level from Google Places API (handles both string and int formats)"""
        if price_level_raw is None:
            return None

        # Handle string format (new Google Places API)
        if isinstance(price_level_raw, str):
            price_level_mapping = {
                "PRICE_LEVEL_FREE": 0,
                "PRICE_LEVEL_INEXPENSIVE": 1,
                "PRICE_LEVEL_MODERATE": 2,
                "PRICE_LEVEL_EXPENSIVE": 3,
                "PRICE_LEVEL_VERY_EXPENSIVE": 4,
            }
            return price_level_mapping.get(price_level_raw)

        # Handle numeric format (legacy or direct numeric values)
        if isinstance(price_level_raw, int) and 0 <= price_level_raw <= 4:
            return price_level_raw

        return None

    def _get_price_level_description(self, price_level: Optional[int]) -> Optional[str]:
        """Convert price level number to human-readable description"""
        if price_level is None:
            return None

        price_descriptions = {
            0: "Free",
            1: "Inexpensive",
            2: "Moderate",
            3: "Expensive",
            4: "Very Expensive",
        }
        return price_descriptions.get(price_level, "Price not specified")

    def _extract_review_summary(self, place_data: dict) -> Optional[str]:
        """Extract review summary from editorial summary if available"""
        if "editorialSummary" in place_data:
            return place_data["editorialSummary"].get("text", "")
        return None

    def _extract_individual_reviews(
        self, place_data: dict, max_reviews: int = 5
    ) -> List[Dict[str, Any]]:
        """Extract individual user reviews from place data"""
        reviews = []

        if "reviews" in place_data and place_data["reviews"]:
            for review_data in place_data["reviews"][:max_reviews]:
                review = {}

                # Extract author information
                if "authorAttribution" in review_data:
                    author = review_data["authorAttribution"]
                    review["author_name"] = author.get("displayName", "Anonymous")
                    review["author_photo"] = author.get("photoUri", "")

                # Extract review details
                review["rating"] = review_data.get("rating", None)
                review["text"] = review_data.get("text", {}).get("text", "")
                review["relative_time"] = review_data.get(
                    "relativePublishTimeDescription", ""
                )
                review["publish_time"] = review_data.get("publishTime", "")

                # Only add reviews with meaningful content
                if review.get("text") and len(review["text"].strip()) > 10:
                    reviews.append(review)

        return reviews

    def _extract_popular_times(self, place_data: dict) -> Optional[Dict[str, str]]:
        """Extract popular times information - Note: This field may not be available in the API"""
        # Popular times data is not consistently available in Google Places API (New)
        # This method is kept for future compatibility if the field becomes available
        return None

    def _extract_accessibility_info(self, place_data: dict) -> Optional[str]:
        """Extract accessibility information"""
        if "accessibilityOptions" in place_data:
            options = place_data["accessibilityOptions"]
            features = []

            if options.get("wheelchairAccessibleEntrance"):
                features.append("Wheelchair accessible entrance")
            if options.get("wheelchairAccessibleParking"):
                features.append("Wheelchair accessible parking")
            if options.get("wheelchairAccessibleRestroom"):
                features.append("Wheelchair accessible restroom")
            if options.get("wheelchairAccessibleSeating"):
                features.append("Wheelchair accessible seating")

            return ", ".join(features) if features else None
        return None

    def _get_cost_range(self, price_level: Optional[int]) -> Optional[str]:
        """Simple cost range based on price level from API"""
        if price_level is None:
            return "Free"

        # Simple cost ranges
        cost_ranges = {0: "Free", 1: "$5-15", 2: "$15-30", 3: "$30-50", 4: "$50+"}

        return cost_ranges.get(price_level, "Free")


# Register the tool
from app.tools.base_tool import tool_registry

attraction_search_tool = AttractionSearchTool()
tool_registry.register(attraction_search_tool)
