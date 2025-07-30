"""
Attraction Search Tool - API Query and Response Example

This example shows the exact API queries and responses for the attraction search tool.
"""

import json
import sys
import os

# Add the app directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.attraction_search import (
    AttractionSearchTool,
    AttractionSearchInput,
    AttractionSearchOutput,
)


def show_api_query_example():
    """Show example API queries that the tool would make"""

    print("🔍 Attraction Search Tool - API Query Examples")
    print("=" * 60)

    tool = AttractionSearchTool()

    # Example 1: Text Search Query
    print("\n📤 Example 1: Text Search API Query")
    print("-" * 40)

    input_data = AttractionSearchInput(
        location="Paris, France",
        query="Eiffel Tower",
        max_results=3,
        include_photos=True,
    )

    print("Input Parameters:")
    print(json.dumps(input_data.model_dump(), indent=2))

    print("\nAPI Request Body (Google Places API v1):")
    request_body = {
        "textQuery": "Eiffel Tower in Paris, France",
        "maxResultCount": 3,
        "locationBias": {
            "rectangle": {
                "low": {"latitude": -90, "longitude": -180},
                "high": {"latitude": 90, "longitude": 180},
            }
        },
    }
    print(json.dumps(request_body, indent=2))

    print("\nAPI Headers:")
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "YOUR_API_KEY_HERE",
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri,places.photos",
    }
    print(json.dumps(headers, indent=2))

    # Example 2: Nearby Search Query
    print("\n📤 Example 2: Nearby Search API Query")
    print("-" * 40)

    input_data = AttractionSearchInput(
        location="Tokyo, Japan",
        max_results=3,
        radius_meters=5000,
        min_rating=4.0,
        include_photos=True,
    )

    print("Input Parameters:")
    print(json.dumps(input_data.model_dump(), indent=2))

    print("\nAPI Request Body (Google Places API v1):")
    request_body = {
        "maxResultCount": 3,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": 35.6762,  # Tokyo coordinates (after geocoding)
                    "longitude": 139.6503,
                },
                "radius": 5000,
            }
        },
        "includedTypes": [
            "tourist_attraction",
            "museum",
            "park",
            "zoo",
            "aquarium",
            "amusement_park",
        ],
    }
    print(json.dumps(request_body, indent=2))


def show_api_response_example():
    """Show example API responses that the tool would receive"""

    print("\n📥 Attraction Search Tool - API Response Examples")
    print("=" * 60)

    # Example 1: Google Places API Response
    print("\n📥 Example 1: Google Places API Response")
    print("-" * 40)

    api_response = {
        "places": [
            {
                "id": "ChIJD7fiBh9u5kcRYJSMaMOCCwQ",
                "displayName": {"text": "Eiffel Tower", "languageCode": "en"},
                "formattedAddress": "Champ de Mars, 5 Av. Anatole France, 75007 Paris, France",
                "rating": 4.6,
                "userRatingCount": 234567,
                "priceLevel": 2,
                "types": ["tourist_attraction", "point_of_interest", "establishment"],
                "location": {"latitude": 48.8584, "longitude": 2.2945},
                "internationalPhoneNumber": "+33 892 70 12 39",
                "websiteUri": "https://www.toureiffel.paris/",
                "photos": [
                    {
                        "name": "places/ChIJD7fiBh9u5kcRYJSMaMOCCwQ/photos/AQ",
                        "widthPx": 1920,
                        "heightPx": 1080,
                    }
                ],
                "regularOpeningHours": {
                    "periods": [
                        {
                            "open": {"day": 1, "time": "0900"},
                            "close": {"day": 1, "time": "2359"},
                        },
                        {
                            "open": {"day": 2, "time": "0900"},
                            "close": {"day": 2, "time": "2359"},
                        },
                    ]
                },
            }
        ]
    }

    print("Raw Google Places API Response:")
    print(json.dumps(api_response, indent=2))

    # Example 2: Tool's Processed Response
    print("\n📥 Example 2: Tool's Processed Response")
    print("-" * 40)

    tool_response = AttractionSearchOutput(
        success=True,
        attractions=[
            {
                "name": "Eiffel Tower",
                "description": "Eiffel Tower is a tourist attraction located at Champ de Mars, 5 Av. Anatole France, 75007 Paris, France with a 4.6/5 rating based on 234567 reviews.",
                "location": "Champ de Mars, 5 Av. Anatole France, 75007 Paris, France",
                "category": "Tourist Attraction",
                "rating": 4.6,
                "price_level": 2,
                "opening_hours": {"Monday": "09:00-23:59", "Tuesday": "09:00-23:59"},
                "place_id": "ChIJD7fiBh9u5kcRYJSMaMOCCwQ",
                "formatted_address": "Champ de Mars, 5 Av. Anatole France, 75007 Paris, France",
                "phone_number": "+33 892 70 12 39",
                "website": "https://www.toureiffel.paris/",
                "photo_urls": [
                    "https://places.googleapis.com/v1/places/ChIJD7fiBh9u5kcRYJSMaMOCCwQ/photos/AQ/media?maxHeightPx=400&maxWidthPx=400&key=YOUR_API_KEY"
                ],
                "reviews_count": 234567,
                "latitude": 48.8584,
                "longitude": 2.2945,
            }
        ],
        total_results=1,
        search_location="Paris, France",
        data={
            "search_type": "text",
            "original_results": 1,
            "filtered_results": 1,
            "search_parameters": {
                "location": "Paris, France",
                "query": "Eiffel Tower",
                "max_results": 3,
                "include_photos": True,
            },
        },
    )

    print("Tool's Processed Response:")
    # Convert to dict and handle any non-serializable objects
    response_dict = tool_response.model_dump()
    print(json.dumps(response_dict, indent=2, default=str))


def show_usage_example():
    """Show how to use the attraction search tool"""

    print("\n💡 How to Use the Attraction Search Tool")
    print("=" * 60)

    print("\n1. Set up your API key:")
    print("   export ATTRACTION_SEARCH_API_KEY='your_google_places_api_key'")

    print("\n2. Basic usage in Python:")
    print("""
from app.tools.attraction_search import AttractionSearchTool, AttractionSearchInput

# Create the tool
tool = AttractionSearchTool()

# Search for a specific attraction
input_data = AttractionSearchInput(
    location="Paris, France",
    query="Eiffel Tower",
    max_results=5
)

# Execute the search
result = await tool._execute(input_data, context)

# Process results
if result.success:
    for attraction in result.attractions:
        print(f"Name: {attraction.name}")
        print(f"Rating: {attraction.rating}/5")
        print(f"Address: {attraction.formatted_address}")
""")

    print("\n3. Available search options:")
    print("   - location: Required - city, country, or coordinates")
    print("   - query: Optional - specific attraction name")
    print("   - category: Optional - tourist_attraction, museum, park, etc.")
    print("   - radius_meters: Optional - search radius (default: 5000)")
    print("   - min_rating: Optional - minimum rating filter (0-5)")
    print("   - max_results: Optional - max results to return (default: 10)")
    print("   - include_photos: Optional - include photo URLs (default: True)")
    print("   - price_levels: Optional - filter by price level (0-4)")


def main():
    """Main function to show all examples"""
    show_api_query_example()
    show_api_response_example()
    show_usage_example()

    print("\n" + "=" * 60)
    print("✨ Example complete! The attraction search tool is ready to use.")


if __name__ == "__main__":
    main()
