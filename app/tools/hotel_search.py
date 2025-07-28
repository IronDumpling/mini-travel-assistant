"""
Hotel Search Tool - AMADEUS Hotel API Integration

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
import aiohttp
import os
from pydantic import BaseModel
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, ToolMetadata
from app.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

class Hotel(BaseModel):
    name: str
    location: str
    price_per_night: Optional[float] = None
    currency: Optional[str] = None
    rating: Optional[float] = None
    amenities: List[str] = []
    hotel_id: Optional[str] = None
    chain_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[Dict[str, Any]] = None

class HotelSearchInput(ToolInput):
    """Hotel search input"""
    location: str  # This should be a 3-character IATA city code (e.g., "PAR", "LON")
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
    """Hotel search tool class using AMADEUS API"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="hotel_search",
            description="Search for hotel accommodation information using AMADEUS API, with multiple filtering conditions",
            category="accommodation",
            tags=["hotel", "accommodation", "search", "booking", "amadeus"],
            timeout=30
        )
        super().__init__(metadata)
        
        # AMADEUS API configuration
        self.api_key = os.getenv("HOTEL_SEARCH_API_KEY", "YOUR_AMADEUS_API_KEY")
        self.api_secret = os.getenv("HOTEL_SEARCH_API_SECRET", "YOUR_AMADEUS_API_SECRET")
        self.base_url = "https://test.api.amadeus.com/v1"  # Test environment
        # self.base_url = "https://api.amadeus.com/v1"  # Production environment
        self.access_token = None
        
        # Log API configuration status
        if self.api_key != "YOUR_AMADEUS_API_KEY":
            logger.info("✅ AMADEUS API credentials loaded from environment variables")
        else:
            logger.warning("⚠️ AMADEUS API credentials not found in environment variables")
        
        logger.info("🏨 HotelSearchTool initialized with AMADEUS API configuration")

    async def _get_access_token(self) -> Optional[str]:
        """Get AMADEUS API access token"""
        try:
            logger.info("🔑 Requesting AMADEUS API access token")
            logger.info(f"🔑 Using API Key: {self.api_key[:20]}...")
            logger.info(f"🔑 Using API Secret: {self.api_secret[:10]}...")
            
            async with aiohttp.ClientSession() as session:
                auth_data = {
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret
                }
                
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                
                async with session.post(
                    "https://test.api.amadeus.com/v1/security/oauth2/token",
                    data=auth_data,
                    headers=headers
                ) as response:
                    logger.info(f"🔑 AMADEUS auth response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get("access_token")
                        logger.info("✅ AMADEUS access token obtained successfully")
                        return self.access_token
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to get AMADEUS access token. Status: {response.status}, Error: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Exception while getting AMADEUS access token: {str(e)}")
            return None

    async def _execute(self, input_data: HotelSearchInput, context: ToolExecutionContext) -> HotelSearchOutput:
        """Execute hotel search using AMADEUS API"""
        try:
            logger.info(f"🏨 Starting hotel search for location: {input_data.location}")
            logger.info(f"📅 Check-in: {input_data.check_in}, Check-out: {input_data.check_out}")
            logger.info(f"👥 Guests: {input_data.guests}, Rooms: {input_data.rooms}")
            
            # Get access token if not available
            if not self.access_token:
                self.access_token = await self._get_access_token()
                if not self.access_token:
                    logger.error("❌ Cannot proceed without AMADEUS access token")
                    return HotelSearchOutput(
                        success=False,
                        error="Failed to authenticate with AMADEUS API",
                        hotels=[]
                    )
            
            hotels = await self._search_hotels_amadeus(
                input_data.location,
                input_data.check_in,
                input_data.check_out,
                input_data.guests
            )
            
            logger.info(f"🏨 Found {len(hotels)} hotels from AMADEUS API")
            
            # Filter hotels based on search criteria
            filtered_hotels = self._filter_hotels(hotels, input_data)
            
            logger.info(f"🏨 After filtering: {len(filtered_hotels)} hotels match criteria")
            
            return HotelSearchOutput(
                success=True,
                hotels=filtered_hotels,
                total_results=len(filtered_hotels),
                search_location=input_data.location,
                data={
                    "total_results": len(hotels),
                    "filtered_results": len(filtered_hotels),
                    "api_source": "amadeus"
                }
            )
        except Exception as e:
            logger.error(f"❌ Exception in hotel search execution: {str(e)}")
            return HotelSearchOutput(
                success=False,
                error=str(e),
                hotels=[]
            )
    
    async def _search_hotels_amadeus(
        self,
        city_code: str,
        check_in: datetime,
        check_out: datetime,
        guests: int = 1
    ) -> List[Hotel]:
        """Search for hotels using AMADEUS Hotel List API"""
        try:
            # Validate and clean city code
            if not city_code:
                logger.error("❌ City code is empty or None")
                return []
            
            # Clean the city code - remove whitespace, uppercase, ensure it's exactly 3 characters
            cleaned_city_code = city_code.strip().upper()
            if len(cleaned_city_code) != 3:
                logger.error(f"❌ Invalid city code length: '{cleaned_city_code}' (length: {len(cleaned_city_code)})")
                return []
            
            if not cleaned_city_code.isalpha():
                logger.error(f"❌ City code contains non-alphabetic characters: '{cleaned_city_code}'")
                return []
            
            logger.info(f"🔍 Searching hotels in city code: '{cleaned_city_code}' (original: '{city_code}')")
            
            async with aiohttp.ClientSession() as session:
                # AMADEUS Hotel List API endpoint
                endpoint = f"{self.base_url}/reference-data/locations/hotels/by-city"
                
                # Simplified parameters - only city code
                params = {
                    "cityCode": cleaned_city_code
                }
                
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                logger.info(f"🌐 Making request to AMADEUS API: {endpoint}")
                logger.info(f"📋 Request parameters: {params}")
                
                # Debug: Show the exact URL being constructed
                import urllib.parse
                query_string = urllib.parse.urlencode(params)
                full_url = f"{endpoint}?{query_string}"
                logger.info(f"🔗 Full URL: {full_url}")
                logger.info(f"🔗 City code being sent: '{cleaned_city_code}' (type: {type(cleaned_city_code)})")
                
                async with session.get(endpoint, params=params, headers=headers) as response:
                    logger.info(f"📡 AMADEUS API response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"📊 AMADEUS API response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                        hotels = []
                        if "data" in data and isinstance(data["data"], list):
                            for hotel_data in data["data"]:
                                hotel = self._parse_amadeus_hotel(hotel_data)
                                if hotel:
                                    hotels.append(hotel)
                            
                            logger.info(f"✅ Successfully parsed {len(hotels)} hotels from AMADEUS response")
                        else:
                            logger.warning(f"⚠️ Unexpected AMADEUS response structure: {data}")
                        
                        return hotels
                    elif response.status == 401:
                        logger.warning("⚠️ AMADEUS access token expired, refreshing...")
                        self.access_token = None
                        # Try to get new token and retry once
                        new_token = await self._get_access_token()
                        if new_token:
                            return await self._search_hotels_amadeus(city_code, check_in, check_out, guests)
                        else:
                            logger.error("❌ Failed to refresh AMADEUS access token")
                            return []
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ AMADEUS API error. Status: {response.status}, Error: {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Exception while searching hotels with AMADEUS API: {str(e)}")
            return []

    def _parse_amadeus_hotel(self, hotel_data: dict) -> Optional[Hotel]:
        """Parse hotel data from AMADEUS API response"""
        try:
            logger.debug(f"🔍 Parsing hotel data: {hotel_data.get('name', 'Unknown')}")
            
            # Extract address information
            address = {}
            if "address" in hotel_data:
                addr = hotel_data["address"]
                address = {
                    "street": addr.get("lines", []),
                    "city": addr.get("cityName"),
                    "postal_code": addr.get("postalCode"),
                    "country": addr.get("countryCode")
                }
            
            # Extract coordinates
            latitude = None
            longitude = None
            if "geoCode" in hotel_data:
                geo = hotel_data["geoCode"]
                latitude = geo.get("latitude")
                longitude = geo.get("longitude")
            
            hotel = Hotel(
                name=hotel_data.get("name", "Unknown Hotel"),
                location=hotel_data.get("address", {}).get("cityName", "Unknown"),
                hotel_id=hotel_data.get("hotelId"),
                chain_code=hotel_data.get("chainCode"),
                latitude=latitude,
                longitude=longitude,
                address=address,
                # Note: AMADEUS Hotel List API doesn't provide pricing, rating, or amenities
                # These would need to be fetched from additional API calls
                price_per_night=None,
                currency=None,
                rating=None,
                amenities=[]
            )
            
            logger.debug(f"✅ Successfully parsed hotel: {hotel.name}")
            return hotel
            
        except Exception as e:
            logger.error(f"❌ Error parsing hotel data: {str(e)}")
            return None
    
    def _filter_hotels(self, hotels: List[Hotel], criteria: HotelSearchInput) -> List[Hotel]:
        """Filter hotels based on search criteria"""
        logger.info(f"🔍 Filtering {len(hotels)} hotels based on criteria")
        
        filtered = hotels
        
        # Skip rating filter since AMADEUS Hotel List API doesn't provide ratings
        # if criteria.min_rating:
        #     filtered = [h for h in filtered if h.rating and h.rating >= criteria.min_rating]
        #     logger.info(f"📊 After rating filter: {len(filtered)} hotels")
        
        # Skip price filter since AMADEUS Hotel List API doesn't provide pricing
        # if criteria.max_price:
        #     filtered = [h for h in filtered if h.price_per_night and h.price_per_night <= criteria.max_price]
        #     logger.info(f"💰 After price filter: {len(filtered)} hotels")
        
        # Skip amenities filter since AMADEUS Hotel List API doesn't provide amenities
        # if criteria.required_amenities:
        #     filtered = [
        #         h for h in filtered 
        #         if all(amenity in h.amenities for amenity in criteria.required_amenities)
        #     ]
        #     logger.info(f"🏊 After amenities filter: {len(filtered)} hotels")
        
        # Limit results to first 20 hotels for better performance
        if len(filtered) > 20:
            filtered = filtered[:20]
            logger.info(f"📊 Limited to first 20 hotels for performance")
        
        logger.info(f"✅ Final filtered results: {len(filtered)} hotels")
        return filtered


    def get_input_schema(self) -> Dict[str, Any]:
        return HotelSearchInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        return HotelSearchOutput.model_json_schema()

from app.tools.base_tool import tool_registry
hotel_search_tool = HotelSearchTool()
tool_registry.register(hotel_search_tool) 