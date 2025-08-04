"""
Geographical Data Configuration - Centralized mapping data for travel planning

This module contains all geographical mappings used throughout the travel planning system:
- Region to cities mappings (for multi-destination planning)
- City to IATA code mappings (for flight and hotel searches)
- City to airport code mappings (for more specific flight searches)

Moved from travel_agent.py for better maintainability and centralized configuration.
"""

from typing import Dict, List, Union

class GeographicalMappings:
    """Centralized geographical mapping data"""
    
    # ✅ Region/Continent to multiple cities mapping (based on available documents)
    REGION_TO_CITIES = {
        # European destinations with multiple cities
        'europe': ['london', 'paris', 'berlin', 'rome', 'barcelona', 'amsterdam', 'vienna', 'prague'],
        'european': ['london', 'paris', 'berlin', 'rome', 'barcelona', 'amsterdam', 'vienna', 'prague'],
        'western europe': ['london', 'paris', 'amsterdam', 'barcelona'],
        'central europe': ['berlin', 'vienna', 'prague', 'budapest'],
        'southern europe': ['rome', 'barcelona'],
        
        # Asian destinations with multiple cities  
        'asia': ['tokyo', 'beijing', 'shanghai', 'seoul', 'singapore'],
        'asian': ['tokyo', 'beijing', 'shanghai', 'seoul', 'singapore'],
        'east asia': ['tokyo', 'beijing', 'shanghai', 'seoul'],
        'southeast asia': ['singapore'],
        'japan': ['tokyo', 'kyoto'],
        'china': ['beijing', 'shanghai'],
        
        # Other regions (keeping single cities for now)
        'north america': ['new york'],
        'north american': ['new york'],
        'south america': ['sao paulo'],
        'south american': ['sao paulo'],
        'africa': ['johannesburg'],
        'african': ['johannesburg'],
        'oceania': ['sydney'],
        'middle east': ['dubai'],
        'middle eastern': ['dubai'],
        
        # ✅ Special area mappings to available cities
        'swiss alps': ['munich', 'vienna'],     # Nearby European cities
        'alps': ['munich', 'vienna'],           # Nearby European cities
        'himalaya': ['beijing'],                # Closest available city
        'himalayas': ['beijing'],               # Closest available city
        'rockies': ['new york'],                # Default North American city
        'rocky mountains': ['new york'],        # Default North American city
        'andes': ['sao paulo'],                 # Default South American city
        'sahara': ['cairo'],                    # Not in our documents
        'sahara desert': ['cairo'],             # Not in our documents
        'scottish highlands': ['london'],       # UK -> London
        'lake district': ['london'],            # UK -> London
        'tuscany': ['rome'],                    # Italy -> Rome
        'provence': ['paris'],                  # France -> Paris
        'bavarian alps': ['munich'],            # Germany -> Munich
        'black forest': ['munich'],             # Germany -> Munich
        
        # Country-level mappings to available cities
        'france': ['paris'],
        'germany': ['berlin', 'munich'], 
        'italy': ['rome'],
        'spain': ['barcelona'],
        'netherlands': ['amsterdam'],
        'austria': ['vienna'],
        'czech republic': ['prague'],
        'hungary': ['budapest'],
        'united kingdom': ['london'],
        'england': ['london'],
        'scotland': ['london'],  # Use London as we don't have Edinburgh data
        'ireland': ['london'],   # Use London as we don't have Dublin data
        'switzerland': ['munich'],  # Use nearby city in our dataset
        'belgium': ['amsterdam'], # Use nearby city in our dataset
        'poland': ['berlin'],     # Use nearby city in our dataset
        'sweden': ['london'],     # Use available city
        'norway': ['london'],     # Use available city
        'denmark': ['london'],    # Use available city
        'finland': ['london']     # Use available city
    }
    
    # City to IATA code mapping (comprehensive version)
    CITY_TO_IATA = {
        # European cities (available in our documents)
        'amsterdam': 'AMS',
        'barcelona': 'BCN', 
        'berlin': 'BER',
        'budapest': 'BUD',
        'london': 'LON',
        'munich': 'MUC',
        'paris': 'PAR',
        'prague': 'PRG',
        'rome': 'ROM',
        'vienna': 'VIE',
        
        # Asian cities (available in our documents)
        'beijing': 'PEK',
        'kyoto': 'ITM',  # Kyoto doesn't have a major airport, using nearby Osaka
        'seoul': 'SEL',
        'shanghai': 'SHA',
        'singapore': 'SIN',
        'tokyo': 'TYO',
        'osaka': 'ITM',
        
        # North American cities
        'new york': 'JFK',
        'los angeles': 'LAX',
        'chicago': 'ORD',
        'miami': 'MIA',
        'san francisco': 'SFO',
        'toronto': 'YYZ',
        'vancouver': 'YVR',
        'montreal': 'YUL',
        
        # Oceania cities
        'sydney': 'SYD',
        'melbourne': 'MEL',
        'brisbane': 'BNE',
        'perth': 'PER',
        'adelaide': 'ADL',
        'auckland': 'AKL',
        'wellington': 'WLG',
        
        # African cities
        'cape town': 'CPT',
        'johannesburg': 'JNB',
        'nairobi': 'NBO',
        'cairo': 'CAI',
        'marrakech': 'RAK',
        'casablanca': 'CMN',
        
        # South American cities
        'rio de janeiro': 'GIG',
        'sao paulo': 'GRU',
        'buenos aires': 'EZE',
        'santiago': 'SCL',
        'lima': 'LIM',
        'bogota': 'BOG',
        'medellin': 'MDE',
        'caracas': 'CCS',
        
        # Middle Eastern cities
        'dubai': 'DXB',
        'doha': 'DOH',
        'riyadh': 'RUH',
        'jeddah': 'JED',
        'kuwait': 'KWI',
        'beirut': 'BEY',
        'tel aviv': 'TLV',
        'istanbul': 'IST',
        
        # Indian cities
        'mumbai': 'BOM',
        'delhi': 'DEL',
        'bangalore': 'BLR',
        'chennai': 'MAA',
        'kolkata': 'CCU',
        'hyderabad': 'HYD',
        'pune': 'PNQ',
        
        # Eastern European cities
        'moscow': 'SVO',
        'st petersburg': 'LED',
        'kiev': 'KBP',
        'warsaw': 'WAW',
        'krakow': 'KRK',
        'bratislava': 'BTS',
        'ljubljana': 'LJU',
        'zagreb': 'ZAG',
        'belgrade': 'BEG',
        'sofia': 'SOF',
        'bucharest': 'OTP',
        
        # Nordic countries
        'tallinn': 'TLL',
        'riga': 'RIX',
        'vilnius': 'VNO',
        'helsinki': 'HEL',
        'stockholm': 'ARN',
        'oslo': 'OSL',
        'copenhagen': 'CPH',
        'reykjavik': 'KEF',
        
        # Additional UK cities
        'dublin': 'DUB',
        'glasgow': 'GLA',
        'edinburgh': 'EDI',
        'manchester': 'MAN',
        'birmingham': 'BHX',
        'leeds': 'LBA',
        'liverpool': 'LPL',
        
        # Additional European cities
        'zurich': 'ZUR',
        'geneva': 'GVA',
        'brussels': 'BRU',
        'luxembourg': 'LUX',
        'milan': 'MXP',
        'florence': 'FLR',
        'venice': 'VCE',
        'naples': 'NAP',
        'madrid': 'MAD',
        'seville': 'SVQ',
        'valencia': 'VLC',
        'lisbon': 'LIS',
        'porto': 'OPO',
        'athens': 'ATH',
        'thessaloniki': 'SKG',
        
        # Mexican cities
        'mexico city': 'MEX',
        'guadalajara': 'GDL',
        'monterrey': 'MTY',
        'cancun': 'CUN',
        'puerto vallarta': 'PVR'
    }
    
    # Alternative airport codes for more specific flight searches
    CITY_TO_AIRPORT_CODES = {
        'london': ['LHR', 'LGW', 'STN', 'LTN'],  # Multiple London airports
        'paris': ['CDG', 'ORY'],                 # Multiple Paris airports
        'new york': ['JFK', 'LGA', 'EWR'],       # Multiple NYC airports
        'tokyo': ['NRT', 'HND'],                 # Multiple Tokyo airports
        'milan': ['MXP', 'LIN', 'BGY'],          # Multiple Milan airports
        'moscow': ['SVO', 'DME', 'VKO'],         # Multiple Moscow airports
        'bangkok': ['BKK', 'DMK'],               # Multiple Bangkok airports
        'seoul': ['ICN', 'GMP'],                 # Multiple Seoul airports
        'shanghai': ['PVG', 'SHA'],              # Multiple Shanghai airports
        'beijing': ['PEK', 'PKX'],               # Multiple Beijing airports
        'rome': ['FCO', 'CIA'],                  # Multiple Rome airports
        'chicago': ['ORD', 'MDW'],               # Multiple Chicago airports
        'los angeles': ['LAX', 'BUR', 'SNA'],   # Multiple LA airports
        'dublin': ['DUB'],
        'stockholm': ['ARN', 'BMA'],
        'oslo': ['OSL', 'TRF'],
        'barcelona': ['BCN'],
        'madrid': ['MAD'],
        'amsterdam': ['AMS'],
        'berlin': ['BER', 'SXF'],
        'munich': ['MUC'],
        'zurich': ['ZUR'],
        'vienna': ['VIE'],
        'prague': ['PRG'],
        'budapest': ['BUD'],
        'warsaw': ['WAW'],
        'athens': ['ATH'],
        'istanbul': ['IST', 'SAW']
    }

    @classmethod
    def get_region_cities(cls, region: str) -> List[str]:
        """Get cities for a region (case-insensitive)"""
        return cls.REGION_TO_CITIES.get(region.lower().strip(), [])
    
    @classmethod
    def get_iata_code(cls, city: str) -> str:
        """Get IATA code for a city (case-insensitive)"""
        return cls.CITY_TO_IATA.get(city.lower().strip(), '')
    
    @classmethod
    def get_airport_codes(cls, city: str) -> List[str]:
        """Get all airport codes for a city (case-insensitive)"""
        return cls.CITY_TO_AIRPORT_CODES.get(city.lower().strip(), [])
    
    @classmethod
    def get_primary_airport_code(cls, city: str) -> str:
        """Get primary airport code for a city"""
        codes = cls.get_airport_codes(city)
        return codes[0] if codes else cls.get_iata_code(city)
    
    # Reverse mapping: IATA code to city name
    IATA_TO_CITY = {
        # European cities (available in our documents)
        'AMS': 'Amsterdam',
        'BCN': 'Barcelona', 
        'BER': 'Berlin',
        'BUD': 'Budapest',
        'LON': 'London',
        'MUC': 'Munich',
        'PAR': 'Paris',
        'PRG': 'Prague',
        'ROM': 'Rome',
        'VIE': 'Vienna',
        
        # Asian cities (available in our documents)
        'PEK': 'Beijing',
        'ITM': 'Kyoto',  # Kyoto doesn't have a major airport, using nearby Osaka
        'SEL': 'Seoul',
        'SHA': 'Shanghai',
        'SIN': 'Singapore',
        'TYO': 'Tokyo',
        
        # North American cities
        'JFK': 'New York',
        'LAX': 'Los Angeles',
        'ORD': 'Chicago',
        'MIA': 'Miami',
        'SFO': 'San Francisco',
        'YYZ': 'Toronto',
        'YVR': 'Vancouver',
        'YUL': 'Montreal',
        
        # Oceania cities
        'SYD': 'Sydney',
        'MEL': 'Melbourne',
        'BNE': 'Brisbane',
        'PER': 'Perth',
        'ADL': 'Adelaide',
        'AKL': 'Auckland',
        'WLG': 'Wellington',
        
        # African cities
        'CPT': 'Cape Town',
        'JNB': 'Johannesburg',
        'NBO': 'Nairobi',
        'CAI': 'Cairo',
        'RAK': 'Marrakech',
        'CMN': 'Casablanca',
        
        # South American cities
        'GIG': 'Rio de Janeiro',
        'GRU': 'São Paulo',
        'EZE': 'Buenos Aires',
        'SCL': 'Santiago',
        'LIM': 'Lima',
        'BOG': 'Bogotá',
        'MDE': 'Medellín',
        'CCS': 'Caracas',
        
        # Middle Eastern cities
        'DXB': 'Dubai',
        'DOH': 'Doha',
        'RUH': 'Riyadh',
        'JED': 'Jeddah',
        'KWI': 'Kuwait',
        'BEY': 'Beirut',
        'TLV': 'Tel Aviv',
        'IST': 'Istanbul',
        
        # Additional European cities
        'ZUR': 'Zurich',
        'GVA': 'Geneva',
        'BRU': 'Brussels',
        'LUX': 'Luxembourg',
        'MXP': 'Milan',
        'FLR': 'Florence',
        'VCE': 'Venice',
        'NAP': 'Naples',
        'MAD': 'Madrid',
        'SVQ': 'Seville',
        'VLC': 'Valencia',
        'LIS': 'Lisbon',
        'OPO': 'Porto',
        'ATH': 'Athens',
        'SKG': 'Thessaloniki'
    }
    
    # Location-specific fallback coordinates for geocoding when API fails
    LOCATION_FALLBACKS = {
        "london": {"lat": 51.5074, "lng": -0.1278},
        "paris": {"lat": 48.8566, "lng": 2.3522},
        "new york": {"lat": 40.7128, "lng": -74.0060},
        "tokyo": {"lat": 35.6762, "lng": 139.6503},
        "singapore": {"lat": 1.3521, "lng": 103.8198},
        "sydney": {"lat": -33.8688, "lng": 151.2093},
        "rome": {"lat": 41.9028, "lng": 12.4964},
        "madrid": {"lat": 40.4168, "lng": -3.7038},
        "barcelona": {"lat": 41.3851, "lng": 2.1734},
        "amsterdam": {"lat": 52.3676, "lng": 4.9041},
        "berlin": {"lat": 52.5200, "lng": 13.4050},
        "munich": {"lat": 48.1351, "lng": 11.5820},
        "vienna": {"lat": 48.2082, "lng": 16.3738},
        "prague": {"lat": 50.0755, "lng": 14.4378},
        "budapest": {"lat": 47.4979, "lng": 19.0402},
        "seoul": {"lat": 37.5665, "lng": 126.9780},
        "shanghai": {"lat": 31.2304, "lng": 121.4737},
        "beijing": {"lat": 39.9042, "lng": 116.4074},
        "hong kong": {"lat": 22.3193, "lng": 114.1694},
        "bangkok": {"lat": 13.7563, "lng": 100.5018},
        "dubai": {"lat": 25.2048, "lng": 55.2708},
        "cairo": {"lat": 30.0444, "lng": 31.2357},
        "johannesburg": {"lat": -26.2041, "lng": 28.0473},
        "cape town": {"lat": -33.9249, "lng": 18.4241},
        "são paulo": {"lat": -23.5505, "lng": -46.6333},
        "rio de janeiro": {"lat": -22.9068, "lng": -43.1729},
        "buenos aires": {"lat": -34.6118, "lng": -58.3960},
        "santiago": {"lat": -33.4489, "lng": -70.6693},
        "mexico city": {"lat": 19.4326, "lng": -99.1332},
        "toronto": {"lat": 43.6532, "lng": -79.3832},
        "vancouver": {"lat": 49.2827, "lng": -123.1207},
        "montreal": {"lat": 45.5017, "lng": -73.5673},
    }

    @classmethod
    def get_city_name(cls, iata_code: str) -> str:
        """Get city name from IATA code (case-insensitive)"""
        return cls.IATA_TO_CITY.get(iata_code.upper().strip(), iata_code)
    
    @classmethod
    def get_fallback_coordinates(cls, location: str) -> Dict[str, float]:
        """Get fallback coordinates for a location (case-insensitive)"""
        from typing import Dict
        location_lower = location.lower().strip()
        
        # Check direct matches in fallback map
        for city_key, coords in cls.LOCATION_FALLBACKS.items():
            if city_key in location_lower:
                return coords
        
        # If no match found, raise exception
        raise KeyError(f"No fallback coordinates available for location: {location}")

    @classmethod
    def is_region(cls, location: str) -> bool:
        """Check if a location is a region/area rather than a specific city"""
        return location.lower().strip() in cls.REGION_TO_CITIES
    
    # User preference tracking destinations (smaller list for efficiency)
    PREFERENCE_TRACKING_DESTINATIONS = [
        "tokyo", "kyoto", "paris", "london", "new york", "beijing", 
        "shanghai", "rome", "barcelona"
    ]
    
    # Intent analysis destinations (medium list for fallback analysis)
    INTENT_ANALYSIS_DESTINATIONS = [
        "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing",
        "shanghai", "rome", "barcelona", "amsterdam", "vienna", "prague",
        "budapest", "berlin"
    ]
    
    # Comprehensive destination extraction list (large list for comprehensive parsing)
    COMPREHENSIVE_DESTINATIONS = [
        "tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", 
        "shanghai", "rome", "barcelona", "amsterdam", "vienna", "prague", 
        "budapest", "berlin", "bangkok", "singapore", "seoul", "sydney", 
        "melbourne", "munich", "madrid", "athens", "dubai", "istanbul",
        "hong kong", "hongkong", "san francisco", "los angeles", "chicago",
        "miami", "toronto", "vancouver", "montreal", "sydney", "melbourne",
        "brisbane", "perth", "adelaide", "auckland", "wellington", "cape town",
        "johannesburg", "nairobi", "cairo", "marrakech", "casablanca", "mumbai",
        "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune",
        "mexico city", "guadalajara", "monterrey", "rio de janeiro", "sao paulo",
        "buenos aires", "santiago", "lima", "bogota", "medellin", "caracas",
        "moscow", "st petersburg", "kiev", "warsaw", "krakow", "bratislava",
        "ljubljana", "zagreb", "belgrade", "sofia", "bucharest", "budapest",
        "tallinn", "riga", "vilnius", "helsinki", "stockholm", "oslo", "copenhagen",
        "reykjavik", "dublin", "glasgow", "edinburgh", "manchester", "birmingham",
        "leeds", "liverpool", "newcastle", "cardiff", "belfast", "brussels",
        "antwerp", "rotterdam", "the hague", "utrecht", "luxembourg", "geneva",
        "zurich", "bern", "basel", "lucerne", "milan", "florence", "venice",
        "naples", "palermo", "catania", "bologna", "turin", "genoa", "bari",
        "lisbon", "porto", "faro", "funchal", "pontadelgada", "valencia",
        "bilbao", "seville", "granada", "malaga", "alicante", "palma",
        "ibiza", "tenerife", "las palmas", "marrakech", "fes", "tangier",
        "agadir", "rabat", "casablanca", "tunis", "algiers", "cairo",
        "alexandria", "giza", "luxor", "aswan", "sharm el sheikh", "hurghada",
        "tel aviv", "jerusalem", "haifa", "beer sheva", "amman", "petra",
        "aqaba", "damascus", "beirut", "baghdad", "basra", "erbil", "sulaymaniyah",
        "tehran", "mashhad", "isfahan", "shiraz", "tabriz", "kerman", "yazd",
        "kabul", "kandahar", "herat", "mazar e sharif", "jalalabad", "kunduz",
        "peshawar", "lahore", "karachi", "islamabad", "rawalpindi", "faisalabad",
        "multan", "gujranwala", "sialkot", "quetta", "peshawar", "abbottabad",
        "murree", "gilgit", "skardu", "chitral", "swat", "hunza", "kashmir",
        "srinagar", "leh", "manali", "shimla", "mussoorie", "nainital", "ranikhet",
        "almora", "pithoragarh", "chamoli", "rudraprayag", "tehri", "uttarkashi",
        "dehradun", "haridwar", "rishikesh", "kedarnath", "badrinath", "gangotri",
        "yamunotri", "hemkund", "valley of flowers", "auli", "joshimath", "karnaprayag",
        "gauchar", "karanprayag"
    ]

    @classmethod
    def get_available_cities(cls) -> List[str]:
        """Get all available cities in our documents"""
        # Cities from our knowledge base documents
        return [
            'amsterdam', 'barcelona', 'berlin', 'budapest', 'london', 
            'munich', 'paris', 'prague', 'rome', 'vienna',  # Europe
            'beijing', 'kyoto', 'seoul', 'shanghai', 'singapore', 'tokyo'  # Asia
        ]
    
    @classmethod
    def get_preference_tracking_destinations(cls) -> List[str]:
        """Get destinations list for user preference tracking"""
        return cls.PREFERENCE_TRACKING_DESTINATIONS
    
    @classmethod
    def get_intent_analysis_destinations(cls) -> List[str]:
        """Get destinations list for intent analysis"""
        return cls.INTENT_ANALYSIS_DESTINATIONS
    
    @classmethod
    def get_comprehensive_destinations(cls) -> List[str]:
        """Get comprehensive destinations list for destination extraction"""
        return cls.COMPREHENSIVE_DESTINATIONS
    
    @classmethod
    def get_nearest_available_city(cls, target_region: str) -> str:
        """Get the nearest available city for a region/country"""
        available_cities = cls.get_available_cities()
        region_cities = cls.get_region_cities(target_region)
        
        # Find intersection between region cities and available cities
        for city in region_cities:
            if city in available_cities:
                return city
        
        # If no direct match, return first available city from region
        return region_cities[0] if region_cities else 'london'  # Default fallback

# Global instance for easy access
geo_mappings = GeographicalMappings()

# Convenience functions for backward compatibility
def get_region_to_cities() -> Dict[str, List[str]]:
    """Get the region to cities mapping"""
    return GeographicalMappings.REGION_TO_CITIES

def get_city_to_iata() -> Dict[str, str]:
    """Get the city to IATA mapping"""
    return GeographicalMappings.CITY_TO_IATA

def get_city_to_airport_codes() -> Dict[str, List[str]]:
    """Get the city to airport codes mapping"""
    return GeographicalMappings.CITY_TO_AIRPORT_CODES