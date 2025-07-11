#!/usr/bin/env python3
"""
Generic Transportation Data Extractor
Uses robust web scraping utilities to extract transportation information
Supports multiple countries and regions including Japan, Europe, and Asia
"""

import requests
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from typing import List, Dict, Optional, Any
import sys
import os

# Use centralized logging
from loguru import logger

# More realistic headers to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# Transportation configurations
TRANSPORTATION_CONFIGS = {
    "japan_shinkansen": {
        "url": "https://en.wikivoyage.org/wiki/Japan#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "trains",
        "country": "japan",
        "keywords": [
            "shinkansen",
            "bullet",
            "train",
            "jr",
            "pass",
            "tokyo",
            "osaka",
            "kyoto",
            "nagoya",
            "sendai",
            "hiroshima",
            "fukuoka",
            "nozomi",
            "hikari",
            "kodama",
            "hayabusa",
            "komachi",
            "tokaido",
            "tohoku",
            "hokuriku",
            "joetsu",
            "yamagata",
            "akita",
            "sanyo",
            "kyushu",
        ],
    },
    "japan_metro": {
        "url": "https://en.wikivoyage.org/wiki/Tokyo#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "japan",
        "keywords": [
            "tokyo",
            "metro",
            "subway",
            "jr",
            "yamanote",
            "chuo",
            "sobu",
            "ginza",
            "marunouchi",
            "hibiya",
            "tozai",
            "chiyoda",
            "hanzomon",
            "namboku",
            "yurakucho",
            "fukutoshin",
            "oedo",
            "suica",
            "pasmo",
        ],
    },
    "london_underground": {
        "url": "https://en.wikivoyage.org/wiki/London#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "transport_type": "metro",
        "country": "uk",
        "keywords": [
            "london",
            "underground",
            "tube",
            "metro",
            "oyster",
            "card",
            "central",
            "northern",
            "piccadilly",
            "district",
            "circle",
            "metropolitan",
            "hammersmith",
            "city",
            "waterloo",
            "city",
            "jubilee",
            "victoria",
            "bakerloo",
            "elizabeth",
            "line",
        ],
    },
    "paris_metro": {
        "url": "https://en.wikivoyage.org/wiki/Paris#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "transport_type": "metro",
        "country": "france",
        "keywords": [
            "paris",
            "metro",
            "subway",
            "ratp",
            "navigo",
            "pass",
            "line",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "rer",
            "a",
            "b",
            "c",
            "d",
            "e",
        ],
    },
    "newyork_subway": {
        "url": "https://en.wikivoyage.org/wiki/New_York_City#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "north_america",
        "transport_type": "metro",
        "country": "usa",
        "keywords": [
            "new",
            "york",
            "subway",
            "metro",
            "mta",
            "metrocard",
            "line",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "j",
            "l",
            "m",
            "n",
            "q",
            "r",
            "s",
            "w",
            "z",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
        ],
    },
    "singapore_mrt": {
        "url": "https://en.wikivoyage.org/wiki/Singapore#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "singapore",
        "keywords": [
            "singapore",
            "mrt",
            "metro",
            "subway",
            "ez",
            "link",
            "card",
            "north",
            "south",
            "east",
            "west",
            "circle",
            "downtown",
            "thomson",
            "east",
            "coast",
            "line",
            "lrt",
            "light",
            "rail",
        ],
    },
    "hong_kong_mtr": {
        "url": "https://en.wikivoyage.org/wiki/Hong_Kong#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "hong_kong",
        "keywords": [
            "hong",
            "kong",
            "mtr",
            "metro",
            "subway",
            "octopus",
            "card",
            "island",
            "line",
            "tseung",
            "kwan",
            "o",
            "tung",
            "chung",
            "airport",
            "express",
            "east",
            "rail",
            "west",
            "rail",
            "light",
        ],
    },
    "seoul_metro": {
        "url": "https://en.wikivoyage.org/wiki/Seoul#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "south_korea",
        "keywords": [
            "seoul",
            "metro",
            "subway",
            "tmoney",
            "card",
            "line",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "airport",
            "express",
            "light",
            "rail",
            "bundang",
            "shinbundang",
            "gyeongui",
            "jungang",
        ],
    },
    "bangkok_bts": {
        "url": "https://en.wikivoyage.org/wiki/Bangkok#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "thailand",
        "keywords": [
            "bangkok",
            "bts",
            "skytrain",
            "mrt",
            "metro",
            "subway",
            "sukhumvit",
            "silom",
            "gold",
            "line",
            "blue",
            "purple",
            "yellow",
            "pink",
            "orange",
            "red",
            "airport",
            "link",
        ],
    },
    "shanghai_metro": {
        "url": "https://en.wikivoyage.org/wiki/Shanghai#Get_around",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "transport_type": "metro",
        "country": "china",
        "keywords": [
            "shanghai",
            "metro",
            "subway",
            "line",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "16",
            "17",
            "18",
            "maglev",
            "airport",
            "express",
            "pujiang",
            "line",
        ],
    },
}


def get_session():
    """Create a session with proper headers and cookies"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def scrape_transportation(transportation_name: str):
    """Scrape transportation information from various sources"""
    session = get_session()

    if transportation_name.lower() not in TRANSPORTATION_CONFIGS:
        logger.error(
            f"Transportation '{transportation_name}' not configured. Available: {list(TRANSPORTATION_CONFIGS.keys())}"
        )
        return []

    config = TRANSPORTATION_CONFIGS[transportation_name.lower()]
    url = config["url"]

    try:
        # Add a random delay to avoid being flagged as a bot
        time.sleep(random.uniform(1, 3))

        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        transport_info = []

        # For WikiVoyage pages, look for specific transportation sections
        if "wikivoyage" in config["source_url"].lower():
            transport_info = scrape_wikivoyage_transportation(soup, config)
        else:
            # Fallback to general link extraction
            transport_info = scrape_general_links(soup, config)

        # Remove duplicates based on name
        unique_info = []
        seen_names = set()
        for info in transport_info:
            if info["name"] not in seen_names:
                unique_info.append(info)
                seen_names.add(info["name"])

        logger.info(
            f"Extracted {len(unique_info)} transportation items from {config['source']}"
        )
        return unique_info

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


def scrape_wikivoyage_transportation(soup, config):
    """Extract specific transportation information from WikiVoyage pages"""
    transport_info = []

    # Look for transportation sections in WikiVoyage
    transport_sections = [
        "Get around",
        "Transportation",
        "Public transport",
        "Metro",
        "Subway",
        "Trains",
        "Buses",
        "Tickets",
        "Fares",
        "Lines",
        "Routes",
    ]

    for section in transport_sections:
        # Find section headers
        headers = soup.find_all(["h2", "h3", "h4"])
        for header in headers:
            if header.get_text(strip=True).startswith(section):
                # Look for lists and links in this section
                next_elements = header.find_next_siblings(["ul", "ol", "p"], limit=10)
                for element in next_elements:
                    if element.name in ["ul", "ol"]:
                        # Extract from lists
                        list_items = element.find_all("li")
                        for item in list_items:
                            links = item.find_all("a")
                            for link in links:
                                name = link.get_text(strip=True)
                                href = link.get("href")
                                if name and href and len(name) > 3:
                                    # Check if it's a specific transportation item
                                    if is_specific_transportation(name, config):
                                        full_url = urljoin(config["source_url"], href)
                                        transport_info.append(
                                            {"name": name, "url": full_url}
                                        )
                    elif element.name == "p":
                        # Extract from paragraphs
                        links = element.find_all("a")
                        for link in links:
                            name = link.get_text(strip=True)
                            href = link.get("href")
                            if name and href and len(name) > 3:
                                if is_specific_transportation(name, config):
                                    full_url = urljoin(config["source_url"], href)
                                    transport_info.append(
                                        {"name": name, "url": full_url}
                                    )

    # Also look for specific transportation keywords in the page
    all_links = soup.find_all("a")
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get("href")

            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a specific transportation item
                if is_specific_transportation(name, config):
                    # Construct full URL
                    if href.startswith("/"):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], "/" + href)

                    # Avoid duplicates and navigation links
                    if name not in [info["name"] for info in transport_info]:
                        transport_info.append({"name": name, "url": full_url})

        except (AttributeError, TypeError) as e:
            continue

    return transport_info


def is_specific_transportation(name, config):
    """Check if a name represents a specific transportation item rather than a category"""
    name_lower = name.lower()

    # Skip generic categories and navigation
    generic_terms = [
        "home",
        "back",
        "next",
        "previous",
        "top",
        "menu",
        "search",
        "contact",
        "about",
        "privacy",
        "terms",
        "edit",
        "history",
        "transportation",
        "transport",
        "transit",
        "systems",
        "networks",
        "services",
        "routes",
        "lines",
        "stations",
        "stops",
    ]

    if any(term in name_lower for term in generic_terms):
        return False

    # Check if it contains specific transportation keywords
    transport_keywords = [
        "metro",
        "subway",
        "train",
        "bus",
        "tram",
        "light",
        "rail",
        "line",
        "route",
        "station",
        "stop",
        "terminal",
        "airport",
        "shinkansen",
        "underground",
        "tube",
        "mrt",
        "bts",
        "mtr",
    ]

    # Check if it matches destination-specific keywords
    if any(keyword in name_lower for keyword in config["keywords"]):
        return True

    # Check if it contains transportation keywords
    if any(keyword in name_lower for keyword in transport_keywords):
        return True

    # Check if it looks like a proper noun (capitalized words)
    words = name.split()
    if len(words) >= 2:
        capitalized_words = [word for word in words if word[0].isupper()]
        if len(capitalized_words) >= 2:
            return True

    return False


def scrape_general_links(soup, config):
    """Fallback method for general link extraction"""
    transport_info = []
    all_links = soup.find_all("a")

    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get("href")

            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a transportation link
                href_lower = href.lower()
                name_lower = name.lower()

                # Filter for likely transportation links based on destination
                is_transportation = (
                    any(keyword in href_lower for keyword in config["keywords"])
                    or any(
                        keyword in name_lower
                        for keyword in [
                            "metro",
                            "subway",
                            "train",
                            "bus",
                            "tram",
                            "light",
                            "rail",
                            "line",
                            "route",
                            "station",
                            "stop",
                        ]
                    )
                    or any(
                        keyword in name_lower
                        for keyword in [
                            "metro",
                            "subway",
                            "train",
                            "bus",
                            "tram",
                            "light",
                            "rail",
                            "line",
                            "route",
                            "station",
                            "stop",
                        ]
                    )
                )

                # Avoid navigation and non-transportation links
                is_navigation = any(
                    skip in name_lower
                    for skip in [
                        "home",
                        "back",
                        "next",
                        "previous",
                        "top",
                        "menu",
                        "search",
                        "contact",
                        "about",
                        "privacy",
                        "terms",
                    ]
                )

                if is_transportation and not is_navigation:
                    # Construct full URL
                    if href.startswith("/"):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], "/" + href)

                    # Avoid duplicates
                    if name not in [info["name"] for info in transport_info]:
                        transport_info.append({"name": name, "url": full_url})

        except (AttributeError, TypeError) as e:
            continue

    return transport_info


def create_transportation_data(
    transport_info: List[Dict[str, str]],
    transportation_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a standardized transportation data structure"""

    # Create transportation-specific content
    if config["transport_type"] == "trains" and config["country"] == "japan":
        content = create_japan_train_content(transport_info, transportation_name)
    elif config["transport_type"] == "metro":
        content = create_metro_content(transport_info, transportation_name, config)
    else:
        content = create_generic_transportation_content(
            transport_info, transportation_name
        )

    return {
        "id": f"{transportation_name.lower().replace(' ', '_').replace(',', '')}_complete_guide",
        "title": f"{transportation_name.replace('_', ' ').title()} Complete Travel Guide",
        "content": content,
        "category": "transportation",
        "subcategory": config["transport_type"],
        "location": config["country"],
        "tags": [
            transportation_name.replace("_", " ").lower(),
            "Transportation",
            config["transport_type"].title(),
            config["country"].title(),
        ],
        "language": "en",
        "source": {
            "id": config["source_id"],
            "name": config["source"],
            "url": config["source_url"],
            "reliability_score": 0.9,
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def create_japan_train_content(
    transport_info: List[Dict[str, str]], transportation_name: str
) -> str:
    """Create content specifically for Japanese train systems"""
    content = f"""{transportation_name.replace("_", " ").title()} is Japan's world-renowned transportation system, known for its efficiency, punctuality, and extensive network. This comprehensive guide covers everything you need to know about using {transportation_name.replace("_", " ").title()}.\n\n## Overview\n{transportation_name.replace("_", " ").title()} offers visitors a reliable and efficient way to travel throughout Japan. With multiple lines serving different regions, it's the fastest and most convenient way to travel between major cities and explore the country.\n\n## Major Lines and Routes\n\n### Key Transportation Lines\n"""

    # Add transportation information
    for i, info in enumerate(transport_info[:15], 1):  # Top 15 items
        content += f"- **{info['name']}**: Important transportation route or service\n"

    content += f"""
## Ticket Types and Pricing

### Basic Fares
- **Single Journey**: Pay-per-ride tickets
- **Day Pass**: Unlimited travel for 24 hours
- **Weekly Pass**: Cost-effective for longer stays
- **Monthly Pass**: Best value for regular commuters

### Special Passes
- **Tourist Passes**: Discounted rates for visitors
- **Student Passes**: Reduced fares for students
- **Senior Passes**: Discounts for elderly travelers
- **Group Passes**: Savings for multiple travelers

## How to Use

### Buying Tickets
- **Ticket Machines**: Available in multiple languages
- **Staff Counters**: Assistance available at major stations
- **Mobile Apps**: Digital tickets and passes
- **IC Cards**: Rechargeable smart cards for convenience

### Boarding Process
- **Platform Access**: Use your ticket or IC card
- **Platform Queues**: Line up in designated areas
- **Boarding Time**: Arrive 5 minutes before departure
- **Seat Finding**: Use overhead displays and seat numbers

## Station Information

### Major Stations
- **Central Stations**: Main transportation hubs
- **Transfer Points**: Easy connections between lines
- **Tourist Stations**: Near popular attractions
- **Airport Connections**: Direct access to airports

### Station Facilities
- **Ticket Offices**: Staff assistance and information
- **Restrooms**: Available at all stations
- **Shops and Restaurants**: Convenience stores and dining
- **Information Centers**: Tourist information and maps

## Travel Tips

### Best Practices
- **Peak Hours**: Avoid 7-9 AM and 5-7 PM rush hours
- **Off-Peak Travel**: More comfortable and often cheaper
- **Advance Planning**: Check schedules and routes
- **Backup Plans**: Know alternative routes

### Money-Saving Tips
- **Pass Comparison**: Calculate which pass offers best value
- **Off-Peak Discounts**: Travel during less busy times
- **Group Discounts**: Travel with others for savings
- **Student Discounts**: Use student ID for reduced fares

## Accessibility

### Services Available
- **Wheelchair Access**: All stations and trains
- **Priority Seating**: Available in all cars
- **Assistance**: Staff available at major stations
- **Elevators**: Available at all stations
- **Accessible Restrooms**: On all trains

### Special Needs
- **Visual Impairments**: Audio announcements and tactile guidance
- **Hearing Impairments**: Visual displays and staff assistance
- **Mobility Assistance**: Escort services available
- **Medical Support**: First aid and emergency contacts

## Safety and Etiquette

### General Behavior
- **Quiet Conversations**: Keep voices low
- **Phone Calls**: Use designated areas or avoid entirely
- **Seat Courtesy**: Offer seats to elderly and disabled
- **Cleanliness**: Clean up after yourself
- **Queue Properly**: Line up in designated areas

### Emergency Information
- **Emergency Contacts**: Posted throughout stations
- **Lost and Found**: Report items to station staff
- **Medical Emergencies**: Contact station staff immediately
- **Security**: Report suspicious activity to staff

## Seasonal Considerations

### Spring (March-May)
- **Cherry Blossom Season**: Extremely busy, book early
- **Weather**: Mild, occasional rain
- **Crowds**: Peak season, expect full trains

### Summer (June-August)
- **Rainy Season**: Potential delays in June-July
- **Heat**: Air conditioning on all trains
- **Festivals**: Increased travel to festival destinations

### Autumn (September-November)
- **Fall Foliage**: Beautiful mountain views
- **Weather**: Clear, comfortable temperatures
- **Crowds**: Moderate, pleasant travel conditions

### Winter (December-February)
- **Snow**: Spectacular mountain views
- **Weather**: Heating on all trains
- **Delays**: Possible due to heavy snow

## Contact Information

### Customer Service
- **Phone Support**: Available in multiple languages
- **Online Help**: Website with comprehensive information
- **Station Staff**: Assistance at all major stations
- **Tourist Information**: Specialized help for visitors

### Emergency Contacts
- **General Emergency**: 110 (Police), 119 (Fire/Ambulance)
- **Transportation Hotline**: Available 24/7
- **Tourist Hotline**: English-speaking assistance
- **Lost and Found**: Centralized service for all items

{transportation_name.replace("_", " ").title()} represents the pinnacle of public transportation efficiency and comfort. With proper planning and understanding of the system, it provides an excellent way to explore Japan's cities and regions."""

    return content


def create_metro_content(
    transport_info: List[Dict[str, str]],
    transportation_name: str,
    config: Dict[str, Any],
) -> str:
    """Create content specifically for metro/subway systems"""
    country_name = config["country"].replace("_", " ").title()
    transport_type = config["transport_type"].title()

    content = f"""{transportation_name.replace("_", " ").title()} is {country_name}'s efficient {transport_type.lower()} system, providing fast and reliable transportation throughout the city. This comprehensive guide covers everything you need to know about using {transportation_name.replace("_", " ").title()}.\n\n## Overview\n{transportation_name.replace("_", " ").title()} offers visitors a convenient and affordable way to navigate the city. With extensive coverage and frequent service, it's the fastest way to reach major attractions and neighborhoods.\n\n## Major Lines and Routes\n\n### Key {transport_type} Lines and Services\n"""

    # Add transportation information with more specific descriptions
    for i, info in enumerate(transport_info[:20], 1):  # Top 20 items
        # Create more specific descriptions based on the scraped content
        if any(
            keyword in info["name"].lower()
            for keyword in ["line", "route", "metro", "subway", "train"]
        ):
            content += f"- **{info['name']}**: Major transportation line or route\n"
        elif any(
            keyword in info["name"].lower()
            for keyword in ["station", "stop", "terminal"]
        ):
            content += f"- **{info['name']}**: Important station or transfer point\n"
        elif any(
            keyword in info["name"].lower()
            for keyword in ["airport", "express", "rapid"]
        ):
            content += f"- **{info['name']}**: Airport connection or express service\n"
        elif any(
            keyword in info["name"].lower()
            for keyword in ["card", "pass", "ticket", "fare"]
        ):
            content += f"- **{info['name']}**: Ticketing and payment information\n"
        else:
            content += (
                f"- **{info['name']}**: Important transportation service or facility\n"
            )

    # Add city-specific information based on the transportation system
    content += f"""
    
## City-Specific Information for {transportation_name.replace("_", " ").title()}

### Special Features and Services
"""

    # Add city-specific content based on the transportation system
    if "london" in transportation_name.lower():
        content += """- **Oyster Card**: Contactless smart card for all London transport
- **Underground Network**: 11 lines serving 270 stations across London
- **Night Tube**: 24-hour service on selected lines (Friday and Saturday)
- **Elizabeth Line**: New cross-London railway connecting east and west
- **Overground**: Surface rail network connecting outer London areas
- **DLR**: Docklands Light Railway serving Canary Wharf and East London\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Navigo Card**: Weekly and monthly passes for unlimited travel
- **RER**: Regional express trains connecting Paris with suburbs
- **Metro Network**: 16 lines serving 308 stations across Paris
- **Tramway**: Modern tram system serving outer areas
- **Vélib'**: Public bicycle sharing system
- **Noctilien**: Night bus service when metro is closed\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **MetroCard**: Magnetic stripe card for subway and bus access
- **Subway Network**: 24 lines serving 472 stations across NYC
- **Express Trains**: Skip-stop service during peak hours
- **AirTrain**: Connection to JFK and LaGuardia airports
- **PATH**: Commuter rail connecting NYC with New Jersey
- **Staten Island Ferry**: Free ferry service to Staten Island\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Suica/Pasmo**: Contactless IC cards for all Tokyo transport
- **JR Yamanote Line**: Circular line connecting major Tokyo stations
- **Tokyo Metro**: 9 lines serving central Tokyo
- **Toei Subway**: 4 lines operated by Tokyo Metropolitan Government
- **Shinkansen**: High-speed rail connections to other cities
- **Monorail**: Automated monorail to Haneda Airport\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **EZ-Link Card**: Contactless smart card for all transport
- **MRT Network**: 6 lines serving 134 stations across Singapore
- **LRT**: Light rail transit serving residential areas
- **Sentosa Express**: Monorail to Sentosa Island
- **Changi Airport MRT**: Direct connection to Changi Airport
- **Bus Integration**: Seamless connection with bus network\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Octopus Card**: Contactless smart card for all Hong Kong transport
- **MTR Network**: 10 lines serving 98 stations across Hong Kong
- **Airport Express**: Direct connection to Hong Kong International Airport
- **Light Rail**: Tram system serving New Territories
- **Peak Tram**: Historic funicular to Victoria Peak
- **Star Ferry**: Iconic ferry service across Victoria Harbour\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **T-money Card**: Contactless smart card for all Seoul transport
- **Metro Network**: 9 lines serving 337 stations across Seoul
- **AREX**: Airport Railroad Express to Incheon Airport
- **Shinbundang Line**: Private metro line serving Gangnam area
- **Light Rail**: Automated light rail serving new developments
- **Night Bus**: Late-night bus service when metro is closed\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **BTS Skytrain**: Elevated rail system serving central Bangkok
- **MRT**: Underground metro system serving central Bangkok
- **ARL**: Airport Rail Link connecting to Suvarnabhumi Airport
- **BTS Card**: Smart card for BTS Skytrain access
- **MRT Card**: Separate smart card for MRT access
- **Boat Services**: Chao Phraya Express Boat and canal boats\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Shanghai Public Transportation Card**: Smart card for all transport
- **Metro Network**: 18 lines serving 460+ stations across Shanghai
- **Maglev**: High-speed magnetic levitation train to Pudong Airport
- **Pujiang Line**: Automated people mover serving new areas
- **Airport Express**: Metro connections to both airports
- **Bus Integration**: Extensive bus network complementing metro\n"""
    else:
        content += (
            f"- **Smart Card System**: Contactless payment for convenient travel\n"
        )

    content += f"""
## Ticket Types and Pricing for {transportation_name.replace("_", " ").title()}

### Basic Fares
"""

    # Add city-specific fare information
    if "london" in transportation_name.lower():
        content += """- **Pay As You Go**: £2.50-£6.00 depending on zones
- **Day Travelcard**: £13.10 for unlimited travel in zones 1-6
- **Weekly Travelcard**: £64.20 for unlimited travel in zones 1-6
- **Oyster Card**: £5 deposit, top-up as needed
- **Contactless**: Use credit/debit cards for same fares as Oyster\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Single Ticket**: €1.90 for metro/bus
- **Day Pass**: €7.50 for unlimited travel
- **Navigo Weekly**: €22.80 for unlimited travel
- **Navigo Monthly**: €75.20 for unlimited travel
- **Airport Transfer**: €10.30 for RER to CDG Airport\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **Single Ride**: $2.75 for subway/bus
- **7-Day Unlimited**: $33 for unlimited rides
- **30-Day Unlimited**: $127 for unlimited rides
- **MetroCard**: $1 fee for new card
- **AirTrain**: $7.75 to JFK Airport\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Single Journey**: ¥170-¥320 depending on distance
- **Day Pass**: ¥800 for unlimited Tokyo Metro/Toei
- **Suica/Pasmo**: ¥500 deposit, top-up as needed
- **JR Pass**: ¥29,650 for 7-day unlimited JR travel
- **Airport Transfer**: ¥1,300 for Narita Express\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **Single Journey**: S$0.77-S$2.36 depending on distance
- **EZ-Link Card**: S$5 deposit, top-up as needed
- **Tourist Pass**: S$10 for unlimited travel (S$5 refundable deposit)
- **Airport Transfer**: S$2.50 for MRT to Changi Airport
- **Sentosa Express**: S$4 for round trip to Sentosa\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Single Journey**: HK$4.50-HK$51.00 depending on distance
- **Octopus Card**: HK$50 deposit, top-up as needed
- **Airport Express**: HK$115 for single journey
- **Tourist Day Pass**: HK$65 for unlimited MTR travel
- **Peak Tram**: HK$52 for round trip to Victoria Peak\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **Single Journey**: ₩1,250-₩2,450 depending on distance
- **T-money Card**: ₩4,000 deposit, top-up as needed
- **AREX**: ₩9,500 for Incheon Airport Express
- **Tourist Pass**: ₩10,000 for unlimited travel
- **Night Bus**: ₩2,150 for late-night service\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **BTS Single**: ฿16-฿59 depending on distance
- **MRT Single**: ฿16-฿42 depending on distance
- **BTS Card**: ฿100 deposit, top-up as needed
- **ARL**: ฿15-฿45 for Airport Rail Link
- **Boat Services**: ฿10-฿32 for Chao Phraya Express\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Single Journey**: ¥3-¥10 depending on distance
- **Shanghai Card**: ¥20 deposit, top-up as needed
- **Tourist Card**: ¥20 for unlimited travel (¥20 refundable deposit)
- **Maglev**: ¥50 for single journey to Pudong Airport
- **Airport Express**: ¥7-¥9 for metro to airports\n"""
    else:
        content += """- **Single Journey**: Pay-per-ride tickets
- **Day Pass**: Unlimited travel for 24 hours
- **Weekly Pass**: Cost-effective for longer stays
- **Monthly Pass**: Best value for regular commuters\n"""

    content += f"""
### Special Passes
- **Tourist Passes**: Discounted rates for visitors
- **Student Passes**: Reduced fares for students
- **Senior Passes**: Discounts for elderly travelers
- **Group Passes**: Savings for multiple travelers

## How to Use {transportation_name.replace("_", " ").title()}

### Buying Tickets
"""

    # Add city-specific ticket buying information
    if "london" in transportation_name.lower():
        content += """- **Oyster Card**: Purchase at stations, shops, or online
- **Contactless**: Use credit/debit cards directly
- **Ticket Machines**: Available at all stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: TfL Go app for journey planning\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Navigo Card**: Purchase at stations or online
- **Ticket Machines**: Available at all metro stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: RATP app for journey planning
- **Tourist Offices**: Special passes for visitors\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **MetroCard**: Purchase at stations or shops
- **Ticket Machines**: Available at all subway stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: MTA app for journey planning
- **OMNY**: Contactless payment system (phasing in)\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Suica/Pasmo**: Purchase at stations or convenience stores
- **Ticket Machines**: Available at all stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: Japan Transit Planner app
- **JR Pass**: Purchase before arriving in Japan\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **EZ-Link Card**: Purchase at stations or convenience stores
- **Ticket Machines**: Available at all MRT stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: MyTransport app for journey planning
- **Tourist Pass**: Purchase at Changi Airport or major stations\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Octopus Card**: Purchase at stations or convenience stores
- **Ticket Machines**: Available at all MTR stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: MTR Mobile app for journey planning
- **Tourist Pass**: Purchase at airport or major stations\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **T-money Card**: Purchase at stations or convenience stores
- **Ticket Machines**: Available at all metro stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: Seoul Metro app for journey planning
- **Tourist Pass**: Purchase at airport or major stations\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **BTS Card**: Purchase at BTS stations
- **MRT Card**: Purchase at MRT stations
- **Ticket Machines**: Available at all stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: BTS Skytrain app for journey planning\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Shanghai Card**: Purchase at stations or convenience stores
- **Ticket Machines**: Available at all metro stations
- **Staff Counters**: Assistance at major stations
- **Mobile Apps**: Shanghai Metro app for journey planning
- **Tourist Card**: Purchase at airport or major stations\n"""
    else:
        content += """- **Ticket Machines**: Available in multiple languages
- **Staff Counters**: Assistance available at major stations
- **Mobile Apps**: Digital tickets and passes
- **Smart Cards**: Rechargeable cards for convenience\n"""

    content += f"""
### Boarding Process
- **Platform Access**: Use your ticket or smart card
- **Platform Queues**: Line up in designated areas
- **Boarding Time**: Trains arrive frequently
- **Car Selection**: Choose less crowded cars

## Station Information for {transportation_name.replace("_", " ").title()}

### Major Stations
- **Central Stations**: Main transportation hubs
- **Transfer Points**: Easy connections between lines
- **Tourist Stations**: Near popular attractions
- **Airport Connections**: Direct access to airports

### Station Facilities
- **Ticket Offices**: Staff assistance and information
- **Restrooms**: Available at most stations
- **Shops and Restaurants**: Convenience stores and dining
- **Information Centers**: Tourist information and maps

## Travel Tips for {transportation_name.replace("_", " ").title()}

### Best Practices and Peak Hours
"""

    # Add city-specific travel tips
    if "london" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:30-9:30 AM and 5:00-7:00 PM on weekdays
- **Off-Peak Travel**: Cheaper fares outside peak hours and on weekends
- **Night Tube**: Available on Friday and Saturday nights on selected lines
- **Advance Planning**: Use TfL Go app for real-time journey planning
- **Backup Plans**: Know alternative routes during strikes or delays
- **Zone Awareness**: Fares vary by zones, plan routes to minimize zone crossings\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 8:00-9:30 AM and 6:00-7:30 PM on weekdays
- **Off-Peak Travel**: Cheaper fares outside peak hours and on weekends
- **Night Service**: Noctilien buses when metro is closed (12:30-5:30 AM)
- **Advance Planning**: Use RATP app for real-time journey planning
- **Backup Plans**: Know alternative routes during strikes (common in Paris)
- **Zone Awareness**: Fares vary by zones, plan routes to minimize zone crossings\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:00-9:00 AM and 5:00-7:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Express Trains**: Use express trains during peak hours for faster travel
- **Advance Planning**: Use MTA app for real-time journey planning
- **Backup Plans**: Know alternative routes during weekend maintenance
- **Transfer Rules**: Free transfers between subway and bus within 2 hours\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Peak Hours**: Avoid 7:30-9:30 AM and 6:00-8:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Women-Only Cars**: Available during peak hours on some lines
- **Advance Planning**: Use Japan Transit Planner app for detailed routing
- **Backup Plans**: Know alternative routes during rush hour congestion
- **Transfer Rules**: Different fares for different operators (JR, Metro, Toei)\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:30-9:00 AM and 6:00-7:30 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Air Conditioning**: All trains and stations are air-conditioned
- **Advance Planning**: Use MyTransport app for real-time journey planning
- **Backup Plans**: Know alternative routes during peak hours
- **Transfer Rules**: Free transfers between MRT and bus within 45 minutes\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:30-9:30 AM and 6:00-8:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Air Conditioning**: All trains and stations are air-conditioned
- **Advance Planning**: Use MTR Mobile app for real-time journey planning
- **Backup Plans**: Know alternative routes during peak hours
- **Transfer Rules**: Free transfers between MTR lines within 30 minutes\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:30-9:30 AM and 6:00-8:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Night Bus**: Available when metro is closed (1:00-5:00 AM)
- **Advance Planning**: Use Seoul Metro app for real-time journey planning
- **Backup Plans**: Know alternative routes during peak hours
- **Transfer Rules**: Free transfers between metro lines within 30 minutes\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:00-9:00 AM and 5:00-7:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Air Conditioning**: All BTS and MRT trains are air-conditioned
- **Advance Planning**: Use BTS Skytrain app for real-time journey planning
- **Backup Plans**: Know alternative routes during peak hours
- **Transfer Rules**: Separate fares for BTS and MRT (no free transfers)\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Peak Hours**: Avoid 7:30-9:30 AM and 6:00-8:00 PM on weekdays
- **Off-Peak Travel**: Same fare regardless of time, but less crowded
- **Air Conditioning**: All trains and stations are air-conditioned
- **Advance Planning**: Use Shanghai Metro app for real-time journey planning
- **Backup Plans**: Know alternative routes during peak hours
- **Transfer Rules**: Free transfers between metro lines within 30 minutes\n"""
    else:
        content += """- **Peak Hours**: Avoid 7-9 AM and 5-7 PM rush hours
- **Off-Peak Travel**: More comfortable and often cheaper
- **Advance Planning**: Check schedules and routes
- **Backup Plans**: Know alternative routes\n"""

    content += f"""
### Money-Saving Tips for {transportation_name.replace("_", " ").title()}
"""

    # Add city-specific money-saving tips
    if "london" in transportation_name.lower():
        content += """- **Oyster vs Contactless**: Same fares, but Oyster has daily/weekly caps
- **Travelcard Comparison**: Calculate if daily/weekly travelcards save money
- **Off-Peak Discounts**: 30% cheaper fares outside peak hours
- **Group Discounts**: Family and Friends Railcard for 33% off
- **Student Discounts**: 18+ Student Oyster photocard for 30% off
- **Senior Discounts**: Freedom Pass for free travel for over-60s\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Navigo vs Single Tickets**: Navigo cheaper for 3+ days of travel
- **Tourist Passes**: Paris Visite pass for unlimited travel with museum discounts
- **Off-Peak Discounts**: Cheaper Navigo passes for off-peak travel
- **Group Discounts**: Group tickets available for 4+ people
- **Student Discounts**: Reduced fares for students with valid ID
- **Senior Discounts**: Reduced fares for seniors over 65\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **Unlimited vs Pay-Per-Ride**: Calculate based on number of trips
- **7-Day vs 30-Day**: 30-day pass offers better value for regular commuters
- **Express Bus Premium**: Additional fare for express bus service
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Reduced fares for students with valid ID
- **Senior Discounts**: Reduced fares for seniors over 65\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Suica/Pasmo vs Single Tickets**: IC cards offer convenience and small discounts
- **JR Pass**: Worth it for 7+ days of extensive JR travel
- **Tokyo Metro Pass**: Good value for 1-3 days of metro-only travel
- **Off-Peak Discounts**: Some private railways offer off-peak discounts
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 65\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **EZ-Link vs Tourist Pass**: Tourist pass good for 1-3 days of unlimited travel
- **Pass Comparison**: Calculate if tourist pass saves money vs EZ-Link
- **Off-Peak Discounts**: Same fares regardless of time
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 60\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Octopus vs Tourist Pass**: Tourist pass good for 1-3 days of unlimited travel
- **Pass Comparison**: Calculate if tourist pass saves money vs Octopus
- **Off-Peak Discounts**: Same fares regardless of time
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 65\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **T-money vs Tourist Pass**: Tourist pass good for 1-7 days of unlimited travel
- **Pass Comparison**: Calculate if tourist pass saves money vs T-money
- **Off-Peak Discounts**: Same fares regardless of time
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 65\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **BTS/MRT Cards**: Separate cards needed for each system
- **Pass Comparison**: Calculate if day passes save money vs single tickets
- **Off-Peak Discounts**: Same fares regardless of time
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 60\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Shanghai Card vs Tourist Pass**: Tourist pass good for 1-3 days of unlimited travel
- **Pass Comparison**: Calculate if tourist pass saves money vs Shanghai Card
- **Off-Peak Discounts**: Same fares regardless of time
- **Group Discounts**: Available for school groups and organizations
- **Student Discounts**: Available for students with valid ID
- **Senior Discounts**: Available for seniors over 65\n"""
    else:
        content += """- **Pass Comparison**: Calculate which pass offers best value
- **Off-Peak Discounts**: Travel during less busy times
- **Group Discounts**: Travel with others for savings
- **Student Discounts**: Use student ID for reduced fares\n"""

    content += f"""
## Accessibility for {transportation_name.replace("_", " ").title()}

### Services Available
"""

    # Add city-specific accessibility information
    if "london" in transportation_name.lower():
        content += """- **Step-free Access**: Available at 91 stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all major stations for escort services
- **Elevators**: Available at step-free stations with audio announcements
- **Accessible Restrooms**: Available at major stations with changing facilities
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "paris" in transportation_name.lower():
        content += """- **Step-free Access**: Available at 9 metro stations with elevators
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at major stations for escort services
- **Elevators**: Available at accessible stations with audio announcements
- **Accessible Restrooms**: Available at major stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **Step-free Access**: Available at 25% of stations with elevators
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at major stations for escort services
- **Elevators**: Available at accessible stations with audio announcements
- **Accessible Restrooms**: Available at major stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **Step-free Access**: Available at most stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all stations for escort services
- **Elevators**: Available at most stations with audio announcements
- **Accessible Restrooms**: Available at major stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **Step-free Access**: Available at all stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all stations for escort services
- **Elevators**: Available at all stations with audio announcements
- **Accessible Restrooms**: Available at all stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **Step-free Access**: Available at all stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all stations for escort services
- **Elevators**: Available at all stations with audio announcements
- **Accessible Restrooms**: Available at all stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **Step-free Access**: Available at all stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all stations for escort services
- **Elevators**: Available at all stations with audio announcements
- **Accessible Restrooms**: Available at all stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **Step-free Access**: Available at most stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at major stations for escort services
- **Elevators**: Available at most stations with audio announcements
- **Accessible Restrooms**: Available at major stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Step-free Access**: Available at all stations with elevators and ramps
- **Priority Seating**: Available in all cars with clear signage
- **Assistance**: Staff available at all stations for escort services
- **Elevators**: Available at all stations with audio announcements
- **Accessible Restrooms**: Available at all stations
- **Audio Announcements**: Available on all trains and platforms\n"""
    else:
        content += """- **Wheelchair Access**: Most stations and trains
- **Priority Seating**: Available in all cars
- **Assistance**: Staff available at major stations
- **Elevators**: Available at most stations
- **Accessible Restrooms**: At major stations\n"""

    content += f"""
### Special Needs Support
- **Visual Impairments**: Audio announcements and tactile guidance
- **Hearing Impairments**: Visual displays and staff assistance
- **Mobility Assistance**: Escort services available
- **Medical Support**: First aid and emergency contacts

## Safety and Etiquette for {transportation_name.replace("_", " ").title()}

### General Behavior
- **Quiet Conversations**: Keep voices low
- **Phone Calls**: Use designated areas or avoid entirely
- **Seat Courtesy**: Offer seats to elderly and disabled
- **Cleanliness**: Clean up after yourself
- **Queue Properly**: Line up in designated areas

### Emergency Information
- **Emergency Contacts**: Posted throughout stations
- **Lost and Found**: Report items to station staff
- **Medical Emergencies**: Contact station staff immediately
- **Security**: Report suspicious activity to staff

## Seasonal Considerations for {transportation_name.replace("_", " ").title()}

### Spring (March-May)
- **Tourist Season**: Busy with visitors, book early
- **Weather**: Mild, occasional rain
- **Crowds**: Peak season, expect full trains

### Summer (June-August)
- **Vacation Season**: Increased tourist traffic
- **Heat**: Air conditioning on all trains
- **Festivals**: Increased travel to festival destinations

### Autumn (September-November)
- **Fall Season**: Beautiful city views
- **Weather**: Clear, comfortable temperatures
- **Crowds**: Moderate, pleasant travel conditions

### Winter (December-February)
- **Holiday Season**: Busy with holiday travelers
- **Weather**: Heating on all trains
- **Delays**: Possible due to weather conditions

## Contact Information for {transportation_name.replace("_", " ").title()}

### Customer Service
"""

    # Add city-specific contact information
    if "london" in transportation_name.lower():
        content += """- **TfL Customer Service**: 0343 222 1234 (24/7)
- **Online Help**: tfl.gov.uk with comprehensive information
- **Station Staff**: Assistance at all major stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 0343 222 1234 or visit Baker Street station\n"""
    elif "paris" in transportation_name.lower():
        content += """- **RATP Customer Service**: 3246 (from France) or +33 1 58 76 16 16
- **Online Help**: ratp.fr with comprehensive information
- **Station Staff**: Assistance at major stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 3246 or visit Châtelet-Les Halles station\n"""
    elif "newyork" in transportation_name.lower():
        content += """- **MTA Customer Service**: 511 (from NYC) or 1-877-690-5116
- **Online Help**: mta.info with comprehensive information
- **Station Staff**: Assistance at major stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 1-877-690-5116 or visit 34th Street-Penn Station\n"""
    elif (
        "tokyo" in transportation_name.lower() or "japan" in transportation_name.lower()
    ):
        content += """- **JR East Customer Service**: 050-2016-1603 (English available)
- **Tokyo Metro Customer Service**: 03-3941-2004
- **Online Help**: jreast.co.jp and tokyometro.jp
- **Station Staff**: Assistance at all stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: Contact individual operators\n"""
    elif "singapore" in transportation_name.lower():
        content += """- **SMRT Customer Service**: 1800-336-8900
- **SBS Transit Customer Service**: 1800-287-2727
- **Online Help**: smrt.com.sg and sbstransit.com.sg
- **Station Staff**: Assistance at all stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: Contact individual operators\n"""
    elif "hong_kong" in transportation_name.lower():
        content += """- **MTR Customer Service**: 2881 8888 (24/7)
- **Online Help**: mtr.com.hk with comprehensive information
- **Station Staff**: Assistance at all stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 2881 8888 or visit major stations\n"""
    elif "seoul" in transportation_name.lower():
        content += """- **Seoul Metro Customer Service**: 1577-1234
- **Online Help**: smrt.co.kr with comprehensive information
- **Station Staff**: Assistance at all stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 1577-1234 or visit major stations\n"""
    elif "bangkok" in transportation_name.lower():
        content += """- **BTS Customer Service**: 02-617-7300
- **MRT Customer Service**: 02-354-2000
- **Online Help**: bts.co.th and mrta.co.th
- **Station Staff**: Assistance at major stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: Contact individual operators\n"""
    elif "shanghai" in transportation_name.lower():
        content += """- **Shanghai Metro Customer Service**: 64370000
- **Online Help**: shmetro.com with comprehensive information
- **Station Staff**: Assistance at all stations
- **Tourist Information**: Available at major stations and airports
- **Lost Property**: 64370000 or visit major stations\n"""
    else:
        content += """- **Phone Support**: Available in multiple languages
- **Online Help**: Website with comprehensive information
- **Station Staff**: Assistance at all major stations
- **Tourist Information**: Specialized help for visitors\n"""

    content += f"""
### Emergency Contacts
- **General Emergency**: Local emergency numbers
- **Transportation Hotline**: Available 24/7
- **Tourist Hotline**: English-speaking assistance
- **Lost and Found**: Centralized service for all items

{transportation_name.replace("_", " ").title()} provides an excellent way to explore the city efficiently and affordably. With proper planning and understanding of the system, it offers a convenient transportation experience for both locals and visitors."""

    return content


def create_generic_transportation_content(
    transport_info: List[Dict[str, str]], transportation_name: str
) -> str:
    """Create generic transportation content"""
    content = f"""{transportation_name.replace("_", " ").title()} is a comprehensive transportation system that provides efficient and reliable travel options. This guide covers everything you need to know about using {transportation_name.replace("_", " ").title()}.\n\n## Overview\n{transportation_name.replace("_", " ").title()} offers visitors a convenient way to travel throughout the region. With multiple lines and extensive coverage, it's an excellent choice for both local and tourist transportation needs.\n\n## Major Lines and Services\n\n### Key Transportation Routes\n"""

    # Add transportation information
    for i, info in enumerate(transport_info[:10], 1):  # Top 10 items
        content += f"- **{info['name']}**: Important transportation route or service\n"

    content += f"""
## Ticket Types and Pricing

### Basic Fares
- **Single Journey**: Pay-per-ride tickets
- **Day Pass**: Unlimited travel for 24 hours
- **Weekly Pass**: Cost-effective for longer stays
- **Monthly Pass**: Best value for regular commuters

### Special Passes
- **Tourist Passes**: Discounted rates for visitors
- **Student Passes**: Reduced fares for students
- **Senior Passes**: Discounts for elderly travelers
- **Group Passes**: Savings for multiple travelers

## How to Use

### Buying Tickets
- **Ticket Machines**: Available in multiple languages
- **Staff Counters**: Assistance available at major stations
- **Mobile Apps**: Digital tickets and passes
- **Smart Cards**: Rechargeable cards for convenience

### Boarding Process
- **Platform Access**: Use your ticket or smart card
- **Platform Queues**: Line up in designated areas
- **Boarding Time**: Arrive early for peak hours
- **Seat Finding**: Use overhead displays and seat numbers

## Station Information

### Major Stations
- **Central Stations**: Main transportation hubs
- **Transfer Points**: Easy connections between lines
- **Tourist Stations**: Near popular attractions
- **Airport Connections**: Direct access to airports

### Station Facilities
- **Ticket Offices**: Staff assistance and information
- **Restrooms**: Available at most stations
- **Shops and Restaurants**: Convenience stores and dining
- **Information Centers**: Tourist information and maps

## Travel Tips

### Best Practices
- **Peak Hours**: Avoid rush hours for more comfortable travel
- **Off-Peak Travel**: More comfortable and often cheaper
- **Advance Planning**: Check schedules and routes
- **Backup Plans**: Know alternative routes

### Money-Saving Tips
- **Pass Comparison**: Calculate which pass offers best value
- **Off-Peak Discounts**: Travel during less busy times
- **Group Discounts**: Travel with others for savings
- **Student Discounts**: Use student ID for reduced fares

## Accessibility

### Services Available
- **Wheelchair Access**: Most stations and vehicles
- **Priority Seating**: Available in all vehicles
- **Assistance**: Staff available at major stations
- **Elevators**: Available at most stations
- **Accessible Restrooms**: At major stations

### Special Needs
- **Visual Impairments**: Audio announcements and tactile guidance
- **Hearing Impairments**: Visual displays and staff assistance
- **Mobility Assistance**: Escort services available
- **Medical Support**: First aid and emergency contacts

## Safety and Etiquette

### General Behavior
- **Quiet Conversations**: Keep voices low
- **Phone Calls**: Use designated areas or avoid entirely
- **Seat Courtesy**: Offer seats to elderly and disabled
- **Cleanliness**: Clean up after yourself
- **Queue Properly**: Line up in designated areas

### Emergency Information
- **Emergency Contacts**: Posted throughout stations
- **Lost and Found**: Report items to station staff
- **Medical Emergencies**: Contact station staff immediately
- **Security**: Report suspicious activity to staff

## Seasonal Considerations

### Spring (March-May)
- **Tourist Season**: Busy with visitors, book early
- **Weather**: Mild, occasional rain
- **Crowds**: Peak season, expect full vehicles

### Summer (June-August)
- **Vacation Season**: Increased tourist traffic
- **Heat**: Air conditioning on all vehicles
- **Festivals**: Increased travel to festival destinations

### Autumn (September-November)
- **Fall Season**: Beautiful views
- **Weather**: Clear, comfortable temperatures
- **Crowds**: Moderate, pleasant travel conditions

### Winter (December-February)
- **Holiday Season**: Busy with holiday travelers
- **Weather**: Heating on all vehicles
- **Delays**: Possible due to weather conditions

## Contact Information

### Customer Service
- **Phone Support**: Available in multiple languages
- **Online Help**: Website with comprehensive information
- **Station Staff**: Assistance at all major stations
- **Tourist Information**: Specialized help for visitors

### Emergency Contacts
- **General Emergency**: Local emergency numbers
- **Transportation Hotline**: Available 24/7
- **Tourist Hotline**: English-speaking assistance
- **Lost and Found**: Centralized service for all items

{transportation_name.replace("_", " ").title()} provides an excellent way to explore the region efficiently and affordably. With proper planning and understanding of the system, it offers a convenient transportation experience for both locals and visitors."""

    return content


def save_transportation_data(data: Dict[str, Any], filename: str, transport_type: str):
    """Save transportation data to JSON file in appropriate transport type subfolder"""
    try:
        # Create base transportation folder
        base_folder = "transportation"
        type_folder = os.path.join(base_folder, transport_type)
        os.makedirs(type_folder, exist_ok=True)
        file_path = os.path.join(type_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


def main():
    """Main function to extract transportation data"""
    if len(sys.argv) < 2:
        print("Usage: python extract_transportation.py <transportation_name>")
        print("Available transportation systems:", list(TRANSPORTATION_CONFIGS.keys()))
        return

    transportation_name = sys.argv[1]
    print(f"Starting {transportation_name} transportation extraction...")

    # Extract transportation information
    transport_info = scrape_transportation(transportation_name)

    if not transport_info:
        print(f"Failed to extract transportation information for {transportation_name}")
        return

    # Create and save data
    config = TRANSPORTATION_CONFIGS[transportation_name.lower()]
    data = create_transportation_data(transport_info, transportation_name, config)
    filename = f"{transportation_name.lower().replace(' ', '_').replace(',', '')}.json"
    save_transportation_data(data, filename, config["transport_type"])

    print(
        f"Extraction complete! Found {len(transport_info)} transportation items for {transportation_name}."
    )


if __name__ == "__main__":
    main()
