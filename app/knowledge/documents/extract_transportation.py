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
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "Cache-Control": "max-age=0"
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
            'shinkansen', 'bullet', 'train', 'jr', 'pass', 'tokyo', 'osaka',
            'kyoto', 'nagoya', 'sendai', 'hiroshima', 'fukuoka', 'nozomi',
            'hikari', 'kodama', 'hayabusa', 'komachi', 'tokaido', 'tohoku',
            'hokuriku', 'joetsu', 'yamagata', 'akita', 'sanyo', 'kyushu'
        ]
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
            'tokyo', 'metro', 'subway', 'jr', 'yamanote', 'chuo', 'sobu',
            'ginza', 'marunouchi', 'hibiya', 'tozai', 'chiyoda', 'hanzomon',
            'namboku', 'yurakucho', 'fukutoshin', 'oedo', 'suica', 'pasmo'
        ]
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
            'london', 'underground', 'tube', 'metro', 'oyster', 'card',
            'central', 'northern', 'piccadilly', 'district', 'circle',
            'metropolitan', 'hammersmith', 'city', 'waterloo', 'city',
            'jubilee', 'victoria', 'bakerloo', 'elizabeth', 'line'
        ]
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
            'paris', 'metro', 'subway', 'ratp', 'navigo', 'pass',
            'line', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', 'rer', 'a', 'b', 'c', 'd', 'e'
        ]
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
            'new', 'york', 'subway', 'metro', 'mta', 'metrocard',
            'line', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'j', 'l', 'm',
            'n', 'q', 'r', 's', 'w', 'z', '1', '2', '3', '4', '5', '6', '7'
        ]
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
            'singapore', 'mrt', 'metro', 'subway', 'ez', 'link', 'card',
            'north', 'south', 'east', 'west', 'circle', 'downtown',
            'thomson', 'east', 'coast', 'line', 'lrt', 'light', 'rail'
        ]
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
            'hong', 'kong', 'mtr', 'metro', 'subway', 'octopus', 'card',
            'island', 'line', 'tseung', 'kwan', 'o', 'tung', 'chung',
            'airport', 'express', 'east', 'rail', 'west', 'rail', 'light'
        ]
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
            'seoul', 'metro', 'subway', 'tmoney', 'card', 'line', '1', '2',
            '3', '4', '5', '6', '7', '8', '9', 'airport', 'express', 'light',
            'rail', 'bundang', 'shinbundang', 'gyeongui', 'jungang'
        ]
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
            'bangkok', 'bts', 'skytrain', 'mrt', 'metro', 'subway',
            'sukhumvit', 'silom', 'gold', 'line', 'blue', 'purple',
            'yellow', 'pink', 'orange', 'red', 'airport', 'link'
        ]
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
            'shanghai', 'metro', 'subway', 'line', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10', '11', '12', '13', '16', '17', '18',
            'maglev', 'airport', 'express', 'pujiang', 'line'
        ]
    }
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
        logger.error(f"Transportation '{transportation_name}' not configured. Available: {list(TRANSPORTATION_CONFIGS.keys())}")
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
            if info['name'] not in seen_names:
                unique_info.append(info)
                seen_names.add(info['name'])
        
        logger.info(f"Extracted {len(unique_info)} transportation items from {config['source']}")
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
        'Get around', 'Transportation', 'Public transport', 'Metro', 'Subway',
        'Trains', 'Buses', 'Tickets', 'Fares', 'Lines', 'Routes'
    ]
    
    for section in transport_sections:
        # Find section headers
        headers = soup.find_all(['h2', 'h3', 'h4'])
        for header in headers:
            if header.get_text(strip=True).startswith(section):
                # Look for lists and links in this section
                next_elements = header.find_next_siblings(['ul', 'ol', 'p'], limit=10)
                for element in next_elements:
                    if element.name in ['ul', 'ol']:
                        # Extract from lists
                        list_items = element.find_all('li')
                        for item in list_items:
                            links = item.find_all('a')
                            for link in links:
                                name = link.get_text(strip=True)
                                href = link.get('href')
                                if name and href and len(name) > 3:
                                    # Check if it's a specific transportation item
                                    if is_specific_transportation(name, config):
                                        full_url = urljoin(config["source_url"], href)
                                        transport_info.append({
                                            "name": name,
                                            "url": full_url
                                        })
                    elif element.name == 'p':
                        # Extract from paragraphs
                        links = element.find_all('a')
                        for link in links:
                            name = link.get_text(strip=True)
                            href = link.get('href')
                            if name and href and len(name) > 3:
                                if is_specific_transportation(name, config):
                                    full_url = urljoin(config["source_url"], href)
                                    transport_info.append({
                                        "name": name,
                                        "url": full_url
                                    })
    
    # Also look for specific transportation keywords in the page
    all_links = soup.find_all('a')
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get('href')
            
            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a specific transportation item
                if is_specific_transportation(name, config):
                    # Construct full URL
                    if href.startswith('/'):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], '/' + href)
                    
                    # Avoid duplicates and navigation links
                    if name not in [info['name'] for info in transport_info]:
                        transport_info.append({
                            "name": name,
                            "url": full_url
                        })
                        
        except (AttributeError, TypeError) as e:
            continue
    
    return transport_info


def is_specific_transportation(name, config):
    """Check if a name represents a specific transportation item rather than a category"""
    name_lower = name.lower()
    
    # Skip generic categories and navigation
    generic_terms = [
        'home', 'back', 'next', 'previous', 'top', 'menu', 'search', 
        'contact', 'about', 'privacy', 'terms', 'edit', 'history',
        'transportation', 'transport', 'transit', 'systems', 'networks',
        'services', 'routes', 'lines', 'stations', 'stops'
    ]
    
    if any(term in name_lower for term in generic_terms):
        return False
    
    # Check if it contains specific transportation keywords
    transport_keywords = [
        'metro', 'subway', 'train', 'bus', 'tram', 'light', 'rail',
        'line', 'route', 'station', 'stop', 'terminal', 'airport',
        'shinkansen', 'underground', 'tube', 'mrt', 'bts', 'mtr'
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
    all_links = soup.find_all('a')
    
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get('href')
            
            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a transportation link
                href_lower = href.lower()
                name_lower = name.lower()
                
                # Filter for likely transportation links based on destination
                is_transportation = (
                    any(keyword in href_lower for keyword in config["keywords"]) or
                    any(keyword in name_lower for keyword in ['metro', 'subway', 'train', 'bus', 'tram', 'light', 'rail', 'line', 'route', 'station', 'stop']) or
                    any(keyword in name_lower for keyword in ['metro', 'subway', 'train', 'bus', 'tram', 'light', 'rail', 'line', 'route', 'station', 'stop'])
                )
                
                # Avoid navigation and non-transportation links
                is_navigation = any(skip in name_lower for skip in ['home', 'back', 'next', 'previous', 'top', 'menu', 'search', 'contact', 'about', 'privacy', 'terms'])
                
                if is_transportation and not is_navigation:
                    # Construct full URL
                    if href.startswith('/'):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], '/' + href)
                    
                    # Avoid duplicates
                    if name not in [info['name'] for info in transport_info]:
                        transport_info.append({
                            "name": name,
                            "url": full_url
                        })
                        
        except (AttributeError, TypeError) as e:
            continue
    
    return transport_info


def create_transportation_data(transport_info: List[Dict[str, str]], 
                              transportation_name: str,
                              config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized transportation data structure"""
    
    # Create transportation-specific content
    if config["transport_type"] == "trains" and config["country"] == "japan":
        content = create_japan_train_content(transport_info, transportation_name)
    elif config["transport_type"] == "metro":
        content = create_metro_content(transport_info, transportation_name, config)
    else:
        content = create_generic_transportation_content(transport_info, transportation_name)
    
    return {
        "id": f"{transportation_name.lower().replace(' ', '_').replace(',', '')}_complete_guide",
        "title": f"{transportation_name.replace('_', ' ').title()} Complete Travel Guide",
        "content": content,
        "category": "transportation",
        "subcategory": config["transport_type"],
        "location": config["country"],
        "tags": [
            transportation_name.replace('_', ' ').lower(),
            "Transportation",
            config["transport_type"].title(),
            config["country"].title()
        ],
        "language": "en",
        "source": {
            "id": config["source_id"],
            "name": config["source"],
            "url": config["source_url"],
            "reliability_score": 0.9
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def create_japan_train_content(transport_info: List[Dict[str, str]], transportation_name: str) -> str:
    """Create content specifically for Japanese train systems"""
    content = f"""{transportation_name.replace('_', ' ').title()} is Japan's world-renowned transportation system, known for its efficiency, punctuality, and extensive network. This comprehensive guide covers everything you need to know about using {transportation_name.replace('_', ' ').title()}.\n\n## Overview\n{transportation_name.replace('_', ' ').title()} offers visitors a reliable and efficient way to travel throughout Japan. With multiple lines serving different regions, it's the fastest and most convenient way to travel between major cities and explore the country.\n\n## Major Lines and Routes\n\n### Key Transportation Lines\n"""
    
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

{transportation_name.replace('_', ' ').title()} represents the pinnacle of public transportation efficiency and comfort. With proper planning and understanding of the system, it provides an excellent way to explore Japan's cities and regions."""
    
    return content


def create_metro_content(transport_info: List[Dict[str, str]], transportation_name: str, config: Dict[str, Any]) -> str:
    """Create content specifically for metro/subway systems"""
    country_name = config["country"].replace('_', ' ').title()
    transport_type = config["transport_type"].title()
    
    content = f"""{transportation_name.replace('_', ' ').title()} is {country_name}'s efficient {transport_type.lower()} system, providing fast and reliable transportation throughout the city. This comprehensive guide covers everything you need to know about using {transportation_name.replace('_', ' ').title()}.\n\n## Overview\n{transportation_name.replace('_', ' ').title()} offers visitors a convenient and affordable way to navigate the city. With extensive coverage and frequent service, it's the fastest way to reach major attractions and neighborhoods.\n\n## Major Lines and Routes\n\n### Key {transport_type} Lines\n"""
    
    # Add transportation information
    for i, info in enumerate(transport_info[:15], 1):  # Top 15 items
        content += f"- **{info['name']}**: Important {transport_type.lower()} line or route\n"
    
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
- **Boarding Time**: Trains arrive frequently
- **Car Selection**: Choose less crowded cars

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
- **Wheelchair Access**: Most stations and trains
- **Priority Seating**: Available in all cars
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

{transportation_name.replace('_', ' ').title()} provides an excellent way to explore the city efficiently and affordably. With proper planning and understanding of the system, it offers a convenient transportation experience for both locals and visitors."""
    
    return content


def create_generic_transportation_content(transport_info: List[Dict[str, str]], transportation_name: str) -> str:
    """Create generic transportation content"""
    content = f"""{transportation_name.replace('_', ' ').title()} is a comprehensive transportation system that provides efficient and reliable travel options. This guide covers everything you need to know about using {transportation_name.replace('_', ' ').title()}.\n\n## Overview\n{transportation_name.replace('_', ' ').title()} offers visitors a convenient way to travel throughout the region. With multiple lines and extensive coverage, it's an excellent choice for both local and tourist transportation needs.\n\n## Major Lines and Services\n\n### Key Transportation Routes\n"""
    
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

{transportation_name.replace('_', ' ').title()} provides an excellent way to explore the region efficiently and affordably. With proper planning and understanding of the system, it offers a convenient transportation experience for both locals and visitors."""
    
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
    filename = f"{transportation_name.lower().replace(' ', '_').replace(',', '')}_attractions.json"
    save_transportation_data(data, filename, config["transport_type"])
    
    print(f"Extraction complete! Found {len(transport_info)} transportation items for {transportation_name}.")


if __name__ == "__main__":
    main() 