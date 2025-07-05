#!/usr/bin/env python3
"""
Generic Destination Data Extractor
Uses robust web scraping utilities to extract destination information
Supports multiple countries and regions including Japan and Europe
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

# Destination configurations
DESTINATION_CONFIGS = {
    "kyoto": {
        "url": "https://www.japan-guide.com/e/e2158.html",
        "source": "Japan Guide",
        "source_url": "https://www.japan-guide.com",
        "source_id": "japan_guide_official",
        "region": "asia",
        "keywords": [
            'kinkakuji', 'fushimi', 'kiyomizu', 'arashiyama', 'ginkakuji',
            'nijo', 'imperial', 'ryoanji', 'tenryuji', 'nishiki', 'gion',
            'heian', 'philosopher', 'kodaiji', 'yasaka', 'maruyama',
            'nanzenji', 'kenninji', 'shorenin', 'chionin', 'eikando',
            'shugakuin', 'ninnaji', 'kibune', 'kurama', 'ohara', 'sanzenin',
            'takao', 'hieizan', 'daitokuji', 'kitano', 'kamo', 'enkoji',
            'manshuin', 'kokedera', 'katsura', 'yoshiminedera', 'daikakuji',
            'myoshinji', 'toei', 'hozugawa', 'sagano', 'yamazaki'
        ]
    },
    "paris": {
        "url": "https://en.wikivoyage.org/wiki/Paris",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'eiffel', 'louvre', 'notre-dame', 'arc', 'champs', 'montmartre',
            'versailles', 'seine', 'marais', 'latin', 'sacre', 'pompidou',
            'orsay', 'tuileries', 'luxembourg', 'pantheon', 'sainte-chapelle',
            'musee', 'palace', 'cathedral', 'museum', 'garden', 'park'
        ]
    },
    "rome": {
        "url": "https://en.wikivoyage.org/wiki/Rome",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'colosseum', 'vatican', 'pantheon', 'forum', 'trevi', 'piazza',
            'sistine', 'st-peter', 'castel', 'baths', 'catacombs', 'palatine',
            'capitoline', 'trajan', 'circus', 'appian', 'basilica', 'church'
        ]
    },
    "barcelona": {
        "url": "https://en.wikivoyage.org/wiki/Barcelona",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'sagrada', 'park', 'guell', 'ramblas', 'gothic', 'barceloneta',
            'montjuic', 'tibidabo', 'casa', 'mila', 'batllo', 'pedrera',
            'camp', 'nou', 'olympic', 'port', 'vell', 'cathedral', 'museum'
        ]
    },
    "amsterdam": {
        "url": "https://en.wikivoyage.org/wiki/Amsterdam",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'rijksmuseum', 'van', 'gogh', 'anne', 'frank', 'dam', 'square',
            'canals', 'jordaan', 'vondelpark', 'red', 'light', 'district',
            'museums', 'plein', 'begijnhof', 'westerkerk', 'nemo', 'palace'
        ]
    },
    "london": {
        "url": "https://en.wikivoyage.org/wiki/London",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'buckingham', 'palace', 'tower', 'bridge', 'westminster', 'abbey',
            'big', 'ben', 'london', 'eye', 'trafalgar', 'square', 'hyde', 'park',
            'british', 'museum', 'natural', 'history', 'science', 'museum',
            'tate', 'modern', 'national', 'gallery', 'covent', 'garden'
        ]
    },
    "berlin": {
        "url": "https://en.wikivoyage.org/wiki/Berlin",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'brandenburg', 'gate', 'reichstag', 'berlin', 'wall', 'memorial',
            'museum', 'island', 'checkpoint', 'charlie', 'alexanderplatz',
            'unter', 'den', 'linden', 'tiergarten', 'victory', 'column',
            'holocaust', 'memorial', 'east', 'side', 'gallery'
        ]
    },
    "prague": {
        "url": "https://en.wikivoyage.org/wiki/Prague",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'charles', 'bridge', 'prague', 'castle', 'old', 'town', 'square',
            'astronomical', 'clock', 'wenceslas', 'square', 'jewish', 'quarter',
            'st', 'vitus', 'cathedral', 'petrin', 'tower', 'national', 'museum'
        ]
    },
    "vienna": {
        "url": "https://en.wikivoyage.org/wiki/Vienna",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'schonbrunn', 'palace', 'st', 'stephens', 'cathedral', 'belvedere',
            'hofburg', 'palace', 'prater', 'vienna', 'state', 'opera',
            'kunsthistorisches', 'museum', 'natural', 'history', 'museum',
            'albertina', 'museum', 'stephensplatz', 'ring', 'strasse'
        ]
    },
    "budapest": {
        "url": "https://en.wikivoyage.org/wiki/Budapest",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "keywords": [
            'parliament', 'building', 'budapest', 'castle', 'chain', 'bridge',
            'fishermans', 'bastion', 'st', 'stephens', 'basilica', 'heroes',
            'square', 'szechenyi', 'baths', 'gellert', 'baths', 'vajdahunyad',
            'castle', 'andrasy', 'avenue', 'opera', 'house'
        ]
    },
    "beijing": {
        "url": "https://en.wikivoyage.org/wiki/Beijing",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'forbidden', 'city', 'great', 'wall', 'temple', 'heaven', 'summer',
            'palace', 'tiananmen', 'square', 'beijing', 'hutong', 'olympic',
            'park', 'bird', 'nest', 'water', 'cube', 'ming', 'tombs',
            'lama', 'temple', 'confucius', 'temple', 'bell', 'drum', 'tower',
            'beihai', 'park', 'jingshan', 'park', 'yuanmingyuan', 'ruins'
        ]
    },
    "shanghai": {
        "url": "https://en.wikivoyage.org/wiki/Shanghai",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'bund', 'shanghai', 'tower', 'pearl', 'oriental', 'pearl',
            'nanjing', 'road', 'peoples', 'square', 'yuyuan', 'garden',
            'temple', 'city', 'god', 'xintiandi', 'tianzifang', 'french',
            'concession', 'jade', 'buddha', 'temple', 'longhua', 'pagoda',
            'shanghai', 'museum', 'science', 'technology', 'museum'
        ]
    },
    "seoul": {
        "url": "https://en.wikivoyage.org/wiki/Seoul",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'gyeongbokgung', 'palace', 'changdeokgung', 'deoksugung',
            'namsan', 'tower', 'seoul', 'myeongdong', 'dongdaemun',
            'insadong', 'hongdae', 'gangnam', 'bukchon', 'hanok',
            'namsangol', 'village', 'seoul', 'forest', 'park', 'lotte',
            'world', 'tower', 'coex', 'mall', 'aquarium', 'n', 'seoul'
        ]
    },
    "singapore": {
        "url": "https://en.wikivoyage.org/wiki/Singapore",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'marina', 'bay', 'sands', 'gardens', 'cloud', 'forest',
            'flower', 'dome', 'sentosa', 'island', 'universal', 'studios',
            'singapore', 'zoo', 'night', 'safari', 'river', 'safari',
            'merlion', 'park', 'chinatown', 'little', 'india', 'arab',
            'street', 'orchard', 'road', 'clarke', 'quay', 'boat', 'quay'
        ]
    },
    "bangkok": {
        "url": "https://en.wikivoyage.org/wiki/Bangkok",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'grand', 'palace', 'wat', 'phra', 'kaew', 'emerald', 'buddha',
            'wat', 'arun', 'temple', 'dawn', 'wat', 'pho', 'reclining',
            'buddha', 'chinatown', 'yaowarat', 'khaosan', 'road', 'siam',
            'paragon', 'central', 'world', 'chatuchak', 'weekend', 'market',
            'lumphini', 'park', 'vimanmek', 'mansion', 'ananta', 'samakhom'
        ]
    },
    "hong_kong": {
        "url": "https://en.wikivoyage.org/wiki/Hong_Kong",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "keywords": [
            'victoria', 'peak', 'hong', 'kong', 'disneyland', 'ocean',
            'park', 'tian', 'tan', 'buddha', 'big', 'buddha', 'lantau',
            'island', 'ngong', 'ping', 'village', 'central', 'mid',
            'levels', 'escalator', 'temple', 'street', 'night', 'market',
            'stanley', 'market', 'repeal', 'bay', 'avenue', 'stars'
        ]
    }
}


def get_session():
    """Create a session with proper headers and cookies"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def scrape_destination(destination_name: str):
    """Scrape destination attractions from various sources"""
    session = get_session()
    
    if destination_name.lower() not in DESTINATION_CONFIGS:
        logger.error(f"Destination '{destination_name}' not configured. Available: {list(DESTINATION_CONFIGS.keys())}")
        return []
    
    config = DESTINATION_CONFIGS[destination_name.lower()]
    url = config["url"]
    
    try:
        # Add a random delay to avoid being flagged as a bot
        time.sleep(random.uniform(1, 3))
        
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        attractions = []
        
        # For WikiVoyage pages, look for specific attraction sections
        if "wikivoyage" in config["source_url"].lower():
            attractions = scrape_wikivoyage_attractions(soup, config)
        else:
            # Fallback to general link extraction
            attractions = scrape_general_links(soup, config)
        
        # Remove duplicates based on name
        unique_attractions = []
        seen_names = set()
        for attraction in attractions:
            if attraction['name'] not in seen_names:
                unique_attractions.append(attraction)
                seen_names.add(attraction['name'])
        
        logger.info(f"Extracted {len(unique_attractions)} attractions from {config['source']}")
        return unique_attractions
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


def scrape_wikivoyage_attractions(soup, config):
    """Extract specific attractions from WikiVoyage pages"""
    attractions = []
    
    # Look for attraction sections in WikiVoyage
    attraction_sections = [
        'See', 'Do', 'Buy', 'Eat', 'Drink', 'Sleep', 'Connect', 'Go next'
    ]
    
    for section in attraction_sections:
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
                                    # Check if it's a specific attraction
                                    if is_specific_attraction(name, config):
                                        full_url = urljoin(config["source_url"], href)
                                        attractions.append({
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
                                if is_specific_attraction(name, config):
                                    full_url = urljoin(config["source_url"], href)
                                    attractions.append({
                                        "name": name,
                                        "url": full_url
                                    })
    
    # Also look for specific attraction keywords in the page
    all_links = soup.find_all('a')
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get('href')
            
            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a specific attraction
                if is_specific_attraction(name, config):
                    # Construct full URL
                    if href.startswith('/'):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], '/' + href)
                    
                    # Avoid duplicates and navigation links
                    if name not in [att['name'] for att in attractions]:
                        attractions.append({
                            "name": name,
                            "url": full_url
                        })
                        
        except (AttributeError, TypeError) as e:
            continue
    
    return attractions


def is_specific_attraction(name, config):
    """Check if a name represents a specific attraction rather than a category"""
    name_lower = name.lower()
    
    # Skip generic categories and navigation
    generic_terms = [
        'home', 'back', 'next', 'previous', 'top', 'menu', 'search', 
        'contact', 'about', 'privacy', 'terms', 'edit', 'history',
        'monuments', 'museums', 'squares', 'churches', 'basilicas',
        'attractions', 'sights', 'landmarks', 'places', 'areas',
        'districts', 'neighborhoods', 'quarters', 'zones'
    ]
    
    if any(term in name_lower for term in generic_terms):
        return False
    
    # Check if it contains specific attraction keywords
    attraction_keywords = [
        'museum', 'palace', 'castle', 'temple', 'shrine', 'park', 
        'square', 'church', 'cathedral', 'monument', 'tower', 
        'bridge', 'market', 'basilica', 'forum', 'colosseum',
        'vatican', 'pantheon', 'trevi', 'fountain', 'gallery',
        'opera', 'theater', 'theatre', 'garden', 'palace'
    ]
    
    # Check if it matches destination-specific keywords
    if any(keyword in name_lower for keyword in config["keywords"]):
        return True
    
    # Check if it contains attraction keywords
    if any(keyword in name_lower for keyword in attraction_keywords):
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
    attractions = []
    all_links = soup.find_all('a')
    
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get('href')
            
            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like an attraction link
                href_lower = href.lower()
                name_lower = name.lower()
                
                # Filter for likely attraction links based on destination
                is_attraction = (
                    any(keyword in href_lower for keyword in config["keywords"]) or
                    any(keyword in name_lower for keyword in ['museum', 'palace', 'castle', 'temple', 'shrine', 'park', 'square', 'church', 'cathedral', 'monument', 'tower', 'bridge', 'market']) or
                    any(keyword in name_lower for keyword in ['museum', 'palace', 'castle', 'temple', 'shrine', 'park', 'square', 'church', 'cathedral', 'monument', 'tower', 'bridge', 'market'])
                )
                
                # Avoid navigation and non-attraction links
                is_navigation = any(skip in name_lower for skip in ['home', 'back', 'next', 'previous', 'top', 'menu', 'search', 'contact', 'about', 'privacy', 'terms'])
                
                if is_attraction and not is_navigation:
                    # Construct full URL
                    if href.startswith('/'):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], '/' + href)
                    
                    # Avoid duplicates
                    if name not in [att['name'] for att in attractions]:
                        attractions.append({
                            "name": name,
                            "url": full_url
                        })
                        
        except (AttributeError, TypeError) as e:
            continue
    
    return attractions


def create_destination_data(attractions: List[Dict[str, str]], 
                          destination_name: str,
                          config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized destination data structure"""
    
    # Create destination-specific content
    if destination_name.lower() == "kyoto":
        content = create_kyoto_content(attractions, destination_name)
    elif destination_name.lower() in ["paris", "rome", "barcelona", "amsterdam"]:
        content = create_european_content(attractions, destination_name)
    else:
        content = create_generic_content(attractions, destination_name)
    
    return {
        "id": f"{destination_name.lower().replace(' ', '_').replace(',', '')}_complete_guide",
        "title": f"{destination_name} Complete Travel Guide",
        "content": content,
        "category": "destinations",
        "subcategory": config["region"],
        "location": destination_name,
        "tags": [destination_name.split(',')[0].strip(), "Travel Guide"],
        "language": "en",
        "source": {
            "id": config["source_id"],
            "name": config["source"],
            "url": config["source_url"],
            "reliability_score": 0.9
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def create_kyoto_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Kyoto-specific content"""
    content = f"""{destination_name} is Japan's cultural heart and former imperial capital, home to over 1,600 Buddhist temples, 400 Shinto shrines, palaces, and gardens. This ancient city offers visitors a glimpse into traditional Japan with its well-preserved historic districts and cultural treasures.

## Overview
{destination_name} served as Japan's capital for over 1,000 years (794-1868) and remains the country's cultural and spiritual center. With its stunning temples, traditional gardens, and preserved historic districts, Kyoto offers an authentic Japanese experience unlike any other city.

## Top Attractions

### Temples & Shrines
"""
    
    # Add temple and shrine information
    temples = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['temple', 'shrine', 'ji', 'dera', 'jinja'])]
    for i, temple in enumerate(temples[:10], 1):  # Top 10 temples
        content += f"- **{temple['name']}**: One of Kyoto's most important religious sites\n"
    
    content += """
### Historic Districts
- **Gion**: Famous geisha district with traditional tea houses
- **Arashiyama**: Scenic area with bamboo groves and temples
- **Higashiyama**: Preserved historic district around Kiyomizudera
- **Pontocho**: Narrow alley with atmospheric dining options

### Castles & Palaces
"""
    
    # Add castle and palace information
    castles = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['castle', 'palace', 'imperial'])]
    for castle in castles:
        content += f"- **{castle['name']}**: Historic royal residence and architectural marvel\n"
    
    content += """
## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-20°C)
- **Highlights**: Cherry blossom season (sakura) at Maruyama Park and Philosopher's Path
- **Crowds**: Very busy, especially during Golden Week
- **Recommendation**: Book accommodations early

### Summer (June - August)
- **Weather**: Hot and humid (20-35°C)
- **Highlights**: Gion Matsuri festival in July
- **Considerations**: Rainy season in June-July

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-25°C)
- **Highlights**: Spectacular fall foliage at temples and gardens
- **Advantages**: Clear skies, beautiful colors

### Winter (December - February)
- **Weather**: Cold but dry (0-10°C)
- **Advantages**: Fewer crowds, peaceful temple visits
- **Special**: New Year celebrations and traditional events

## Transportation

### Getting Around Kyoto
- **Bus System**: Extensive network covering all major attractions
- **Subway**: Two lines connecting key areas
- **Bicycle Rental**: Popular and convenient way to explore
- **Walking**: Many attractions are within walking distance in historic districts

### From Other Cities
- **From Tokyo**: 2.5 hours by Shinkansen
- **From Osaka**: 30 minutes by JR train
- **From Nara**: 45 minutes by JR train

## Cultural Etiquette

### Temple & Shrine Visits
- **Purification**: Wash hands and rinse mouth at entrance
- **Prayer**: Bow twice, clap twice, bow once at shrines
- **Photography**: Check for restrictions, especially indoors
- **Clothing**: Dress modestly, remove hats

### Traditional Experiences
- **Tea Ceremony**: Experience traditional Japanese tea culture
- **Zen Meditation**: Available at several temples
- **Kimono Rental**: Walk through historic districts in traditional dress
- **Geisha Culture**: Respectful observation in Gion district

## Food & Dining

### Must-Try Kyoto Foods
- **Kaiseki**: Traditional multi-course dining
- **Yudofu**: Hot pot with tofu and vegetables
- **Kyoto-style Sushi**: Different from Tokyo-style
- **Matcha**: Green tea and related sweets
- **Kyoto Pickles**: Traditional preserved vegetables

### Dining Districts
- **Pontocho**: Traditional restaurants and tea houses
- **Nishiki Market**: Food market with local specialties
- **Gion**: High-end traditional dining
- **Kyoto Station**: Modern restaurants and food courts

## Shopping

### Traditional Crafts
- **Kiyomizu Pottery**: Traditional ceramics
- **Nishijin Textiles**: Traditional silk weaving
- **Kyoto Dolls**: Traditional Japanese dolls
- **Fans and Umbrellas**: Handcrafted accessories

### Shopping Areas
- **Teramachi & Shinkyogoku**: Covered shopping arcades
- **Nishiki Market**: Food and traditional items
- **Kyoto Station**: Modern shopping complex
- **Gion**: High-end traditional goods

## Accommodation

### Traditional Options
- **Ryokan**: Traditional Japanese inns with tatami rooms
- **Machiya**: Traditional townhouses converted to guesthouses
- **Temple Lodging**: Stay at Buddhist temples (shukubo)

### Modern Options
- **Hotels**: Range from budget to luxury
- **Hostels**: Budget-friendly options
- **Apartments**: Self-catering options

## Seasonal Highlights

### Spring
- Cherry blossoms at Maruyama Park, Philosopher's Path, and temples
- Aoi Matsuri festival in May

### Summer
- Gion Matsuri festival in July
- Evening illuminations at temples

### Autumn
- Fall foliage at temples and gardens
- Jidai Matsuri festival in October

### Winter
- New Year celebrations
- Snow-covered temples (rare but beautiful)

## Practical Information

### Important Numbers
- **Emergency (Fire/Ambulance)**: 119
- **Police**: 110
- **Tourist Information**: 075-343-6655

### Useful Apps
- **Google Translate**: For signs and menus
- **Hyperdia**: Train route planning
- **Kyoto City Bus**: Bus route information

### Money & Payments
- **Cash**: Still widely used, especially at smaller establishments
- **Credit Cards**: Accepted at hotels and larger restaurants
- **IC Cards**: Suica/Pasmo for public transportation

## Weather by Season
- **Spring**: 10-20°C, cherry blossoms, occasional rain
- **Summer**: 20-35°C, high humidity, rainy season in June-July
- **Autumn**: 10-25°C, clear skies, beautiful fall colors
- **Winter**: 0-10°C, dry, occasional snow, peaceful temple visits"""
    
    return content


def create_european_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create European destination content"""
    content = f"""{destination_name} is a vibrant European destination with rich history, culture, and modern attractions. This city offers visitors a perfect blend of historical landmarks, cultural experiences, and contemporary amenities.

## Overview
{destination_name} is a must-visit destination in Europe, known for its iconic landmarks, rich cultural heritage, and vibrant atmosphere. From historic monuments to modern attractions, this city provides an unforgettable travel experience.

## Top Attractions

### Historic Landmarks
"""
    
    # Add landmark information
    landmarks = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['museum', 'palace', 'castle', 'church', 'cathedral', 'monument', 'tower', 'square', 'basilica', 'forum', 'colosseum', 'vatican', 'pantheon', 'trevi', 'fountain', 'gallery', 'opera', 'theater', 'theatre', 'garden'])]
    for i, landmark in enumerate(landmarks[:20], 1):  # Top 20 landmarks
        content += f"- **{landmark['name']}**: Iconic landmark and cultural treasure\n"
    
    content += f"""
### Cultural Districts
- **Historic Center**: Explore the heart of {destination_name}
- **Cultural Quarter**: Museums, galleries, and performance venues
- **Shopping Districts**: Modern retail and traditional markets
- **Entertainment Areas**: Restaurants, cafes, and nightlife

## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-20°C)
- **Highlights**: Blooming gardens and outdoor festivals
- **Crowds**: Moderate, good time to visit
- **Advantages**: Pleasant weather, fewer crowds than summer

### Summer (June - August)
- **Weather**: Warm temperatures (20-30°C)
- **Highlights**: Outdoor events and festivals
- **Crowds**: Peak tourist season
- **Considerations**: Book accommodations early

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-20°C)
- **Highlights**: Fall colors and cultural events
- **Advantages**: Fewer crowds, pleasant weather

### Winter (December - February)
- **Weather**: Cold temperatures (0-10°C)
- **Highlights**: Holiday markets and indoor attractions
- **Advantages**: Fewer crowds, lower prices

## Transportation

### Getting Around
- **Public Transport**: Efficient metro, bus, and tram systems
- **Walking**: Many attractions are within walking distance
- **Bicycle**: Bike rental available throughout the city
- **Taxi**: Readily available but can be expensive

### From Other Cities
- **International Airport**: Well-connected to major cities
- **Train**: High-speed rail connections to neighboring countries
- **Bus**: Affordable long-distance connections

## Cultural Etiquette

### General Behavior
- **Greetings**: Handshakes are common, learn local greetings
- **Dress**: Smart casual is appropriate for most places
- **Photography**: Ask permission before taking photos of people
- **Tipping**: Check local customs for tipping practices

### Visiting Religious Sites
- **Dress Code**: Modest clothing required
- **Behavior**: Quiet and respectful behavior expected
- **Photography**: Check for restrictions

## Food & Dining

### Local Cuisine
- **Traditional Dishes**: Sample local specialties
- **Street Food**: Affordable and delicious options
- **Fine Dining**: High-end restaurants available
- **Cafes**: Perfect for coffee and light meals

### Dining Districts
- **Historic Quarter**: Traditional restaurants
- **Modern Areas**: International cuisine
- **Waterfront**: Seafood and scenic dining
- **University District**: Budget-friendly options

## Shopping

### Traditional Items
- **Local Crafts**: Handmade souvenirs
- **Textiles**: Traditional fabrics and clothing
- **Art**: Local artwork and prints
- **Food**: Regional specialties and treats

### Shopping Areas
- **Historic Markets**: Traditional shopping experience
- **Modern Malls**: International brands
- **Boutique Districts**: Unique and local shops
- **Artisan Quarters**: Handcrafted goods

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury
- **Hostels**: Budget-friendly accommodation
- **Apartments**: Self-catering options
- **Boutique Hotels**: Unique and charming stays

## Seasonal Highlights

### Spring
- Flower festivals and outdoor events
- Cultural celebrations

### Summer
- Music festivals and outdoor performances
- Long daylight hours for sightseeing

### Autumn
- Cultural festivals and events
- Beautiful fall colors

### Winter
- Holiday markets and celebrations
- Indoor cultural activities

## Practical Information

### Important Numbers
- **Emergency**: 112 (EU emergency number)
- **Police**: Local emergency number
- **Tourist Information**: Available at visitor centers

### Useful Apps
- **Google Translate**: For language assistance
- **Local Transport**: Public transport apps
- **Maps**: Offline maps recommended

### Money & Payments
- **Currency**: Euro (€) in most European countries
- **Credit Cards**: Widely accepted
- **Cash**: Still useful for small purchases

## Weather by Season
- **Spring**: 10-20°C, occasional rain, pleasant weather
- **Summer**: 20-30°C, warm and sunny, peak season
- **Autumn**: 10-20°C, comfortable temperatures, fewer crowds
- **Winter**: 0-10°C, cold but manageable, holiday atmosphere"""
    
    return content


def create_generic_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create generic destination content"""
    content = f"""{destination_name} is a fascinating destination with unique attractions and cultural experiences waiting to be discovered.

## Overview
{destination_name} offers visitors a diverse range of experiences, from historic landmarks to modern attractions. This destination provides an excellent opportunity to explore local culture, history, and contemporary life.

## Top Attractions

### Key Landmarks
"""
    
    # Add attraction information
    for i, attraction in enumerate(attractions[:10], 1):  # Top 10 attractions
        content += f"- **{attraction['name']}**: Notable attraction worth visiting\n"
    
    content += f"""
## Best Time to Visit

### Spring (March - May)
- **Weather**: Generally mild and pleasant
- **Highlights**: Blooming nature and outdoor activities
- **Advantages**: Good weather, moderate crowds

### Summer (June - August)
- **Weather**: Warm to hot temperatures
- **Highlights**: Peak tourist season with many events
- **Considerations**: Book accommodations early

### Autumn (September - November)
- **Weather**: Comfortable temperatures
- **Highlights**: Cultural events and festivals
- **Advantages**: Fewer crowds, pleasant weather

### Winter (December - February)
- **Weather**: Cooler temperatures
- **Highlights**: Indoor attractions and seasonal events
- **Advantages**: Lower prices, fewer tourists

## Transportation

### Getting Around
- **Public Transport**: Efficient local transportation systems
- **Walking**: Many attractions are accessible on foot
- **Taxis**: Available but check local rates
- **Rental Cars**: Available for exploring surrounding areas

## Cultural Etiquette

### General Guidelines
- **Respect**: Show respect for local customs and traditions
- **Language**: Learn basic local phrases
- **Photography**: Ask permission when appropriate
- **Dress**: Follow local dress codes for religious sites

## Food & Dining

### Local Cuisine
- **Traditional Dishes**: Sample local specialties
- **Street Food**: Affordable and authentic options
- **Restaurants**: Range from casual to fine dining
- **Markets**: Fresh local produce and ingredients

## Practical Information

### Important Numbers
- **Emergency**: Local emergency number
- **Tourist Information**: Available at visitor centers
- **Police**: Local police contact

### Useful Tips
- **Language**: English may not be widely spoken
- **Currency**: Check local currency and exchange rates
- **Safety**: Follow local safety guidelines
- **Health**: Check vaccination requirements

## Weather Information
- **Spring**: Generally mild with occasional rain
- **Summer**: Warm to hot, peak tourist season
- **Autumn**: Comfortable temperatures, good for sightseeing
- **Winter**: Cooler weather, fewer crowds"""
    
    return content


def save_destination_data(data: Dict[str, Any], filename: str, region: str):
    """Save destination data to JSON file in appropriate region subfolder"""
    try:
        # Create base destination folder
        base_folder = "destinations"
        region_folder = os.path.join(base_folder, region)
        os.makedirs(region_folder, exist_ok=True)
        file_path = os.path.join(region_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


def main():
    """Main function to extract destination data"""
    if len(sys.argv) < 2:
        print("Usage: python extract_destination.py <destination_name>")
        print("Available destinations:", list(DESTINATION_CONFIGS.keys()))
        return
    
    destination_name = sys.argv[1]
    print(f"Starting {destination_name} destination extraction...")
    
    # Extract attractions
    attractions = scrape_destination(destination_name)
    
    if not attractions:
        print(f"Failed to extract attractions for {destination_name}")
        return
    
    # Create and save data
    config = DESTINATION_CONFIGS[destination_name.lower()]
    data = create_destination_data(attractions, destination_name, config)
    filename = f"{destination_name.lower().replace(' ', '_').replace(',', '')}_attractions.json"
    save_destination_data(data, filename, config["region"])
    
    print(f"Extraction complete! Found {len(attractions)} attractions for {destination_name}.")


if __name__ == "__main__":
    main() 