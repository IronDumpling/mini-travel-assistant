#!/usr/bin/env python3
"""
Generic Destination Data Extractor
Uses robust web scraping utilities to extract destination information
Supports multiple countries and regions including Japan and Europe
"""

import requests
import time
import random
import re
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
    
    # Look for specific attraction keywords in the page content
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
    
    # If we didn't find enough attractions, try a more aggressive approach
    if len(attractions) < 5:
        # Look for any links that contain destination-specific keywords
        for link in all_links:
            try:
                name = link.get_text(strip=True)
                href = link.get('href')
                
                if href and name and isinstance(href, str) and len(name) > 3:
                    name_lower = name.lower()
                    
                    # Check for destination-specific keywords
                    if any(keyword in name_lower for keyword in config["keywords"]):
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
    
    # Filter out any remaining navigation or non-attraction links
    filtered_attractions = []
    for attraction in attractions:
        name_lower = attraction['name'].lower()
        
        # Skip if it contains navigation terms
        navigation_terms = [
            'what\'s nearby', 'get shortened url', 'bahasa indonesia',
            'tiếng việt', 'jump to content', 'main page', 'travel destinations',
            'random page', 'recent changes', 'community portal', 'maintenance panel',
            'interlingual lounge', 'donate', 'edit', 'history', 'watch', 'read',
            'view source', 'view history', 'what links here', 'related changes',
            'special pages', 'printable version', 'permanent link', 'page information',
            'wikidata item', 'cite this page', 'create a book', 'download as pdf',
            'version for printing', 'in other projects', 'wikimedia commons',
            'wikisource', 'wiktionary', 'wikibooks', 'wikiquote', 'wikinews',
            'wikiversity', 'wikidata', 'mediawiki', 'meta-wiki', 'wikispecies',
            'wikivoyage', 'wikimedia', 'foundation', 'disclaimers', 'contact',
            'mobile view', 'developers', 'statistics', 'cookie statement',
            'terms of use', 'privacy policy', 'code of conduct'
        ]
        
        if not any(term in name_lower for term in navigation_terms):
            filtered_attractions.append(attraction)
    
    return filtered_attractions


def is_specific_attraction(name, config):
    """Check if a name represents a specific attraction rather than a category"""
    name_lower = name.lower()
    
    # Skip generic categories and navigation
    generic_terms = [
        'home', 'back', 'next', 'previous', 'top', 'menu', 'search', 
        'contact', 'about', 'privacy', 'terms', 'edit', 'history',
        'monuments', 'museums', 'squares', 'churches', 'basilicas',
        'attractions', 'sights', 'landmarks', 'places', 'areas',
        'districts', 'neighborhoods', 'quarters', 'zones',
        'what\'s nearby', 'get shortened url', 'bahasa indonesia',
        'tiếng việt', 'rural', 'transportation', 'card', 'theaters',
        'concert halls', 'jump to content', 'main page', 'travel destinations',
        'random page', 'recent changes', 'community portal', 'maintenance panel',
        'interlingual lounge', 'donate', 'edit', 'history', 'watch', 'read',
        'view source', 'view history', 'what links here', 'related changes',
        'special pages', 'printable version', 'permanent link', 'page information',
        'wikidata item', 'cite this page', 'create a book', 'download as pdf',
        'version for printing', 'in other projects', 'wikimedia commons',
        'wikisource', 'wiktionary', 'wikibooks', 'wikiquote', 'wikinews',
        'wikiversity', 'wikidata', 'mediawiki', 'meta-wiki', 'wikispecies',
        'wikivoyage', 'wikimedia', 'foundation', 'disclaimers', 'contact',
        'mobile view', 'developers', 'statistics', 'cookie statement',
        'terms of use', 'privacy policy', 'code of conduct', 'disclaimers',
        'contact wikimedia', 'mobile view', 'developers', 'statistics',
        'cookie statement', 'terms of use', 'privacy policy', 'code of conduct'
    ]
    
    if any(term in name_lower for term in generic_terms):
        return False
    
    # Skip numbered sections and navigation
    if re.match(r'^\d+\.\d+', name) or re.match(r'^\d+\.', name):
        return False
    
    # Skip very short names
    if len(name.strip()) < 4:
        return False
    
    # Check if it contains specific attraction keywords
    attraction_keywords = [
        'museum', 'palace', 'castle', 'temple', 'shrine', 'park', 
        'square', 'church', 'cathedral', 'monument', 'tower', 
        'bridge', 'market', 'basilica', 'forum', 'colosseum',
        'vatican', 'pantheon', 'trevi', 'fountain', 'gallery',
        'opera', 'theater', 'theatre', 'garden', 'palace', 'wall',
        'forbidden', 'summer', 'tiananmen', 'ming', 'lama', 'confucius',
        'bell', 'drum', 'tower', 'beihai', 'jingshan', 'yuanmingyuan',
        'olympic', 'bird', 'nest', 'water', 'cube', 'hutong', 'nanluoguxiang',
        'wangfujing', 'sanlitun', 'houhai', 'yuyuantan', 'panjiayuan'
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
    
    # Normalize destination name for comparison
    normalized_name = destination_name.lower().strip()
    
    logger.info(f"Creating content for destination: '{destination_name}' (normalized: '{normalized_name}')")
    logger.info(f"Found {len(attractions)} attractions")
    
    # Create destination-specific content
    if normalized_name == "kyoto":
        logger.info("Using Kyoto-specific content")
        content = create_kyoto_content(attractions, destination_name)
    elif normalized_name == "beijing":
        logger.info("Using Beijing-specific content")
        content = create_beijing_content(attractions, destination_name)
    elif normalized_name == "seoul":
        logger.info("Using Seoul-specific content")
        content = create_seoul_content(attractions, destination_name)
    elif normalized_name == "shanghai":
        logger.info("Using Shanghai-specific content")
        content = create_shanghai_content(attractions, destination_name)
    elif normalized_name == "paris":
        logger.info("Using Paris-specific content")
        content = create_paris_content(attractions, destination_name)
    elif normalized_name == "london":
        logger.info("Using London-specific content")
        content = create_london_content(attractions, destination_name)
    elif normalized_name == "rome":
        logger.info("Using Rome-specific content")
        content = create_rome_content(attractions, destination_name)
    elif normalized_name == "barcelona":
        logger.info("Using Barcelona-specific content")
        content = create_barcelona_content(attractions, destination_name)
    elif normalized_name == "amsterdam":
        logger.info("Using Amsterdam-specific content")
        content = create_amsterdam_content(attractions, destination_name)
    elif normalized_name == "berlin":
        logger.info("Using Berlin-specific content")
        content = create_berlin_content(attractions, destination_name)
    elif normalized_name == "prague":
        logger.info("Using Prague-specific content")
        content = create_prague_content(attractions, destination_name)
    elif normalized_name == "vienna":
        logger.info("Using Vienna-specific content")
        content = create_vienna_content(attractions, destination_name)
    elif normalized_name == "budapest":
        logger.info("Using Budapest-specific content")
        content = create_budapest_content(attractions, destination_name)
    elif config["region"] == "europe":
        logger.info("Using European content")
        content = create_european_content(attractions, destination_name)
    else:
        logger.info("Using generic content")
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


def create_beijing_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Beijing-specific content"""
    content = f"""{destination_name} is China's historic capital and political center, home to some of the world's most iconic landmarks including the Great Wall, Forbidden City, and Temple of Heaven. This ancient city seamlessly blends imperial grandeur with modern development.

## Overview
{destination_name} has served as China's capital for over 800 years and remains the country's political, cultural, and educational heart. With its rich history spanning dynasties, {destination_name} offers visitors an unparalleled glimpse into China's imperial past and contemporary culture.

## Top Attractions

### Imperial Landmarks
"""
    
    # Add imperial landmark information
    imperial_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['forbidden', 'temple', 'summer', 'palace', 'tiananmen', 'great wall', 'ming', 'lama', 'confucius'])]
    
    if imperial_sites:
        for i, site in enumerate(imperial_sites[:10], 1):
            content += f"- **{site['name']}**: Iconic imperial landmark and UNESCO World Heritage site\n"
    else:
        content += """- **Forbidden City**: Imperial palace complex and UNESCO World Heritage site
- **Great Wall of China**: Ancient defensive wall and world wonder
- **Temple of Heaven**: Imperial complex for ceremonies and prayers
- **Summer Palace**: Imperial garden and palace complex
- **Tiananmen Square**: Historic square and political center
- **Ming Tombs**: Imperial burial complex
- **Lama Temple**: Tibetan Buddhist temple
- **Confucius Temple**: Traditional temple honoring Confucius\n"""
    
    content += """
### Historic Districts
- **Hutongs**: Traditional narrow alleyways and courtyard homes
- **Wangfujing**: Famous shopping street and food district
- **Nanluoguxiang**: Hip hutong area with cafes and boutiques
- **Dongcheng**: Historic district with many traditional sites

### Modern Attractions
"""
    
    # Add modern attraction information
    modern_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['olympic', 'bird', 'water', 'cube', 'park'])]
    
    if modern_sites:
        for site in modern_sites:
            content += f"- **{site['name']}**: Modern architectural marvel and cultural venue\n"
    else:
        content += """- **Beijing Olympic Park**: Venue of the 2008 Summer Olympics
- **Bird's Nest Stadium**: Iconic Olympic stadium
- **Water Cube**: Olympic swimming venue
- **Beijing National Stadium**: Modern sports complex\n"""
    
    content += """
## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-25°C), occasional sandstorms
- **Highlights**: Cherry blossoms at Yuyuantan Park, clear skies
- **Advantages**: Pleasant weather, fewer crowds than summer
- **Considerations**: March can be windy with sandstorms

### Summer (June - August)
- **Weather**: Hot and humid (25-35°C), frequent rain
- **Highlights**: Long daylight hours, outdoor activities
- **Considerations**: Peak tourist season, book accommodations early
- **Air Quality**: Can be affected by pollution

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-25°C), clear skies
- **Highlights**: Golden autumn colors, National Day celebrations
- **Advantages**: Best weather, beautiful scenery, moderate crowds
- **Special**: Golden Week in October (very busy)

### Winter (December - February)
- **Weather**: Cold and dry (-5 to 10°C), occasional snow
- **Highlights**: Snow-covered Great Wall, indoor attractions
- **Advantages**: Fewer crowds, lower prices, unique winter views
- **Considerations**: Very cold, some outdoor sites may be less accessible

## Transportation

### Getting Around Beijing
- **Metro**: Extensive 23-line system covering most attractions
- **Bus**: Comprehensive network, very affordable
- **Taxi**: Readily available, use ride-hailing apps
- **Bicycle**: Bike-sharing available throughout the city

### From Other Cities
- **International Airport**: Capital International Airport (PEK)
- **High-Speed Rail**: Connections to Shanghai, Guangzhou, Xi'an
- **Domestic Flights**: Well-connected to all major Chinese cities

## Cultural Etiquette

### General Behavior
- **Greetings**: Handshakes common, slight bow for elders
- **Dress**: Modest clothing, remove hats in temples
- **Photography**: Ask permission before taking photos of people
- **Queue**: Be patient in lines and crowds

### Visiting Religious Sites
- **Dress Code**: Modest clothing, cover shoulders and knees
- **Behavior**: Quiet and respectful, no smoking
- **Photography**: Check for restrictions, especially in temples
- **Donations**: Optional but appreciated

## Food & Dining

### Must-Try Beijing Foods
- **Peking Duck**: Beijing's most famous dish
- **Jiaozi**: Chinese dumplings
- **Hot Pot**: Spicy or mild broth with meat and vegetables
- **Noodles**: Hand-pulled noodles in various styles
- **Street Food**: Jianbing (savory crepes), baozi (steamed buns)

### Dining Districts
- **Wangfujing**: Food street with traditional and modern options
- **Nanluoguxiang**: Hip cafes and restaurants
- **Sanlitun**: International cuisine and nightlife
- **Houhai**: Lakeside dining with traditional atmosphere

## Shopping

### Traditional Items
- **Silk**: Traditional Chinese silk products
- **Tea**: Various types of Chinese tea
- **Calligraphy**: Traditional writing supplies
- **Antiques**: Reproductions and genuine items (be careful)

### Shopping Areas
- **Wangfujing**: Modern shopping street
- **Sanlitun**: International brands and boutiques
- **Panjiayuan**: Antique and flea market
- **Silk Market**: Traditional goods and bargaining

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury international chains
- **Hutong Hotels**: Traditional courtyard accommodations
- **Hostels**: Budget-friendly options in popular areas
- **Serviced Apartments**: Good for longer stays

## Seasonal Highlights

### Spring
- Cherry blossoms at Yuyuantan Park
- Clear skies and pleasant weather
- Cultural festivals and events

### Summer
- Long daylight hours for sightseeing
- Outdoor activities and festivals
- Summer Palace at its best

### Autumn
- Golden autumn colors
- National Day celebrations
- Best weather for outdoor activities

### Winter
- Snow-covered Great Wall
- Indoor cultural activities
- Traditional winter foods

## Practical Information

### Important Numbers
- **Emergency**: 110 (Police), 120 (Ambulance), 119 (Fire)
- **Tourist Information**: 12301 (24-hour hotline)
- **Weather**: 12121

### Useful Apps
- **WeChat**: Essential for payments and communication
- **DiDi**: Chinese ride-hailing app
- **Baidu Maps**: Better than Google Maps in China
- **Pleco**: Chinese-English dictionary

### Money & Payments
- **Currency**: Chinese Yuan (CNY/RMB)
- **WeChat Pay/Alipay**: Digital payments widely accepted
- **Cash**: Still useful for small purchases
- **Credit Cards**: Accepted at hotels and larger establishments

## Weather by Season
- **Spring**: 10-25°C, occasional sandstorms, pleasant weather
- **Summer**: 25-35°C, high humidity, frequent rain, peak season
- **Autumn**: 10-25°C, clear skies, beautiful colors, best weather
- **Winter**: -5 to 10°C, cold and dry, occasional snow, fewer crowds"""
    
    return content


def create_seoul_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Seoul-specific content"""
    content = f"""{destination_name} is South Korea's dynamic capital, a fascinating blend of ancient traditions and cutting-edge technology. From historic palaces and temples to modern skyscrapers and K-pop culture, {destination_name} offers visitors an exciting mix of old and new.

## Overview
{destination_name} has been Korea's capital for over 600 years and is now a global metropolis of over 10 million people. The city seamlessly combines its rich cultural heritage with modern innovation, making it one of Asia's most exciting destinations.

## Top Attractions

### Historic Palaces
"""
    
    # Add palace information
    palaces = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['palace', 'gung', 'changdeok', 'gyeongbok', 'deoksugung'])]
    
    if palaces:
        for i, palace in enumerate(palaces[:5], 1):
            content += f"- **{palace['name']}**: Magnificent royal palace and cultural treasure\n"
    else:
        content += """- **Gyeongbokgung Palace**: Main royal palace of the Joseon dynasty
- **Changdeokgung Palace**: UNESCO World Heritage palace complex
- **Deoksugung Palace**: Historic palace with Western architecture
- **Gyeonghuigung Palace**: Smaller palace with beautiful gardens\n"""
    
    content += """
### Traditional Districts
- **Bukchon Hanok Village**: Traditional Korean houses and culture
- **Insadong**: Traditional arts, crafts, and tea houses
- **Myeongdong**: Shopping and street food district
- **Hongdae**: Youth culture and entertainment area

### Modern Attractions
"""
    
    # Add modern attraction information
    modern_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['tower', 'skytree', 'observatory', 'museum', 'park'])]
    
    if modern_sites:
        for site in modern_sites[:5]:
            content += f"- **{site['name']}**: Modern landmark and cultural venue\n"
    else:
        content += """- **N Seoul Tower**: Iconic tower with panoramic city views
- **Lotte World Tower**: Tallest building in Korea
- **COEX Mall**: Large underground shopping complex
- **Seoul Forest Park**: Urban park and recreation area\n"""
    
    content += """
## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-25°C), cherry blossoms
- **Highlights**: Cherry blossom festivals, Yeouido Spring Flower Festival
- **Advantages**: Pleasant weather, beautiful scenery
- **Special**: Cherry blossoms peak in early April

### Summer (June - August)
- **Weather**: Hot and humid (20-35°C), monsoon season
- **Highlights**: Summer festivals, outdoor activities
- **Considerations**: Rainy season in June-July, very humid
- **Air Conditioning**: Essential for comfort

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-25°C), clear skies
- **Highlights**: Fall foliage, cultural festivals
- **Advantages**: Best weather, beautiful autumn colors
- **Special**: Chuseok holiday in September/October

### Winter (December - February)
- **Weather**: Cold and dry (-5 to 10°C), occasional snow
- **Highlights**: Winter festivals, indoor attractions
- **Advantages**: Fewer crowds, lower prices
- **Considerations**: Very cold, some outdoor activities limited

## Transportation

### Getting Around Seoul
- **Metro**: Extensive 9-line system, clean and efficient
- **Bus**: Comprehensive network, very affordable
- **Taxi**: Readily available, use Kakao T app
- **Bicycle**: Bike-sharing available in many areas

### From Other Cities
- **International Airport**: Incheon International Airport (ICN)
- **High-Speed Rail**: KTX connections to Busan, Daegu, Gwangju
- **Domestic Flights**: Well-connected to all major Korean cities

## Cultural Etiquette

### General Behavior
- **Greetings**: Bow slightly when meeting people
- **Dress**: Smart casual is appropriate for most places
- **Photography**: Ask permission before taking photos of people
- **Queue**: Be patient and orderly in lines

### Visiting Cultural Sites
- **Dress Code**: Modest clothing for temples and palaces
- **Behavior**: Quiet and respectful, remove shoes when required
- **Photography**: Check for restrictions, especially in palaces
- **Donations**: Optional but appreciated at temples

## Food & Dining

### Must-Try Seoul Foods
- **Korean BBQ**: Grilled meat at your table
- **Bibimbap**: Mixed rice bowl with vegetables and meat
- **Kimchi**: Fermented vegetables, Korea's national dish
- **Tteokbokki**: Spicy rice cakes
- **Korean Fried Chicken**: Crispy fried chicken with various sauces

### Dining Districts
- **Myeongdong**: Street food and international cuisine
- **Hongdae**: Youth-oriented restaurants and cafes
- **Gangnam**: High-end dining and international cuisine
- **Insadong**: Traditional Korean restaurants and tea houses

## Shopping

### Traditional Items
- **Hanbok**: Traditional Korean clothing
- **Ceramics**: Traditional Korean pottery
- **Tea**: Various types of Korean tea
- **Cosmetics**: K-beauty products

### Shopping Areas
- **Myeongdong**: International brands and cosmetics
- **Dongdaemun**: 24-hour shopping and wholesale
- **Gangnam**: Luxury brands and high-end shopping
- **Insadong**: Traditional arts and crafts

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury international chains
- **Hanok Stays**: Traditional Korean house accommodations
- **Hostels**: Budget-friendly options in popular areas
- **Guesthouses**: Family-run accommodations

## Seasonal Highlights

### Spring
- Cherry blossoms throughout the city
- Yeouido Spring Flower Festival
- Pleasant weather for outdoor activities

### Summer
- Summer festivals and events
- Outdoor activities and nightlife
- Monsoon season brings rain and humidity

### Autumn
- Beautiful fall foliage
- Cultural festivals and events
- Best weather for sightseeing

### Winter
- Winter festivals and events
- Indoor cultural activities
- Traditional winter foods

## Practical Information

### Important Numbers
- **Emergency**: 112 (Police), 119 (Fire/Ambulance)
- **Tourist Information**: 1330 (24-hour hotline)
- **Weather**: 131

### Useful Apps
- **Kakao T**: Korean ride-hailing app
- **Naver Maps**: Better than Google Maps in Korea
- **Google Translate**: For language assistance
- **Seoul Metro**: Subway route planning

### Money & Payments
- **Currency**: Korean Won (KRW)
- **Credit Cards**: Widely accepted
- **Cash**: Still useful for small purchases
- **T-money Card**: For public transportation

## Weather by Season
- **Spring**: 10-25°C, cherry blossoms, pleasant weather
- **Summer**: 20-35°C, high humidity, monsoon season
- **Autumn**: 10-25°C, clear skies, beautiful fall colors
- **Winter**: -5 to 10°C, cold and dry, occasional snow"""
    
    return content


def create_shanghai_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Shanghai-specific content"""
    content = f"""{destination_name} is China's largest city and global financial hub, where East meets West in spectacular fashion. From the historic Bund waterfront to the futuristic Pudong skyline, {destination_name} showcases China's rapid modernization while preserving its cultural heritage.

## Overview
{destination_name} is a city of contrasts - colonial architecture along the Bund, ultra-modern skyscrapers in Pudong, and traditional gardens and temples scattered throughout. As China's most cosmopolitan city, it offers visitors a unique blend of history, culture, and innovation.

## Top Attractions

### Historic Landmarks
"""
    
    # Add historic landmark information
    historic_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['bund', 'yuyuan', 'temple', 'garden', 'pagoda', 'museum'])]
    for i, site in enumerate(historic_sites[:8], 1):
        content += f"- **{site['name']}**: Historic landmark and cultural treasure\n"
    
    content += """
### Modern Attractions
"""
    
    # Add modern attraction information
    modern_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['tower', 'pearl', 'oriental', 'skytree', 'observatory'])]
    for site in modern_sites[:5]:
        content += f"- **{site['name']}**: Modern architectural marvel\n"
    
    content += """
### Cultural Districts
- **The Bund**: Historic waterfront with colonial architecture
- **French Concession**: Charming tree-lined streets and cafes
- **Xintiandi**: Hip area with restored shikumen houses
- **Tianzifang**: Artsy maze of narrow alleys and boutiques

## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-25°C), occasional rain
- **Highlights**: Cherry blossoms, pleasant weather
- **Advantages**: Good weather, moderate crowds
- **Considerations**: March can be windy

### Summer (June - August)
- **Weather**: Hot and humid (25-35°C), frequent rain
- **Highlights**: Long daylight hours, outdoor activities
- **Considerations**: Peak tourist season, very humid
- **Air Quality**: Can be affected by pollution

### Autumn (September - November)
- **Weather**: Comfortable temperatures (15-25°C), clear skies
- **Highlights**: Golden autumn colors, cultural festivals
- **Advantages**: Best weather, beautiful scenery
- **Special**: National Day holiday in October

### Winter (December - February)
- **Weather**: Cold and damp (0-15°C), occasional snow
- **Highlights**: Indoor attractions, winter festivals
- **Advantages**: Fewer crowds, lower prices
- **Considerations**: Cold and damp weather

## Transportation

### Getting Around Shanghai
- **Metro**: Extensive 18-line system, clean and efficient
- **Bus**: Comprehensive network, very affordable
- **Taxi**: Readily available, use ride-hailing apps
- **Bicycle**: Bike-sharing available throughout the city

### From Other Cities
- **International Airport**: Pudong International Airport (PVG)
- **High-Speed Rail**: Connections to Beijing, Guangzhou, Hangzhou
- **Domestic Flights**: Well-connected to all major Chinese cities

## Cultural Etiquette

### General Behavior
- **Greetings**: Handshakes common, slight bow for elders
- **Dress**: Smart casual is appropriate for most places
- **Photography**: Ask permission before taking photos of people
- **Queue**: Be patient in lines and crowds

### Visiting Cultural Sites
- **Dress Code**: Modest clothing for temples and museums
- **Behavior**: Quiet and respectful, no smoking
- **Photography**: Check for restrictions
- **Donations**: Optional but appreciated

## Food & Dining

### Must-Try Shanghai Foods
- **Xiaolongbao**: Soup dumplings
- **Shanghai Hairy Crab**: Seasonal delicacy
- **Hong Shao Rou**: Braised pork belly
- **Shengjianbao**: Pan-fried soup dumplings
- **Noodles**: Various styles of Shanghai noodles

### Dining Districts
- **The Bund**: High-end dining with river views
- **French Concession**: International cuisine and cafes
- **Xintiandi**: Modern restaurants and bars
- **Nanjing Road**: Traditional and modern options

## Shopping

### Traditional Items
- **Silk**: Traditional Chinese silk products
- **Tea**: Various types of Chinese tea
- **Jade**: Traditional jade jewelry and carvings
- **Antiques**: Reproductions and genuine items

### Shopping Areas
- **Nanjing Road**: Famous shopping street
- **Xintiandi**: Boutique shopping
- **Tianzifang**: Arts and crafts
- **The Bund**: Luxury shopping

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury international chains
- **Boutique Hotels**: Unique accommodations in historic areas
- **Hostels**: Budget-friendly options in popular areas
- **Serviced Apartments**: Good for longer stays

## Seasonal Highlights

### Spring
- Cherry blossoms in parks
- Pleasant weather for outdoor activities
- Cultural festivals and events

### Summer
- Long daylight hours for sightseeing
- Outdoor activities and festivals
- Summer heat and humidity

### Autumn
- Golden autumn colors
- Cultural festivals and events
- Best weather for outdoor activities

### Winter
- Indoor cultural activities
- Winter festivals and events
- Traditional winter foods

## Practical Information

### Important Numbers
- **Emergency**: 110 (Police), 120 (Ambulance), 119 (Fire)
- **Tourist Information**: 12301 (24-hour hotline)
- **Weather**: 12121

### Useful Apps
- **WeChat**: Essential for payments and communication
- **DiDi**: Chinese ride-hailing app
- **Baidu Maps**: Better than Google Maps in China
- **Shanghai Metro**: Subway route planning

### Money & Payments
- **Currency**: Chinese Yuan (CNY/RMB)
- **WeChat Pay/Alipay**: Digital payments widely accepted
- **Cash**: Still useful for small purchases
- **Credit Cards**: Accepted at hotels and larger establishments

## Weather by Season
- **Spring**: 10-25°C, occasional rain, pleasant weather
- **Summer**: 25-35°C, high humidity, frequent rain, peak season
- **Autumn**: 15-25°C, clear skies, beautiful colors, best weather
- **Winter**: 0-15°C, cold and damp, occasional snow, fewer crowds"""
    
    return content


def create_paris_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Paris-specific content"""
    content = f"""{destination_name} is the City of Light, France's romantic capital and one of the world's most beautiful cities. From the iconic Eiffel Tower to the historic Louvre Museum, {destination_name} offers visitors an unparalleled cultural and artistic experience.

## Overview
{destination_name} has been a center of art, culture, and fashion for centuries. With its stunning architecture, world-class museums, and charming neighborhoods, the city embodies the perfect blend of history and modernity.

## Top Attractions

### Iconic Landmarks
"""
    
    # Add iconic landmark information
    iconic_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['eiffel', 'louvre', 'notre-dame', 'arc', 'champs', 'versailles'])]
    for i, site in enumerate(iconic_sites[:8], 1):
        content += f"- **{site['name']}**: World-famous landmark and cultural icon\n"
    
    content += """
### Cultural Districts
- **Le Marais**: Historic district with trendy boutiques
- **Montmartre**: Artistic neighborhood with Sacré-Cœur
- **Latin Quarter**: Student area with historic charm
- **Saint-Germain-des-Prés**: Literary and artistic quarter

### Museums & Galleries
"""
    
    # Add museum information
    museums = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['museum', 'musee', 'gallery', 'palace'])]
    for museum in museums[:5]:
        content += f"- **{museum['name']}**: World-class cultural institution\n"
    
    content += """
## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-20°C), cherry blossoms
- **Highlights**: Spring flowers, outdoor cafes
- **Advantages**: Pleasant weather, moderate crowds
- **Special**: Paris Fashion Week in March

### Summer (June - August)
- **Weather**: Warm temperatures (20-30°C), long days
- **Highlights**: Outdoor festivals, river cruises
- **Considerations**: Peak tourist season, book early
- **Special**: Bastille Day celebrations in July

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-20°C), fall colors
- **Highlights**: Cultural events, wine harvest
- **Advantages**: Fewer crowds, beautiful autumn scenery
- **Special**: Paris Fashion Week in September

### Winter (December - February)
- **Weather**: Cold temperatures (0-10°C), occasional snow
- **Highlights**: Christmas markets, indoor attractions
- **Advantages**: Fewer crowds, lower prices
- **Special**: Christmas lights and decorations

## Transportation

### Getting Around Paris
- **Metro**: Extensive 16-line system, efficient and clean
- **Bus**: Comprehensive network, scenic routes
- **RER**: Regional trains for longer distances
- **Walking**: Many attractions are within walking distance

### From Other Cities
- **International Airport**: Charles de Gaulle (CDG) and Orly (ORY)
- **High-Speed Rail**: TGV connections to major European cities
- **Eurostar**: Direct train to London

## Cultural Etiquette

### General Behavior
- **Greetings**: Bonjour (hello) is essential
- **Dress**: Smart casual, Parisians dress well
- **Photography**: Ask permission before taking photos
- **Queue**: Be patient and orderly

### Visiting Cultural Sites
- **Dress Code**: Modest clothing for churches
- **Behavior**: Quiet and respectful
- **Photography**: Check for restrictions
- **Tickets**: Book major attractions in advance

## Food & Dining

### Must-Try Paris Foods
- **Croissants**: Fresh from local bakeries
- **Macarons**: Colorful French pastries
- **Escargots**: Traditional French snails
- **Coq au Vin**: Classic French chicken dish
- **Crêpes**: Sweet and savory options

### Dining Districts
- **Le Marais**: Trendy restaurants and cafes
- **Saint-Germain**: Traditional French bistros
- **Montmartre**: Charming neighborhood dining
- **Champs-Élysées**: High-end restaurants

## Shopping

### Traditional Items
- **Fashion**: Designer clothing and accessories
- **Perfume**: French fragrances
- **Wine**: French wines and champagne
- **Art**: Prints and reproductions

### Shopping Areas
- **Champs-Élysées**: Luxury shopping
- **Le Marais**: Boutique shopping
- **Galeries Lafayette**: Department store
- **Rue de Rivoli**: Traditional shopping

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury
- **Boutique Hotels**: Charming small hotels
- **Hostels**: Budget-friendly options
- **Apartments**: Self-catering options

## Seasonal Highlights

### Spring
- Cherry blossoms in parks
- Outdoor cafes and terraces
- Spring fashion shows

### Summer
- Long daylight hours
- Outdoor festivals and events
- River cruises on the Seine

### Autumn
- Fall colors in parks
- Cultural festivals
- Wine harvest celebrations

### Winter
- Christmas markets
- Indoor cultural activities
- Winter sales in January

## Practical Information

### Important Numbers
- **Emergency**: 112 (EU emergency number)
- **Police**: 17
- **Tourist Information**: Available at visitor centers

### Useful Apps
- **RATP**: Public transport app
- **Google Translate**: For language assistance
- **Paris Metro**: Subway route planning

### Money & Payments
- **Currency**: Euro (€)
- **Credit Cards**: Widely accepted
- **Cash**: Still useful for small purchases
- **Tipping**: Service included, extra tip appreciated

## Weather by Season
- **Spring**: 10-20°C, cherry blossoms, pleasant weather
- **Summer**: 20-30°C, long days, peak tourist season
- **Autumn**: 10-20°C, fall colors, fewer crowds
- **Winter**: 0-10°C, cold, Christmas atmosphere"""
    
    return content


def create_london_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create London-specific content"""
    content = f"""{destination_name} is the historic capital of England and the United Kingdom, a global city that seamlessly blends centuries of history with modern innovation. From the iconic Big Ben to the cutting-edge Tate Modern, {destination_name} offers visitors an unparalleled cultural experience.

## Overview
{destination_name} has been a center of power, culture, and commerce for over 2,000 years. With its royal palaces, world-class museums, and diverse neighborhoods, the city offers something for every visitor.

## Top Attractions

### Historic Landmarks
"""
    
    # Add historic landmark information
    historic_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['buckingham', 'tower', 'westminster', 'big ben', 'trafalgar', 'hyde', 'st paul'])]
    for i, site in enumerate(historic_sites[:8], 1):
        content += f"- **{site['name']}**: Historic landmark and cultural icon\n"
    
    content += """
### Museums & Galleries
"""
    
    # Add museum information
    museums = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['museum', 'gallery', 'tate', 'british', 'natural', 'science'])]
    for museum in museums[:5]:
        content += f"- **{museum['name']}**: World-class cultural institution\n"
    
    content += """
### Cultural Districts
- **Westminster**: Government and royal landmarks
- **Soho**: Entertainment and nightlife district
- **Camden**: Alternative culture and markets
- **Shoreditch**: Hipster area with street art

## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-20°C), cherry blossoms
- **Highlights**: Spring flowers, outdoor events
- **Advantages**: Pleasant weather, moderate crowds
- **Special**: Chelsea Flower Show in May

### Summer (June - August)
- **Weather**: Warm temperatures (15-25°C), long days
- **Highlights**: Outdoor festivals, royal events
- **Considerations**: Peak tourist season, book early
- **Special**: Wimbledon tennis tournament

### Autumn (September - November)
- **Weather**: Comfortable temperatures (10-20°C), fall colors
- **Highlights**: Cultural events, fashion week
- **Advantages**: Fewer crowds, beautiful autumn scenery
- **Special**: London Fashion Week in September

### Winter (December - February)
- **Weather**: Cold temperatures (0-10°C), occasional snow
- **Highlights**: Christmas markets, indoor attractions
- **Advantages**: Fewer crowds, lower prices
- **Special**: New Year's Eve celebrations

## Transportation

### Getting Around London
- **Underground**: Extensive tube network, efficient
- **Bus**: Comprehensive network, scenic routes
- **Overground**: Surface rail network
- **Walking**: Many attractions are within walking distance

### From Other Cities
- **International Airport**: Heathrow (LHR), Gatwick (LGW), Stansted (STN)
- **Eurostar**: Direct train to Paris and Brussels
- **Domestic Rail**: Connections to all major UK cities

## Cultural Etiquette

### General Behavior
- **Greetings**: Handshakes common, polite and reserved
- **Dress**: Smart casual, Londoners dress well
- **Photography**: Ask permission before taking photos
- **Queue**: British people are very particular about queuing

### Visiting Cultural Sites
- **Dress Code**: Modest clothing for churches
- **Behavior**: Quiet and respectful
- **Photography**: Check for restrictions
- **Tickets**: Book major attractions in advance

## Food & Dining

### Must-Try London Foods
- **Fish and Chips**: Traditional British dish
- **Sunday Roast**: Traditional Sunday meal
- **Full English Breakfast**: Hearty morning meal
- **Afternoon Tea**: Traditional British tea service
- **Pie and Mash**: Traditional London dish

### Dining Districts
- **Soho**: International cuisine and trendy restaurants
- **Camden**: Street food and alternative dining
- **Mayfair**: High-end restaurants
- **Brick Lane**: Indian and Bangladeshi cuisine

## Shopping

### Traditional Items
- **Tea**: Traditional British tea
- **Tweed**: Traditional British fabric
- **Antiques**: From Portobello Road market
- **Fashion**: From Oxford Street and Bond Street

### Shopping Areas
- **Oxford Street**: Main shopping street
- **Bond Street**: Luxury shopping
- **Camden Market**: Alternative and vintage
- **Portobello Road**: Antiques and vintage

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury
- **Boutique Hotels**: Charming small hotels
- **Hostels**: Budget-friendly options
- **Apartments**: Self-catering options

## Seasonal Highlights

### Spring
- Cherry blossoms in parks
- Chelsea Flower Show
- Spring fashion shows

### Summer
- Long daylight hours
- Outdoor festivals and events
- Royal events and ceremonies

### Autumn
- Fall colors in parks
- London Fashion Week
- Cultural festivals

### Winter
- Christmas markets
- New Year's Eve celebrations
- Indoor cultural activities

## Practical Information

### Important Numbers
- **Emergency**: 999 (UK emergency number)
- **Police**: 101 (non-emergency)
- **Tourist Information**: Available at visitor centers

### Useful Apps
- **TfL Go**: Public transport app
- **Google Translate**: For language assistance
- **London Underground**: Tube route planning

### Money & Payments
- **Currency**: British Pound (£)
- **Credit Cards**: Widely accepted
- **Cash**: Still useful for small purchases
- **Contactless**: Very common for payments

## Weather by Season
- **Spring**: 10-20°C, cherry blossoms, pleasant weather
- **Summer**: 15-25°C, long days, peak tourist season
- **Autumn**: 10-20°C, fall colors, fewer crowds
- **Winter**: 0-10°C, cold, Christmas atmosphere"""
    
    return content


def create_rome_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Rome-specific content"""
    content = f"""{destination_name} is the Eternal City, Italy's historic capital and one of the world's most beautiful cities. From the ancient Colosseum to the magnificent Vatican, {destination_name} offers visitors an unparalleled journey through history and art.

## Overview
{destination_name} has been a center of power, culture, and religion for over 2,500 years. With its ancient ruins, Renaissance art, and vibrant modern culture, the city offers a unique blend of past and present.

## Top Attractions

### Ancient Landmarks
"""
    
    # Add ancient landmark information
    ancient_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['colosseum', 'forum', 'pantheon', 'palatine', 'circus', 'appian', 'catacombs'])]
    for i, site in enumerate(ancient_sites[:8], 1):
        content += f"- **{site['name']}**: Ancient Roman landmark and archaeological treasure\n"
    
    content += """
### Vatican City
"""
    
    # Add Vatican information
    vatican_sites = [att for att in attractions if any(keyword in att['name'].lower() for keyword in ['vatican', 'sistine', 'st peter', 'basilica'])]
    for site in vatican_sites[:3]:
        content += f"- **{site['name']}**: Sacred site and artistic masterpiece\n"
    
    content += """
### Cultural Districts
- **Centro Storico**: Historic center with major landmarks
- **Trastevere**: Charming neighborhood with authentic atmosphere
- **Vatican**: Religious and cultural center
- **Testaccio**: Traditional Roman neighborhood

## Best Time to Visit

### Spring (March - May)
- **Weather**: Mild temperatures (10-25°C), pleasant
- **Highlights**: Spring flowers, outdoor cafes
- **Advantages**: Good weather, moderate crowds
- **Special**: Easter celebrations at the Vatican

### Summer (June - August)
- **Weather**: Hot temperatures (20-35°C), dry
- **Highlights**: Long daylight hours, outdoor dining
- **Considerations**: Peak tourist season, very hot
- **Special**: Summer festivals and events

### Autumn (September - November)
- **Weather**: Comfortable temperatures (15-25°C), pleasant
- **Highlights**: Cultural events, wine harvest
- **Advantages**: Fewer crowds, beautiful weather
- **Special**: Grape harvest in surrounding regions

### Winter (December - February)
- **Weather**: Cool temperatures (5-15°C), occasional rain
- **Highlights**: Christmas celebrations, indoor attractions
- **Advantages**: Fewer crowds, lower prices
- **Special**: Christmas markets and decorations

## Transportation

### Getting Around Rome
- **Metro**: Three-line system, efficient for major sites
- **Bus**: Comprehensive network, scenic routes
- **Tram**: Limited but useful routes
- **Walking**: Many attractions are within walking distance

### From Other Cities
- **International Airport**: Fiumicino (FCO) and Ciampino (CIA)
- **High-Speed Rail**: Connections to Milan, Florence, Naples
- **Domestic Flights**: Well-connected to major Italian cities

## Cultural Etiquette

### General Behavior
- **Greetings**: Handshakes common, warm and friendly
- **Dress**: Smart casual, Italians dress well
- **Photography**: Ask permission before taking photos
- **Queue**: Be patient, Italians are more relaxed about queuing

### Visiting Religious Sites
- **Dress Code**: Modest clothing required, cover shoulders and knees
- **Behavior**: Quiet and respectful
- **Photography**: Check for restrictions, especially in churches
- **Tickets**: Book Vatican and major attractions in advance

## Food & Dining

### Must-Try Roman Foods
- **Pizza Romana**: Thin-crust Roman-style pizza
- **Pasta alla Carbonara**: Traditional Roman pasta dish
- **Saltimbocca**: Veal with prosciutto and sage
- **Gelato**: Italian ice cream
- **Espresso**: Traditional Italian coffee

### Dining Districts
- **Trastevere**: Authentic Roman restaurants
- **Testaccio**: Traditional Roman cuisine
- **Centro Storico**: Tourist-friendly restaurants
- **Monti**: Hip neighborhood with trendy restaurants

## Shopping

### Traditional Items
- **Fashion**: Italian designer clothing
- **Leather Goods**: Bags, shoes, and accessories
- **Wine**: Italian wines from surrounding regions
- **Art**: Prints and reproductions

### Shopping Areas
- **Via del Corso**: Main shopping street
- **Via Condotti**: Luxury shopping
- **Campo de' Fiori**: Local market
- **Porta Portese**: Sunday flea market

## Accommodation

### Options Available
- **Hotels**: Range from budget to luxury
- **Boutique Hotels**: Charming small hotels
- **Hostels**: Budget-friendly options
- **Apartments**: Self-catering options

## Seasonal Highlights

### Spring
- Spring flowers in parks
- Easter celebrations
- Outdoor cafes and terraces

### Summer
- Long daylight hours
- Summer festivals and events
- Outdoor dining and nightlife

### Autumn
- Wine harvest celebrations
- Cultural festivals
- Beautiful autumn weather

### Winter
- Christmas markets
- Indoor cultural activities
- Traditional winter foods

## Practical Information

### Important Numbers
- **Emergency**: 112 (EU emergency number)
- **Police**: 113
- **Tourist Information**: Available at visitor centers

### Useful Apps
- **ATAC**: Public transport app
- **Google Translate**: For language assistance
- **Roma Metro**: Subway route planning

### Money & Payments
- **Currency**: Euro (€)
- **Credit Cards**: Widely accepted
- **Cash**: Still useful for small purchases
- **Tipping**: Service included, extra tip appreciated

## Weather by Season
- **Spring**: 10-25°C, pleasant weather, moderate crowds
- **Summer**: 20-35°C, hot and dry, peak tourist season
- **Autumn**: 15-25°C, beautiful weather, fewer crowds
- **Winter**: 5-15°C, cool and wet, fewer crowds"""
    
    return content


def create_barcelona_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Barcelona-specific content"""
    return create_european_content(attractions, destination_name)


def create_amsterdam_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Amsterdam-specific content"""
    return create_european_content(attractions, destination_name)


def create_berlin_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Berlin-specific content"""
    return create_european_content(attractions, destination_name)


def create_prague_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Prague-specific content"""
    return create_european_content(attractions, destination_name)


def create_vienna_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Vienna-specific content"""
    return create_european_content(attractions, destination_name)


def create_budapest_content(attractions: List[Dict[str, str]], destination_name: str) -> str:
    """Create Budapest-specific content"""
    return create_european_content(attractions, destination_name)


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