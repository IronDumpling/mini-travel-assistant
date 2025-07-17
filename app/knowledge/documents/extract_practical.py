#!/usr/bin/env python3
"""
Generic Practical Travel Information Extractor
Uses robust web scraping utilities to extract practical travel information
Supports multiple countries and regions including visa requirements, customs, health info
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

# Practical information configurations
PRACTICAL_CONFIGS = {
    "japan_visa": {
        "url": "https://en.wikivoyage.org/wiki/Japan#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "info_type": "visa",
        "country": "japan",
        "keywords": [
            "visa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "exemption",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
        ],
    },
    "usa_visa": {
        "url": "https://en.wikivoyage.org/wiki/United_States_of_America#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "north_america",
        "info_type": "visa",
        "country": "usa",
        "keywords": [
            "visa",
            "esta",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
    "schengen_visa": {
        "url": "https://en.wikivoyage.org/wiki/Schengen_visa",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "info_type": "visa",
        "country": "schengen",
        "keywords": [
            "schengen",
            "visa",
            "europe",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
            "schengen",
            "area",
        ],
    },
    "uk_visa": {
        "url": "https://en.wikivoyage.org/wiki/United_Kingdom#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "europe",
        "info_type": "visa",
        "country": "uk",
        "keywords": [
            "visa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "exemption",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
            "brexit",
        ],
    },
    "australia_visa": {
        "url": "https://en.wikivoyage.org/wiki/Australia#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "oceania",
        "info_type": "visa",
        "country": "australia",
        "keywords": [
            "visa",
            "eta",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
            "quarantine",
        ],
    },
    "canada_visa": {
        "url": "https://en.wikivoyage.org/wiki/Canada#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "north_america",
        "info_type": "visa",
        "country": "canada",
        "keywords": [
            "visa",
            "eta",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
    "china_visa": {
        "url": "https://en.wikivoyage.org/wiki/China#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "info_type": "visa",
        "country": "china",
        "keywords": [
            "visa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "exemption",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
    "india_visa": {
        "url": "https://en.wikivoyage.org/wiki/India#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "asia",
        "info_type": "visa",
        "country": "india",
        "keywords": [
            "visa",
            "evisa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
    "brazil_visa": {
        "url": "https://en.wikivoyage.org/wiki/Brazil#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "south_america",
        "info_type": "visa",
        "country": "brazil",
        "keywords": [
            "visa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "exemption",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
    "south_africa_visa": {
        "url": "https://en.wikivoyage.org/wiki/South_Africa#Get_in",
        "source": "WikiVoyage",
        "source_url": "https://en.wikivoyage.org",
        "source_id": "wikivoyage_official",
        "region": "africa",
        "info_type": "visa",
        "country": "south_africa",
        "keywords": [
            "visa",
            "passport",
            "immigration",
            "entry",
            "requirements",
            "tourist",
            "business",
            "work",
            "student",
            "transit",
            "waiver",
            "exemption",
            "application",
            "documents",
            "fee",
            "duration",
            "validity",
            "extension",
            "border",
            "customs",
            "declaration",
        ],
    },
}


def get_session():
    """Create a session with proper headers and cookies"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def scrape_practical_info(practical_name: str):
    """Scrape practical travel information from various sources"""
    session = get_session()

    if practical_name.lower() not in PRACTICAL_CONFIGS:
        logger.error(
            f"Practical info '{practical_name}' not configured. Available: {list(PRACTICAL_CONFIGS.keys())}"
        )
        return []

    config = PRACTICAL_CONFIGS[practical_name.lower()]
    url = config["url"]

    try:
        # Add a random delay to avoid being flagged as a bot
        time.sleep(random.uniform(1, 3))

        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        practical_info = []

        # For WikiVoyage pages, look for specific practical information sections
        if "wikivoyage" in config["source_url"].lower():
            practical_info = scrape_wikivoyage_practical(soup, config)
        else:
            # Fallback to general link extraction
            practical_info = scrape_general_links(soup, config)

        # Remove duplicates based on name
        unique_info = []
        seen_names = set()
        for info in practical_info:
            if info["name"] not in seen_names:
                unique_info.append(info)
                seen_names.add(info["name"])

        logger.info(
            f"Extracted {len(unique_info)} practical items from {config['source']}"
        )
        return unique_info

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


def scrape_wikivoyage_practical(soup, config):
    """Extract specific practical information from WikiVoyage pages"""
    practical_info = []

    # Look for practical information sections in WikiVoyage
    practical_sections = [
        "Get in",
        "Visa",
        "Passport",
        "Immigration",
        "Entry requirements",
        "Customs",
        "Health",
        "Vaccinations",
        "Insurance",
        "Documents",
        "Requirements",
        "Application",
        "Fees",
        "Duration",
        "Validity",
    ]

    for section in practical_sections:
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
                                    # Check if it's a specific practical item
                                    if is_specific_practical(name, config):
                                        full_url = urljoin(config["source_url"], href)
                                        practical_info.append(
                                            {"name": name, "url": full_url}
                                        )
                    elif element.name == "p":
                        # Extract from paragraphs
                        links = element.find_all("a")
                        for link in links:
                            name = link.get_text(strip=True)
                            href = link.get("href")
                            if name and href and len(name) > 3:
                                if is_specific_practical(name, config):
                                    full_url = urljoin(config["source_url"], href)
                                    practical_info.append(
                                        {"name": name, "url": full_url}
                                    )

    # Also look for specific practical keywords in the page
    all_links = soup.find_all("a")
    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get("href")

            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a specific practical item
                if is_specific_practical(name, config):
                    # Construct full URL
                    if href.startswith("/"):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], "/" + href)

                    # Avoid duplicates and navigation links
                    if name not in [info["name"] for info in practical_info]:
                        practical_info.append({"name": name, "url": full_url})

        except (AttributeError, TypeError) as e:
            continue

    return practical_info


def is_specific_practical(name, config):
    """Check if a name represents a specific practical item rather than a category"""
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
        "practical",
        "information",
        "travel",
        "guide",
        "help",
        "support",
    ]

    if any(term in name_lower for term in generic_terms):
        return False

    # Check if it contains specific practical keywords
    practical_keywords = [
        "visa",
        "passport",
        "immigration",
        "entry",
        "requirements",
        "tourist",
        "business",
        "work",
        "student",
        "transit",
        "waiver",
        "exemption",
        "application",
        "documents",
        "fee",
        "duration",
        "validity",
        "extension",
        "border",
        "customs",
        "declaration",
        "health",
        "vaccination",
        "insurance",
    ]

    # Check if it matches destination-specific keywords
    if any(keyword in name_lower for keyword in config["keywords"]):
        return True

    # Check if it contains practical keywords
    if any(keyword in name_lower for keyword in practical_keywords):
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
    practical_info = []
    all_links = soup.find_all("a")

    for link in all_links:
        try:
            name = link.get_text(strip=True)
            href = link.get("href")

            if href and name and isinstance(href, str) and len(name) > 3:
                # Check if this looks like a practical link
                href_lower = href.lower()
                name_lower = name.lower()

                # Filter for likely practical links based on destination
                is_practical = any(
                    keyword in href_lower for keyword in config["keywords"]
                ) or any(
                    keyword in name_lower
                    for keyword in [
                        "visa",
                        "passport",
                        "immigration",
                        "entry",
                        "requirements",
                        "tourist",
                        "business",
                        "work",
                        "student",
                        "transit",
                        "waiver",
                        "exemption",
                        "application",
                        "documents",
                        "fee",
                        "duration",
                        "validity",
                        "extension",
                        "border",
                        "customs",
                        "declaration",
                        "health",
                        "vaccination",
                        "insurance",
                    ]
                )

                # Avoid navigation and non-practical links
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

                if is_practical and not is_navigation:
                    # Construct full URL
                    if href.startswith("/"):
                        full_url = urljoin(config["source_url"], href)
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = urljoin(config["source_url"], "/" + href)

                    # Avoid duplicates
                    if name not in [info["name"] for info in practical_info]:
                        practical_info.append({"name": name, "url": full_url})

        except (AttributeError, TypeError) as e:
            continue

    return practical_info


def create_practical_data(
    practical_info: List[Dict[str, str]], practical_name: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a standardized practical data structure"""

    # Create practical-specific content
    if config["info_type"] == "visa":
        content = create_visa_content(practical_info, practical_name, config)
    else:
        content = create_generic_practical_content(practical_info, practical_name)

    return {
        "id": f"{practical_name.lower().replace(' ', '_').replace(',', '')}_complete_guide",
        "title": f"{practical_name.replace('_', ' ').title()} Complete Travel Guide",
        "content": content,
        "category": "practical",
        "subcategory": config["info_type"],
        "location": config["country"],
        "tags": [
            practical_name.replace("_", " ").lower(),
            "Practical",
            config["info_type"].title(),
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


def create_visa_content(
    practical_info: List[Dict[str, str]], practical_name: str, config: Dict[str, Any]
) -> str:
    """Create content specifically for visa information"""
    country_name = config["country"].replace("_", " ").title()

    content = f"""{practical_name.replace("_", " ").title()} is essential information for travelers planning to visit {country_name}. This comprehensive guide covers everything you need to know about visa requirements, application processes, and entry procedures.\n\n## Overview\n{country_name} has specific visa requirements for different nationalities and purposes of visit. Understanding these requirements is crucial for a smooth travel experience.\n\n## Key Entry Requirements and Information\n\n### Important Requirements and Procedures\n"""

    # Add practical information with more specific descriptions
    for i, info in enumerate(practical_info[:20], 1):  # Top 20 items
        # Create more specific descriptions based on the scraped content
        if any(
            keyword in info["name"].lower()
            for keyword in ["visa", "entry", "requirements", "immigration"]
        ):
            content += f"- **{info['name']}**: Essential entry requirement and visa information\n"
        elif any(
            keyword in info["name"].lower()
            for keyword in ["customs", "quarantine", "health"]
        ):
            content += f"- **{info['name']}**: Important customs, health, and quarantine procedures\n"
        elif any(
            keyword in info["name"].lower()
            for keyword in ["work", "employment", "business"]
        ):
            content += (
                f"- **{info['name']}**: Work visa and employment-related requirements\n"
            )
        elif any(
            keyword in info["name"].lower()
            for keyword in ["student", "study", "education"]
        ):
            content += (
                f"- **{info['name']}**: Student visa and educational requirements\n"
            )
        elif any(
            keyword in info["name"].lower()
            for keyword in ["vaccination", "medical", "health"]
        ):
            content += f"- **{info['name']}**: Health and medical requirements\n"
        else:
            content += f"- **{info['name']}**: Important travel and entry information\n"

    # Add country-specific information based on the country
    content += f"""
    
## Country-Specific Information for {country_name}

### Special Requirements and Considerations
"""

    # Add country-specific content based on the country
    if config["country"] == "usa":
        content += """- **ESTA (Electronic System for Travel Authorization)**: Required for visa waiver program countries
- **Visa Waiver Program**: Available for citizens of 40+ countries
- **Customs Declaration**: Strict customs and border protection procedures
- **TSA Security**: Enhanced airport security measures
- **Border Control**: Comprehensive immigration inspection process\n"""
    elif config["country"] == "australia":
        content += """- **ETA (Electronic Travel Authority)**: Electronic visa for eligible countries
- **Quarantine Requirements**: Strict biosecurity and quarantine procedures
- **Working Holiday Visas**: Available for young travelers from partner countries
- **Student Visas**: Popular for international students
- **Health Requirements**: Medical examinations may be required\n"""
    elif config["country"] == "canada":
        content += """- **eTA (Electronic Travel Authorization)**: Required for visa-exempt countries
- **Express Entry**: Points-based immigration system
- **Study Permits**: Popular for international students
- **Work Permits**: Various categories available
- **Family Sponsorship**: Family reunification programs\n"""
    elif config["country"] == "japan":
        content += """- **Visa Exemptions**: Available for many nationalities
- **Working Holiday Visas**: Bilateral agreements with partner countries
- **Student Visas**: Popular for language and academic studies
- **Business Visas**: For commercial and trade activities
- **Cultural Activities**: Special visas for cultural exchange\n"""
    elif config["country"] == "china":
        content += """- **Visa Categories**: Multiple visa types available
- **Invitation Letters**: Often required for business visas
- **Health Declarations**: Medical examination requirements
- **Registration**: Local police registration required
- **Extension Procedures**: Visa extension processes\n"""
    elif config["country"] == "india":
        content += """- **e-Visa System**: Online visa application available
- **Tourist Visas**: Multiple entry options available
- **Business Visas**: For commercial activities
- **Medical Visas**: For medical treatment
- **Conference Visas**: For attending conferences\n"""
    elif config["country"] == "brazil":
        content += """- **Visa Exemptions**: Available for many nationalities
- **Business Visas**: For commercial activities
- **Student Visas**: For academic studies
- **Work Visas**: Employment authorization
- **Tourist Visas**: For leisure travel\n"""
    else:
        content += f"- **Standard Requirements**: Follow standard visa application procedures for {country_name}\n"

    content += f"""
## Visa Application Process for {country_name}

### Required Documents
"""

    # Add country-specific document requirements
    if config["country"] == "usa":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **DS-160 Application Form**: Complete online application form
- **Photographs**: 2x2 inch color photos with white background
- **Financial Proof**: Bank statements, employment verification, tax returns
- **Travel Itinerary**: Flight bookings and accommodation details
- **ESTA Authorization**: For visa waiver program countries
- **Interview Appointment**: Required at US embassy/consulate
- **SEVIS Fee Receipt**: For student and exchange visitor visas\n"""
    elif config["country"] == "canada":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **Application Forms**: Complete all required forms accurately
- **Photographs**: Recent passport-style photos (35mm x 45mm)
- **Financial Proof**: Bank statements, employment letters, tax returns
- **Travel Itinerary**: Flight bookings and accommodation details
- **eTA Authorization**: For visa-exempt countries
- **Biometric Information**: Fingerprints and photo required
- **Medical Examination**: May be required for certain visa types\n"""
    elif config["country"] == "australia":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **Application Forms**: Complete online or paper application
- **Photographs**: Recent passport-style photos
- **Financial Proof**: Bank statements, employment verification
- **Travel Itinerary**: Flight bookings and accommodation details
- **ETA Authorization**: For eligible countries
- **Health Requirements**: Medical examination if required
- **Character Requirements**: Police certificates if required\n"""
    elif config["country"] == "japan":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **Visa Application Form**: Complete all sections in English or Japanese
- **Photographs**: 2-inch square color photos with white background
- **Financial Proof**: Bank statements, employment certificates
- **Travel Itinerary**: Detailed day-by-day travel plan
- **Invitation Letter**: If visiting friends/family
- **Hotel Reservations**: Confirmed bookings for entire stay
- **Employment Certificate**: On company letterhead with official stamp\n"""
    elif config["country"] == "china":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **Visa Application Form**: Complete all sections accurately
- **Photographs**: Recent passport-style photos
- **Financial Proof**: Bank statements, employment verification
- **Travel Itinerary**: Detailed travel plan with dates
- **Invitation Letter**: Often required for business visas
- **Hotel Reservations**: Confirmed bookings
- **Health Declaration**: Medical examination may be required\n"""
    elif config["country"] == "india":
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **e-Visa Application**: Complete online application form
- **Photographs**: Recent passport-style photos
- **Financial Proof**: Bank statements, employment letters
- **Travel Itinerary**: Flight bookings and accommodation details
- **Yellow Fever Certificate**: If arriving from affected countries
- **Business Documents**: For business visa applications
- **Conference Registration**: For conference visa applications\n"""
    else:
        content += """- **Passport**: Valid for at least 6 months beyond intended stay
- **Application Form**: Complete all sections accurately
- **Photographs**: Recent passport-style photos
- **Financial Proof**: Bank statements and employment verification
- **Travel Itinerary**: Flight bookings and accommodation details
- **Insurance**: Travel insurance coverage proof\n"""

    content += f"""
### Application Steps
"""

    # Add country-specific application steps
    if config["country"] == "usa":
        content += """- **Complete DS-160**: Fill out online application form
- **Pay Visa Fee**: Pay application fee at designated bank
- **Schedule Interview**: Book appointment at US embassy/consulate
- **Gather Documents**: Prepare all required documentation
- **Attend Interview**: Visit embassy/consulate for visa interview
- **Biometric Collection**: Provide fingerprints and photo
- **Wait for Processing**: Processing time varies by visa type
- **Passport Return**: Collect passport with visa or rejection notice\n"""
    elif config["country"] == "canada":
        content += """- **Choose Application Method**: Online or paper application
- **Complete Forms**: Fill out all required application forms
- **Pay Fees**: Pay application and biometric fees
- **Submit Application**: Submit online or at visa office
- **Biometric Appointment**: Provide fingerprints and photo
- **Medical Examination**: If required for your visa type
- **Wait for Processing**: Processing time varies by application type
- **Passport Submission**: Submit passport for visa stamping\n"""
    elif config["country"] == "australia":
        content += """- **Choose Visa Type**: Select appropriate visa category
- **Complete Application**: Fill out online or paper application
- **Pay Visa Fee**: Pay application fee online
- **Submit Documents**: Upload or submit required documents
- **Health Examination**: Complete medical examination if required
- **Character Assessment**: Provide police certificates if required
- **Wait for Decision**: Processing time varies by visa type
- **Visa Grant**: Receive visa grant notice or decision\n"""
    else:
        content += """- **Document Preparation**: Gather all required documents
- **Form Completion**: Fill out application forms accurately
- **Submission**: Submit to embassy, consulate, or online portal
- **Processing**: Wait for application review and approval
- **Collection**: Receive passport with visa or rejection notice\n"""

    content += f"""
## Visa Types Available for {country_name}
"""

    # Add country-specific visa types
    if config["country"] == "usa":
        content += """
### B1/B2 Tourist and Business Visa
- **Purpose**: Tourism, business meetings, medical treatment
- **Duration**: Up to 6 months per visit
- **Requirements**: Strong ties to home country, sufficient funds
- **Processing Time**: 3-5 business days after interview

### F1 Student Visa
- **Purpose**: Academic studies at US institutions
- **Duration**: Length of study program
- **Requirements**: SEVIS fee, acceptance letter, financial proof
- **Processing Time**: 2-3 weeks

### H1B Work Visa
- **Purpose**: Specialty occupation employment
- **Duration**: Up to 6 years (renewable)
- **Requirements**: Job offer, labor certification, qualifications
- **Processing Time**: 6-12 months

### J1 Exchange Visitor Visa
- **Purpose**: Cultural exchange, research, teaching
- **Duration**: Varies by program type
- **Requirements**: Sponsorship, program acceptance
- **Processing Time**: 2-4 weeks\n"""
    elif config["country"] == "canada":
        content += """
### Temporary Resident Visa (TRV)
- **Purpose**: Tourism, visiting family, business
- **Duration**: Up to 6 months per visit
- **Requirements**: Proof of funds, travel insurance, return ticket
- **Processing Time**: 2-4 weeks

### Study Permit
- **Purpose**: Academic studies at Canadian institutions
- **Duration**: Length of study program
- **Requirements**: Acceptance letter, financial proof, medical exam
- **Processing Time**: 4-8 weeks

### Work Permit
- **Purpose**: Temporary employment in Canada
- **Duration**: Varies by work permit type
- **Requirements**: Job offer, LMIA (if required), qualifications
- **Processing Time**: 2-6 months

### Express Entry
- **Purpose**: Permanent residence through skilled worker program
- **Duration**: Permanent residence
- **Requirements**: Points-based system, language proficiency
- **Processing Time**: 6-12 months\n"""
    elif config["country"] == "australia":
        content += """
### Tourist Visa (Subclass 600)
- **Purpose**: Tourism, visiting family, business
- **Duration**: 3, 6, or 12 months
- **Requirements**: Sufficient funds, health insurance, return ticket
- **Processing Time**: 1-4 weeks

### Student Visa (Subclass 500)
- **Purpose**: Academic studies at Australian institutions
- **Duration**: Length of study program
- **Requirements**: COE, financial proof, health insurance
- **Processing Time**: 2-8 weeks

### Working Holiday Visa (Subclass 417/462)
- **Purpose**: Work and travel for young people
- **Duration**: 12 months (extendable)
- **Requirements**: Age 18-30, sufficient funds, health requirements
- **Processing Time**: 2-4 weeks

### Skilled Worker Visa (Subclass 189/190/491)
- **Purpose**: Permanent residence for skilled workers
- **Duration**: Permanent residence
- **Requirements**: Points-based system, occupation on list
- **Processing Time**: 6-12 months\n"""
    else:
        content += """
### Tourist Visa
- **Purpose**: Sightseeing, visiting friends/family
- **Duration**: Usually 30-90 days
- **Requirements**: Standard documentation package
- **Processing Time**: 5-15 business days

### Business Visa
- **Purpose**: Business meetings, conferences, trade
- **Duration**: Varies by country and purpose
- **Requirements**: Business invitation, company documents
- **Processing Time**: 7-20 business days

### Student Visa
- **Purpose**: Academic studies, language courses
- **Duration**: Length of study program
- **Requirements**: Acceptance letter, financial support proof
- **Processing Time**: 10-30 business days

### Work Visa
- **Purpose**: Employment, professional activities
- **Duration**: Employment contract period
- **Requirements**: Job offer, work permit, qualifications
- **Processing Time**: 15-45 business days\n"""

    content += f"""
## Entry Requirements for {country_name}

### Passport Validity
"""

    # Add country-specific passport requirements
    if config["country"] == "usa":
        content += """- **Minimum Validity**: 6 months beyond intended stay
- **Blank Pages**: At least 2 blank pages for visa stamps
- **Condition**: Passport must be in good condition
- **Machine Readable**: Must be machine-readable passport
- **Biometric**: E-passport preferred for visa waiver program\n"""
    elif config["country"] == "canada":
        content += """- **Minimum Validity**: 6 months beyond intended stay
- **Blank Pages**: At least 2 blank pages for visa stamps
- **Condition**: Passport must be in good condition
- **Biometric**: E-passport required for eTA applications
- **Validity**: Must be valid throughout entire stay\n"""
    elif config["country"] == "australia":
        content += """- **Minimum Validity**: 6 months beyond intended stay
- **Blank Pages**: At least 2 blank pages for visa stamps
- **Condition**: Passport must be in good condition
- **Biometric**: E-passport preferred for ETA applications
- **Validity**: Must be valid for entire stay period\n"""
    else:
        content += """- **Minimum Validity**: Usually 6 months beyond stay
- **Blank Pages**: At least 2-4 blank pages required
- **Condition**: Passport must be in good condition\n"""

    content += f"""
### Health Requirements
"""

    # Add country-specific health requirements
    if config["country"] == "usa":
        content += """- **COVID-19 Requirements**: Current vaccination requirements
- **Medical Insurance**: Recommended but not mandatory
- **Medical Examination**: Required for certain visa types
- **Vaccinations**: Check CDC recommendations
- **Health Declaration**: May be required at entry\n"""
    elif config["country"] == "canada":
        content += """- **Medical Examination**: Required for stays over 6 months
- **Health Insurance**: Recommended for all visitors
- **COVID-19 Requirements**: Current vaccination requirements
- **Biosecurity**: Strict quarantine procedures
- **Health Screening**: May be required at entry\n"""
    elif config["country"] == "australia":
        content += """- **Health Requirements**: Medical examination may be required
- **Quarantine**: Strict biosecurity and quarantine procedures
- **Health Insurance**: Recommended for all visitors
- **Vaccinations**: Check Australian health requirements
- **Medical Certificates**: May be required for certain visas\n"""
    else:
        content += """- **Vaccinations**: Check country-specific requirements
- **Medical Certificates**: May be required for certain visas
- **COVID-19**: Current pandemic-related requirements
- **Insurance**: Health insurance may be mandatory\n"""

    content += f"""
### Financial Requirements
"""

    # Add country-specific financial requirements
    if config["country"] == "usa":
        content += """- **Bank Statements**: 3-6 months of statements
- **Minimum Balance**: Sufficient to cover entire stay
- **Income Proof**: Employment letter, tax returns
- **Sponsorship**: If applicable, sponsor's financial documents
- **Return Ticket**: Proof of return travel arrangements\n"""
    elif config["country"] == "canada":
        content += """- **Bank Statements**: 4-6 months of statements
- **Minimum Balance**: CAD $10,000+ for extended stays
- **Income Proof**: Employment letter, pay stubs
- **Sponsorship**: If applicable, sponsor's financial documents
- **Travel Insurance**: Recommended for all visitors\n"""
    elif config["country"] == "australia":
        content += """- **Bank Statements**: 3-6 months of statements
- **Minimum Balance**: AUD $5,000+ for tourist visas
- **Income Proof**: Employment letter, tax returns
- **Sponsorship**: If applicable, sponsor's financial documents
- **Health Insurance**: Required for student visas\n"""
    else:
        content += """- **Bank Statements**: Usually 3-6 months of statements
- **Minimum Balance**: Varies by country and stay duration
- **Income Proof**: Employment letter, tax returns
- **Sponsorship**: If applicable, sponsor's financial documents\n"""

    content += f"""
## Application Fees for {country_name}
"""

    # Add country-specific fees
    if config["country"] == "usa":
        content += """
### Standard Visa Fees
- **B1/B2 Tourist/Business Visa**: $160 USD
- **F1 Student Visa**: $160 USD
- **H1B Work Visa**: $190 USD
- **J1 Exchange Visitor**: $160 USD
- **ESTA (Visa Waiver)**: $21 USD

### Additional Costs
- **SEVIS Fee**: $350 USD (for student/exchange visas)
- **Processing Fee**: Varies by location
- **Express Service**: Not available for most visa types
- **Document Translation**: $20-$50 per document\n"""
    elif config["country"] == "canada":
        content += """
### Standard Visa Fees
- **Temporary Resident Visa**: CAD $100
- **Study Permit**: CAD $150
- **Work Permit**: CAD $155
- **eTA (Electronic Travel Authorization)**: CAD $7
- **Biometric Fee**: CAD $85

### Additional Costs
- **Processing Fee**: Varies by application type
- **Medical Examination**: CAD $200-$400
- **Document Translation**: CAD $30-$80 per document
- **Travel Insurance**: CAD $50-$200\n"""
    elif config["country"] == "australia":
        content += """
### Standard Visa Fees
- **Tourist Visa (Subclass 600)**: AUD $145
- **Student Visa (Subclass 500)**: AUD $620
- **Working Holiday Visa**: AUD $495
- **ETA (Electronic Travel Authority)**: AUD $20
- **Skilled Worker Visa**: AUD $4,115

### Additional Costs
- **Health Examination**: AUD $200-$400
- **Police Certificate**: AUD $50-$100
- **Document Translation**: AUD $50-$100 per document
- **Health Insurance**: AUD $500-$1,500\n"""
    else:
        content += """
### Standard Fees
- **Tourist Visa**: $50-$200 USD (varies by country)
- **Business Visa**: $100-$300 USD
- **Student Visa**: $150-$400 USD
- **Work Visa**: $200-$500 USD

### Additional Costs
- **Processing Fee**: $20-$100 USD
- **Express Service**: 50-200% additional fee
- **Document Translation**: $20-$50 per document
- **Travel to Embassy**: Transportation and accommodation costs\n"""

    content += f"""
## Processing Times for {country_name}
"""

    # Add country-specific processing times
    if config["country"] == "usa":
        content += """
### Regular Processing Times
- **B1/B2 Tourist/Business**: 3-5 business days after interview
- **F1 Student Visa**: 2-3 weeks
- **H1B Work Visa**: 6-12 months
- **J1 Exchange Visitor**: 2-4 weeks
- **ESTA (Visa Waiver)**: Usually immediate

### Peak Season Delays
- **Summer (June-August)**: 1-2 weeks additional
- **Holiday Season**: 2-3 weeks additional
- **Backlog Periods**: May extend processing times\n"""
    elif config["country"] == "canada":
        content += """
### Regular Processing Times
- **Temporary Resident Visa**: 2-4 weeks
- **Study Permit**: 4-8 weeks
- **Work Permit**: 2-6 months
- **Express Entry**: 6-12 months
- **eTA**: Usually immediate

### Peak Season Delays
- **Summer (June-August)**: 1-2 weeks additional
- **Student Season**: 2-4 weeks additional for study permits
- **Holiday Season**: 1-2 weeks additional\n"""
    elif config["country"] == "australia":
        content += """
### Regular Processing Times
- **Tourist Visa**: 1-4 weeks
- **Student Visa**: 2-8 weeks
- **Working Holiday Visa**: 2-4 weeks
- **Skilled Worker Visa**: 6-12 months
- **ETA**: Usually immediate

### Peak Season Delays
- **Student Season**: 2-4 weeks additional for student visas
- **Holiday Season**: 1-2 weeks additional
- **Working Holiday Season**: 1-2 weeks additional\n"""
    else:
        content += """
### Regular Processing
- **Standard Time**: 5-15 business days
- **Peak Season**: 10-25 business days
- **Complex Cases**: 20-45 business days
- **Background Checks**: Additional 1-4 weeks

### Expedited Processing
- **Express Service**: 2-5 business days
- **Emergency Cases**: 1-3 business days
- **Additional Fee**: 100-300% of regular fee
- **Availability**: Not available for all visa types\n"""

    content += f"""
## Common Reasons for Rejection in {country_name}

### Documentation Issues
- **Incomplete Application**: Missing required documents
- **Expired Documents**: Passport, bank statements, etc.
- **Inconsistent Information**: Conflicting details in documents
- **Poor Quality Documents**: Illegible or damaged documents

### Financial Issues
- **Insufficient Funds**: Cannot support travel expenses
- **Unstable Income**: Irregular employment or earnings
- **Suspicious Activity**: Large sudden deposits or withdrawals
- **Lack of Ties**: No strong reason to return home

### Travel History Issues
- **Previous Violations**: Overstay, work on tourist visa
- **Suspicious Patterns**: Frequent short trips
- **Criminal Background**: Previous legal issues
- **Security Concerns**: National security considerations

## Tips for Successful Application to {country_name}

### Documentation Tips
- **Organize Carefully**: Use document checklist
- **Provide Originals**: Don't submit only photocopies
- **Translate Documents**: Use certified translation services
- **Be Consistent**: Ensure all information matches

### Financial Tips
- **Maintain Stable Balance**: Avoid large withdrawals before application
- **Show Multiple Sources**: Salary, investments, rental income
- **Include Extra Funds**: Show more than minimum required
- **Provide Explanation**: For any unusual financial activity

### Application Tips
- **Apply Early**: Allow plenty of time for processing
- **Be Honest**: Don't provide false information
- **Follow Instructions**: Read all requirements carefully
- **Keep Copies**: Maintain copies of all submitted documents

## Special Considerations for {country_name}

### First-Time Applicants
- **Use Professional Help**: Consider visa service agencies
- **Allow Extra Time**: Processing may take longer
- **Provide Extra Documentation**: Show strong ties to home country
- **Consider Guided Tour**: May improve approval chances

### Frequent Travelers
- **Maintain Good Record**: No overstays or violations
- **Keep Documentation**: Previous visas and travel records
- **Update Information**: Changes in employment, address, etc.
- **Consider Multiple Entry**: More convenient for regular visits

### Business Travelers
- **Separate Documentation**: Don't mix with tourism
- **Letter from Employer**: Explaining business purpose
- **Company Invitation**: If applicable
- **Conference Registration**: If attending events

## Contact Information for {country_name}

### Embassy/Consulate
- **Website**: Official government website
- **Phone**: Direct contact numbers
- **Email**: Visa inquiry email addresses
- **Address**: Physical location and hours

### Visa Service Centers
- **Authorized Agencies**: Professional visa services
- **Online Portals**: E-visa application systems
- **Customer Service**: Phone and email support
- **Operating Hours**: Vary by location

### Emergency Contacts
- **24/7 Hotline**: Emergency assistance
- **Travel Insurance**: Medical and emergency support
- **Local Embassy**: In-country assistance
- **Tourist Police**: Local law enforcement

{practical_name.replace("_", " ").title()} provides essential information for travelers visiting {country_name}. Proper preparation and understanding of requirements will ensure a smooth and successful travel experience."""

    return content


def create_generic_practical_content(
    practical_info: List[Dict[str, str]], practical_name: str
) -> str:
    """Create generic practical content"""
    content = f"""{practical_name.replace("_", " ").title()} is essential practical information for travelers. This comprehensive guide covers important travel requirements, procedures, and tips for a successful journey.\n\n## Overview\n{practical_name.replace("_", " ").title()} provides crucial information that travelers need to know before and during their trip. Understanding these requirements is essential for a smooth travel experience.\n\n## Key Information\n\n### Important Requirements\n"""

    # Add practical information
    for i, info in enumerate(practical_info[:10], 1):  # Top 10 items
        content += f"- **{info['name']}**: Important practical information\n"

    content += f"""
## Essential Requirements

### Documentation
- **Passport**: Valid travel document
- **Visa**: If required for your nationality
- **Insurance**: Travel and health insurance
- **Vaccinations**: Required immunizations
- **Medical Certificates**: If applicable

### Financial Requirements
- **Sufficient Funds**: Proof of financial means
- **Bank Statements**: Recent financial records
- **Credit Cards**: International payment methods
- **Emergency Funds**: Backup financial resources

### Health Requirements
- **Vaccinations**: Required immunizations
- **Medical Insurance**: Health coverage
- **Prescriptions**: Medication documentation
- **Health Declarations**: If required

## Travel Preparation

### Before You Go
- **Research Requirements**: Check all entry requirements
- **Gather Documents**: Prepare all necessary paperwork
- **Book Accommodations**: Confirm lodging arrangements
- **Purchase Insurance**: Travel and health coverage
- **Check Health Requirements**: Vaccinations and medical needs

### Documentation Checklist
- **Passport**: Valid for required duration
- **Visa**: If required for your nationality
- **Insurance**: Travel and health coverage
- **Vaccinations**: Required immunizations
- **Financial Proof**: Bank statements, employment letters
- **Travel Itinerary**: Flight and accommodation details

## Important Tips

### Documentation Tips
- **Make Copies**: Keep copies of all important documents
- **Digital Backups**: Store documents electronically
- **Translation**: Have documents translated if needed
- **Validity**: Ensure all documents are current

### Financial Tips
- **Multiple Sources**: Carry various payment methods
- **Emergency Funds**: Keep backup financial resources
- **Exchange Rates**: Understand local currency
- **Banking**: Inform your bank of travel plans

### Health Tips
- **Medical Check**: Visit doctor before travel
- **Prescriptions**: Bring necessary medications
- **Insurance**: Ensure adequate health coverage
- **Emergency Contacts**: Know local emergency numbers

## Common Issues and Solutions

### Documentation Problems
- **Expired Documents**: Renew before travel
- **Missing Documents**: Apply for replacements early
- **Translation Issues**: Use certified translation services
- **Validity Problems**: Check all expiration dates

### Financial Issues
- **Insufficient Funds**: Plan budget carefully
- **Payment Problems**: Carry multiple payment methods
- **Currency Issues**: Research exchange rates
- **Banking Problems**: Inform banks of travel plans

### Health Issues
- **Missing Vaccinations**: Get required immunizations
- **Medication Problems**: Bring prescriptions and documentation
- **Insurance Issues**: Ensure adequate coverage
- **Medical Emergencies**: Know local healthcare options

## Emergency Information

### Emergency Contacts
- **Local Emergency**: Country-specific emergency numbers
- **Embassy/Consulate**: Your country's diplomatic mission
- **Travel Insurance**: Emergency assistance hotline
- **Tourist Police**: Local law enforcement

### Medical Emergencies
- **Local Hospitals**: Know nearest medical facilities
- **Pharmacies**: Location of pharmacies
- **Emergency Services**: Ambulance and medical transport
- **Insurance Claims**: How to file insurance claims

### Travel Emergencies
- **Lost Documents**: What to do if documents are lost
- **Stolen Items**: Reporting theft and getting help
- **Flight Issues**: Dealing with flight problems
- **Accommodation Problems**: Resolving lodging issues

## Contact Information

### Government Resources
- **Embassy/Consulate**: Official government assistance
- **Tourist Information**: Local tourism offices
- **Immigration**: Border and entry requirements
- **Customs**: Import/export regulations

### Travel Services
- **Travel Agencies**: Professional travel assistance
- **Insurance Companies**: Travel and health coverage
- **Transportation**: Local transport information
- **Accommodation**: Lodging assistance

{practical_name.replace("_", " ").title()} provides essential information for successful travel. Proper preparation and understanding of requirements will ensure a smooth and enjoyable journey."""

    return content


def save_practical_data(data: Dict[str, Any], filename: str, info_type: str):
    """Save practical data to JSON file in appropriate info type subfolder"""
    try:
        # Create base practical folder
        base_folder = "practical"
        type_folder = os.path.join(base_folder, info_type)
        os.makedirs(type_folder, exist_ok=True)
        file_path = os.path.join(type_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


def main():
    """Main function to extract practical information"""
    if len(sys.argv) < 2:
        print("Usage: python extract_practical.py <practical_name>")
        print("Available practical information:", list(PRACTICAL_CONFIGS.keys()))
        return

    practical_name = sys.argv[1]
    print(f"Starting {practical_name} practical information extraction...")

    # Extract practical information
    practical_info = scrape_practical_info(practical_name)

    if not practical_info:
        print(f"Failed to extract practical information for {practical_name}")
        return

    # Create and save data
    config = PRACTICAL_CONFIGS[practical_name.lower()]
    data = create_practical_data(practical_info, practical_name, config)
    filename = f"{practical_name.lower().replace(' ', '_').replace(',', '')}.json"
    save_practical_data(data, filename, config["info_type"])

    print(
        f"Extraction complete! Found {len(practical_info)} practical items for {practical_name}."
    )


if __name__ == "__main__":
    main()
