#!/usr/bin/env python3
"""
Travel Data Generator Script

A developer-friendly script to generate travel knowledge data files for the RAG system.
Supports creating structured JSON files from various sources including online data,
manual input, and template generation.

Usage:
    python generate_travel_data.py --help
    python generate_travel_data.py --template destination
    python generate_travel_data.py --interactive
    python generate_travel_data.py --from-url "https://example.com/travel-guide"
"""

import json
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import requests
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TravelKnowledgeTemplate:
    """Template for travel knowledge data"""
    id: str
    title: str
    content: str
    category: str
    location: Optional[str] = None
    tags: List[str] = None
    language: str = "en"
    source: Optional[Dict[str, Any]] = None
    last_updated: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class TravelDataGenerator:
    """Main travel data generator class"""
    
    def __init__(self, output_dir: str = "documents"):
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Category definitions
        self.categories = {
            "destinations": "Travel destinations, attractions, and cultural information",
            "transportation": "Flights, trains, buses, and other transportation options", 
            "accommodation": "Hotels, hostels, and other lodging options",
            "activities": "Tourist activities, tours, and experiences",
            "practical": "Visas, currency, weather, safety, and other practical info"
        }
        
        # Subcategory mappings
        self.subcategories = {
            "destinations": ["asia", "europe", "america", "africa", "oceania"],
            "transportation": ["flights", "trains", "buses", "local_transport"],
            "accommodation": ["hotels", "hostels", "bnb", "resorts"],
            "activities": ["sightseeing", "adventure", "cultural", "entertainment"],
            "practical": ["visa", "currency", "weather", "safety", "health"]
        }
        
        logger.info(f"Travel data generator initialized, output directory: {self.output_dir}")
    
    def generate_template(self, category: str, subcategory: str = None) -> Dict[str, Any]:
        """Generate a template for a specific category"""
        if category not in self.categories:
            raise ValueError(f"Invalid category. Available: {list(self.categories.keys())}")
        
        templates = {
            "destinations": self._get_destination_template,
            "transportation": self._get_transportation_template,
            "accommodation": self._get_accommodation_template,
            "activities": self._get_activities_template,
            "practical": self._get_practical_template
        }
        
        template_func = templates[category]
        return template_func(subcategory)
    
    def _get_destination_template(self, subcategory: str = "asia") -> Dict[str, Any]:
        """Generate destination template"""
        return {
            "id": "destination_example_001",
            "title": "Example Destination - City Name",
            "content": """This is a comprehensive travel guide for [DESTINATION_NAME].

## Overview
[DESTINATION_NAME] is a [beautiful/historic/modern] city located in [COUNTRY]. Known for [MAIN_ATTRACTIONS], it offers visitors [UNIQUE_EXPERIENCES].

## Main Attractions
1. **[ATTRACTION_1]**: [Description and visiting info]
2. **[ATTRACTION_2]**: [Description and visiting info]
3. **[ATTRACTION_3]**: [Description and visiting info]

## Best Time to Visit
- **Peak Season**: [MONTHS] - [DESCRIPTION]
- **Shoulder Season**: [MONTHS] - [DESCRIPTION]
- **Off Season**: [MONTHS] - [DESCRIPTION]

## Getting There
- **By Air**: [Airport info and flight details]
- **By Train**: [Train station and route info]
- **By Car**: [Driving directions and parking info]

## Budget Guide
- **Budget Range**: $[X]-[Y] per day
- **Accommodation**: $[X]-[Y] per night
- **Food**: $[X]-[Y] per day
- **Activities**: $[X]-[Y] per day
- **Transportation**: $[X]-[Y] per day

## Local Tips
- [TIP_1]
- [TIP_2]
- [TIP_3]

## Cultural Notes
[Important cultural information, customs, etiquette]

## Emergency Information
- Emergency Number: [NUMBER]
- Tourist Police: [NUMBER]
- Embassy Contact: [CONTACT_INFO]""",
            "category": "destinations",
            "subcategory": subcategory,
            "location": "City, Country",
            "tags": ["city_name", "country", "tourism", "attractions"],
            "language": "en",
            "source": {
                "name": "Official Tourism Board",
                "url": "https://example.com/travel-guide",
                "reliability_score": 0.9
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_transportation_template(self, subcategory: str = "flights") -> Dict[str, Any]:
        """Generate transportation template"""
        return {
            "id": "transport_example_001",
            "title": "Transportation Guide - [ROUTE/SERVICE]",
            "content": """Complete transportation guide for [TRANSPORTATION_TYPE] between [ORIGIN] and [DESTINATION].

## Service Overview
[SERVICE_NAME] operates [FREQUENCY] between [ORIGIN] and [DESTINATION]. Journey time is approximately [DURATION].

## Booking Information
- **Advance Booking**: [RECOMMENDATION]
- **Online Booking**: [WEBSITE_LINKS]
- **Phone Booking**: [PHONE_NUMBERS]
- **In-Person**: [OFFICE_LOCATIONS]

## Pricing
- **Economy**: $[PRICE_RANGE]
- **Business**: $[PRICE_RANGE]
- **First Class**: $[PRICE_RANGE]
- **Peak Season Surcharge**: [PERCENTAGE]%

## Schedule
- **Departure Times**: [TIMES]
- **Frequency**: [DAILY/WEEKLY_FREQUENCY]
- **Seasonal Variations**: [DESCRIPTIONS]

## Stations/Terminals
### [ORIGIN_STATION]
- **Address**: [FULL_ADDRESS]
- **Facilities**: [AMENITIES_LIST]
- **Getting There**: [TRANSPORT_OPTIONS]

### [DESTINATION_STATION]
- **Address**: [FULL_ADDRESS]
- **Facilities**: [AMENITIES_LIST]
- **Getting There**: [TRANSPORT_OPTIONS]

## Tips for Travelers
- [TIP_1]
- [TIP_2]
- [TIP_3]

## Luggage Policy
- **Carry-on**: [RESTRICTIONS]
- **Checked Luggage**: [ALLOWANCE_AND_FEES]
- **Special Items**: [POLICIES]""",
            "category": "transportation",
            "subcategory": subcategory,
            "location": "Route: Origin to Destination",
            "tags": ["transportation", subcategory, "booking", "schedule"],
            "language": "en",
            "source": {
                "name": "Official Transportation Provider",
                "url": "https://example.com/transportation",
                "reliability_score": 0.95
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_accommodation_template(self, subcategory: str = "hotels") -> Dict[str, Any]:
        """Generate accommodation template"""
        return {
            "id": "accommodation_example_001",
            "title": "Accommodation Guide - [PROPERTY_NAME]",
            "content": """Comprehensive accommodation guide for [PROPERTY_NAME] in [LOCATION].

## Property Overview
[PROPERTY_NAME] is a [STAR_RATING]-star [PROPERTY_TYPE] located in [NEIGHBORHOOD], [CITY]. Built in [YEAR], it offers [NUMBER] rooms and suites with [STYLE_DESCRIPTION].

## Room Types & Rates
### [ROOM_TYPE_1]
- **Size**: [SQUARE_METERS] sqm
- **Occupancy**: Up to [NUMBER] guests
- **Amenities**: [AMENITIES_LIST]
- **Rate**: $[PRICE_RANGE] per night

### [ROOM_TYPE_2]
- **Size**: [SQUARE_METERS] sqm
- **Occupancy**: Up to [NUMBER] guests
- **Amenities**: [AMENITIES_LIST]
- **Rate**: $[PRICE_RANGE] per night

## Hotel Facilities
- **Dining**: [RESTAURANT_DESCRIPTIONS]
- **Fitness**: [GYM_SPA_DETAILS]
- **Business**: [MEETING_ROOM_DETAILS]
- **Recreation**: [POOL_ACTIVITIES]
- **Services**: [CONCIERGE_SERVICES]

## Location & Transportation
- **Address**: [FULL_ADDRESS]
- **Nearest Airport**: [AIRPORT_NAME] - [DISTANCE/TIME]
- **Public Transport**: [METRO_BUS_DETAILS]
- **Parking**: [PARKING_OPTIONS_AND_FEES]

## Nearby Attractions
- **[ATTRACTION_1]**: [DISTANCE] away
- **[ATTRACTION_2]**: [DISTANCE] away
- **[ATTRACTION_3]**: [DISTANCE] away

## Booking Information
- **Check-in**: [TIME]
- **Check-out**: [TIME]
- **Cancellation**: [POLICY_DETAILS]
- **Payment**: [ACCEPTED_METHODS]
- **Deposit**: [DEPOSIT_REQUIREMENTS]

## Guest Reviews Summary
- **Overall Rating**: [RATING]/5
- **Cleanliness**: [RATING]/5
- **Service**: [RATING]/5
- **Location**: [RATING]/5
- **Value**: [RATING]/5

## Contact Information
- **Phone**: [PHONE_NUMBER]
- **Email**: [EMAIL_ADDRESS]
- **Website**: [WEBSITE_URL]""",
            "category": "accommodation",
            "subcategory": subcategory,
            "location": "City, Country",
            "tags": ["accommodation", subcategory, "booking", "amenities"],
            "language": "en",
            "source": {
                "name": "Hotel Official Website",
                "url": "https://example.com/hotel",
                "reliability_score": 0.9
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_activities_template(self, subcategory: str = "sightseeing") -> Dict[str, Any]:
        """Generate activities template"""
        return {
            "id": "activity_example_001", 
            "title": "Activity Guide - [ACTIVITY_NAME]",
            "content": """Complete guide for [ACTIVITY_NAME] in [LOCATION].

## Activity Overview
[ACTIVITY_NAME] is a [TYPE_OF_ACTIVITY] that [DESCRIPTION]. Suitable for [TARGET_AUDIENCE], this [DURATION] experience offers [HIGHLIGHTS].

## What's Included
- [INCLUSION_1]
- [INCLUSION_2]
- [INCLUSION_3]
- [INCLUSION_4]

## Schedule & Duration
- **Duration**: [TOTAL_TIME]
- **Start Times**: [AVAILABLE_TIMES]
- **Meeting Point**: [LOCATION_DETAILS]
- **End Point**: [LOCATION_DETAILS]

## Pricing
- **Adult**: $[PRICE]
- **Child** ([AGE_RANGE]): $[PRICE]
- **Senior** ([AGE_RANGE]): $[PRICE]
- **Group Discount**: [DISCOUNT_DETAILS]

## Requirements & Restrictions
- **Age Limit**: [AGE_REQUIREMENTS]
- **Physical Requirements**: [FITNESS_LEVEL]
- **What to Bring**: [ITEMS_LIST]
- **What to Wear**: [CLOTHING_RECOMMENDATIONS]
- **Weather Policy**: [CANCELLATION_POLICY]

## Booking Information
- **Advance Booking**: [REQUIRED/RECOMMENDED]
- **Cancellation**: [POLICY_DETAILS]
- **Languages Available**: [LANGUAGE_LIST]
- **Group Size**: Maximum [NUMBER] people

## Safety Information
- [SAFETY_POINT_1]
- [SAFETY_POINT_2]
- [SAFETY_POINT_3]

## Tips for Best Experience
- [TIP_1]
- [TIP_2]
- [TIP_3]

## Contact & Booking
- **Phone**: [PHONE_NUMBER]
- **Email**: [EMAIL_ADDRESS]
- **Website**: [WEBSITE_URL]
- **Address**: [MEETING_POINT_ADDRESS]""",
            "category": "activities",
            "subcategory": subcategory,
            "location": "City, Country",
            "tags": ["activities", subcategory, "booking", "tour"],
            "language": "en",
            "source": {
                "name": "Activity Provider",
                "url": "https://example.com/activity",
                "reliability_score": 0.85
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_practical_template(self, subcategory: str = "visa") -> Dict[str, Any]:
        """Generate practical information template"""
        return {
            "id": "practical_example_001",
            "title": "Practical Guide - [TOPIC] for [COUNTRY]",
            "content": """Essential practical information about [TOPIC] for travelers to [COUNTRY].

## Overview
[TOPIC_DESCRIPTION] for [COUNTRY]. This guide covers [SCOPE_OF_INFORMATION].

## Requirements
### For [NATIONALITY_1] Citizens
- [REQUIREMENT_1]
- [REQUIREMENT_2]
- [REQUIREMENT_3]

### For [NATIONALITY_2] Citizens
- [REQUIREMENT_1]
- [REQUIREMENT_2]
- [REQUIREMENT_3]

## Application Process
1. **Step 1**: [DETAILED_DESCRIPTION]
2. **Step 2**: [DETAILED_DESCRIPTION]
3. **Step 3**: [DETAILED_DESCRIPTION]
4. **Step 4**: [DETAILED_DESCRIPTION]

## Required Documents
- **Primary Documents**:
  - [DOCUMENT_1]: [SPECIFICATIONS]
  - [DOCUMENT_2]: [SPECIFICATIONS]
  - [DOCUMENT_3]: [SPECIFICATIONS]

- **Supporting Documents**:
  - [DOCUMENT_1]: [SPECIFICATIONS]
  - [DOCUMENT_2]: [SPECIFICATIONS]

## Fees and Processing Time
- **Standard Processing**: [TIME] - $[FEE]
- **Express Processing**: [TIME] - $[FEE]
- **Emergency Processing**: [TIME] - $[FEE]

## Where to Apply
### [LOCATION_1]
- **Address**: [FULL_ADDRESS]
- **Hours**: [OPERATING_HOURS]
- **Phone**: [PHONE_NUMBER]
- **Email**: [EMAIL_ADDRESS]

### [LOCATION_2]
- **Address**: [FULL_ADDRESS]
- **Hours**: [OPERATING_HOURS]
- **Phone**: [PHONE_NUMBER]
- **Email**: [EMAIL_ADDRESS]

## Important Notes
- [IMPORTANT_NOTE_1]
- [IMPORTANT_NOTE_2]
- [IMPORTANT_NOTE_3]

## Common Mistakes to Avoid
- [MISTAKE_1]
- [MISTAKE_2]
- [MISTAKE_3]

## Useful Links
- [OFFICIAL_WEBSITE]: [URL]
- [ADDITIONAL_RESOURCE]: [URL]
- [FAQ_PAGE]: [URL]""",
            "category": "practical",
            "subcategory": subcategory,
            "location": "Country",
            "tags": ["practical", subcategory, "requirements", "official"],
            "language": "en",
            "source": {
                "name": "Official Government Source",
                "url": "https://example.com/official-info",
                "reliability_score": 0.98
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def interactive_creation(self) -> Dict[str, Any]:
        """Interactive mode for creating travel knowledge"""
        print("\n🌍 Interactive Travel Knowledge Creator")
        print("=" * 50)
        
        # Get basic information
        print("\n1. Basic Information")
        title = input("Enter the title: ").strip()
        if not title:
            title = "New Travel Knowledge Item"
        
        # Select category
        print(f"\n2. Select Category:")
        for i, (cat_id, desc) in enumerate(self.categories.items(), 1):
            print(f"   {i}. {cat_id.title()}: {desc}")
        
        while True:
            try:
                cat_choice = int(input("Enter category number (1-5): "))
                if 1 <= cat_choice <= 5:
                    category = list(self.categories.keys())[cat_choice - 1]
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Select subcategory
        subcats = self.subcategories.get(category, [])
        subcategory = None
        if subcats:
            print(f"\n3. Select Subcategory for {category}:")
            for i, subcat in enumerate(subcats, 1):
                print(f"   {i}. {subcat}")
            
            while True:
                try:
                    subcat_choice = int(input(f"Enter subcategory number (1-{len(subcats)}): "))
                    if 1 <= subcat_choice <= len(subcats):
                        subcategory = subcats[subcat_choice - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(subcats)}")
                except ValueError:
                    print("Please enter a valid number")
        
        # Get location
        location = input(f"\n4. Enter location (e.g., 'Paris, France'): ").strip()
        
        # Get tags
        tags_input = input("\n5. Enter tags (comma-separated): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Get content
        print("\n6. Content Creation")
        print("Choose content creation method:")
        print("   1. Use template (recommended)")
        print("   2. Enter custom content")
        print("   3. Load from file")
        
        while True:
            try:
                content_choice = int(input("Enter choice (1-3): "))
                if 1 <= content_choice <= 3:
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        if content_choice == 1:
            # Use template
            template_data = self.generate_template(category, subcategory)
            content = template_data["content"]
            print("✅ Template content loaded. You can edit the generated file later.")
        elif content_choice == 2:
            # Custom content
            print("Enter your content (press Ctrl+D when finished):")
            content_lines = []
            try:
                while True:
                    line = input()
                    content_lines.append(line)
            except EOFError:
                content = "\n".join(content_lines)
        else:
            # Load from file
            file_path = input("Enter file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print("✅ Content loaded from file")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                content = "Content loaded from file (edit as needed)"
        
        # Generate ID
        knowledge_id = self._generate_id(title, category, location)
        
        # Create knowledge data
        knowledge_data = {
            "id": knowledge_id,
            "title": title,
            "content": content,
            "category": category,
            "location": location or None,
            "tags": tags,
            "language": "en",
            "source": {
                "name": "Manual Entry",
                "url": None,
                "reliability_score": 0.8
            },
            "last_updated": datetime.now().isoformat()
        }
        
        if subcategory:
            knowledge_data["subcategory"] = subcategory
        
        return knowledge_data
    
    def save_knowledge_data(self, knowledge_data: Dict[str, Any], filename: str = None) -> Path:
        """Save knowledge data to file"""
        if not filename:
            # Generate filename from category and title
            category = knowledge_data.get("category", "general")
            subcategory = knowledge_data.get("subcategory", "")
            title_clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in knowledge_data.get("title", "unnamed"))
            
            if subcategory:
                dir_path = self.output_dir / category / subcategory
                filename = f"{title_clean.lower()}.json"
            else:
                dir_path = self.output_dir / category
                filename = f"{title_clean.lower()}.json"
        else:
            dir_path = self.output_dir
        
        # Create directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = dir_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge data saved to: {file_path}")
        return file_path
    
    def create_sample_data(self):
        """Create comprehensive sample data for all categories"""
        print("🚀 Creating sample travel knowledge data...")
        
        samples = [
            # Destinations
            {
                "id": "tokyo_travel_guide",
                "title": "Tokyo Complete Travel Guide",
                "category": "destinations",
                "subcategory": "asia",
                "location": "Tokyo, Japan",
                "tags": ["Tokyo", "Japan", "Asia", "Megacity", "Culture", "Technology"],
                "content": """Tokyo, Japan's bustling capital, blends ultramodern and traditional elements perfectly. This comprehensive guide covers everything you need to know for an amazing Tokyo experience.

## Overview
Tokyo is one of the world's most vibrant cities, home to over 13 million people. From neon-lit skyscrapers to peaceful temples, Tokyo offers endless discoveries for every type of traveler.

## Top Attractions

### Must-Visit Areas
- **Shibuya**: Famous for the world's busiest pedestrian crossing
- **Harajuku**: Youth culture and fashion district
- **Asakusa**: Traditional area with Senso-ji Temple
- **Ginza**: Upscale shopping and dining district
- **Shinjuku**: Entertainment and business hub

### Iconic Landmarks
- **Tokyo Tower**: 333m red steel tower with observation decks
- **Tokyo Skytree**: World's second-tallest structure at 634m
- **Meiji Shrine**: Peaceful Shinto shrine in Shibuya
- **Imperial Palace**: Primary residence of the Emperor
- **Tsukiji Outer Market**: Fresh seafood and street food

## Best Time to Visit
- **Spring (March-May)**: Cherry blossom season, mild weather
- **Autumn (September-November)**: Comfortable temperatures, fall colors
- **Summer (June-August)**: Hot and humid, festival season
- **Winter (December-February)**: Cold but clear, fewer crowds

## Transportation
- **JR Pass**: Unlimited travel on JR trains for tourists
- **Tokyo Metro**: Extensive subway system covering the city
- **IC Cards**: Suica or Pasmo for convenient travel
- **Taxi**: Expensive but available everywhere

## Budget Guide
- **Budget**: $50-80/day (hostels, convenience store meals)
- **Mid-range**: $100-150/day (business hotels, restaurant meals)
- **Luxury**: $200+/day (high-end hotels and dining)

## Cultural Tips
- Bow when greeting people
- Remove shoes when entering homes/temples
- Don't eat or drink while walking
- Keep voices low on public transport
- Tipping is not customary

## Emergency Information
- **Emergency Number**: 119 (Fire/Ambulance), 110 (Police)
- **Tourist Hotline**: 050-3816-2787
- **English Support**: Available at major stations""",
                "source": {
                    "name": "Japan National Tourism Organization",
                    "url": "https://www.jnto.go.jp",
                    "reliability_score": 0.95
                }
            },
            
            # Transportation
            {
                "id": "shinkansen_guide",
                "title": "Japan Shinkansen (Bullet Train) Complete Guide",
                "category": "transportation",
                "subcategory": "trains",
                "location": "Japan",
                "tags": ["Shinkansen", "Bullet Train", "Japan", "Transportation", "JR Pass"],
                "content": """The Shinkansen bullet train is Japan's high-speed rail network, connecting major cities at speeds up to 320 km/h. This guide covers everything you need to know about riding the Shinkansen.

## Overview
The Shinkansen has been operating since 1964 and is renowned for its punctuality, safety, and comfort. With multiple lines serving different regions, it's the fastest way to travel between Japan's major cities.

## Major Routes
### Tokaido Shinkansen (Tokyo-Osaka)
- **Journey Time**: 2 hours 15 minutes (Nozomi)
- **Major Stops**: Tokyo, Shinagawa, Yokohama, Nagoya, Kyoto, Osaka
- **Frequency**: Every 10 minutes during peak hours

### Tohoku Shinkansen (Tokyo-Aomori)
- **Journey Time**: 3 hours 10 minutes to Sendai
- **Major Stops**: Tokyo, Ueno, Sendai, Morioka, Hachinohe, Shin-Aomori
- **Frequency**: 1-2 trains per hour

### Hokuriku Shinkansen (Tokyo-Kanazawa)
- **Journey Time**: 2 hours 30 minutes
- **Major Stops**: Tokyo, Ueno, Takasaki, Karuizawa, Nagano, Toyama, Kanazawa
- **Frequency**: 1 train per hour

## Train Types
- **Nozomi**: Fastest, limited stops (not covered by JR Pass)
- **Hikari**: Fast, some intermediate stops
- **Kodama**: Local, stops at all stations

## Seat Classes
### Ordinary Cars (Non-Reserved)
- **Price**: Standard fare
- **Features**: Basic seating, first-come-first-served

### Ordinary Cars (Reserved)
- **Price**: +530 yen reservation fee
- **Features**: Guaranteed seat, same comfort as non-reserved

### Green Cars (First Class)
- **Price**: +4,000-5,000 yen premium
- **Features**: Larger seats, more legroom, complimentary refreshments

### Gran Class (Premium)
- **Price**: +8,000-10,000 yen premium
- **Features**: Luxury seats, premium meals, personal service

## Booking and Tickets
### JR Pass Holders
- Free rides on Hikari and Kodama trains
- Seat reservations recommended for busy periods
- Cannot use Nozomi or Mizuho trains

### Individual Tickets
- Purchase at JR stations or online
- IC cards cannot be used for Shinkansen
- Tickets include base fare + express surcharge

## Travel Tips
- Arrive at the platform 5 minutes early
- Queue in designated areas on the platform
- Seat numbers are marked on the platform
- Large luggage requires advance reservation
- Food and drinks can be purchased onboard

## Etiquette
- Keep conversations quiet
- Put phones on silent mode
- Recline seats carefully
- Don't eat strong-smelling food
- Clean up after yourself""",
                "source": {
                    "name": "JR East Official Guide",
                    "url": "https://www.jreast.co.jp",
                    "reliability_score": 0.98
                }
            }
        ]
        
        for sample in samples:
            # Add missing fields
            sample["language"] = "en"
            sample["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            file_path = self.save_knowledge_data(sample)
            print(f"✅ Created: {file_path}")
        
        print(f"\n🎉 Sample data creation completed!")
        print(f"📁 Files saved to: {self.output_dir}")
    
    def _generate_id(self, title: str, category: str, location: str = "") -> str:
        """Generate a unique ID for knowledge item"""
        # Create a clean version of the title
        clean_title = "".join(c if c.isalnum() else "_" for c in title.lower())
        clean_title = clean_title[:30]  # Limit length
        
        # Add category prefix
        prefix = category[:4]
        
        # Add location if provided
        if location:
            location_part = "".join(c if c.isalnum() else "_" for c in location.split(",")[0].lower())[:10]
            return f"{prefix}_{location_part}_{clean_title}"
        else:
            return f"{prefix}_{clean_title}"


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate travel knowledge data for the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample data
  python generate_travel_data.py --sample

  # Interactive mode
  python generate_travel_data.py --interactive

  # Generate template
  python generate_travel_data.py --template destinations --subcategory asia

  # Generate multiple templates
  python generate_travel_data.py --batch-templates
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        default="documents",
        help="Output directory for generated files (default: documents)"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create comprehensive sample data"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true", 
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--template",
        choices=["destinations", "transportation", "accommodation", "activities", "practical"],
        help="Generate template for specific category"
    )
    
    parser.add_argument(
        "--subcategory",
        help="Subcategory for template generation"
    )
    
    parser.add_argument(
        "--batch-templates",
        action="store_true",
        help="Generate templates for all categories"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TravelDataGenerator(args.output_dir)
    
    try:
        if args.sample:
            generator.create_sample_data()
        
        elif args.interactive:
            knowledge_data = generator.interactive_creation()
            file_path = generator.save_knowledge_data(knowledge_data)
            print(f"\n✅ Knowledge data created: {file_path}")
        
        elif args.template:
            template_data = generator.generate_template(args.template, args.subcategory)
            filename = f"template_{args.template}"
            if args.subcategory:
                filename += f"_{args.subcategory}"
            filename += ".json"
            
            file_path = generator.save_knowledge_data(template_data, filename)
            print(f"✅ Template created: {file_path}")
            print("\n📝 Edit the template file to add your specific travel information")
        
        elif args.batch_templates:
            print("🚀 Generating templates for all categories...")
            for category in generator.categories.keys():
                for subcategory in generator.subcategories.get(category, [None]):
                    template_data = generator.generate_template(category, subcategory)
                    filename = f"template_{category}"
                    if subcategory:
                        filename += f"_{subcategory}"
                    filename += ".json"
                    
                    file_path = generator.save_knowledge_data(template_data, filename)
                    print(f"✅ Created: {file_path}")
            
            print("\n🎉 All templates generated successfully!")
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 