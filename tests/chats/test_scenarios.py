"""
Test Scenarios Configuration for Advanced Chat API Testing
Defines scenarios for both single-query and multi-query tests
"""

# Single-query test scenarios (different sessions, single question each)
SINGLE_QUERY_SCENARIOS = [
    {
        "name": "london_7_days_budget",
        "message": "Plan a 7-day trip to London for 2 people with a budget of $3000",
        "expected_keywords": ["london", "7 days", "budget", "£", "$3000", "attractions", "hotels"]
    },
    {
        "name": "paris_romantic_weekend",
        "message": "Plan a romantic weekend in Paris for 2 people, focusing on fine dining and cultural experiences",
        "expected_keywords": ["paris", "romantic", "weekend", "dining", "culture", "museums"]
    },
    {
        "name": "tokyo_business_trip",
        "message": "I need a 3-day business trip to Tokyo with meetings in Shibuya and Shinjuku areas",
        "expected_keywords": ["tokyo", "business", "shibuya", "shinjuku", "meetings", "hotels"]
    },
    {
        "name": "family_singapore",
        "message": "Plan a 5-day family trip to Singapore with 2 adults and 2 children (ages 8 and 12)",
        "expected_keywords": ["singapore", "family", "children", "kids", "attractions", "5 days"]
    },
    {
        "name": "budget_backpacking_europe",
        "message": "Create a 14-day backpacking itinerary across Europe for a student with a $1500 budget",
        "expected_keywords": ["europe", "backpacking", "budget", "student", "$1500", "14 days"]
    },
    {
        "name": "luxury_dubai",
        "message": "Plan a luxury 4-day trip to Dubai for celebrating our anniversary, budget is flexible",
        "expected_keywords": ["dubai", "luxury", "anniversary", "flexible", "4 days"]
    },
    {
        "name": "solo_travel_barcelona",
        "message": "I'm planning a solo trip to Barcelona for 6 days, interested in art, architecture, and nightlife",
        "expected_keywords": ["barcelona", "solo", "art", "architecture", "nightlife", "6 days"]
    },
    {
        "name": "winter_ski_trip",
        "message": "Plan a 5-day ski trip to the Swiss Alps for 4 people in February",
        "expected_keywords": ["ski", "swiss alps", "winter", "february", "5 days", "4 people"]
    },
    {
        "name": "german_summer_trip",
        "message": "Plan a 7-day trip to Berlin and Munich for 2 people with a budget of $7000 in this summer",
        "expected_keywords": ["summer", "berlin", "munich", "$7000", "7 days", "2 people"]
    }
]

# Multi-query test scenarios (same session, multiple questions)
MULTI_QUERY_SCENARIOS = [
    {
        "title": "Detailed Tokyo Trip Planning",
        "description": "Comprehensive planning for a Tokyo trip with multiple detailed questions",
        "queries": [
            "I'm planning a 7-day trip to Tokyo in March with my partner. We're interested in traditional culture, modern attractions, and great food. Budget is around $4000 total.",
            "What are the best neighborhoods to stay in Tokyo for easy access to both traditional sites and modern attractions?",
            "Can you recommend specific restaurants for authentic Japanese cuisine? We want to try sushi, ramen, tempura, and kaiseki.",
            "What's the best way to get around Tokyo? Should we get a JR Pass?",
            "Can you create a day-by-day itinerary that balances temples, museums, shopping, and food experiences?",
            "What should we pack for Tokyo in March? Any cultural etiquette we should be aware of?",
            "Are there any day trips from Tokyo you'd recommend? Maybe to Mount Fuji or nearby cities?"
        ]
    },
    {
        "title": "European Multi-City Adventure",
        "description": "Planning a comprehensive European trip across multiple cities",
        "queries": [
            "I want to plan a 21-day European trip visiting Paris, Amsterdam, Berlin, Prague, and Vienna. Budget is $6000 for 2 people.",
            "How should I organize the order of cities to minimize travel time and costs?",
            "What's the best transportation method between these cities? Train vs flights?",
            "Can you suggest 3-4 days worth of activities for each city, focusing on history, art, and local culture?",
            "What are the accommodation recommendations for each city? Mix of budget and mid-range options.",
            "How much should I budget for food in each city? Any must-try local specialties?",
            "What's the best time of year for this trip considering weather and tourist crowds?"
        ]
    },
    {
        "title": "Family Beach Vacation Planning",
        "description": "Detailed planning for a family beach vacation with children",
        "queries": [
            "Plan a 10-day beach vacation for a family of 4 (2 adults, kids aged 6 and 10) in the Caribbean, budget $5000.",
            "Which Caribbean island would be best for families with young children?",
            "What kind of accommodation should we book? Resort vs vacation rental pros and cons?",
            "What activities and excursions are suitable for children these ages?",
            "How do we handle meals with picky eaters? Resort meal plans vs cooking?",
            "What should we pack for a Caribbean trip with kids? Any safety considerations?",
            "Can you create a sample daily schedule that balances beach time, activities, and rest?"
        ]
    },
    {
        "title": "Business Travel Optimization",
        "description": "Optimizing a complex business travel schedule",
        "queries": [
            "I need to plan business travel to New York, Chicago, and Los Angeles with meetings scheduled. The trip is 2 weeks long.",
            "What's the most efficient flight routing to minimize travel time and jet lag?",
            "Recommend business hotels near major business districts in each city.",
            "What are the best transportation options from airports to city centers in each location?",
            "Can you suggest some quick networking dinner spots and business lunch venues?",
            "How should I schedule the cities to optimize for meeting efficiency?",
            "Any tips for maintaining productivity during extended business travel?"
        ]
    },
    {
        "title": "Adventure Backpacking Trip",
        "description": "Planning an adventure-focused backpacking trip",
        "queries": [
            "Plan a 3-week adventure backpacking trip through Southeast Asia for 2 people, focusing on outdoor activities. Budget $3000.",
            "Which countries and regions offer the best hiking, diving, and adventure sports?",
            "What's the best route to take considering seasons and weather patterns?",
            "What gear should we pack for multiple climates and activities?",
            "How do we balance budget accommodation with safety and comfort?",
            "What are the visa requirements and health considerations for this region?",
            "Can you suggest a mix of guided tours vs independent exploration?"
        ]
    }
]

# Configuration for refinement testing
REFINEMENT_TEST_CONFIG = {
    "quality_thresholds": [0.6, 0.75, 0.85],
    "max_iterations": [2, 3, 5],
    "timeout_seconds": 300.0
}

# Test categories for analysis
TEST_CATEGORIES = {
    "trip_length": {
        "short": ["weekend", "3-day", "4-day"],
        "medium": ["5-day", "7-day", "10-day"],
        "long": ["14-day", "21-day", "3-week"]
    },
    "trip_type": {
        "leisure": ["romantic", "family", "solo"],
        "business": ["business", "meetings"],
        "adventure": ["backpacking", "ski", "hiking"],
        "luxury": ["luxury", "anniversary"]
    },
    "destination_type": {
        "city": ["london", "paris", "tokyo", "singapore", "dubai", "barcelona"],
        "region": ["europe", "caribbean", "southeast asia"],
        "nature": ["alps", "mount fuji", "beach"]
    }
} 