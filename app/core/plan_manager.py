"""
Plan Manager - Manages travel plans and calendar events for sessions

This module handles:
- Plan creation and updates
- Calendar event management
- Plan-session association
- Async plan generation from chat responses
"""

import uuid
import json
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from app.api.schemas import SessionTravelPlan, CalendarEvent, TravelPlanMetadata, CalendarEventType
from app.core.logging_config import get_logger
from app.core.prompt_manager import prompt_manager, PromptType

logger = get_logger(__name__)


class PlanManager:
    """Manages travel plans for sessions"""
    
    def __init__(self, storage_path: str = "./data/plans"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active plans
        self.plans: Dict[str, SessionTravelPlan] = {}
        self.session_to_plan: Dict[str, str] = {}  # session_id -> plan_id
        
        # Load existing plans
        self._load_plans()
        
        logger.info(f"✅ PlanManager initialized with {len(self.plans)} plans")
    
    def create_plan_for_session(self, session_id: str) -> str:
        """Create a new empty plan for a session"""
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        
        plan = SessionTravelPlan(
            plan_id=plan_id,
            session_id=session_id,
            events=[],
            metadata=TravelPlanMetadata(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        self.plans[plan_id] = plan
        self.session_to_plan[session_id] = plan_id
        self._save_plan(plan)
        
        logger.info(f"✅ Created plan {plan_id} for session {session_id}")
        return plan_id
    
    def get_plan_by_session(self, session_id: str) -> Optional[SessionTravelPlan]:
        """Get plan for a session, create if doesn't exist"""
        plan_id = self.session_to_plan.get(session_id)
        
        if not plan_id:
            # Create new plan if none exists
            plan_id = self.create_plan_for_session(session_id)
        
        return self.plans.get(plan_id)
    
    def get_plan_by_id(self, plan_id: str) -> Optional[SessionTravelPlan]:
        """Get plan by ID"""
        return self.plans.get(plan_id)
    
    async def update_plan_from_chat_response(
        self, 
        session_id: str, 
        user_message: str, 
        agent_response: str,
        response_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:  # Return update summary instead of bool
        """
        Enhanced plan update with detailed change tracking
        """
        try:
            # Get existing plan
            existing_plan = self.get_plan_by_session(session_id)
            if not existing_plan:
                logger.warning(f"No existing plan found for session {session_id}, creating new plan")
                self.create_plan_for_session(session_id)
                existing_plan = self.get_plan_by_session(session_id)
            
            logger.info(f"Starting plan update for session {session_id}:")
            logger.info(f"  - Existing plan has {len(existing_plan.events) if existing_plan.events else 0} events")
            logger.info(f"  - User message: {user_message[:150]}{'...' if len(user_message) > 150 else ''}")
            logger.info(f"  - Agent response length: {len(agent_response)} characters")
            
            update_summary = {
                "success": True,
                "changes_made": [],
                "events_added": 0,
                "events_updated": 0,
                "events_deleted": 0,
                "metadata_updated": False,
                "plan_modifications": {}
            }

            # Extract modifications using enhanced LLM method
            logger.info(f"Attempting to extract plan modifications...")
            modifications = await self._extract_plan_modifications(
                user_message, agent_response, response_metadata, existing_plan
            )
            
            logger.info(f"Plan modification extraction result:")
            logger.info(f"  - New events: {len(modifications.get('new_events', []))}")
            logger.info(f"  - Updated events: {len(modifications.get('updated_events', []))}")
            logger.info(f"  - Deleted events: {len(modifications.get('deleted_event_ids', []))}")
            logger.info(f"  - Modification reason: {modifications.get('plan_modifications', {}).get('reason', 'Unknown')}")
            
            # Apply modifications
            modifications_applied = False
            
            if modifications.get("new_events"):
                existing_plan.events.extend(modifications["new_events"])
                update_summary["events_added"] = len(modifications["new_events"])
                update_summary["changes_made"].append(f"Added {len(modifications['new_events'])} new events")
                logger.info(f"Added {len(modifications['new_events'])} new events to plan {existing_plan.plan_id}")
                modifications_applied = True

            if modifications.get("updated_events"):
                for updated_event in modifications["updated_events"]:
                    self._update_existing_event(existing_plan, updated_event)
                update_summary["events_updated"] = len(modifications["updated_events"])
                update_summary["changes_made"].append(f"Updated {len(modifications['updated_events'])} events")
                logger.info(f"Updated {len(modifications['updated_events'])} events in plan {existing_plan.plan_id}")
                modifications_applied = True

            if modifications.get("deleted_event_ids"):
                for event_id in modifications["deleted_event_ids"]:
                    existing_plan.events = [e for e in existing_plan.events if e.id != event_id]
                update_summary["events_deleted"] = len(modifications["deleted_event_ids"])
                update_summary["changes_made"].append(f"Deleted {len(modifications['deleted_event_ids'])} events")
                logger.info(f"Deleted {len(modifications['deleted_event_ids'])} events from plan {existing_plan.plan_id}")
                modifications_applied = True
            
            # 🔥 CRITICAL FIX: Apply time conflict resolution after any modifications
            if modifications_applied:
                logger.info(f"Applying time conflict resolution to updated plan with {len(existing_plan.events)} events...")
                try:
                    existing_plan.events = self._resolve_time_conflicts(existing_plan.events)
                    update_summary["changes_made"].append("Applied time conflict resolution")
                    logger.info(f"Time conflict resolution completed: {len(existing_plan.events)} events final")
                except Exception as e:
                    logger.error(f"Time conflict resolution failed: {e}")
                    update_summary["changes_made"].append(f"Time conflict resolution failed: {str(e)}")

            # Update metadata
            metadata_updates = self._extract_metadata_updates(user_message, agent_response, response_metadata)
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(existing_plan.metadata, key):
                        setattr(existing_plan.metadata, key, value)
                update_summary["metadata_updated"] = True
                update_summary["changes_made"].append("Updated plan metadata")
                logger.info(f"Updated plan metadata: {list(metadata_updates.keys())}")

            # Store modification details
            update_summary["plan_modifications"] = modifications.get("plan_modifications", {})

            # Save updated plan if any changes were made
            if update_summary["changes_made"]:
                existing_plan.updated_at = datetime.now(timezone.utc)
                self._save_plan(existing_plan)
                logger.info(f"Plan {existing_plan.plan_id} updated successfully: {update_summary['changes_made']}")
            else:
                logger.info(f"No changes needed for plan {existing_plan.plan_id}")

            return update_summary
            
        except Exception as e:
            logger.error(f"Failed to update plan from chat response: {e}")
            return {
                "success": False, 
                "error": str(e),
                "changes_made": [],
                "events_added": 0,
                "events_updated": 0,
                "events_deleted": 0,
                "metadata_updated": False
            }
    
    async def generate_plan_from_tool_results(
        self, 
        session_id: str, 
        tool_results: Dict[str, Any], 
        destination: str, 
        user_message: str,
        intent: Dict[str, Any] = None,
        multi_destinations: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate travel plan events directly from tool results (Fast non-LLM approach)
        Moved from travel_agent to centralize plan generation logic
        """
        from datetime import datetime, timedelta
        import uuid
        
        try:
            logger.info(f"Generating plan from tool results for destination: {destination}")
            if multi_destinations and len(multi_destinations) > 1:
                logger.info(f"🌍 Planning multi-destination trip with {len(multi_destinations)} destinations")
            
            # Get existing plan
            plan = self.get_plan_by_session(session_id)
            if not plan:
                logger.warning(f"No plan found for session {session_id}, creating new one")
                plan_id = self.create_plan_for_session(session_id)
                plan = self.get_plan_by_id(plan_id)
            
            # ✅ Extract duration from user intent or message, not hardcoded
            duration = self._extract_trip_duration(user_message, intent)
            travelers = self._extract_travelers_count(user_message, intent)
            
            # ✅ Extract dates from user message or use smart defaults
            start_date = self._extract_trip_start_date(user_message, intent)
            
            logger.info(f"Plan parameters: destination={destination}, duration={duration}, travelers={travelers}, start_date={start_date}")
            
            # Generate events from tool results
            events = self._create_events_from_tools(
                tool_results, 
                destination, 
                start_date, 
                duration, 
                travelers, 
                user_message,
                multi_destinations
            )
            
            # Update plan metadata (TravelPlanMetadata is a Pydantic model, not a dict)
            plan.metadata.destination = destination
            plan.metadata.duration_days = duration
            plan.metadata.travelers = travelers
            plan.metadata.budget = self._estimate_budget(duration, travelers)
            plan.metadata.budget_currency = "USD"
            plan.metadata.last_updated = datetime.now()
            plan.metadata.completion_status = "complete"
            plan.metadata.confidence = 0.88
            
            # Add events to plan (filter out None events)
            events_added = 0
            for event in events:
                if event is not None:  # _create_calendar_event_from_data can return None
                    plan.events.append(event)
                    events_added += 1
            
            # Save plan
            self._save_plan(plan)
            
            logger.info(f"Generated {events_added} events for {duration}-day trip to {destination}")
            
            return {
                "success": True,
                "changes_made": [f"Generated {events_added} events from tool results"],
                "events_added": events_added,
                "events_updated": 0,
                "events_deleted": 0,
                "metadata_updated": True,
                "plan_generation_method": "fast_tool_based",
                "duration": duration,
                "travelers": travelers
            }
            
        except Exception as e:
            logger.error(f"Failed to generate plan from tool results: {e}")
            return {
                "success": False,
                "error": str(e),
                "changes_made": []
            }

    def _extract_trip_duration(self, user_message: str, intent: Dict[str, Any] = None) -> int:
        """Extract trip duration from user message or intent, with smart defaults"""
        import re
        
        # Try to extract from intent first
        if intent and "travel_details" in intent:
            duration = intent["travel_details"].get("duration")
            if duration and isinstance(duration, (int, float)) and duration > 0:
                return int(duration)
        
        # Extract from user message using regex
        user_message_lower = user_message.lower()
        
        # Look for explicit day mentions
        day_patterns = [
            r'(\d+)[- ]days?',
            r'(\d+)[- ]day trip',
            r'for (\d+) days?',
            r'spend (\d+) days?',
            r'stay (\d+) days?'
        ]
        
        for pattern in day_patterns:
            match = re.search(pattern, user_message_lower)
            if match:
                duration = int(match.group(1))
                if 1 <= duration <= 30:  # Reasonable range
                    logger.info(f"Extracted duration from message: {duration} days")
                    return duration
        
        # Look for week mentions
        week_patterns = [
            r'(\d+)[- ]weeks?',
            r'for (\d+) weeks?',
            r'spend (\d+) weeks?'
        ]
        
        for pattern in week_patterns:
            match = re.search(pattern, user_message_lower)
            if match:
                duration = int(match.group(1)) * 7
                if duration <= 30:  # Max 30 days
                    logger.info(f"Extracted duration from message: {duration} days ({match.group(1)} weeks)")
                    return duration
        
        # Look for weekend/trip keywords
        if any(word in user_message_lower for word in ['weekend', 'short trip', 'quick trip']):
            logger.info("Detected weekend/short trip: defaulting to 3 days")
            return 3
        elif any(word in user_message_lower for word in ['vacation', 'holiday', 'long trip']):
            logger.info("Detected vacation/holiday: defaulting to 7 days")
            return 7
        
        # Default based on destination type or general default
        logger.info("No duration specified, defaulting to 5 days")
        return 5

    def _extract_travelers_count(self, user_message: str, intent: Dict[str, Any] = None) -> int:
        """Extract number of travelers from user message or intent"""
        import re
        
        # Try to extract from intent first
        if intent and "travel_details" in intent:
            travelers = intent["travel_details"].get("travelers")
            if travelers and isinstance(travelers, (int, float)) and travelers > 0:
                return int(travelers)
        
        # Extract from user message
        user_message_lower = user_message.lower()
        
        # Look for explicit numbers
        patterns = [
            r'for (\d+) people',
            r'(\d+) travelers?',
            r'(\d+) persons?',
            r'(\d+) adults?',
            r'group of (\d+)',
            r'(\d+) of us'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_message_lower)
            if match:
                count = int(match.group(1))
                if 1 <= count <= 20:  # Reasonable range
                    logger.info(f"Extracted travelers count: {count}")
                    return count
        
        # Look for couple/pair indicators
        if any(word in user_message_lower for word in ['couple', 'romantic', 'honeymoon', 'anniversary', 'two of us', 'my partner']):
            logger.info("Detected couple trip: 2 travelers")
            return 2
        
        # Look for family indicators
        if any(word in user_message_lower for word in ['family', 'kids', 'children']):
            logger.info("Detected family trip: 4 travelers")
            return 4
        
        # Default to solo travel
        logger.info("No traveler count specified, defaulting to 1")
        return 1

    def _extract_trip_start_date(self, user_message: str, intent: Dict[str, Any] = None) -> datetime:
        """Extract start date from user message or use smart default"""
        from datetime import datetime, timedelta
        import re
        
        # Try to extract from intent first
        if intent and "travel_details" in intent and "dates" in intent["travel_details"]:
            departure = intent["travel_details"]["dates"].get("departure")
            if departure and departure != "unknown":
                try:
                    return datetime.strptime(departure, "%Y-%m-%d")
                except:
                    pass
        
        # ✅ Look for explicit date mentions in user message
        user_message_lower = user_message.lower()
        
        # Check for "today", "tomorrow", "next week" etc.
        if 'today' in user_message_lower:
            return datetime.now()
        elif 'tomorrow' in user_message_lower:
            return datetime.now() + timedelta(days=1)
        elif 'next week' in user_message_lower:
            return datetime.now() + timedelta(days=7)
        elif 'next month' in user_message_lower:
            return datetime.now() + timedelta(days=30)
        
        # ✅ Default to tomorrow instead of next week for immediate planning
        default_start = datetime.now() + timedelta(days=1)
        logger.info(f"Using default start date: {default_start.strftime('%Y-%m-%d')} (tomorrow)")
        return default_start

    def _estimate_budget(self, duration: int, travelers: int) -> int:
        """Estimate budget based on duration and travelers"""
        base_per_day_per_person = 150  # $150 per day per person
        total = base_per_day_per_person * duration * travelers
        return max(500, min(total, 10000))  # Cap between $500-$10000

    def _create_events_from_tools(
        self, 
        tool_results: Dict[str, Any], 
        destination: str, 
        start_date: datetime, 
        duration: int, 
        travelers: int,
        user_message: str,
        multi_destinations: List[str] = None
    ) -> List:
        """Create calendar events from tool results with accurate timing"""
        events = []
        from datetime import timedelta
        import uuid
        
        current_time = start_date.replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Handle multi-destination trips (passed as parameter)
        if multi_destinations and len(multi_destinations) > 1:
            logger.info(f"🌍 Planning multi-destination trip: {multi_destinations}")
            return self._create_multi_destination_events(
                tool_results, multi_destinations, start_date, duration, travelers, user_message
            )
        else:
            if multi_destinations:
                logger.warning(f"⚠️ Multi-destinations provided but too short: {multi_destinations}")
            else:
                logger.info(f"🏙️ No multi-destinations provided, creating single destination events")
        
        try:
            # Add flight events (outbound and return)
            if "flight_search" in tool_results:
                flight_result = tool_results["flight_search"]
                if hasattr(flight_result, 'flights') and len(flight_result.flights) > 0:
                    # Outbound flight
                    outbound_flight = flight_result.flights[0]
                    outbound_time = current_time.replace(hour=10, minute=0)  # 10 AM departure
                    
                    event = self._create_calendar_event_from_data({
                        "id": f"flight_outbound_{str(uuid.uuid4())[:8]}",
                        "title": f"Flight to {destination}",
                        "description": f"Flight details: {getattr(outbound_flight, 'airline', 'TBA')} - Duration: {getattr(outbound_flight, 'duration', 'TBA')} minutes",
                        "event_type": "flight",
                        "start_time": outbound_time.isoformat(),
                        "end_time": (outbound_time + timedelta(hours=int(getattr(outbound_flight, 'duration', 360) / 60))).isoformat(),
                        "location": f"Airport → {destination}",
                        "details": {
                            "source": "flight_search",
                            "airline": getattr(outbound_flight, 'airline', 'TBA'),
                            "price": {"amount": getattr(outbound_flight, 'price', 500), "currency": "USD"},
                            "flight_number": getattr(outbound_flight, 'flight_number', 'TBA'),
                            "duration_minutes": getattr(outbound_flight, 'duration', 360)
                        }
                    })
                    if event:
                        events.append(event)
                    
                    # Return flight (on last day)
                    return_time = (start_date + timedelta(days=duration-1)).replace(hour=18, minute=0)
                    if len(flight_result.flights) > 1:
                        return_flight = flight_result.flights[1]
                    else:
                        return_flight = outbound_flight  # Use same flight as template
                    
                    event = self._create_calendar_event_from_data({
                        "id": f"flight_return_{str(uuid.uuid4())[:8]}",
                        "title": f"Return Flight",
                        "description": f"Return flight: {getattr(return_flight, 'airline', 'TBA')}",
                        "event_type": "flight",
                        "start_time": return_time.isoformat(),
                        "end_time": (return_time + timedelta(hours=int(getattr(return_flight, 'duration', 360) / 60))).isoformat(),
                        "location": f"{destination} → Home",
                        "details": {
                            "source": "flight_search",
                            "airline": getattr(return_flight, 'airline', 'TBA'),
                            "price": {"amount": getattr(return_flight, 'price', 500), "currency": "USD"},
                            "flight_number": getattr(return_flight, 'flight_number', 'TBA'),
                            "duration_minutes": getattr(return_flight, 'duration', 360)
                        }
                    })
                    if event:
                        events.append(event)
            
            # Add hotel events (full duration - 1 day to account for departure)
            if "hotel_search" in tool_results:
                hotel_result = tool_results["hotel_search"]
                if hasattr(hotel_result, 'hotels') and len(hotel_result.hotels) > 0:
                    hotel = hotel_result.hotels[0]
                    checkin_time = current_time.replace(hour=15, minute=0)  # 3 PM check-in
                    checkout_time = (start_date + timedelta(days=duration-1)).replace(hour=11, minute=0)  # 11 AM checkout
                    
                    event = self._create_calendar_event_from_data({
                        "id": f"hotel_stay_{str(uuid.uuid4())[:8]}",
                        "title": f"Hotel: {getattr(hotel, 'name', 'Hotel in ' + destination)}",
                        "description": f"Accommodation in {destination} for {duration-1} nights",
                        "event_type": "hotel",
                        "start_time": checkin_time.isoformat(),
                        "end_time": checkout_time.isoformat(),
                        "location": getattr(hotel, 'location', destination),
                        "details": {
                            "source": "hotel_search",
                            "rating": getattr(hotel, 'rating', 4.0),
                            "price_per_night": {"amount": getattr(hotel, 'price_per_night', 150), "currency": "USD"},
                            "nights": duration - 1
                        }
                    })
                    if event:
                        events.append(event)
            
            # Add attraction events (spread across middle days)
            if "attraction_search" in tool_results:
                attraction_result = tool_results["attraction_search"]
                if hasattr(attraction_result, 'attractions') and len(attraction_result.attractions) > 0:
                    # Day 1: Arrival, Day N: Departure, Days 2 to N-1: Activities
                    available_activity_days = max(1, duration - 2)  # At least 1 day of activities
                    max_attractions_per_day = 2  # Allow multiple attractions per day
                    max_attractions = available_activity_days * max_attractions_per_day
                    
                    # Use available attractions up to our calculated maximum
                    attractions = attraction_result.attractions[:min(len(attraction_result.attractions), max_attractions)]
                    
                    logger.info(f"📅 Creating {len(attractions)} attraction events across {available_activity_days} days (max {max_attractions_per_day} per day)")
                    
                    for i, attraction in enumerate(attractions):
                        # Distribute attractions across available activity days
                        day_offset = (i // max_attractions_per_day) + 1  # Start from day 2
                        attraction_of_day = i % max_attractions_per_day  # 0 or 1 (morning/afternoon)
                        
                        visit_day = start_date + timedelta(days=day_offset)  
                        
                        # Schedule attractions at different times of day
                        if attraction_of_day == 0:
                            visit_hour = 10  # Morning activity (10 AM)
                        else:
                            visit_hour = 14  # Afternoon activity (2 PM)
                            
                        visit_time = visit_day.replace(hour=visit_hour, minute=0)
                        
                        event = self._create_calendar_event_from_data({
                            "id": f"attraction_{i+1}_{str(uuid.uuid4())[:8]}",
                            "title": f"Visit {getattr(attraction, 'name', f'Attraction in {destination}')}",
                            "description": getattr(attraction, 'category', 'Sightseeing activity'),
                            "event_type": "attraction",
                            "start_time": visit_time.isoformat(),
                            "end_time": (visit_time + timedelta(hours=3)).isoformat(),
                            "location": getattr(attraction, 'location', destination),
                            "details": {
                                "source": "attraction_search",
                                "rating": getattr(attraction, 'rating', 4.5),
                                "category": getattr(attraction, 'category', 'Tourism')
                            }
                        })
                        if event:
                            events.append(event)
            
            # Add default activities if no specific events were created
            if len(events) <= 2:  # Only flights
                # Create activities for all available days, not just 3
                available_activity_days = max(1, duration - 2)  # At least 1 day of activities
                
                for i in range(available_activity_days):
                    day = start_date + timedelta(days=i+1)
                    event_time = day.replace(hour=10, minute=0)  # Consistent 10 AM start
                    event = self._create_calendar_event_from_data({
                        "id": f"activity_{i+1}_{str(uuid.uuid4())[:8]}",
                        "title": f"Explore {destination} - Day {i+1}",
                        "description": f"Discover the best of {destination}",
                        "event_type": "activity",
                        "start_time": event_time.isoformat(),
                        "end_time": (event_time + timedelta(hours=6)).isoformat(),  # Full day activity
                        "location": destination,
                        "details": {
                            "source": "default_generation",
                            "recommendations": ["Bring camera", "Wear comfortable shoes"]
                        }
                    })
                    if event:
                        events.append(event)
            
            logger.info(f"Created {len(events)} events for {duration}-day trip")
            return events
            
        except Exception as e:
            logger.error(f"Error creating events from tool results: {e}")
            # Return basic fallback event
            fallback_event = self._create_calendar_event_from_data({
                "id": f"basic_trip_{str(uuid.uuid4())[:8]}",
                "title": f"Trip to {destination}",
                "description": f"{duration}-day travel to {destination}",
                "event_type": "activity",
                "start_time": current_time.isoformat(),
                "end_time": (current_time + timedelta(hours=4)).isoformat(),
                "location": destination,
                "details": {"source": "fallback_generation"}
            })
            return [fallback_event] if fallback_event else []
    
    def _create_multi_destination_events(
        self, 
        tool_results: Dict[str, Any], 
        destinations: List[str], 
        start_date: datetime, 
        duration: int, 
        travelers: int,
        user_message: str
    ) -> List:
        """Create calendar events for multi-destination trips with complete flight chain"""
        events = []
        from datetime import timedelta
        import uuid
        
        logger.info(f"🗺️ === MULTI-DESTINATION EVENT CREATION START ===")
        logger.info(f"🗺️ Destinations: {destinations}")
        logger.info(f"🗺️ Duration: {duration} days")
        logger.info(f"🗺️ Travelers: {travelers}")
        logger.info(f"🗺️ Start date: {start_date}")
        logger.info(f"🗺️ Tool results keys: {list(tool_results.keys())}")
        
        try:
            logger.info(f"🗺️ Creating multi-destination itinerary for {len(destinations)} cities: {destinations}")
            
            # ✅ Extract origin from user message, flight search data, or use default
            origin = self._extract_origin_from_message(user_message)
            
            # ✅ If no origin found in message, try to get it from flight search data
            if not origin and "flight_search" in tool_results:
                flight_result = tool_results["flight_search"]
                if hasattr(flight_result, 'data') and flight_result.data:
                    flight_data = flight_result.data
                    if flight_data.get("search_type") == "flight_chain":
                        flight_chain = flight_data.get("flight_chain", [])
                        if len(flight_chain) > 0:
                            origin = flight_chain[0]  # First city in chain is origin
                            logger.info(f"🛫 Extracted origin from flight chain: {origin}")
            
            # ✅ Final fallback
            if not origin:
                origin = "YYZ"  # Default to Toronto
                logger.warning(f"🛫 No origin found, using default: {origin}")
            else:
                logger.info(f"🛫 Multi-city trip starting from: {origin}")
            
            # Calculate days per destination with proper distribution
            # Ensure total days equals requested duration, not compressed by destination count
            base_days_per_destination = max(1, duration // len(destinations))
            extra_days = duration % len(destinations)  # Distribute remaining days
            
            logger.info(f"📅 Distributing {duration} days across {len(destinations)} destinations:")
            logger.info(f"   Base allocation: {base_days_per_destination} days per destination")
            if extra_days > 0:
                logger.info(f"   Extra days to distribute: {extra_days}")
            
            # Create a list of days for each destination
            days_allocation = [base_days_per_destination] * len(destinations)
            # Distribute extra days to first few destinations
            for i in range(extra_days):
                days_allocation[i] += 1
            
            logger.info(f"   Final allocation: {days_allocation}")
            
            # Verify total days
            total_allocated_days = sum(days_allocation)
            logger.info(f"   Total days allocated: {total_allocated_days} (requested: {duration})")
            
            current_date = start_date
            
            # Create complete flight chain: Start → A → B → C → ... → N → Start
            flight_chain = [origin] + destinations + [origin]
            logger.info(f"✈️ Flight chain: {' → '.join(flight_chain)}")
            
            for i, destination in enumerate(destinations):  
                
                destination_days = days_allocation[i]
                logger.info(f"🏙️ Planning destination {i+1}/{len(destinations)}: {destination} ({destination_days} days)")
                
                # Create flight to this destination
                # Get proper city names from IATA codes
                from app.knowledge.geographical_data import GeographicalMappings
                destination_city_name = GeographicalMappings.get_city_name(destination)
                logger.info(f"🌍 Converting IATA '{destination}' to city name: '{destination_city_name}'")
                
                if i == 0:
                    # First flight: Origin → First Destination
                    departure_city = origin
                    departure_city_name = GeographicalMappings.get_city_name(departure_city)
                    flight_time = current_date.replace(hour=8, minute=0)
                    flight_title = f"Flight: {departure_city_name} → {destination_city_name}"
                    flight_description = f"Departure from {departure_city_name} to {destination_city_name} - Start of multi-city adventure"
                    flight_sequence = 1  # First flight in chain
                else:
                    # Connecting flights: Previous Destination → Current Destination
                    departure_city = destinations[i-1]
                    departure_city_name = GeographicalMappings.get_city_name(departure_city)
                    flight_time = current_date.replace(hour=14, minute=0)
                    flight_title = f"Flight: {departure_city_name} → {destination_city_name}"
                    flight_description = f"Connecting flight from {departure_city_name} to {destination_city_name}"
                    flight_sequence = i + 1  # Sequence number in chain
                
                # Get flight info from tool results if available
                flight_details = self._extract_flight_details_for_route(
                    tool_results, departure_city, destination, flight_sequence
                )
                
                logger.info(f"✈️ Creating flight event: {flight_title}")
                flight_event_data = {
                    "id": f"flight_{departure_city}_to_{destination}_{str(uuid.uuid4())[:8]}",
                    "title": flight_title,
                    "description": flight_description,
                    "event_type": "flight",
                    "start_time": flight_time.isoformat(),
                    "end_time": (flight_time + timedelta(hours=flight_details.get('duration_hours', 3))).isoformat(),
                    "location": f"{departure_city_name} → {destination_city_name}",
                    "details": {
                        "source": "multi_destination_planning",
                        "flight_sequence": flight_sequence,  # ✅ Use correct sequence
                        "total_flights": len(destinations) + 1,  # +1 for return flight
                        "origin": departure_city,
                        "destination": destination,
                        "airline": flight_details.get("airline", "TBA"),
                        "price": flight_details.get("price", {"amount": "TBA", "currency": "USD"}),
                        "flight_number": flight_details.get("flight_number", "TBA"),
                        "duration_minutes": flight_details.get("duration_hours", 3) * 60
                    }
                }
                
                flight_event = self._create_calendar_event_from_data(flight_event_data)
                if flight_event:
                    events.append(flight_event)
                    logger.info(f"✅ Flight event created and added")
                else:
                    logger.error(f"❌ Failed to create flight event")
                
                # ✅ Hotel for each destination
                checkin_time = current_date.replace(hour=15, minute=0)
                checkout_time = (current_date + timedelta(days=destination_days)).replace(hour=11, minute=0)
                
                # ✅ Get proper city name from IATA code
                from app.knowledge.geographical_data import GeographicalMappings
                city_name = GeographicalMappings.get_city_name(destination)
                
                # ✅ Try to get hotel information from tool results
                hotel_name = f"Hotel in {city_name}"
                hotel_location = city_name
                hotel_details = {
                    "source": "multi_destination_planning",
                    "nights": destination_days,
                    "destination_sequence": i + 1
                }
                
                # ✅ Check if we have hotel search results for this destination
                if "hotel_search" in tool_results:
                    hotel_result = tool_results["hotel_search"]
                    if hasattr(hotel_result, 'hotels') and len(hotel_result.hotels) > 0:
                        # Find hotel for this destination (hotels may be tagged with search_location)
                        matching_hotel = None
                        for hotel in hotel_result.hotels:
                            # Check if hotel is for this destination
                            hotel_search_location = getattr(hotel, 'search_location', '') or ''
                            hotel_location_upper = hotel_search_location.upper() if hotel_search_location else ''
                            destination_upper = destination.upper() if destination else ''
                            
                            if (hotel_search_location == destination or 
                                hotel_location_upper == destination_upper):
                                matching_hotel = hotel
                                break
                        
                        # If no specific match, use first hotel as template
                        if not matching_hotel and hotel_result.hotels:
                            matching_hotel = hotel_result.hotels[0]
                        
                        if matching_hotel:
                            hotel_name = getattr(matching_hotel, 'name', f"Hotel in {city_name}")
                            hotel_location = getattr(matching_hotel, 'location', city_name)
                            hotel_details.update({
                                "source": "hotel_search",
                                "rating": getattr(matching_hotel, 'rating', None),
                                "price_per_night": {
                                    "amount": getattr(matching_hotel, 'price_per_night', None), 
                                    "currency": getattr(matching_hotel, 'currency', 'USD')
                                }
                            })
                
                logger.info(f"🏨 Creating hotel event: {hotel_name}")
                hotel_event = self._create_calendar_event_from_data({
                    "id": f"hotel_{destination}_{str(uuid.uuid4())[:8]}",
                    "title": hotel_name,
                    "description": f"Accommodation in {city_name} for {destination_days} nights",
                    "event_type": "hotel",
                    "start_time": checkin_time.isoformat(),
                    "end_time": checkout_time.isoformat(),
                    "location": hotel_location,
                    "details": hotel_details
                })
                if hotel_event:
                    events.append(hotel_event)
                    logger.info(f"✅ Hotel event created and added")
                else:
                    logger.error(f"❌ Failed to create hotel event")
                
                # Create attraction events for this destination using actual search results
                city_attractions = self._get_attractions_for_destination(tool_results, destination, city_name)
                
                if city_attractions:
                    logger.info(f"🎯 Creating {len(city_attractions)} specific attraction events for {city_name}")
                    
                    # Create events for available attractions, distributed across days
                    max_attractions_per_day = 2  # Allow multiple attractions per day
                    attractions_to_use = city_attractions[:destination_days * max_attractions_per_day]
                    
                    for attr_idx, attraction in enumerate(attractions_to_use):
                        # 🔥 FIX: Properly distribute attractions to avoid same-day overlaps
                        # Distribute attractions across destination days
                        day_offset = attr_idx // max_attractions_per_day
                        attraction_of_day = attr_idx % max_attractions_per_day
                        
                        # Ensure we don't exceed destination days
                        if day_offset >= destination_days:
                            logger.warning(f"🎯 ⚠️  Skipping attraction {attr_idx+1} - exceeds {destination_days} days for {city_name}")
                            continue
                        
                        # 🔥 CRITICAL FIX: Ensure different days get different dates
                        activity_date = current_date + timedelta(days=day_offset)
                        logger.info(f"🎯 📅 Day calculation: attr_idx={attr_idx}, day_offset={day_offset}, attraction_of_day={attraction_of_day}")
                        logger.info(f"🎯 📅 Date calculation: current_date={current_date.strftime('%Y-%m-%d')}, activity_date={activity_date.strftime('%Y-%m-%d')}")
                        
                        # 🔥 NEW APPROACH: Don't set fixed times - let scheduler assign them dynamically
                        # Just set clean placeholder times that will be overridden by the scheduler
                        activity_date_start = activity_date.replace(hour=9, minute=0, second=0, microsecond=0)  # Clean placeholder
                        
                        # 🔥 VALIDATION: Ensure activity_date_start uses the correct date
                        if activity_date_start.date() != activity_date.date():
                            logger.error(f"🎯 ❌ DATE MISMATCH: activity_date={activity_date.date()}, activity_date_start={activity_date_start.date()}")
                        else:
                            logger.info(f"🎯 ✅ Date validation passed: {activity_date_start.date()}")
                        
                        activity_title = f"Visit {getattr(attraction, 'name', f'Attraction in {city_name}')}"
                        logger.info(f"🎯 📅 Creating {activity_title} for {activity_date.strftime('%Y-%m-%d')} (day {day_offset+1}/{destination_days})")
                        logger.info(f"🎯 🕐 Using start_time: {activity_date_start.isoformat()}")
                        
                        activity_event = self._create_calendar_event_from_data({
                            "id": f"attraction_{destination}_{attr_idx+1}_{str(uuid.uuid4())[:8]}",
                            "title": activity_title,
                            "description": getattr(attraction, 'description', f'Visit {getattr(attraction, "category", "attraction")} in {city_name}'),
                            "event_type": "attraction",
                            "start_time": activity_date_start.isoformat(),  # Placeholder - will be rescheduled
                            "end_time": (activity_date_start + timedelta(hours=2)).isoformat(),  # Minimum 2h placeholder
                            "location": getattr(attraction, 'location', city_name),
                            "details": {
                                "source": "attraction_search",
                                "destination_sequence": i + 1,
                                "day_in_destination": day_offset + 1,
                                "attraction_of_day": attraction_of_day + 1,
                                "rating": getattr(attraction, 'rating', None),
                                "category": getattr(attraction, 'category', 'Tourism'),
                                "place_id": getattr(attraction, 'place_id', None)
                            }
                        })
                        if activity_event:
                            events.append(activity_event)
                            logger.info(f"✅ Specific attraction event created and added")
                        else:
                            logger.error(f"❌ Failed to create attraction event")
                else:
                    # Fallback to generic events if no specific attractions found
                    logger.info(f"🎯 No specific attractions found for {city_name}, creating generic activity events")
                    for day in range(destination_days):
                        activity_date = current_date + timedelta(days=day)
                        activity_time = activity_date.replace(hour=10, minute=0)
                        
                        activity_title = f"Explore {city_name} - Day {day+1}"
                        logger.info(f"🎯 Creating fallback activity event: {activity_title}")
                        
                        activity_event = self._create_calendar_event_from_data({
                            "id": f"activity_{destination}_day{day+1}_{str(uuid.uuid4())[:8]}",
                            "title": activity_title,
                            "description": f"Discover attractions and culture in {city_name}",
                            "event_type": "attraction",
                            "start_time": activity_time.isoformat(),
                            "end_time": (activity_time + timedelta(hours=4)).isoformat(),
                            "location": city_name,
                            "details": {
                                "source": "multi_destination_planning_fallback",
                                "day_in_destination": day + 1,
                                "destination_sequence": i + 1,
                                "total_days_in_destination": destination_days,
                                "recommendations": ["Visit main attractions", "Try local cuisine", "Explore cultural sites"]
                            }
                        })
                        if activity_event:
                            events.append(activity_event)
                            logger.info(f"✅ Fallback activity event created and added")
                        else:
                            logger.error(f"❌ Failed to create fallback activity event")
                
                # Move to next destination period
                current_date += timedelta(days=destination_days)
            
            # Final return flight: Last Destination → Origin
            final_destination = destinations[-1]
            final_destination_city_name = GeographicalMappings.get_city_name(final_destination)
            origin_city_name = GeographicalMappings.get_city_name(origin)
            return_flight_time = current_date.replace(hour=18, minute=0)
            
            # Get return flight details
            return_flight_sequence = len(destinations) + 1  # ✅ Correct sequence for return flight
            return_flight_details = self._extract_flight_details_for_route(
                tool_results, final_destination, origin, return_flight_sequence
            )
            
            return_flight_title = f"Return Flight: {final_destination_city_name} → {origin_city_name}"
            logger.info(f"✈️ Creating return flight event: {return_flight_title}")
            
            return_flight_event = self._create_calendar_event_from_data({
                "id": f"flight_{final_destination}_to_{origin}_{str(uuid.uuid4())[:8]}",
                "title": return_flight_title,
                "description": f"Return journey from {final_destination_city_name} to {origin_city_name} - End of multi-city trip",
                "event_type": "flight",
                "start_time": return_flight_time.isoformat(),
                "end_time": (return_flight_time + timedelta(hours=return_flight_details.get('duration_hours', 6))).isoformat(),
                "location": f"{final_destination_city_name} → {origin_city_name}",
                "details": {
                    "source": "multi_destination_planning",
                    "flight_sequence": return_flight_sequence,  # ✅ Use correct sequence
                    "total_flights": len(destinations) + 1,
                    "origin": final_destination,
                    "destination": origin,
                    "trip_conclusion": True,
                    "cities_visited": destinations,
                    "airline": return_flight_details.get("airline", "TBA"),
                    "price": return_flight_details.get("price", {"amount": "TBA", "currency": "USD"}),
                    "flight_number": return_flight_details.get("flight_number", "TBA"),
                    "duration_minutes": return_flight_details.get("duration_hours", 6) * 60
                }
            })
            if return_flight_event:
                events.append(return_flight_event)
                logger.info(f"✅ Return flight event created and added")
            else:
                logger.error(f"❌ Failed to create return flight event")
            
            logger.info(f"✅ Created {len(events)} events for multi-destination trip:")
            logger.info(f"   📍 {len(destinations)} destinations: {', '.join(destinations)}")
            logger.info(f"   ✈️ {len(destinations) + 1} flights: {' → '.join(flight_chain)}")
            logger.info(f"   🏨 {len(destinations)} hotel stays")
            logger.info(f"   🎯 {sum(days_allocation)} total activity days ({total_allocated_days} days total)")
            logger.info(f"   Days distribution: {dict(zip(destinations, days_allocation))}")
            
            # Apply intelligent time conflict resolution
            try:
                logger.info(f"🕐 === STARTING TIME CONFLICT RESOLUTION ===")
                events = self._resolve_time_conflicts(events)
                logger.info(f"🕐 Time conflict resolution completed: {len(events)} final events")
            except Exception as e:
                logger.error(f"❌ Error in time conflict resolution: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error creating multi-destination events: {e}")
            # Fallback to single destination
            return self._create_single_destination_fallback(destinations[0], start_date, duration)
    
    def _resolve_time_conflicts(self, events: List) -> List:
        """Resolve time conflicts between events, prioritizing flights and respecting business hours"""
        
        # Separate events by type and priority
        flight_events = [e for e in events if e.event_type == 'flight']
        hotel_events = [e for e in events if e.event_type == 'hotel']  # All-day, no conflicts
        attraction_events = [e for e in events if e.event_type == 'attraction']
        meal_events = [e for e in events if e.event_type == 'meal']
        transport_events = [e for e in events if e.event_type == 'transportation']
        activity_events = [e for e in events if e.event_type == 'activity']
        other_events = [e for e in events if e.event_type not in ['flight', 'hotel', 'attraction', 'meal', 'transportation', 'activity']]
        
        logger.info(f"Events breakdown: {len(flight_events)} flights, {len(hotel_events)} hotels, {len(attraction_events)} attractions, {len(meal_events)} meals, {len(transport_events)} transport, {len(activity_events)} activities, {len(other_events)} others")
        
        # Sort by start time for conflict detection (include all non-hotel events)
        # Convert all times to timezone-naive for consistent comparison
        def safe_start_time(event):
            start_time = event.start_time
            if isinstance(start_time, str):
                # Parse ISO string and convert to naive datetime
                if '+' in start_time or 'Z' in start_time:
                    parsed = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    return parsed.replace(tzinfo=None)
                else:
                    return datetime.fromisoformat(start_time)
            elif hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                # Convert timezone-aware to naive
                return start_time.replace(tzinfo=None)
            return start_time
        
        scheduled_events = sorted(flight_events + attraction_events + meal_events + transport_events + activity_events + other_events, 
                                 key=safe_start_time)
        
        # Detect and resolve conflicts
        resolved_events = []
        
        # 🔄 SIMPLE & EFFECTIVE: Build a clean time schedule day by day
        logger.info(f"🔄 🎯 Starting conflict resolution for {len(scheduled_events)} events")
        for i, event in enumerate(scheduled_events):
            logger.info(f"🔄 📋 Event {i+1}: {event.title} ({event.event_type}) - {event.start_time}")
        
        resolved_events = self._build_conflict_free_schedule(scheduled_events)
        
        # Combine all events (resolved scheduled events + all-day hotel events)
        final_events = resolved_events + hotel_events
        
        logger.info(f"Final events composition: {len(resolved_events)} scheduled events + {len(hotel_events)} hotel events = {len(final_events)} total events")
        
        logger.info(f"🔄 ✅ Final schedule: {len(final_events)} total events (NO timezone conversion)")
        return final_events
    
    def _build_conflict_free_schedule(self, events: List):
        """Build a completely conflict-free schedule using iterative day-by-day approach with cross-day conflict resolution"""
        from datetime import datetime, timedelta, date
        from collections import defaultdict
        
        logger.info(f"🔄 === BUILDING CONFLICT-FREE SCHEDULE ===")
        
        # Initial grouping by date
        events_by_date = defaultdict(list)
        for event in events:
            if isinstance(event.start_time, str):
                event_date = datetime.fromisoformat(event.start_time.replace('Z', '+00:00')).date()
            else:
                event_date = event.start_time.date()
            events_by_date[event_date].append(event)
            logger.debug(f"🔄 📅 Grouped event: {event.title} -> {event_date}")
        
        resolved_events = []
        
        # 🔥 NEW APPROACH: Iterative scheduling with cross-day movement tracking
        max_iterations = 3  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 🔄 Starting iteration {iteration} of schedule resolution")
            
            movements_occurred = False
            daily_resolved_events = []
            
            # Process each day separately
            for event_date in sorted(events_by_date.keys()):
                daily_events = events_by_date[event_date]
                logger.info(f"🔄 📅 Processing {event_date}: {len(daily_events)} events (iteration {iteration})")
                
                # Separate fixed vs flexible events
                fixed_events = [e for e in daily_events if e.event_type == 'flight']
                flexible_events = [e for e in daily_events if e.event_type in ['attraction', 'meal', 'transportation', 'activity']]
                hotel_events = [e for e in daily_events if e.event_type == 'hotel']  # All-day events
                other_events = [e for e in daily_events if e.event_type not in ['flight', 'attraction', 'meal', 'transportation', 'activity', 'hotel']]
                
                attraction_count = len([e for e in flexible_events if e.event_type == 'attraction'])
                meal_count = len([e for e in flexible_events if e.event_type == 'meal'])
                
                logger.debug(f"Daily breakdown for {event_date}: {len(fixed_events)} flights, {len(flexible_events)} flexible events ({attraction_count} attractions, {meal_count} meals), {len(hotel_events)} hotels")
                
                # Add fixed events as-is (flights have fixed times)
                for event in fixed_events:
                    daily_resolved_events.append(event)
                
                # Add other events that don't need scheduling (e.g., meeting, free_time)
                for event in other_events:
                    daily_resolved_events.append(event)
                
                # Schedule flexible events with cross-day movement detection
                if flexible_events:
                    daily_resolved, moved_events = self._schedule_daily_attractions_with_movement_tracking(
                        event_date, flexible_events, fixed_events
                    )
                    daily_resolved_events.extend(daily_resolved)
                    
                    # Track movements to next iteration
                    if moved_events:
                        movements_occurred = True
                        logger.info(f"🔄 📤 {len(moved_events)} attractions moved to future days")
                        
                        # Update events_by_date with moved events
                        for moved_event, target_date in moved_events:
                            logger.info(f"🔄 📤 Moving {moved_event.title} to {target_date}")
                            # Update event's actual date
                            moved_event.start_time = moved_event.start_time.replace(
                                year=target_date.year, month=target_date.month, day=target_date.day
                            )
                            moved_event.end_time = moved_event.end_time.replace(
                                year=target_date.year, month=target_date.month, day=target_date.day  
                            )
                            events_by_date[target_date].append(moved_event)
            
            resolved_events = daily_resolved_events
            
            # If no movements occurred, we're done
            if not movements_occurred:
                logger.info(f"🔄 ✅ No cross-day movements in iteration {iteration}. Schedule stabilized.")
                break
            
            # Rebuild events_by_date for next iteration (remove moved events from original dates)
            if movements_occurred:
                logger.info(f"🔄 🔄 Movements detected. Starting iteration {iteration + 1}...")
                # Clear and rebuild to ensure moved events are properly grouped
                new_events_by_date = defaultdict(list)
                for event in resolved_events:
                    if isinstance(event.start_time, str):
                        event_date = datetime.fromisoformat(event.start_time.replace('Z', '+00:00')).date()
                    else:
                        event_date = event.start_time.date()
                    new_events_by_date[event_date].append(event)
                events_by_date = new_events_by_date
        
        logger.info(f"🔄 ✅ Schedule completed: {len(resolved_events)} events after {iteration} iteration(s)")
        return resolved_events
    
    def _schedule_daily_attractions_with_movement_tracking(self, event_date, attractions: List, fixed_events: List):
        """
        🔥 ENHANCED SCHEDULING: Schedule attractions with cross-day movement tracking
        Returns: (scheduled_attractions, moved_attractions_with_dates)
        """
        from datetime import datetime, timedelta
        import random
        
        logger.info(f"🔄 🎯 ENHANCED DYNAMIC scheduling {len(attractions)} attractions for {event_date.strftime('%Y-%m-%d')}")
        logger.info(f"🔄 🔧 Features: Clean 30-min intervals, conflict resolution, business hours enforcement, movement tracking")
        
        # Business hours: 9 AM to 5 PM  
        business_start = datetime.combine(event_date, datetime.min.time().replace(hour=9))
        business_end = datetime.combine(event_date, datetime.min.time().replace(hour=17))
        
        logger.info(f"🔄 📊 Business hours: {business_start.strftime('%H:%M')} - {business_end.strftime('%H:%M')}")
        
        # Track blocked periods from fixed events (flights)
        blocked_periods = []
        for flight in fixed_events:
            flight_start, flight_end = self._extract_naive_times(flight)
            blocked_periods.append((flight_start, flight_end))
            logger.info(f"🔄 🚫 Fixed blocked period: {flight_start.strftime('%H:%M')} - {flight_end.strftime('%H:%M')} ({flight.title})")
        
        # Track all scheduled periods (fixed + scheduled attractions)
        scheduled_periods = blocked_periods.copy()
        scheduled_attractions = []
        moved_attractions = []  # (attraction, target_date) tuples
        
        # Start scheduling attractions from business hours start
        current_time = business_start
        
        for i, attraction in enumerate(attractions):
            # Use clean 30-minute intervals
            # Duration: 2.0h, 2.5h, 3.0h, 3.5h, or 4.0h (30-min multiples)
            duration_options = [2.0, 2.5, 3.0, 3.5, 4]
            duration_hours = random.choice(duration_options)
            duration = timedelta(hours=duration_hours)
            
            # Gap: 30min, 60min, or 90min (clean intervals)
            gap_minutes = random.choice([30, 60, 90])
            gap = timedelta(minutes=gap_minutes)
            
            # Round current_time to nearest 30-minute mark
            current_time = self._round_to_30min(current_time)
            
            required_end_time = current_time + duration
            next_start_time = required_end_time + gap
            
            # Check conflicts with ALL scheduled periods (fixed + attractions)
            conflicts_with_any = any(
                not (required_end_time <= period_start or current_time >= period_end)
                for period_start, period_end in scheduled_periods
            )
            
            if conflicts_with_any:
                # Find the latest end time among conflicting periods
                latest_end = max(
                    period_end for period_start, period_end in scheduled_periods
                    if not (required_end_time <= period_start or current_time >= period_end)
                )
                adjusted_start = self._round_to_30min(latest_end + timedelta(minutes=30))
                required_end_time = adjusted_start + duration
                current_time = adjusted_start
                next_start_time = required_end_time + gap
                logger.info(f"🔄 ⚠️  Conflict detected! Adjusted start time to {adjusted_start.strftime('%H:%M')}")
            
            # Check if fits within business hours
            if required_end_time <= business_end:
                # Success! Schedule the attraction
                attraction.start_time = current_time
                attraction.end_time = required_end_time
                
                scheduled_attractions.append(attraction)
                logger.info(f"🔄 ✅ Scheduled: {attraction.title}")
                logger.info(f"🔄     📅 Time: {current_time.strftime('%H:%M')} - {required_end_time.strftime('%H:%M')} ({duration_hours}h)")
                logger.info(f"🔄     ⏰ Gap after: {gap_minutes} minutes")
                
                # Add this attraction to scheduled periods to prevent future conflicts
                scheduled_periods.append((current_time, required_end_time))
                
                # Move to next start time for next attraction
                current_time = next_start_time
                
            else:
                # Can't fit in today - mark for movement to next day
                target_date = event_date + timedelta(days=1)
                logger.info(f"🔄 📅 Marking for next day: {attraction.title} (would end at {required_end_time.strftime('%H:%M')}, past business hours)")
                moved_attractions.append((attraction, target_date))
        
        # Final conflict verification and resolution for scheduled attractions
        if len(scheduled_attractions) > 1:
            scheduled_attractions = self._verify_and_resolve_conflicts(scheduled_attractions, event_date)
        
        logger.info(f"🔄 ✅ Completed ENHANCED scheduling for {event_date.strftime('%Y-%m-%d')}: {len(scheduled_attractions)} scheduled, {len(moved_attractions)} moved")
        
        return scheduled_attractions, moved_attractions
    
    def _schedule_daily_attractions(self, event_date, attractions: List, fixed_events: List):
        """
        🔥 NEW DYNAMIC SCHEDULING: Schedule attractions with flexible timing
        - Each attraction: minimum 2 hours, flexible duration
        - Between attractions: 30-60 minute gaps
        - Business hours: 9:00-17:00
        - No overlaps, intelligent spacing
        """
        from datetime import datetime, timedelta
        from copy import deepcopy
        import random
        
        logger.info(f"🔄 🎯 ENHANCED DYNAMIC scheduling {len(attractions)} attractions for {event_date}")
        logger.info(f"🔄 🔧 Features: Clean 30-min intervals, conflict resolution, business hours enforcement")
        
        # Business hours: 9 AM - 5 PM (8 hours total)
        business_start = datetime.combine(event_date, datetime.min.time().replace(hour=9))
        business_end = datetime.combine(event_date, datetime.min.time().replace(hour=17))
        
        logger.info(f"🔄 📊 Business hours: {business_start.strftime('%H:%M')} - {business_end.strftime('%H:%M')}")
        
        # Collect blocked periods from fixed events (flights)
        blocked_periods = []
        for fixed_event in fixed_events:
            start, end = self._extract_naive_times(fixed_event)
            blocked_periods.append((start, end))
            logger.info(f"🔄 🚫 Fixed blocked period: {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({fixed_event.title})")
        
        # Start scheduling attractions dynamically
        scheduled_attractions = []
        next_day_attractions = []
        current_time = business_start
        
        # 🔥 TRACK ALL SCHEDULED PERIODS to prevent overlaps
        scheduled_periods = list(blocked_periods)  # Start with fixed events
        
        for i, attraction in enumerate(attractions):
            # Use clean 30-minute intervals
            # Duration: 2.0h, 2.5h, 3.0h, or 3.5h (30-min multiples)
            duration_options = [2.0, 2.5, 3.0, 3.5, 4]
            duration_hours = random.choice(duration_options)
            duration = timedelta(hours=duration_hours)
            
            # Gap: 30min or 60min (clean intervals)
            gap_minutes = random.choice([30, 60, 90])
            gap = timedelta(minutes=gap_minutes)
            
            # Round current_time to nearest 30-minute mark
            current_time = self._round_to_30min(current_time)
            
            # Check if we can fit this attraction in the current slot
            required_end_time = current_time + duration
            next_start_time = required_end_time + gap
            
            # Check conflicts with ALL scheduled periods (fixed + attractions)
            conflicts_with_any = any(
                not (required_end_time <= period_start or current_time >= period_end)
                for period_start, period_end in scheduled_periods
            )
            
            # If there's a conflict, find the next available slot
            if conflicts_with_any:
                # Find the next available time after ALL conflicts
                latest_conflict_end = max(
                    period_end for period_start, period_end in scheduled_periods
                    if not (required_end_time <= period_start or current_time >= period_end)
                )
                current_time = self._round_to_30min(latest_conflict_end + timedelta(minutes=15))
                required_end_time = current_time + duration
                next_start_time = required_end_time + gap
                logger.info(f"🔄 ⚠️  Conflict detected! Adjusted start time to {current_time.strftime('%H:%M')}")
            
            # Check if it fits within business hours
            if required_end_time <= business_end:
                # Schedule this attraction
                new_attraction = deepcopy(attraction)
                new_attraction.start_time = current_time
                new_attraction.end_time = required_end_time
                
                scheduled_attractions.append(new_attraction)
                
                # Add this attraction to scheduled periods to prevent future conflicts
                scheduled_periods.append((current_time, required_end_time))
                
                logger.info(f"🔄 ✅ Scheduled: {attraction.title}")
                logger.info(f"🔄     📅 Time: {current_time.strftime('%H:%M')} - {required_end_time.strftime('%H:%M')} ({duration_hours:.1f}h)")
                logger.info(f"🔄     ⏰ Gap after: {gap_minutes} minutes")
                
                # Update current time for next attraction
                current_time = self._round_to_30min(next_start_time)
                
            else:
                # Can't fit in today - move to next day
                logger.info(f"🔄 📅 Moving to next day: {attraction.title} (would end at {required_end_time.strftime('%H:%M')}, past business hours)")
                next_day_attractions.append(attraction)
        
        # Schedule remaining attractions on next day
        if next_day_attractions:
            logger.info(f"🔄 📅 Scheduling {len(next_day_attractions)} attractions on next day")
            next_day_scheduled = self._schedule_daily_attractions(
                event_date + timedelta(days=1),
                next_day_attractions,
                []  # No fixed events on moved day
            )
            scheduled_attractions.extend(next_day_scheduled)
        
        # 🔥 FIX 1: Final conflict verification and resolution
        if len(scheduled_attractions) > 1:
            scheduled_attractions = self._verify_and_resolve_conflicts(scheduled_attractions, event_date)
        
        logger.info(f"🔄 ✅ Completed DYNAMIC scheduling for {event_date.strftime('%Y-%m-%d')}: {len(scheduled_attractions)} total attractions")
        return scheduled_attractions
    
    def _extract_naive_times(self, event):
        """Extract naive datetime objects from event start/end times"""
        from datetime import datetime
        
        # Handle start time
        if isinstance(event.start_time, str):
            start = datetime.fromisoformat(event.start_time.replace('Z', '+00:00'))
        else:
            start = event.start_time
        
        # Handle end time
        if isinstance(event.end_time, str):
            end = datetime.fromisoformat(event.end_time.replace('Z', '+00:00'))
        else:
            end = event.end_time
        
        # Convert to naive datetime for consistent scheduling
        if hasattr(start, 'tzinfo') and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if hasattr(end, 'tzinfo') and end.tzinfo is not None:
            end = end.replace(tzinfo=None)
        
        return start, end
    
    def _round_to_30min(self, dt):
        """Round datetime to nearest 30-minute mark for clean scheduling"""
        from datetime import datetime, timedelta
        
        # Get current minutes
        current_minutes = dt.minute
        
        # Round to nearest 30-minute mark
        if current_minutes <= 15:
            # Round down to :00
            rounded_minutes = 0
        elif current_minutes <= 45:
            # Round to :30
            rounded_minutes = 30
        else:
            # Round up to next hour :00
            dt = dt + timedelta(hours=1)
            rounded_minutes = 0
        
        # Create clean time
        clean_time = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
        return clean_time
    
    def _verify_and_resolve_conflicts(self, attractions, event_date):
        """
        🔥 FINAL CONFLICT RESOLUTION: Verify and iteratively resolve any remaining conflicts
        """
        from datetime import datetime, timedelta
        from copy import deepcopy
        
        logger.info(f"🔄 🔍 Verifying {len(attractions)} attractions for conflicts on {event_date.strftime('%Y-%m-%d')}")
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            conflicts_found = False
            
            # Check all pairs for conflicts
            for i, attr1 in enumerate(attractions):
                for j, attr2 in enumerate(attractions[i+1:], i+1):
                    start1, end1 = self._extract_naive_times(attr1)
                    start2, end2 = self._extract_naive_times(attr2)
                    
                    # Check if they overlap
                    if not (end1 <= start2 or end2 <= start1):
                        logger.info(f"🔄 ⚠️  CONFLICT DETECTED (iter {iteration}): {attr1.title} vs {attr2.title}")
                        logger.info(f"🔄     📅 {start1.strftime('%H:%M')}-{end1.strftime('%H:%M')} vs {start2.strftime('%H:%M')}-{end2.strftime('%H:%M')}")
                        
                        # 🔥 CRITICAL FIX: Always move the LATER event forward, never backward
                        # This prevents events from being moved to already-occupied earlier times
                        if start2 >= start1:
                            # Move attr2 after attr1 (correct direction)
                            new_start = self._round_to_30min(end1 + timedelta(minutes=30))
                            duration = end2 - start2
                            new_end = new_start + duration
                            
                            # 🔥 CHECK: Ensure we don't move past business hours (5 PM)
                            business_end = datetime.combine(event_date, datetime.min.time().replace(hour=17))
                            if new_end > business_end:
                                logger.warning(f"🔄 ⚠️  Cannot reschedule {attr2.title} - would exceed business hours")
                                # Move to next day instead of forcing into same day
                                continue
                            
                            attr2.start_time = new_start
                            attr2.end_time = new_end
                            
                            logger.info(f"🔄 ✅ Resolved: Moved {attr2.title} to {new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}")
                        else:
                            # Move attr1 after attr2 (correct direction)  
                            new_start = self._round_to_30min(end2 + timedelta(minutes=30))
                            duration = end1 - start1
                            new_end = new_start + duration
                            
                            # 🔥 CHECK: Ensure we don't move past business hours (5 PM)
                            business_end = datetime.combine(event_date, datetime.min.time().replace(hour=17))
                            if new_end > business_end:
                                logger.warning(f"🔄 ⚠️  Cannot reschedule {attr1.title} - would exceed business hours")
                                # Move to next day instead of forcing into same day
                                continue
                            
                            attr1.start_time = new_start
                            attr1.end_time = new_end
                            
                            logger.info(f"🔄 ✅ Resolved: Moved {attr1.title} to {new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}")
                        
                        conflicts_found = True
                        break
                if conflicts_found:
                    break
            
            if not conflicts_found:
                logger.info(f"🔄 ✅ No conflicts found after {iteration} iteration(s)")
                break
        
        if iteration >= max_iterations:
            logger.warning(f"🔄 ⚠️  Max iterations reached, some conflicts may remain")
        
        return attractions
    
    def _generate_time_slots(self, business_start: datetime, business_end: datetime, blocked_periods: list):
        """Generate available time slots, avoiding blocked periods"""
        from datetime import timedelta
        
        # Start with the full business day
        slots = [(business_start, business_end)]
        
        # Remove blocked periods
        for block_start, block_end in blocked_periods:
            new_slots = []
            for slot_start, slot_end in slots:
                # Check if this slot overlaps with the blocked period
                if block_end <= slot_start or block_start >= slot_end:
                    # No overlap, keep the slot
                    new_slots.append((slot_start, slot_end))
                else:
                    # Overlap, split the slot
                    if block_start > slot_start:
                        # Add the part before the blocked period
                        buffer_end = block_start - timedelta(hours=1)  # 1-hour buffer before flight
                        if buffer_end > slot_start:
                            new_slots.append((slot_start, buffer_end))
                    
                    if block_end < slot_end:
                        # Add the part after the blocked period
                        buffer_start = block_end + timedelta(hours=1)  # 1-hour buffer after flight
                        if buffer_start < slot_end:
                            new_slots.append((buffer_start, slot_end))
            
            slots = new_slots
        
        # Filter out slots that are too small (less than 1 hour)
        slots = [(start, end) for start, end in slots if (end - start) >= timedelta(hours=1)]
        
        return sorted(slots)
    
    def _adjust_attraction_time(self, attraction_event, scheduled_events: List):
        """Adjust attraction event time to avoid conflicts with scheduled events"""
        from datetime import datetime, timedelta
        
        # Log the input event for debugging
        logger.debug(f"🕐 🔍 Adjusting time for: {attraction_event.title}")
        logger.debug(f"🕐    Original start: {attraction_event.start_time} (type: {type(attraction_event.start_time)})")
        logger.debug(f"🕐    Original end: {attraction_event.end_time} (type: {type(attraction_event.end_time)})")
        
        # Handle both datetime objects and ISO strings
        if isinstance(attraction_event.start_time, str):
            original_start = datetime.fromisoformat(attraction_event.start_time.replace('Z', '+00:00'))
            logger.debug(f"🕐    Parsed start from string: {original_start}")
        else:
            original_start = attraction_event.start_time
            logger.debug(f"🕐    Using datetime object: {original_start}")
            
        if isinstance(attraction_event.end_time, str):
            original_end = datetime.fromisoformat(attraction_event.end_time.replace('Z', '+00:00'))
            logger.debug(f"🕐    Parsed end from string: {original_end}")
        else:
            original_end = attraction_event.end_time
            logger.debug(f"🕐    Using datetime object: {original_end}")
            
        duration = original_end - original_start
        logger.debug(f"🕐    Duration: {duration}")
        
        # Get the date and find available time slots
        event_date = original_start.date()
        
        # Business hours: 9:00 AM to 6:00 PM (in the same timezone as the original event)
        if original_start.tzinfo:
            # Create timezone-aware business hours
            business_start = original_start.replace(hour=9, minute=0, second=0, microsecond=0)
            business_end = original_start.replace(hour=18, minute=0, second=0, microsecond=0)
            logger.debug(f"🕐    Timezone-aware business hours: {business_start} - {business_end}")
        else:
            # Create naive business hours
            business_start = datetime.combine(event_date, datetime.min.time().replace(hour=9))
            business_end = datetime.combine(event_date, datetime.min.time().replace(hour=18))
            logger.debug(f"🕐    Naive business hours: {business_start} - {business_end}")
            
        logger.debug(f"🕐    Business hours timezone: {business_start.tzinfo}")
        
        # Handle special cases for arrival/departure days
        same_date_flights = []
        for e in scheduled_events:
            if e.event_type == 'flight':
                # Handle both datetime objects and ISO strings for flight start_time
                if isinstance(e.start_time, str):
                    flight_date = datetime.fromisoformat(e.start_time.replace('Z', '+00:00')).date()
                else:
                    flight_date = e.start_time.date()
                
                if flight_date == event_date:
                    same_date_flights.append(e)
        
        # Get destination IATA and city name from attraction event
        attraction_location = attraction_event.location or ''
        
        # Convert IATA to city name for matching
        from app.knowledge.geographical_data import GeographicalMappings
        attraction_city = GeographicalMappings.get_city_name(attraction_location)
        
        # Check for arrival flights (incoming to this destination)
        arrival_flights = []
        departure_flights = []
        
        for flight in same_date_flights:
            flight_details = flight.details or {}
            flight_destination = flight_details.get('destination', '')
            flight_origin = flight_details.get('origin', '')
            
            # Check if this flight is arriving to the attraction's destination
            # Match by IATA code or city name
            if (flight_destination == attraction_location or 
                flight_destination == attraction_city or
                attraction_city.lower() in flight_destination.lower()):
                arrival_flights.append(flight)
            
            # Check if this flight is departing from the attraction's destination
            if (flight_origin == attraction_location or
                flight_origin == attraction_city or
                attraction_city.lower() in flight_origin.lower()):
                departure_flights.append(flight)
        
        # Adjust business hours based on flights
        if arrival_flights:
            # If there's an arrival flight, attractions can only start after arrival + buffer time
            arrival_flight = min(arrival_flights, key=lambda x: x.start_time if isinstance(x.start_time, datetime) else datetime.fromisoformat(x.start_time.replace('Z', '+00:00')))
            # Handle both datetime objects and ISO strings for flight end_time
            if isinstance(arrival_flight.end_time, str):
                arrival_end = datetime.fromisoformat(arrival_flight.end_time.replace('Z', '+00:00'))
            else:
                arrival_end = arrival_flight.end_time
            # Add 2-hour buffer for immigration, baggage, transport to city
            earliest_start = arrival_end + timedelta(hours=2)
            business_start = max(business_start, earliest_start)
            logger.info(f"🕐 ✈️  Arrival day adjustment: earliest attraction start at {business_start.strftime('%H:%M')} due to {arrival_flight.title}")
        
        if departure_flights:
            # If there's a departure flight, attractions must end before departure - buffer time
            departure_flight = min(departure_flights, key=lambda x: x.start_time if isinstance(x.start_time, datetime) else datetime.fromisoformat(x.start_time.replace('Z', '+00:00')))
            # Handle both datetime objects and ISO strings for flight start_time
            if isinstance(departure_flight.start_time, str):
                departure_start = datetime.fromisoformat(departure_flight.start_time.replace('Z', '+00:00'))
            else:
                departure_start = departure_flight.start_time
            # Subtract 3-hour buffer for hotel checkout, transport to airport, check-in
            latest_end = departure_start - timedelta(hours=3)
            business_end = min(business_end, latest_end)
            logger.info(f"🕐 ✈️  Departure day adjustment: latest attraction end at {business_end.strftime('%H:%M')} due to {departure_flight.title}")
        
        # Check if we still have enough time for the attraction
        if business_start + duration > business_end:
            logger.warning(f"🕐 ⚠️  Not enough time on {event_date} for {attraction_event.title} (need {duration}, have {business_end - business_start})")
            
            # Try to move to next day instead of giving up
            logger.info(f"🕐 📅 Attempting to move {attraction_event.title} to next day...")
            
            next_day = event_date + timedelta(days=1)
            
            # Create business hours for next day
            if original_start.tzinfo:
                next_business_start = original_start.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
                next_business_end = original_start.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_business_start = datetime.combine(next_day, datetime.min.time().replace(hour=9))
                next_business_end = datetime.combine(next_day, datetime.min.time().replace(hour=18))
                
            # Check for flights or other conflicts on next day
            next_day_conflicts = []
            for e in scheduled_events:
                if isinstance(e.start_time, str):
                    check_date = datetime.fromisoformat(e.start_time.replace('Z', '+00:00')).date()
                else:
                    check_date = e.start_time.date()
                    
                if check_date == next_day:
                    next_day_conflicts.append(e)
            
            # Try multiple strategies for next day
            if len(next_day_conflicts) == 0 and next_business_start + duration <= next_business_end:
                logger.info(f"🕐 ✅ Moving {attraction_event.title} to {next_day} at {next_business_start.strftime('%H:%M')}")
                
                from copy import deepcopy
                adjusted_event = deepcopy(attraction_event)
                adjusted_event.start_time = next_business_start
                adjusted_event.end_time = next_business_start + duration
                return adjusted_event
            
            # Strategy 2: Try reducing duration by 60 minutes if original is too long
            reduced_duration = duration - timedelta(minutes=60)
            if reduced_duration >= timedelta(hours=1) and next_business_start + reduced_duration <= next_business_end and len(next_day_conflicts) <= 1:
                logger.info(f"🕐 ⏱️  Trying reduced duration on next day: {reduced_duration}")
                from copy import deepcopy
                adjusted_event = deepcopy(attraction_event)
                adjusted_event.start_time = next_business_start
                adjusted_event.end_time = next_business_start + reduced_duration
                return adjusted_event
            
            # Strategy 3: Last resort - keep original time but log warning
            logger.error(f"🕐 ❌ All strategies failed for {attraction_event.title} (next day conflicts: {len(next_day_conflicts)}), keeping original time")
            return attraction_event
        
        # Find conflicts on the same date
        same_date_events = []
        for e in scheduled_events:
            if isinstance(e.start_time, str):
                event_date_check = datetime.fromisoformat(e.start_time.replace('Z', '+00:00')).date()
            else:
                event_date_check = e.start_time.date()
            
            if event_date_check == event_date:
                same_date_events.append(e)
        
        # Try to find a conflict-free time slot
        candidate_start = business_start
        
        while candidate_start + duration <= business_end:
            candidate_end = candidate_start + duration
            
            # Check for conflicts with existing events
            has_conflict = False
            for existing_event in same_date_events:
                # Handle both datetime objects and ISO strings
                if isinstance(existing_event.start_time, str):
                    existing_start = datetime.fromisoformat(existing_event.start_time.replace('Z', '+00:00'))
                else:
                    existing_start = existing_event.start_time
                    
                if isinstance(existing_event.end_time, str):
                    existing_end = datetime.fromisoformat(existing_event.end_time.replace('Z', '+00:00'))
                else:
                    existing_end = existing_event.end_time
                
                # Add 1-hour buffer between attraction events (but not for flights/hotels)
                buffer_time = timedelta(hours=1) if hasattr(existing_event, 'event_type') and existing_event.event_type == 'attraction' else timedelta(0)
                
                # Check if times overlap (including buffer for attractions)
                effective_existing_end = existing_end + buffer_time
                effective_candidate_start = candidate_start
                
                if (effective_candidate_start < effective_existing_end and candidate_end > existing_start):
                    has_conflict = True
                    logger.debug(f"🕐 ⚠️  Conflict detected with {getattr(existing_event, 'title', 'Unknown')} (buffer: {buffer_time})")
                    break
            
            if not has_conflict:
                # Double-check: ensure we have enough buffer time after this slot
                next_conflict_start = None
                for check_event in same_date_events:
                    if isinstance(check_event.start_time, str):
                        check_start = datetime.fromisoformat(check_event.start_time.replace('Z', '+00:00'))
                    else:
                        check_start = check_event.start_time
                    
                    if check_start > candidate_end:
                        if next_conflict_start is None or check_start < next_conflict_start:
                            next_conflict_start = check_start
                
                # Ensure we have at least 1-hour buffer before next event
                if next_conflict_start is None or (next_conflict_start - candidate_end) >= timedelta(hours=1):
                    logger.info(f"🕐 ✅ Found valid time slot: {candidate_start.strftime('%H:%M')} - {candidate_end.strftime('%H:%M')}")
                    break
                else:
                    logger.debug(f"🕐 ⚠️  Insufficient buffer to next event at {next_conflict_start.strftime('%H:%M')}")
                    has_conflict = True
            
            # Move to next 30-minute slot and try again (more granular)
            candidate_start += timedelta(minutes=30)
        
        # If we couldn't find a slot, try moving to next day
        if candidate_start + duration > business_end:
            logger.warning(f"🕐 ⚠️  Could not find conflict-free time for {attraction_event.title} on {event_date}")
            
            # 🚨 Try to move to next day
            logger.info(f"🕐 📅 Attempting to move {attraction_event.title} to next day due to conflicts...")
            
            next_day = event_date + timedelta(days=1)
            
            # Create business hours for next day
            if original_start.tzinfo:
                next_business_start = original_start.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
                next_business_end = original_start.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_business_start = datetime.combine(next_day, datetime.min.time().replace(hour=9))
                next_business_end = datetime.combine(next_day, datetime.min.time().replace(hour=18))
                
            # Check for flights or other conflicts on next day
            next_day_conflicts = []
            for e in scheduled_events:
                if isinstance(e.start_time, str):
                    check_date = datetime.fromisoformat(e.start_time.replace('Z', '+00:00')).date()
                else:
                    check_date = e.start_time.date()
                    
                if check_date == next_day:
                    next_day_conflicts.append(e)
            
            # Try multiple strategies for next day  
            if len(next_day_conflicts) == 0 and next_business_start + duration <= next_business_end:
                logger.info(f"🕐 ✅ Moving {attraction_event.title} to {next_day} at {next_business_start.strftime('%H:%M')}")
                
                from copy import deepcopy
                adjusted_event = deepcopy(attraction_event)
                adjusted_event.start_time = next_business_start
                adjusted_event.end_time = next_business_start + duration
                return adjusted_event
            
            # 🚨 Strategy 2: Try reducing duration by 30 minutes if original is too long
            reduced_duration = duration - timedelta(minutes=30)
            if reduced_duration >= timedelta(hours=1) and next_business_start + reduced_duration <= next_business_end and len(next_day_conflicts) <= 1:
                logger.info(f"🕐 ⏱️  Trying reduced duration on next day: {reduced_duration}")
                from copy import deepcopy
                adjusted_event = deepcopy(attraction_event)
                adjusted_event.start_time = next_business_start
                adjusted_event.end_time = next_business_start + reduced_duration
                return adjusted_event
            
            # 🚨 Strategy 3: Last resort - keep original time but log warning
            logger.error(f"🕐 ❌ All strategies failed for {attraction_event.title} (next day conflicts: {len(next_day_conflicts)}), keeping original time")
            return attraction_event
        
        # Create a new CalendarEvent with adjusted times
        from copy import deepcopy
        adjusted_event = deepcopy(attraction_event)
        adjusted_event.start_time = candidate_start
        adjusted_event.end_time = candidate_start + duration
        
        # Log the successful adjustment with timezone info
        logger.info(f"🕐 ✅ Successfully adjusted {attraction_event.title}")
        logger.info(f"🕐    Final time: {adjusted_event.start_time} - {adjusted_event.end_time}")
        logger.info(f"🕐    Timezone: {adjusted_event.start_time.tzinfo}")
        
        return adjusted_event
    
    def _get_attractions_for_destination(self, tool_results: Dict[str, Any], destination_iata: str, city_name: str) -> List:
        """Extract attractions for a specific destination from multi-city attraction search results"""
        if "attraction_search" not in tool_results:
            logger.info(f"🎯 No attraction_search results found in tool_results")
            return []
        
        attraction_result = tool_results["attraction_search"]
        if not hasattr(attraction_result, 'attractions') or not attraction_result.attractions:
            logger.info(f"🎯 No attractions found in attraction_search results")
            return []
        
        # Filter attractions for this specific destination
        # In multi-city mode, each attraction has location set to IATA code
        city_attractions = []
        
        # 🔧 Convert city name to IATA code for matching
        from app.knowledge.geographical_data import GeographicalMappings
        city_iata = None
        
        # Find IATA code for this city
        for iata, city in GeographicalMappings.IATA_TO_CITY.items():
            if city.lower() == city_name.lower():
                city_iata = iata
                break
        
        logger.info(f"🎯 Looking for attractions with location matching:")
        logger.info(f"🎯   - Destination IATA: '{destination_iata}'")
        logger.info(f"🎯   - City name: '{city_name}'")
        logger.info(f"🎯   - City IATA: '{city_iata}'")
        
        for attraction in attraction_result.attractions:
            attraction_location = getattr(attraction, 'location', '')
            logger.debug(f"🎯 Checking attraction location: '{attraction_location}'")
            
            # Match by multiple criteria
            if (attraction_location == destination_iata or           # Direct IATA match
                attraction_location == city_iata or                 # City's IATA code match
                attraction_location.lower() == city_name.lower() or # City name match
                city_name.lower() in attraction_location.lower()):  # Partial city name match
                city_attractions.append(attraction)
                logger.info(f"🎯 ✅ Matched attraction: {getattr(attraction, 'name', 'Unknown')} (location: {attraction_location})")
                
        logger.info(f"🎯 Found {len(city_attractions)} attractions for {city_name} (IATA: {destination_iata})")
        logger.info(f"🎯 Total attractions in search results: {len(attraction_result.attractions)}")
        
        # 🔍 Debug: Show all attraction locations if no matches found
        if len(city_attractions) == 0 and len(attraction_result.attractions) > 0:
            logger.info(f"🎯 ❌ No matches found. All attraction locations in results:")
            for i, attraction in enumerate(attraction_result.attractions[:10]):  # Show first 10
                logger.info(f"🎯   [{i+1}] {getattr(attraction, 'name', 'Unknown')} -> location: '{getattr(attraction, 'location', 'None')}'")
        
        return city_attractions
    
    def _extract_origin_from_message(self, user_message: str) -> Optional[str]:
        """Extract origin city/airport from user message"""
        import re
        
        user_lower = user_message.lower()
        
        # Look for common origin patterns
        origin_patterns = [
            r'from\s+([a-zA-Z\s]+?)(?:\s+to|\s+→)',
            r'starting\s+from\s+([a-zA-Z\s]+?)(?:\s+to|\s+→|\s+and)',
            r'departing\s+from\s+([a-zA-Z\s]+?)(?:\s+to|\s+→)',
            r'leave\s+from\s+([a-zA-Z\s]+?)(?:\s+to|\s+→)',
            r'fly\s+from\s+([a-zA-Z\s]+?)(?:\s+to|\s+→)',
        ]
        
        for pattern in origin_patterns:
            match = re.search(pattern, user_lower)
            if match:
                origin = match.group(1).strip().title()
                # Clean up common words
                origin = re.sub(r'\b(the|city|airport)\b', '', origin, flags=re.IGNORECASE).strip()
                if len(origin) >= 2:  # Valid city name
                    logger.info(f"🛫 Extracted origin from message: {origin}")
                    return origin
        
        # Default origins based on common patterns
        if any(word in user_lower for word in ['singapore', 'sin', 'changi']):
            return "SIN"  # ✅ Add Singapore support
        elif any(word in user_lower for word in ['toronto', 'yyz', 'pearson']):
            return "YYZ"  # ✅ Use IATA code for consistency
        elif any(word in user_lower for word in ['vancouver', 'yvr']):
            return "YVR"
        elif any(word in user_lower for word in ['montreal', 'yul']):
            return "YUL"
        elif any(word in user_lower for word in ['new york', 'nyc', 'jfk', 'lga']):
            return "JFK"
        elif any(word in user_lower for word in ['london', 'lhr', 'heathrow']):
            return "LHR"
        elif any(word in user_lower for word in ['paris', 'cdg']):
            return "CDG"
        elif any(word in user_lower for word in ['tokyo', 'nrt', 'hnd']):
            return "NRT"
        
        logger.info(f"🛫 No origin found in message, using default")
        return None

    def _extract_flight_details_for_route(
        self, tool_results: Dict[str, Any], origin: str, destination: str, sequence: int
    ) -> Dict[str, Any]:
        """Extract flight details for a specific route from tool results"""
        default_details = {
            "duration_hours": 3,
            "airline": "TBA",
            "price": {"amount": "TBA", "currency": "USD"},
            "flight_number": "TBA"
        }
        
        try:
            # Check if we have flight search results
            if "flight_search" not in tool_results:
                return default_details
            
            flight_result = tool_results["flight_search"]
            if not hasattr(flight_result, 'flights') or not flight_result.flights:
                return default_details
            
            # ✅ For flight chain searches, match by route_name in details
            # ✅ Safe handling of origin and destination
            safe_origin = str(origin).upper() if origin else 'UNKNOWN'
            safe_destination = str(destination).upper() if destination else 'UNKNOWN'
            expected_route = f"{safe_origin} → {safe_destination}"
            logger.debug(f"Looking for flight route: {expected_route} (sequence: {sequence})")
            
            # Try to find a flight for this specific route
            for flight in flight_result.flights:
                # ✅ First check if this is a flight chain result with details
                if hasattr(flight, 'details') and flight.details:
                    route_name = flight.details.get('route_name', '')
                    route_sequence = flight.details.get('route_sequence', 0)
                    
                    # Match by exact route name or sequence number
                    if (route_name == expected_route or 
                        route_sequence == sequence):
                        
                        duration_minutes = getattr(flight, 'duration', 180)
                        duration_hours = max(1, duration_minutes // 60)
                        
                        logger.info(f"✅ Found flight chain match: {route_name} (seq: {route_sequence})")
                        return {
                            "duration_hours": duration_hours,
                            "airline": getattr(flight, 'airline', 'TBA'),
                            "price": {
                                "amount": getattr(flight, 'price', 'TBA'),
                                "currency": getattr(flight, 'currency', 'USD')
                            },
                            "flight_number": getattr(flight, 'flight_number', 'TBA')
                        }
                
                # ✅ Fallback: Match by destination (search_destination is set by flight_search.py)
                flight_origin = getattr(flight, 'origin', '') or ''
                flight_dest = getattr(flight, 'destination', '') or ''
                search_dest = getattr(flight, 'search_destination', '') or ''
                
                # ✅ Safe lower() calls with None checks
                flight_origin_lower = flight_origin.lower() if flight_origin else ''
                flight_dest_lower = flight_dest.lower() if flight_dest else ''
                search_dest_lower = search_dest.lower() if search_dest else ''
                
                # ✅ Safe destination matching
                safe_dest_lower = str(destination).lower() if destination else ''

                
                if (safe_dest_lower and (safe_dest_lower in search_dest_lower or 
                    safe_dest_lower in flight_dest_lower)):
                    
                    duration_minutes = getattr(flight, 'duration', 180)  # 3 hours default
                    duration_hours = max(1, duration_minutes // 60)
                    
                    logger.info(f"✅ Found flight destination match: {search_dest_lower}")
                    return {
                        "duration_hours": duration_hours,
                        "airline": getattr(flight, 'airline', 'TBA'),
                        "price": {
                            "amount": getattr(flight, 'price', 'TBA'),
                            "currency": getattr(flight, 'currency', 'USD')
                        },
                        "flight_number": getattr(flight, 'flight_number', 'TBA')
                    }
            
            # If no specific match, use first flight as template
            if flight_result.flights:
                first_flight = flight_result.flights[0]
                duration_minutes = getattr(first_flight, 'duration', 180)
                duration_hours = max(1, duration_minutes // 60)
                
                logger.warning(f"⚠️ No specific match found for {expected_route}, using first flight as template")
                return {
                    "duration_hours": duration_hours,
                    "airline": getattr(first_flight, 'airline', 'TBA'),
                    "price": {
                        "amount": getattr(first_flight, 'price', 'TBA'),
                        "currency": getattr(first_flight, 'currency', 'USD')
                    },
                    "flight_number": getattr(first_flight, 'flight_number', 'TBA')
                }
                
        except Exception as e:
            logger.warning(f"Error extracting flight details for {origin} → {destination}: {e}")
        
        logger.info(f"🔄 Using default flight details for {safe_origin} → {safe_destination}")
        return default_details

    def _create_single_destination_fallback(self, destination: str, start_date: datetime, duration: int) -> List:
        """Create a simple single-destination fallback plan"""
        events = []
        from datetime import timedelta
        import uuid
        
        try:
            # Simple 3-event plan
            arrival_time = start_date.replace(hour=10, minute=0)
            
            # Arrival
            event = self._create_calendar_event_from_data({
                "id": f"fallback_arrival_{str(uuid.uuid4())[:8]}",
                "title": f"Arrive in {destination.title()}",
                "description": f"Begin your {duration}-day trip to {destination.title()}",
                "event_type": "flight",
                "start_time": arrival_time.isoformat(),
                "end_time": (arrival_time + timedelta(hours=2)).isoformat(),
                "location": destination.title(),
                "details": {"source": "fallback_single_destination"}
            })
            if event:
                events.append(event)
            
            # Stay
            checkin_time = start_date.replace(hour=15, minute=0)
            checkout_time = (start_date + timedelta(days=duration)).replace(hour=11, minute=0)
            
            event = self._create_calendar_event_from_data({
                "id": f"fallback_stay_{str(uuid.uuid4())[:8]}",
                "title": f"Stay in {destination.title()}",
                "description": f"Accommodation for {duration} nights",
                "event_type": "hotel",
                "start_time": checkin_time.isoformat(),
                "end_time": checkout_time.isoformat(),
                "location": destination.title(),
                "details": {"source": "fallback_single_destination"}
            })
            if event:
                events.append(event)
            
            # Departure
            departure_time = (start_date + timedelta(days=duration)).replace(hour=18, minute=0)
            
            event = self._create_calendar_event_from_data({
                "id": f"fallback_departure_{str(uuid.uuid4())[:8]}",
                "title": f"Depart from {destination.title()}",
                "description": f"Return journey from {destination.title()}",
                "event_type": "flight",
                "start_time": departure_time.isoformat(),
                "end_time": (departure_time + timedelta(hours=6)).isoformat(),
                "location": destination.title(),
                "details": {"source": "fallback_single_destination"}
            })
            if event:
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error creating fallback events: {e}")
            return []

    async def update_plan_from_structured_response(
        self, 
        session_id: str, 
        agent_response: Any  # Changed from AgentResponse to Any to avoid import issues
    ) -> Dict[str, Any]:
        """Update plan directly from structured response data, avoiding LLM re-parsing"""
        
        try:
            plan = self.get_plan_by_session(session_id)
            if not plan:
                logger.warning(f"No existing plan found for session {session_id}, creating new plan")
                self.create_plan_for_session(session_id)
                plan = self.get_plan_by_session(session_id)
            
            update_summary = {
                "success": True,
                "changes_made": [],
                "events_added": 0,
                "events_updated": 0,
                "events_deleted": 0,
                "metadata_updated": False,
                "direct_structure_used": True
            }
            
            logger.info(f"Processing structured response for session {session_id}")
            
            # Check if response has structured plan data
            if hasattr(agent_response, 'structured_plan') and agent_response.structured_plan:
                # Update plan metadata from structured data
                plan_data = agent_response.structured_plan
                if plan_data.get("destination"):
                    plan.metadata.destination = plan_data["destination"]
                if plan_data.get("duration"):
                    plan.metadata.duration_days = plan_data["duration"]
                if plan_data.get("travelers"):
                    plan.metadata.travelers = plan_data["travelers"]
                # Note: start_date and end_date are not fields in TravelPlanMetadata
                # They are stored as plan_data in the structured_plan itself
                
                update_summary["metadata_updated"] = True
                update_summary["changes_made"].append("Updated plan metadata from structured data")
                logger.info(f"Updated plan metadata: destination={plan_data.get('destination')}")
            
            # Process structured plan events
            if hasattr(agent_response, 'plan_events') and agent_response.plan_events:
                new_events = []
                for event_data in agent_response.plan_events:
                    try:
                        event = self._create_calendar_event_from_structured_data(event_data)
                        if event:
                            new_events.append(event)
                            logger.debug(f"Created event: {event.title}")
                    except Exception as e:
                        logger.warning(f"Failed to create event from structured data: {e}")
                        continue
                
                # Filter out duplicate events
                filtered_events = self._filter_duplicate_events(new_events, plan.events or [])
                
                if filtered_events:
                    if plan.events is None:
                        plan.events = []
                    plan.events.extend(filtered_events)
                    update_summary["events_added"] = len(filtered_events)
                    update_summary["changes_made"].append(f"Added {len(filtered_events)} events from structured data")
                    logger.info(f"Added {len(filtered_events)} new events to plan {plan.plan_id}")
                else:
                    logger.info("No new events to add (all were duplicates or invalid)")
            
            # Save updated plan if any changes were made
            if update_summary["changes_made"]:
                plan.updated_at = datetime.now(timezone.utc)
                self._save_plan(plan)
                logger.info(f"Plan {plan.plan_id} updated successfully via structured data")
            else:
                logger.info(f"No changes needed for plan {plan.plan_id}")
            
            return update_summary
            
        except Exception as e:
            logger.error(f"Failed to update plan from structured response: {e}")
            # Fallback to original LLM-based method only if we have content to parse
            if hasattr(agent_response, 'content') and agent_response.content.strip():
                logger.info("Falling back to LLM-based plan parsing due to structured response failure")
                return await self.update_plan_from_chat_response(
                    session_id, "", agent_response.content, getattr(agent_response, 'metadata', {})
                )
            else:
                logger.warning("No fallback content available, returning failure status")
                return {
                    "success": False,
                    "error": f"Structured plan update failed and no fallback content available: {str(e)}",
                    "changes_made": [],
                    "events_added": 0,
                    "events_updated": 0,
                    "events_deleted": 0,
                    "metadata_updated": False
                }

    def _create_calendar_event_from_structured_data(self, event_data: Dict[str, Any]):
        """Create calendar event from structured event data with event type validation"""
        try:
            # Import required classes
            from app.api.schemas import CalendarEvent
            import uuid
            
            # Parse timestamps
            start_time = event_data.get("start_time")
            end_time = event_data.get("end_time")
            
            if isinstance(start_time, str):
                # Parse and convert to timezone-naive local time
                if start_time.endswith('+00:00') or start_time.endswith('Z'):
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    start_time = datetime.fromisoformat(start_time)
            elif hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
                
            if isinstance(end_time, str):
                # Parse and convert to timezone-naive local time
                if end_time.endswith('+00:00') or end_time.endswith('Z'):
                    end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    end_time = datetime.fromisoformat(end_time)
            elif hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)
            
            # Validate and map event type
            raw_event_type = event_data.get("event_type", "activity")
            validated_event_type = self._map_event_type(raw_event_type)
            
            # Create event
            event = CalendarEvent(
                id=event_data.get("id", str(uuid.uuid4())),
                title=event_data.get("title", ""),
                description=event_data.get("description", ""),
                event_type=validated_event_type,
                start_time=start_time,
                end_time=end_time,
                location=event_data.get("location", ""),
                details=event_data.get("details", {})
            )
            
            # Apply meal timing correction if this is a meal event
            corrected_event = self._correct_meal_timing(event)
            
            return corrected_event
            
        except Exception as e:
            logger.error(f"Failed to create calendar event from structured data: {e}")
            return None
    
    def _map_event_type(self, raw_type: str) -> str:
        """Map and validate event types, providing fallbacks for invalid types"""
        from app.api.schemas import CalendarEventType
        
        # Direct mapping for valid types
        try:
            return CalendarEventType(raw_type.lower())
        except ValueError:
            pass
        
        # Mapping for common variations
        type_mappings = {
            "dining": CalendarEventType.MEAL,
            "food": CalendarEventType.MEAL,
            "breakfast": CalendarEventType.MEAL,
            "lunch": CalendarEventType.MEAL,
            "dinner": CalendarEventType.MEAL,
            "eat": CalendarEventType.MEAL,
            "sightseeing": CalendarEventType.ATTRACTION,
            "tour": CalendarEventType.ACTIVITY,
            "visit": CalendarEventType.ATTRACTION,
            "transport": CalendarEventType.TRANSPORTATION,
            "travel": CalendarEventType.TRANSPORTATION,
            "accommodation": CalendarEventType.HOTEL,
            "stay": CalendarEventType.HOTEL
        }
        
        mapped_type = type_mappings.get(raw_type.lower())
        if mapped_type:
            logger.info(f"Mapped event type '{raw_type}' to '{mapped_type}'")
            return mapped_type
        
        # Default fallback
        logger.warning(f"Unknown event type '{raw_type}', defaulting to 'activity'")
        return CalendarEventType.ACTIVITY

    def _correct_meal_timing(self, event: 'CalendarEvent') -> 'CalendarEvent':
        """Correct meal event timing to ensure proper meal hours and durations"""
        from app.api.schemas import CalendarEventType
        
        if event.event_type != CalendarEventType.MEAL:
            return event  # Only process meal events
        
        # Determine meal type from title/description
        title_lower = event.title.lower()
        description_lower = event.description.lower()
        
        # Map meal types to appropriate times
        meal_timing = {
            'breakfast': {'start_hour': 8, 'duration_hours': 1},
            'brunch': {'start_hour': 10, 'duration_hours': 2}, 
            'lunch': {'start_hour': 12, 'duration_hours': 1.5},
            'afternoon tea': {'start_hour': 15, 'duration_hours': 1},
            'snack': {'start_hour': 15, 'duration_hours': 1},
            'dinner': {'start_hour': 19, 'duration_hours': 2},
            'late dinner': {'start_hour': 20, 'duration_hours': 2},
            'drinks': {'start_hour': 18, 'duration_hours': 2},
            'bar': {'start_hour': 18, 'duration_hours': 2}
        }
        
        # Detect meal type
        detected_type = None
        for meal_type in meal_timing.keys():
            if meal_type in title_lower or meal_type in description_lower:
                detected_type = meal_type
                break
        
        # Default to dinner if no specific type detected
        if not detected_type:
            detected_type = 'dinner'
            logger.info(f"Could not detect meal type for '{event.title}', defaulting to dinner timing")
        
        # Get timing config
        timing_config = meal_timing[detected_type]
        
        # Preserve the original date but correct the time
        original_date = event.start_time.date()
        corrected_start = datetime.combine(
            original_date, 
            datetime.min.time().replace(hour=timing_config['start_hour'], minute=0)
        ).replace(tzinfo=event.start_time.tzinfo)
        
        corrected_end = corrected_start + timedelta(hours=timing_config['duration_hours'])
        
        # Log the correction
        logger.info(f"Corrected meal timing for '{event.title}': {event.start_time.strftime('%H:%M')} -> {corrected_start.strftime('%H:%M')} (detected as {detected_type})")
        
        # Create corrected event
        event.start_time = corrected_start
        event.end_time = corrected_end
        
        return event

    async def _extract_plan_modifications(
        self, 
        user_message: str, 
        agent_response: str, 
        metadata: Dict[str, Any],
        existing_plan
    ) -> Dict[str, Any]:
        """Extract plan modifications using enhanced LLM analysis"""
        logger.info(f"Starting LLM-based plan modification extraction")
        
        try:
            from app.core.llm_service import get_llm_service
            llm_service = get_llm_service()
            
            if not llm_service:
                logger.warning("LLM service not available, falling back to heuristic extraction")
                return await self._fallback_extract_modifications(user_message, agent_response, metadata, existing_plan)
            
            logger.info(f"LLM service available: {type(llm_service).__name__}")
            
            # Format existing events for LLM context
            existing_events_context = ""
            if existing_plan and existing_plan.events:
                existing_events_context = self._format_existing_events_for_llm(existing_plan.events)
                logger.info(f"Formatted {len(existing_plan.events)} existing events for LLM context")
            else:
                logger.info("No existing events to format for LLM context")
            
            prompt = prompt_manager.get_prompt(
                PromptType.PLAN_MODIFICATION,
                existing_events_context=existing_events_context,
                user_message=user_message,
                agent_response=agent_response
            )

            response = await llm_service.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3000  # Increased for multiple meal events
            )
            
            try:
                # Extract content from LLMResponse object
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Add detailed logging for debugging
                logger.debug(f"Raw LLM response: {repr(response)}")
                logger.debug(f"Extracted content: {repr(response_content)}")
                
                # Check if response is empty or None
                if not response_content or response_content.strip() == "":
                    logger.warning("LLM returned empty response, falling back to heuristic extraction")
                    return await self._fallback_extract_modifications(user_message, agent_response, metadata, existing_plan)
                
                # Clean up LLM response that might be wrapped in code blocks
                cleaned_content = response_content.strip()
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]  # Remove ```json
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]  # Remove ending ```
                cleaned_content = cleaned_content.strip()
                
                logger.debug(f"Cleaned JSON content: {cleaned_content[:200]}...")
                
                modifications = json.loads(cleaned_content)
                
                logger.info(f"LLM plan modification extraction successful: {len(modifications.get('new_events', []))} new events, {len(modifications.get('updated_events', []))} updates, {len(modifications.get('deleted_event_ids', []))} deletions")
                
                # Convert new events to CalendarEvent objects
                new_events = []
                for event_data in modifications.get("new_events", []):
                    try:
                        event = self._create_calendar_event_from_structured_data(event_data)
                        if event:
                            new_events.append(event)
                    except Exception as e:
                        logger.warning(f"Error creating event from data {event_data}: {e}")
                        continue
                
                modifications["new_events"] = new_events
                logger.info(f"Successfully converted {len(new_events)} events from LLM response")
                return modifications
                
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response not valid JSON: {e}. Response content: {repr(response_content)}")
                return await self._fallback_extract_modifications(user_message, agent_response, metadata, existing_plan)
                
        except Exception as e:
            logger.error(f"Error in LLM plan modification extraction: {e}")
            return await self._fallback_extract_modifications(user_message, agent_response, metadata, existing_plan)
    
    def _format_existing_events_for_llm(self, events: List) -> str:
        """Format existing events for LLM context"""
        if not events:
            return "No existing events"
        
        event_lines = []
        for event in events:
            event_lines.append(
                f"- {event.id}: {event.title} ({event.event_type}) "
                f"from {event.start_time} to {event.end_time} at {event.location or 'TBD'}"
            )
        
        return "\n".join(event_lines)
    
    def _create_calendar_event_from_data(self, event_data: Dict[str, Any]):
        """Create CalendarEvent from LLM-generated data with validation"""
        try:
            # Validate required fields
            required_fields = ["title", "event_type", "start_time", "end_time"]
            for field in required_fields:
                if not event_data.get(field):
                    logger.warning(f"Event missing required field '{field}': {event_data}")
                    return None
            
            # Parse and validate event type using the mapping function
            event_type_str = event_data.get("event_type", "activity")
            event_type = self._map_event_type(event_type_str)
            
            # Parse and validate datetime fields
            try:
                start_time = datetime.fromisoformat(event_data["start_time"])
                end_time = datetime.fromisoformat(event_data["end_time"])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid datetime format in event {event_data}: {e}")
                return None
            
            event = CalendarEvent(
                id=event_data.get("id", f"event_{uuid.uuid4().hex[:8]}"),
                title=event_data.get("title", "Travel Event"),
                description=event_data.get("description"),
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                location=event_data.get("location"),
                details=event_data.get("details", {}),
                confidence=0.8,
                source="llm_plan_modification"
            )
            return event
            
        except Exception as e:
            logger.warning(f"Error creating calendar event from data {event_data}: {e}")
            return None
    
    async def _fallback_extract_modifications(
        self, user_message: str, agent_response: str, metadata: Dict[str, Any], existing_plan
    ) -> Dict[str, Any]:
        """Fallback plan modification extraction using heuristics - now plan-aware"""
        logger.info(f"Using fallback heuristic extraction for plan modifications")
        
        # Analyze user intent for modifications
        user_lower = user_message.lower()
        response_lower = agent_response.lower()
        
        modifications = {
            "new_events": [],
            "updated_events": [],
            "deleted_event_ids": [],
            "plan_modifications": {
                "reason": "Heuristic extraction used due to LLM unavailability",
                "impact": "Analyzed conversation for plan modifications"
            }
        }
        
        # Check if user is asking for modifications vs new events
        modification_keywords = ["change", "update", "modify", "adjust", "move", "reschedule", "cancel", "remove", "delete"]
        is_modification_request = any(keyword in user_lower for keyword in modification_keywords)
        
        if is_modification_request:
            logger.info(f"Detected modification request in user message: {user_message[:100]}...")
            
            # ✅ Try to handle destination removal heuristically
            removal_keywords = ["remove", "delete", "skip", "don't go", "take out", "exclude", "without"]
            city_removal_detected = any(keyword in user_lower for keyword in removal_keywords)
            
            if city_removal_detected and existing_plan and existing_plan.events:
                logger.info("Attempting heuristic destination removal...")
                deleted_event_ids = self._heuristic_remove_destination_events(
                    user_message, existing_plan.events
                )
                if deleted_event_ids:
                    modifications["deleted_event_ids"] = deleted_event_ids
                    modifications["plan_modifications"]["reason"] = "Heuristic destination removal based on user request"
                    modifications["plan_modifications"]["impact"] = f"Removed {len(deleted_event_ids)} events related to destinations mentioned for removal"
                    logger.info(f"Heuristically identified {len(deleted_event_ids)} events for removal")
                else:
                    logger.warning(f"Could not identify specific events to remove heuristically")
                    modifications["plan_modifications"]["impact"] = "Modification request detected but could not be processed automatically"
            else:
                # For modifications, be conservative and don't add new events
                # Instead, log what we would need to analyze
                logger.warning(f"Modification request detected but LLM analysis failed. Manual review may be needed.")
                modifications["plan_modifications"]["impact"] = "Modification request detected but could not be processed automatically"
            
        else:
            # Only add new events if this seems like a request for additional activities
            new_activity_keywords = ["add", "include", "also", "visit", "see", "go to", "book", "reserve"]
            is_addition_request = any(keyword in user_lower for keyword in new_activity_keywords)
            
            if is_addition_request or not existing_plan or not existing_plan.events:
                logger.info(f"Detected addition request or empty plan, using heuristic extraction")
                # Use existing heuristic method but be more careful
                new_events = self._heuristic_extract_events(user_message, agent_response, metadata)
                
                # Filter out events that might duplicate existing ones
                if existing_plan and existing_plan.events:
                    filtered_events = self._filter_duplicate_events(new_events, existing_plan.events)
                    logger.info(f"Filtered {len(new_events) - len(filtered_events)} potential duplicate events")
                    modifications["new_events"] = filtered_events
                else:
                    modifications["new_events"] = new_events
                    
                modifications["plan_modifications"]["impact"] = f"Added {len(modifications['new_events'])} new events based on conversation analysis"
            else:
                logger.info(f"No clear addition request detected and plan exists, not adding events")
                modifications["plan_modifications"]["impact"] = "No clear plan modifications detected in conversation"
        
        return modifications
    
    def _heuristic_remove_destination_events(self, user_message: str, existing_events: List) -> List[str]:
        """Heuristically identify events to remove based on destination removal requests"""
        import re
        
        deleted_event_ids = []
        user_lower = user_message.lower()
        
        # Extract city names mentioned for removal
        # Look for patterns like "remove Paris", "delete London", "skip Amsterdam"
        removal_patterns = [
            r'(?:remove|delete|skip|exclude|without|take out|don\'t go to)\s+([a-zA-Z\s]+?)(?:\s+and|\s*,|\s*$|\s+from)',
            r'(?:not|no)\s+([a-zA-Z\s]+?)(?:\s+and|\s*,|\s*$)',
            r'(?:without|excluding)\s+([a-zA-Z\s]+?)(?:\s+and|\s*,|\s*$)',
        ]
        
        cities_to_remove = set()
        for pattern in removal_patterns:
            matches = re.findall(pattern, user_lower)
            for match in matches:
                city = match.strip().title()
                if len(city) >= 3:  # Valid city name
                    cities_to_remove.add(city)
                    # Also add common variations
                    if city == "Amsterdam":
                        cities_to_remove.update(["Amsterdam", "AMS"])
                    elif city == "Vienna":
                        cities_to_remove.update(["Vienna", "VIE"])
                    elif city == "London":
                        cities_to_remove.update(["London", "LON"])
                    elif city == "Paris":
                        cities_to_remove.update(["Paris", "PAR"])
                    elif city == "Berlin":
                        cities_to_remove.update(["Berlin", "BER"])
                    elif city == "Rome":
                        cities_to_remove.update(["Rome", "ROM"])
                    elif city == "Barcelona":
                        cities_to_remove.update(["Barcelona", "BCN"])
                    elif city == "Prague":
                        cities_to_remove.update(["Prague", "PRG"])
        
        logger.info(f"Cities identified for removal: {cities_to_remove}")
        
        if not cities_to_remove:
            return deleted_event_ids
        
        # Check each event for removal criteria
        for event in existing_events:
            should_remove = False
            
            # Check event title for city names
            event_title = (event.title or '').lower()
            for city in cities_to_remove:
                if city.lower() in event_title:
                    should_remove = True
                    logger.info(f"Marking event '{event.title}' for removal (title contains '{city}')")
                    break
            
            # Check event location for city names
            if not should_remove:
                event_location = (event.location or '').lower()
                for city in cities_to_remove:
                    if city.lower() in event_location:
                        should_remove = True
                        logger.info(f"Marking event '{event.title}' for removal (location contains '{city}')")
                        break
            
            # Check event details for destination codes or city names
            if not should_remove and hasattr(event, 'details') and event.details:
                event_details_str = str(event.details).lower()
                for city in cities_to_remove:
                    if city.lower() in event_details_str:
                        should_remove = True
                        logger.info(f"Marking event '{event.title}' for removal (details contain '{city}')")
                        break
                
                # Check for destination field specifically
                if not should_remove and isinstance(event.details, dict):
                    destination_field = event.details.get('destination', '').upper()
                    origin_field = event.details.get('origin', '').upper()
                    for city in cities_to_remove:
                        city_upper = city.upper()
                        if city_upper == destination_field or city_upper == origin_field:
                            should_remove = True
                            logger.info(f"Marking event '{event.title}' for removal (destination/origin field matches '{city}')")
                            break
            
            if should_remove:
                deleted_event_ids.append(event.id)
        
        logger.info(f"Identified {len(deleted_event_ids)} events for heuristic removal")
        return deleted_event_ids
    
    def _filter_duplicate_events(self, new_events: List, existing_events: List) -> List:
        """Filter out events that might be duplicates of existing ones"""
        filtered_events = []
        
        for new_event in new_events:
            is_duplicate = False
            
            for existing_event in existing_events:
                # Check for similar titles, types, and overlapping times
                if (existing_event.event_type == new_event.event_type and
                    self._events_are_similar(existing_event, new_event)):
                    logger.info(f"Filtering potential duplicate: {new_event.title} (similar to existing {existing_event.title})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_events.append(new_event)
        
        return filtered_events
    
    def _events_are_similar(self, event1, event2) -> bool:
        """Check if two events are similar enough to be considered duplicates"""
        # Check title similarity
        title1_words = set(event1.title.lower().split())
        title2_words = set(event2.title.lower().split())
        title_overlap = len(title1_words.intersection(title2_words)) / max(len(title1_words), len(title2_words), 1)
        
        # Check time overlap (same day or overlapping times)
        time_overlap = False
        if event1.start_time and event2.start_time:
            # Check if they're on the same day
            if event1.start_time.date() == event2.start_time.date():
                time_overlap = True
        
        # Consider similar if significant title overlap and same day
        return title_overlap > 0.4 and time_overlap
    
    def _update_existing_event(self, plan, updated_event_data: Dict[str, Any]):
        """Update an existing event in the plan"""
        event_id = updated_event_data.get("id")
        if not event_id:
            logger.warning("Cannot update event without ID")
            return
        
        for i, existing_event in enumerate(plan.events):
            if existing_event.id == event_id:
                # Update fields that are provided
                if "title" in updated_event_data:
                    existing_event.title = updated_event_data["title"]
                if "description" in updated_event_data:
                    existing_event.description = updated_event_data["description"]
                if "start_time" in updated_event_data:
                    try:
                        existing_event.start_time = datetime.fromisoformat(updated_event_data["start_time"])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid start_time for update: {e}")
                if "end_time" in updated_event_data:
                    try:
                        existing_event.end_time = datetime.fromisoformat(updated_event_data["end_time"])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid end_time for update: {e}")
                if "location" in updated_event_data:
                    existing_event.location = updated_event_data["location"]
                if "details" in updated_event_data:
                    existing_event.details = updated_event_data["details"]
                
                logger.info(f"Updated event {event_id}: {existing_event.title}")
                return
        
        logger.warning(f"Event with ID {event_id} not found for update")
    
    async def _extract_events_from_response(
        self, 
        user_message: str, 
        agent_response: str, 
        metadata: Dict[str, Any]
    ) -> List[CalendarEvent]:
        """Extract calendar events from agent response using LLM or heuristics"""
        events = []
        
        # Try LLM-based extraction first
        try:
            from app.core.llm_service import get_llm_service
            llm_service = get_llm_service()
            
            if llm_service:
                events = await self._llm_extract_events(
                    user_message, agent_response, metadata, llm_service
                )
        except Exception as e:
            logger.warning(f"LLM event extraction failed: {e}")
        
        # Fallback to heuristic extraction
        if not events:
            events = self._heuristic_extract_events(user_message, agent_response, metadata)
        
        return events
    
    async def _llm_extract_events(
        self, 
        user_message: str, 
        agent_response: str, 
        metadata: Dict[str, Any],
        llm_service
    ) -> List[CalendarEvent]:
        """Use LLM to extract calendar events from response"""
        try:
            prompt = prompt_manager.get_prompt(
                PromptType.EVENT_EXTRACTION,
                user_message=user_message,
                agent_response=agent_response
            )
            
            # Use LLM to extract events
            response = await llm_service.generate_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse response and convert to CalendarEvent objects
            try:
                events_data = json.loads(response.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response not valid JSON: {e}")
                return []
            
            events = []
            
            for event_data in events_data:
                try:
                    # Safely parse start_time
                    start_time_str = event_data.get("start_time")
                    if not start_time_str:
                        logger.warning("Event missing start_time, skipping")
                        continue
                    try:
                        start_time = datetime.fromisoformat(start_time_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid start_time format '{start_time_str}': {e}")
                        continue
                    
                    # Safely parse end_time
                    end_time_str = event_data.get("end_time")
                    if not end_time_str:
                        logger.warning("Event missing end_time, skipping")
                        continue
                    try:
                        end_time = datetime.fromisoformat(end_time_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid end_time format '{end_time_str}': {e}")
                        continue
                    
                    # Safely parse event_type
                    event_type_str = event_data.get("type", "activity")
                    try:
                        event_type = CalendarEventType(event_type_str)
                    except ValueError:
                        logger.warning(f"Unknown event type '{event_type_str}', defaulting to activity")
                        event_type = CalendarEventType.ACTIVITY
                    
                    event = CalendarEvent(
                        id=f"event_{uuid.uuid4().hex[:8]}",
                        title=event_data.get("title", "Travel Event"),
                        description=event_data.get("description"),
                        event_type=event_type,
                        start_time=start_time,
                        end_time=end_time,
                        location=event_data.get("location"),
                        details=event_data.get("details", {}),
                        confidence=0.7,
                        source="llm_extraction"
                    )
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Error processing event data {event_data}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.warning(f"LLM event extraction error: {e}")
            return []
    
    def _heuristic_extract_events(
        self, 
        user_message: str, 
        agent_response: str, 
        metadata: Dict[str, Any]
    ) -> List[CalendarEvent]:
        """Extract events using simple heuristics with intelligent date placement"""
        events = []
        
        # Get trip parameters - use local time without timezone
        start_date = self._extract_start_date(user_message) or datetime.now().replace(tzinfo=None)
        duration_days = self._extract_duration_from_message(user_message) or 3
        destination = metadata.get("destination", "")
        
        # Check if agent mentioned specific attractions/hotels/flights
        response_lower = agent_response.lower()
        
        # Look for flight mentions - place at start and end of trip
        if any(word in response_lower for word in ["flight", "airline", "airport", "departure", "arrival"]):
            # Outbound flight at start of trip
            outbound_departure = start_date.replace(hour=10, minute=0, second=0, microsecond=0)
            outbound_arrival = outbound_departure + timedelta(hours=6)  # Assume 6 hour flight
            
            events.append(CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
                title=f"Flight to {destination}",
                description="Outbound flight",
                event_type=CalendarEventType.FLIGHT,
                start_time=outbound_departure,
                end_time=outbound_arrival,
                location="Airport",
                confidence=0.7,
                source="heuristic_extraction"
            ))
            
            # Return flight at end of trip
            if duration_days > 1:
                return_date = start_date + timedelta(days=duration_days - 1)
                return_departure = return_date.replace(hour=16, minute=0, second=0, microsecond=0)
                return_arrival = return_departure + timedelta(hours=6)
                
                events.append(CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=f"Flight from {destination}",
                    description="Return flight",
                    event_type=CalendarEventType.FLIGHT,
                    start_time=return_departure,
                    end_time=return_arrival,
                    location="Airport",
                    confidence=0.7,
                    source="heuristic_extraction"
                ))
        
        # Look for hotel mentions - create multi-day stay
        if any(word in response_lower for word in ["hotel", "accommodation", "stay", "check-in"]):
            arrival_day = start_date
            if events:  # If we have flights, check in after arrival
                arrival_day = start_date + timedelta(hours=8)  # 8 hours after start for arrival + travel
                
            # Calculate appropriate check-in time
            # If arrival is before 3 PM, check in at 3 PM same day
            # If arrival is 3 PM or later, check in at 3 PM next day
            if arrival_day.hour >= 15:  # 3 PM or later
                check_in_date = arrival_day.date() + timedelta(days=1)
                check_in = datetime.combine(check_in_date, datetime.min.time()).replace(hour=15, minute=0, tzinfo=start_date.tzinfo)
            else:
                check_in_date = arrival_day.date()
                check_in = datetime.combine(check_in_date, datetime.min.time()).replace(hour=15, minute=0, tzinfo=start_date.tzinfo)
                
            check_out = (start_date + timedelta(days=duration_days)).replace(hour=11, minute=0, second=0, microsecond=0)
            
            events.append(CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
                title=f"Hotel Stay in {destination}",
                description=f"Accommodation for {duration_days} days",
                event_type=CalendarEventType.HOTEL,
                start_time=check_in,
                end_time=check_out,
                location=destination,
                confidence=0.7,
                source="heuristic_extraction"
            ))
        
        # Look for attraction mentions - spread across trip days
        if any(word in response_lower for word in ["visit", "attraction", "museum", "park", "tour", "sightseeing"]):
            # ✅ FIXED: Create attractions for all available activity days, not just 3
            available_activity_days = max(1, duration_days - 2)  # At least 1 day of activities
            
            # Create attractions for middle days of the trip
            for day in range(available_activity_days):
                visit_date = start_date + timedelta(days=day + 1)  # Start from day 2
                # Safe time calculation to avoid exceeding 24-hour format
                base_hour = 10
                hour_offset = min(day * 2, 8)  # Max offset 8 hours, avoiding past 18:00
                target_hour = base_hour + hour_offset
                visit_start = visit_date.replace(hour=target_hour, minute=0, second=0, microsecond=0)
                visit_end = visit_start + timedelta(hours=3)  # 3 hour visits
                
                attraction_names = ["City Tour", "Local Attractions", "Cultural Sites", "Historic District", "Shopping District", "Art Galleries", "Local Markets"]
                
                events.append(CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=f"{attraction_names[day % len(attraction_names)]} - {destination}",
                    description="Visit local attractions and landmarks",
                    event_type=CalendarEventType.ATTRACTION,
                    start_time=visit_start,
                    end_time=visit_end,
                    location=destination,
                    confidence=0.6,
                    source="heuristic_extraction"
                ))
        
        # Look for meal/food mentions - create meal events throughout the trip
        if any(word in response_lower for word in ["meal", "food", "restaurant", "dining", "eat", "breakfast", "lunch", "dinner", "cuisine", "specialties"]):
            logger.info(f"Creating meal events for {duration_days} days")
            
            for day in range(duration_days):
                meal_date = start_date + timedelta(days=day)
                
                # Create breakfast
                breakfast_start = meal_date.replace(hour=8, minute=0, second=0, microsecond=0)
                breakfast_end = breakfast_start + timedelta(hours=1)
                
                events.append(CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=f"Breakfast in {destination}",
                    description="Local breakfast experience",
                    event_type=CalendarEventType.MEAL,
                    start_time=breakfast_start,
                    end_time=breakfast_end,
                    location=destination,
                    confidence=0.7,
                    source="heuristic_extraction"
                ))
                
                # Create lunch
                lunch_start = meal_date.replace(hour=12, minute=30, second=0, microsecond=0)
                lunch_end = lunch_start + timedelta(hours=1, minutes=30)
                
                events.append(CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=f"Lunch in {destination}",
                    description="Local lunch specialties",
                    event_type=CalendarEventType.MEAL,
                    start_time=lunch_start,
                    end_time=lunch_end,
                    location=destination,
                    confidence=0.7,
                    source="heuristic_extraction"
                ))
                
                # Create dinner
                dinner_start = meal_date.replace(hour=19, minute=0, second=0, microsecond=0)
                dinner_end = dinner_start + timedelta(hours=2)
                
                events.append(CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=f"Dinner in {destination}",
                    description="Local dinner cuisine",
                    event_type=CalendarEventType.MEAL,
                    start_time=dinner_start,
                    end_time=dinner_end,
                    location=destination,
                    confidence=0.7,
                    source="heuristic_extraction"
                ))
                
            logger.info(f"Created {duration_days * 3} meal events for the trip")
        
        return events
    
    def _extract_duration_from_message(self, user_message: str) -> Optional[int]:
        """Extract trip duration from user message"""
        import re
        
        user_lower = user_message.lower()
        
        # Look for duration patterns
        duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*nights?',
            r'(\d+)\s*weeks?',
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, user_lower)
            for match in matches:
                try:
                    days = int(match)
                    if 'week' in pattern:
                        days *= 7
                    return days
                except ValueError:
                    continue
        
        return None
    
    def _extract_metadata_updates(
        self, 
        user_message: str, 
        agent_response: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata updates from chat"""
        updates = {}
        
        # Extract destination if mentioned
        if "destination" in metadata:
            updates["destination"] = metadata["destination"]
        
        # Extract other metadata from user message
        user_lower = user_message.lower()
        
        # Look for duration mentions
        if "day" in user_lower:
            import re
            days_match = re.search(r'(\d+)\s*days?', user_lower)
            if days_match:
                updates["duration_days"] = int(days_match.group(1))
        
        # Look for traveler count
        if any(word in user_lower for word in ["people", "person", "traveler", "guest"]):
            import re
            travelers_match = re.search(r'(\d+)\s*(?:people|person|traveler|guest)', user_lower)
            if travelers_match:
                updates["travelers"] = int(travelers_match.group(1))
        
        # Look for budget mentions
        if any(word in user_lower for word in ["budget", "$", "dollar", "usd", "eur", "cost"]):
            import re
            budget_match = re.search(r'[\$]?(\d+(?:,\d{3})*(?:\.\d{2})?)', user_lower)
            if budget_match:
                updates["budget"] = float(budget_match.group(1).replace(",", ""))
        
        # Extract travel dates if mentioned
        start_date = self._extract_start_date(user_message)
        if start_date:
            updates["start_date"] = start_date
        
        # Update completion status based on response content
        if agent_response:
            if any(word in agent_response.lower() for word in ["complete", "finalized", "ready"]):
                updates["completion_status"] = "complete"
            elif any(word in agent_response.lower() for word in ["partial", "more information", "additional"]):
                updates["completion_status"] = "partial"
        
        # Always update timestamp - use timezone-naive
        updates["last_updated"] = datetime.now().replace(tzinfo=None)
        
        return updates
    
    def _extract_start_date(self, user_message: str) -> Optional[datetime]:
        """Extract travel start date from user message"""
        import re
        from dateutil import parser
        
        user_lower = user_message.lower()
        
        # Look for specific date patterns
        date_patterns = [
            r'(\w+\s+\d{1,2},?\s+\d{4})',  # "July 26, 2025" or "July 26 2025"
            r'(\d{1,2}/\d{1,2}/\d{4})',    # "7/26/2025"
            r'(\d{4}-\d{1,2}-\d{1,2})',    # "2025-07-26"
            r'(next\s+\w+)',               # "next week", "next month"
            r'(tomorrow|today)',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle relative dates - use timezone-naive datetime
                    if 'tomorrow' in match.lower():
                        return datetime.now().replace(tzinfo=None) + timedelta(days=1)
                    elif 'today' in match.lower():
                        return datetime.now().replace(tzinfo=None)
                    elif 'next week' in match.lower():
                        return datetime.now().replace(tzinfo=None) + timedelta(weeks=1)
                    elif 'next month' in match.lower():
                        return datetime.now().replace(tzinfo=None) + timedelta(days=30)
                    else:
                        # Try to parse absolute dates
                        parsed_date = parser.parse(match, fuzzy=True)
                        return parsed_date.replace(tzinfo=None)
                except Exception:
                    continue
        
        return None
    
    def add_event(self, session_id: str, event: CalendarEvent) -> bool:
        """Add an event to a plan"""
        plan = self.get_plan_by_session(session_id)
        if not plan:
            return False
        
        plan.events.append(event)
        plan.updated_at = datetime.now(timezone.utc)
        self._save_plan(plan)
        return True
    
    def remove_event(self, session_id: str, event_id: str) -> bool:
        """Remove an event from a plan"""
        plan = self.get_plan_by_session(session_id)
        if not plan:
            return False
        
        original_count = len(plan.events)
        plan.events = [e for e in plan.events if e.id != event_id]
        
        if len(plan.events) < original_count:
            plan.updated_at = datetime.now(timezone.utc)
            self._save_plan(plan)
            return True
        
        return False
    
    def update_event(self, session_id: str, event: CalendarEvent) -> bool:
        """Update an existing event"""
        plan = self.get_plan_by_session(session_id)
        if not plan:
            return False
        
        for i, existing_event in enumerate(plan.events):
            if existing_event.id == event.id:
                plan.events[i] = event
                plan.updated_at = datetime.now(timezone.utc)
                self._save_plan(plan)
                return True
        
        return False
    
    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan"""
        if plan_id not in self.plans:
            return False
        
        plan = self.plans[plan_id]
        session_id = plan.session_id
        
        # Remove from memory
        del self.plans[plan_id]
        if session_id in self.session_to_plan:
            del self.session_to_plan[session_id]
        
        # Remove file
        plan_file = self.storage_path / f"{plan_id}.json"
        if plan_file.exists():
            plan_file.unlink()
        
        logger.info(f"Deleted plan {plan_id}")
        return True
    
    def _load_plans(self):
        """Load existing plans from storage"""
        try:
            for plan_file in self.storage_path.glob("*.json"):
                with open(plan_file, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)
                    plan = SessionTravelPlan(**plan_data)
                    self.plans[plan.plan_id] = plan
                    self.session_to_plan[plan.session_id] = plan.plan_id
        except Exception as e:
            logger.error(f"Error loading plans: {e}")
    
    def _save_plan(self, plan: SessionTravelPlan):
        """Save plan to storage with proper datetime serialization"""
        try:
            plan_file = self.storage_path / f"{plan.plan_id}.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(), f, default=self._datetime_serializer, indent=2)
        except Exception as e:
            logger.error(f"Error saving plan {plan.plan_id}: {e}")
    
    def _datetime_serializer(self, obj):
        """Custom serializer for datetime objects to ensure simple ISO format without timezone"""
        if isinstance(obj, datetime):
            # Always return timezone-naive ISO format
            if obj.tzinfo is not None:
                obj = obj.replace(tzinfo=None)
            return obj.isoformat()  # Produces "2025-03-10T10:00:00" format
        return str(obj)


# Global plan manager instance
_plan_manager: Optional[PlanManager] = None

def get_plan_manager() -> PlanManager:
    """Get global plan manager instance"""
    global _plan_manager
    if _plan_manager is None:
        _plan_manager = PlanManager()
    return _plan_manager 