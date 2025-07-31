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
            if modifications.get("new_events"):
                existing_plan.events.extend(modifications["new_events"])
                update_summary["events_added"] = len(modifications["new_events"])
                update_summary["changes_made"].append(f"Added {len(modifications['new_events'])} new events")
                logger.info(f"Added {len(modifications['new_events'])} new events to plan {existing_plan.plan_id}")

            if modifications.get("updated_events"):
                for updated_event in modifications["updated_events"]:
                    self._update_existing_event(existing_plan, updated_event)
                update_summary["events_updated"] = len(modifications["updated_events"])
                update_summary["changes_made"].append(f"Updated {len(modifications['updated_events'])} events")
                logger.info(f"Updated {len(modifications['updated_events'])} events in plan {existing_plan.plan_id}")

            if modifications.get("deleted_event_ids"):
                for event_id in modifications["deleted_event_ids"]:
                    existing_plan.events = [e for e in existing_plan.events if e.id != event_id]
                update_summary["events_deleted"] = len(modifications["deleted_event_ids"])
                update_summary["changes_made"].append(f"Deleted {len(modifications['deleted_event_ids'])} events")
                logger.info(f"Deleted {len(modifications['deleted_event_ids'])} events from plan {existing_plan.plan_id}")

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
        """Create calendar event from structured event data"""
        try:
            # Import required classes
            from app.api.schemas import CalendarEvent
            import uuid
            
            # Parse timestamps
            start_time = event_data.get("start_time")
            end_time = event_data.get("end_time")
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            
            # Create event
            event = CalendarEvent(
                id=event_data.get("id", str(uuid.uuid4())),
                title=event_data.get("title", ""),
                description=event_data.get("description", ""),
                event_type=event_data.get("event_type", "activity"),
                start_time=start_time,
                end_time=end_time,
                location=event_data.get("location", ""),
                details=event_data.get("details", {})
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to create calendar event from structured data: {e}")
            return None

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
            
            prompt = f"""Analyze this travel planning conversation and determine what changes should be made to the existing travel plan.

EXISTING PLAN EVENTS:
{existing_events_context}

USER REQUEST: {user_message}
AGENT RESPONSE: {agent_response}

TASK: Determine what modifications should be made to the existing plan based on this conversation.

Analyze for:
1. NEW events to add (flights, hotels, attractions, restaurants, activities)
2. UPDATES to existing events (time changes, location changes, details updates)
3. DELETIONS of existing events (if user wants to remove something)
4. PLAN METADATA changes (destination, dates, budget, etc.)

For each event, determine:
- Event type (flight/hotel/attraction/restaurant/activity)
- Title and description
- Start and end times (use ISO format: YYYY-MM-DDTHH:MM:SS+00:00)
- Location
- Any specific details mentioned

Return ONLY a valid JSON response with this exact structure:
{{
    "new_events": [
        {{
            "title": "Event Title",
            "description": "Event description",
            "event_type": "flight|hotel|attraction|restaurant|activity",
            "start_time": "2025-07-27T10:00:00+00:00",
            "end_time": "2025-07-27T16:00:00+00:00",
            "location": "Location name",
            "details": {{}}
        }}
    ],
    "updated_events": [
        {{
            "id": "existing_event_id",
            "title": "Updated Title",
            "start_time": "2025-07-27T11:00:00+00:00",
            "end_time": "2025-07-27T17:00:00+00:00"
        }}
    ],
    "deleted_event_ids": ["event_id_to_delete"],
    "plan_modifications": {{
        "reason": "Why these changes were made",
        "impact": "How this affects the overall plan"
    }}
}}

If no changes are needed, return empty arrays for each section."""

            response = await llm_service.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
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
                        event = self._create_calendar_event_from_data(event_data)
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
            
            # Parse and validate event type
            event_type_str = event_data.get("event_type", "activity")
            try:
                event_type = CalendarEventType(event_type_str)
            except ValueError:
                logger.warning(f"Unknown event type '{event_type_str}', defaulting to activity")
                event_type = CalendarEventType.ACTIVITY
            
            # Parse and validate datetime fields
            try:
                start_time = datetime.fromisoformat(event_data["start_time"])
                end_time = datetime.fromisoformat(event_data["end_time"])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid datetime format in event {event_data}: {e}")
                return None
            
            event = CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
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
            prompt = f"""
            Extract calendar events from this travel planning conversation.
            
            User: {user_message}
            Agent: {agent_response}
            
            Extract any specific travel events mentioned (flights, hotels, attractions, restaurants, activities).
            For each event, determine:
            - Title
            - Type (flight/hotel/attraction/restaurant/activity)
            - Start time (estimate if not explicit)
            - Duration/end time
            - Location
            - Description
            
            Return as JSON array of events. If no specific events are mentioned, return empty array.
            """
            
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
        
        # Get trip parameters
        start_date = self._extract_start_date(user_message) or datetime.now(timezone.utc)
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
            # Create attractions for middle days of the trip
            for day in range(min(duration_days, 3)):  # Max 3 attractions
                visit_date = start_date + timedelta(days=day)
                visit_start = visit_date.replace(hour=9 + (day * 2), minute=0, second=0, microsecond=0)  # Stagger times
                visit_end = visit_start + timedelta(hours=2)  # 2 hour visits
                
                attraction_names = ["City Tour", "Local Attractions", "Cultural Sites"]
                
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
        
        # Always update timestamp
        updates["last_updated"] = datetime.now(timezone.utc)
        
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
                    # Handle relative dates
                    if 'tomorrow' in match.lower():
                        return datetime.now(timezone.utc) + timedelta(days=1)
                    elif 'today' in match.lower():
                        return datetime.now(timezone.utc)
                    elif 'next week' in match.lower():
                        return datetime.now(timezone.utc) + timedelta(weeks=1)
                    elif 'next month' in match.lower():
                        return datetime.now(timezone.utc) + timedelta(days=30)
                    else:
                        # Try to parse absolute dates
                        parsed_date = parser.parse(match, fuzzy=True)
                        return parsed_date.replace(tzinfo=timezone.utc)
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
        """Save plan to storage"""
        try:
            plan_file = self.storage_path / f"{plan.plan_id}.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving plan {plan.plan_id}: {e}")


# Global plan manager instance
_plan_manager: Optional[PlanManager] = None

def get_plan_manager() -> PlanManager:
    """Get global plan manager instance"""
    global _plan_manager
    if _plan_manager is None:
        _plan_manager = PlanManager()
    return _plan_manager 