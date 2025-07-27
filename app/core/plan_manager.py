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
    ) -> bool:
        """
        Asynchronously update plan based on chat response
        This runs after the chat response is sent to user
        """
        try:
            # Get or create plan
            plan = self.get_plan_by_session(session_id)
            if not plan:
                logger.warning(f"Could not get/create plan for session {session_id}")
                return False
            
            # Extract events from chat response
            new_events = await self._extract_events_from_response(
                user_message, agent_response, response_metadata
            )
            
            # Update plan metadata
            metadata_updates = self._extract_metadata_updates(
                user_message, agent_response, response_metadata
            )
            
            # Apply updates
            updated = False
            
            if new_events:
                plan.events.extend(new_events)
                updated = True
                logger.info(f"Added {len(new_events)} events to plan {plan.plan_id}")
            
            if metadata_updates:
                # Update metadata fields
                for key, value in metadata_updates.items():
                    if hasattr(plan.metadata, key):
                        setattr(plan.metadata, key, value)
                updated = True
                logger.info(f"Updated plan metadata: {metadata_updates}")
            
            if updated:
                plan.updated_at = datetime.now(timezone.utc)
                self._save_plan(plan)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update plan from chat response: {e}")
            return False
    
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
            events_data = json.loads(response.strip())
            events = []
            
            for event_data in events_data:
                event = CalendarEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    title=event_data.get("title", "Travel Event"),
                    description=event_data.get("description"),
                    event_type=CalendarEventType(event_data.get("type", "activity")),
                    start_time=datetime.fromisoformat(event_data.get("start_time")),
                    end_time=datetime.fromisoformat(event_data.get("end_time")),
                    location=event_data.get("location"),
                    details=event_data.get("details", {}),
                    confidence=0.7,
                    source="llm_extraction"
                )
                events.append(event)
            
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
        """Extract events using simple heuristics"""
        events = []
        
        # Check if agent mentioned specific attractions/hotels/flights
        response_lower = agent_response.lower()
        
        # Look for hotel mentions
        if any(word in response_lower for word in ["hotel", "accommodation", "stay", "check-in"]):
            base_date = datetime.now(timezone.utc)
            check_in = base_date.replace(hour=15, minute=0, second=0, microsecond=0)  # 3 PM today
            check_out = (base_date + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)  # 11 AM next day
            
            event = CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
                title="Hotel Stay",
                description="Accommodation for your trip",
                event_type=CalendarEventType.HOTEL,
                start_time=check_in,
                end_time=check_out,
                location=metadata.get("destination", ""),
                confidence=0.6,
                source="heuristic_extraction"
            )
            events.append(event)
        
        # Look for flight mentions
        if any(word in response_lower for word in ["flight", "airline", "airport", "departure", "arrival"]):
            base_date = datetime.now(timezone.utc)
            departure = base_date.replace(hour=10, minute=0, second=0, microsecond=0)
            arrival = base_date.replace(hour=14, minute=0, second=0, microsecond=0)
            
            event = CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
                title="Flight",
                description="Flight for your trip",
                event_type=CalendarEventType.FLIGHT,
                start_time=departure,
                end_time=arrival,
                location="Airport",
                confidence=0.6,
                source="heuristic_extraction"
            )
            events.append(event)
        
        # Look for attraction mentions
        if any(word in response_lower for word in ["visit", "attraction", "museum", "park", "tour", "sightseeing"]):
            base_date = datetime.now(timezone.utc)
            visit_start = base_date.replace(hour=9, minute=0, second=0, microsecond=0)
            visit_end = base_date.replace(hour=12, minute=0, second=0, microsecond=0)
            
            event = CalendarEvent(
                id=f"event_{uuid.uuid4().hex[:8]}",
                title="Sightseeing",
                description="Visit attractions and landmarks",
                event_type=CalendarEventType.ATTRACTION,
                start_time=visit_start,
                end_time=visit_end,
                location=metadata.get("destination", ""),
                confidence=0.5,
                source="heuristic_extraction"
            )
            events.append(event)
        
        return events
    
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
        
        # Update completion status based on response content
        if agent_response:
            if any(word in agent_response.lower() for word in ["complete", "finalized", "ready"]):
                updates["completion_status"] = "complete"
            elif any(word in agent_response.lower() for word in ["partial", "more information", "additional"]):
                updates["completion_status"] = "partial"
        
        # Always update timestamp
        updates["last_updated"] = datetime.now(timezone.utc)
        
        return updates
    
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