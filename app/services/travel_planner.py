from typing import List, Optional
from datetime import datetime
import openai
from app.models.schemas import TravelPreferences, TravelPlan, DailyPlan, Activity
from app.tools.flight_search import FlightSearch
from app.tools.hotel_search import HotelSearch
from app.tools.attraction_search import AttractionSearch

class TravelPlannerService:
    def __init__(self):
        self.flight_search = FlightSearch()
        self.hotel_search = HotelSearch()
        self.attraction_search = AttractionSearch()

    async def generate_plan(self, preferences: TravelPreferences) -> TravelPlan:
        """
        Generate a complete travel plan based on user preferences.
        """
        # 1. Generate initial plan using LLM
        initial_plan = await self._generate_initial_plan(preferences)
        
        # 2. Validate and refine with real-time data
        refined_plan = await self._refine_plan(initial_plan)
        
        # 3. Calculate total costs
        total_cost = sum(day.total_cost for day in refined_plan.daily_plans)
        
        return TravelPlan(
            id=self._generate_plan_id(),
            preferences=preferences,
            daily_plans=refined_plan.daily_plans,
            total_cost=total_cost
        )

    async def _generate_initial_plan(self, preferences: TravelPreferences) -> TravelPlan:
        """
        Use LLM to generate initial travel plan.
        """
        prompt = self._create_planning_prompt(preferences)
        
        # Call OpenAI API
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a travel planning expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse LLM response into structured plan
        return self._parse_llm_response(response.choices[0].message.content, preferences)

    async def _refine_plan(self, plan: TravelPlan) -> TravelPlan:
        """
        Refine the plan using real-time data from external tools.
        """
        # Check flight availability
        flights = await self.flight_search.search(
            origin=plan.preferences.origin,
            destination=plan.preferences.destination,
            start_date=plan.preferences.start_date,
            end_date=plan.preferences.end_date
        )
        
        # Check hotel availability
        hotels = await self.hotel_search.search(
            location=plan.preferences.destination,
            check_in=plan.preferences.start_date,
            check_out=plan.preferences.end_date,
            guests=len(plan.preferences.travelers)
        )
        
        # Refine activities based on real-time data
        refined_plans = []
        for day_plan in plan.daily_plans:
            refined_activities = []
            for activity in day_plan.activities:
                # Check if activity is available
                availability = await self.attraction_search.check_availability(
                    activity.name,
                    activity.start_time,
                    activity.end_time
                )
                
                if availability.is_available:
                    refined_activities.append(activity)
            
            refined_plans.append(DailyPlan(
                date=day_plan.date,
                activities=refined_activities,
                total_cost=sum(a.estimated_cost for a in refined_activities)
            ))
        
        return TravelPlan(
            id=plan.id,
            preferences=plan.preferences,
            daily_plans=refined_plans,
            total_cost=sum(day.total_cost for day in refined_plans)
        )

    def _create_planning_prompt(self, preferences: TravelPreferences) -> str:
        """
        Create a detailed prompt for the LLM.
        """
        return f"""
        Create a detailed travel plan for the following preferences:
        Destination: {preferences.destination}
        Dates: {preferences.start_date} to {preferences.end_date}
        Budget: {preferences.budget.total} {preferences.budget.currency}
        Trip Style: {preferences.trip_style}
        Travelers: {len(preferences.travelers)} ({', '.join(t.type for t in preferences.travelers)})
        Interests: {', '.join(preferences.interests)}
        Goals: {', '.join(preferences.goals)}
        
        Please provide a day-by-day itinerary with:
        1. Morning, afternoon, and evening activities
        2. Estimated costs for each activity
        3. Transportation between activities
        4. Recommended restaurants
        5. Booking instructions where applicable
        """

    def _parse_llm_response(self, response: str, preferences: TravelPreferences) -> TravelPlan:
        """
        Parse the LLM response into a structured TravelPlan.
        """
        # Implementation of parsing logic
        # This would convert the LLM's text response into structured data
        pass

    def _generate_plan_id(self) -> str:
        """
        Generate a unique plan ID.
        """
        return f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}" 