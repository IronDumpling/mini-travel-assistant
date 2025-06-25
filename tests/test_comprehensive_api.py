#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Travel Planning API

This unified test suite combines all API endpoint testing including:
- Agent response to structured TravelPlan parsing
- Enhanced schema with agent metadata  
- Translation layers between agent and API models
- Chat endpoints with self-refinement
- Structured plan creation and management
- Conversation continuity and session management
"""

import asyncio
import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.api.endpoints.travel_plans import (
    ChatMessage, RefinementConfig, 
    chat_with_agent, create_travel_plan, get_agent_status, configure_agent_refinement,
    _parse_agent_response_to_plan, _preferences_to_message
)
from app.models.schemas import TravelPreferences, Budget, Traveler, TravelerType, TripStyle
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage, AgentResponse
from datetime import datetime, timedelta


class TestResults:
    """Track test results for comprehensive reporting"""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_result(self, test_name: str, success: bool, details: str = ""):
        self.tests.append({
            "name": test_name,
            "success": success,
            "details": details
        })
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"🎯 COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*70}")
        print(f"✅ Passed: {self.passed}/{total}")
        print(f"❌ Failed: {self.failed}/{total}")
        print(f"📊 Success Rate: {(self.passed/total*100):.1f}%")
        
        if self.failed > 0:
            print(f"\n❌ Failed Tests:")
            for test in self.tests:
                if not test["success"]:
                    print(f"  • {test['name']}: {test['details']}")
        
        print(f"\n{'🎊 ALL TESTS PASSED!' if self.failed == 0 else '⚠️ SOME TESTS FAILED'}")


# Global test results tracker
results = TestResults()


async def test_agent_response_parsing():
    """Test the core agent response parsing functionality"""
    print("🧪 Testing Agent Response Parsing")
    print("=" * 60)
    
    # Create a comprehensive mock agent response
    mock_response_content = """
    Here's your 3-day Rome cultural immersion itinerary:

    Day 1: Ancient Rome Exploration
    9:00 - Visit the Colosseum ($25) - Explore this iconic amphitheater with skip-the-line tickets
    12:00 - Lunch at Trattoria da Valentino ($45) - Traditional Roman cuisine in Monti district
    14:00 - Roman Forum tour ($20) - Walk through ancient ruins with audio guide
    17:00 - Palatine Hill visit ($15) - Panoramic views of the city and imperial palaces
    19:30 - Dinner in Trastevere ($60) - Authentic neighborhood dining at Da Enzo

    Day 2: Vatican and Renaissance Art
    Morning: Vatican Museums and Sistine Chapel ($30) - World-class art collection, book early entry
    Afternoon: St. Peter's Basilica (Free) - Magnificent Renaissance architecture, climb the dome
    Evening: Dinner near Vatican ($50) - Local recommendations at Dal Toscano

    Day 3: Roman Culture and Leisure
    • Villa Borghese gardens (Free) - Peaceful morning walk and Galleria Borghese visit
    • Capitoline Museums ($15) - Ancient Roman artifacts and city views
    • Spanish Steps and shopping ($100) - Afternoon leisure and Via del Corso
    • Farewell dinner at rooftop restaurant ($80) - Romantic setting with city views
    """
    
    preferences = TravelPreferences(
        origin="New York",
        destination="Rome", 
        start_date=datetime.now() + timedelta(days=15),
        end_date=datetime.now() + timedelta(days=18),
        budget=Budget(total=2000, currency="USD"),
        trip_style=TripStyle.CULTURAL,
        travelers=[Traveler(age=30, type=TravelerType.ADULT)],
        interests=["history", "art", "cuisine"],
        goals=["cultural exploration", "photography"]
    )
    
    mock_response = AgentResponse(
        success=True,
        content=mock_response_content,
        actions_taken=["search_attractions", "search_restaurants", "check_opening_hours"],
        next_steps=["Book Colosseum tickets", "Make restaurant reservations", "Download city map"],
        confidence=0.88,
        metadata={
            "quality_score": 0.82,
            "refinement_iteration": 2,
            "execution_time": 3.5,
            "session_id": "test_session_001"
        }
    )
    
    try:
        print("🔄 Parsing comprehensive agent response...")
        parsed_plan = await _parse_agent_response_to_plan(mock_response, preferences)
        
        # Validation checks
        assert parsed_plan.id is not None, "Plan ID should be generated"
        assert len(parsed_plan.daily_plans) > 0, "Should have daily plans"
        assert parsed_plan.total_cost > 0, "Should have calculated total cost"
        assert parsed_plan.agent_metadata is not None, "Should have agent metadata"
        
        # Detailed analysis
        total_activities = sum(len(day.activities) for day in parsed_plan.daily_plans)
        
        print("✅ Parsing Successful!")
        print(f"📊 Parsed Plan Analysis:")
        print(f"  Plan ID: {parsed_plan.id}")
        print(f"  Daily Plans: {len(parsed_plan.daily_plans)}")
        print(f"  Total Activities: {total_activities}")
        print(f"  Total Cost: ${parsed_plan.total_cost:.2f}")
        print(f"  Status: {parsed_plan.status}")
        
        # Agent metadata validation
        if parsed_plan.agent_metadata:
            print(f"\n🤖 Agent Metadata:")
            print(f"  Confidence: {parsed_plan.agent_metadata.confidence:.2f}")
            print(f"  Quality Score: {parsed_plan.agent_metadata.quality_score:.2f}")
            print(f"  Refinement Used: {parsed_plan.agent_metadata.refinement_used}")
            print(f"  Actions: {len(parsed_plan.agent_metadata.actions_taken)}")
        
        # Detailed activity breakdown
        print(f"\n📅 Daily Breakdown:")
        for i, daily_plan in enumerate(parsed_plan.daily_plans, 1):
            print(f"  Day {i} ({daily_plan.date.strftime('%Y-%m-%d')}):")
            print(f"    Activities: {len(daily_plan.activities)}")
            print(f"    Daily Cost: ${daily_plan.total_cost:.2f}")
            
            for j, activity in enumerate(daily_plan.activities[:2], 1):  # Show first 2 activities
                print(f"      {j}. {activity.name}")
                print(f"         Time: {activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}")
                print(f"         Location: {activity.location}")
                print(f"         Cost: ${activity.estimated_cost:.2f}")
        
        results.add_result("Agent Response Parsing", True, f"Parsed {total_activities} activities across {len(parsed_plan.daily_plans)} days")
        return parsed_plan
        
    except Exception as e:
        print(f"❌ Agent response parsing failed: {e}")
        import traceback
        traceback.print_exc()
        results.add_result("Agent Response Parsing", False, str(e))
        return None


async def test_preferences_to_message_conversion():
    """Test the enhanced preferences to message conversion"""
    print("\n🧪 Testing Preferences to Message Conversion")
    print("=" * 60)
    
    preferences = TravelPreferences(
        origin="London",
        destination="Barcelona",
        start_date=datetime.now() + timedelta(days=45),
        end_date=datetime.now() + timedelta(days=49),
        budget=Budget(total=2800, currency="EUR"),
        trip_style=TripStyle.ADVENTURE,
        travelers=[
            Traveler(age=25, type=TravelerType.ADULT),
            Traveler(age=27, type=TravelerType.ADULT)
        ],
        interests=["hiking", "beaches", "nightlife", "local cuisine"],
        goals=["adventure sports", "cultural exploration", "relaxation"],
        additional_notes="Looking for a mix of adventure and relaxation"
    )
    
    try:
        print("🔄 Converting preferences to natural language...")
        message = _preferences_to_message(preferences)
        
        # Validation checks
        assert len(message) > 100, "Message should be comprehensive"
        assert preferences.destination in message, "Should contain destination"
        assert str(preferences.budget.total) in message, "Should contain budget"
        assert any(interest in message for interest in preferences.interests), "Should contain interests"
        
        # Analysis
        message_length = len(message)
        word_count = len(message.split())
        duration = (preferences.end_date - preferences.start_date).days
        
        print("✅ Conversion Successful!")
        print(f"📝 Generated Message Analysis:")
        print(f"  Length: {message_length} characters")
        print(f"  Word Count: {word_count} words")
        print(f"  Contains Duration: {f'{duration} days' in message}")
        print(f"  Contains Budget: {str(preferences.budget.total) in message}")
        print(f"  Contains All Interests: {all(interest in message for interest in preferences.interests)}")
        print(f"  Contains All Goals: {all(goal in message for goal in preferences.goals)}")
        
        print(f"\n📝 Sample Message (first 200 chars):")
        print(f"{message[:200]}...")
        
        results.add_result("Preferences to Message", True, f"Generated {word_count} word message")
        return message
        
    except Exception as e:
        print(f"❌ Preferences to message conversion failed: {e}")
        results.add_result("Preferences to Message", False, str(e))
        return None


async def test_chat_endpoint_with_refinement():
    """Test the chat endpoint with self-refinement enabled"""
    print("\n🧪 Testing Chat Endpoint with Self-Refinement")
    print("=" * 60)
    
    message = ChatMessage(
        message="I want to plan a romantic 5-day trip to Paris for my anniversary. Budget is around $4000 for 2 people. We love art, good food, and historic sites. Please provide a detailed day-by-day itinerary.",
        enable_refinement=True
    )
    
    print(f"📝 Test Message: {message.message[:100]}...")
    print(f"🔧 Refinement Enabled: {message.enable_refinement}")
    
    try:
        response = await chat_with_agent(message)
        
        # Validation checks
        assert response.success, "Chat should succeed"
        assert response.confidence > 0, "Should have confidence score"
        assert len(response.content) > 100, "Should have substantial content"
        assert response.session_id is not None, "Should have session ID"
        
        print("✅ Chat Response Received:")
        print(f"📊 Response Analysis:")
        print(f"  Success: {response.success}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Session ID: {response.session_id}")
        print(f"  Content Length: {len(response.content)} characters")
        print(f"  Actions Taken: {len(response.actions_taken)}")
        print(f"  Next Steps: {len(response.next_steps)}")
        
        if response.refinement_details:
            print(f"\n🔍 Refinement Details:")
            print(f"  Final Iteration: {response.refinement_details.get('final_iteration', 'N/A')}")
            print(f"  Quality Score: {response.refinement_details.get('final_quality_score', 0):.2f}")
            print(f"  Status: {response.refinement_details.get('refinement_status', 'N/A')}")
        
        print(f"\n📝 Response Preview (first 200 chars):")
        print(f"{response.content[:200]}...")
        
        if response.next_steps:
            print(f"\n📋 Next Steps (first 3):")
            for i, step in enumerate(response.next_steps[:3], 1):
                print(f"  {i}. {step}")
        
        results.add_result("Chat with Refinement", True, f"Generated {len(response.content)} char response")
        return response.session_id
        
    except Exception as e:
        print(f"❌ Chat endpoint with refinement failed: {e}")
        results.add_result("Chat with Refinement", False, str(e))
        return None


async def test_chat_endpoint_without_refinement():
    """Test the chat endpoint without self-refinement for comparison"""
    print("\n🧪 Testing Chat Endpoint WITHOUT Self-Refinement")
    print("=" * 60)
    
    message = ChatMessage(
        message="Plan a quick weekend trip to New York City for 2 people, budget $2000. Include must-see attractions and restaurant recommendations.",
        enable_refinement=False
    )
    
    try:
        response = await chat_with_agent(message)
        
        # Validation
        assert response.success, "Chat should succeed"
        assert response.refinement_details is None, "Should not have refinement details"
        
        print("✅ Chat Response (No Refinement):")
        print(f"📊 Response Analysis:")
        print(f"  Success: {response.success}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Content Length: {len(response.content)}")
        print(f"  Refinement Details: {response.refinement_details}")
        print(f"  Session ID: {response.session_id}")
        
        results.add_result("Chat without Refinement", True, f"No refinement used as expected")
        return response
        
    except Exception as e:
        print(f"❌ Chat without refinement failed: {e}")
        results.add_result("Chat without Refinement", False, str(e))
        return None


async def test_structured_plan_creation():
    """Test the enhanced structured plan creation with full parsing"""
    print("\n🧪 Testing Enhanced Structured Plan Creation")
    print("=" * 60)
    
    preferences = TravelPreferences(
        origin="San Francisco",
        destination="Tokyo",
        start_date=datetime.now() + timedelta(days=30),
        end_date=datetime.now() + timedelta(days=37),  # 7-day trip
        budget=Budget(total=5000, currency="USD"),
        trip_style=TripStyle.CULTURAL,
        travelers=[
            Traveler(age=32, type=TravelerType.ADULT),
            Traveler(age=29, type=TravelerType.ADULT)
        ],
        interests=["temples", "technology", "cuisine", "gardens", "traditional culture"],
        goals=["cultural immersion", "authentic experiences", "photography"],
        additional_notes="First time visiting Japan, interested in both traditional and modern aspects"
    )
    
    try:
        print("📋 Test Preferences:")
        print(f"  Route: {preferences.origin} → {preferences.destination}")
        print(f"  Duration: {(preferences.end_date - preferences.start_date).days} days")
        print(f"  Budget: {preferences.budget.total} {preferences.budget.currency}")
        print(f"  Travelers: {len(preferences.travelers)} adults")
        print(f"  Interests: {', '.join(preferences.interests)}")
        
        print("\n🔄 Creating structured travel plan...")
        plan = await create_travel_plan(preferences)
        
        # Comprehensive validation
        assert plan.id is not None, "Plan should have ID"
        assert plan.preferences.destination == preferences.destination, "Preferences should match"
        assert len(plan.daily_plans) > 0, "Should have daily plans"
        assert plan.total_cost >= 0, "Should have valid total cost"
        assert plan.agent_metadata is not None, "Should have agent metadata"
        
        print("✅ Structured Plan Created Successfully!")
        print(f"📊 Plan Analysis:")
        print(f"  Plan ID: {plan.id}")
        print(f"  Status: {plan.status}")
        print(f"  Total Cost: ${plan.total_cost:.2f}")
        print(f"  Daily Plans: {len(plan.daily_plans)}")
        print(f"  Quality Score: {plan.quality_score:.2f if plan.quality_score else 'N/A'}")
        
        # Agent metadata analysis
        if plan.agent_metadata:
            print(f"\n🤖 Agent Metadata:")
            print(f"  Confidence: {plan.agent_metadata.confidence:.2f}")
            print(f"  Actions Taken: {len(plan.agent_metadata.actions_taken)}")
            print(f"  Refinement Used: {plan.agent_metadata.refinement_used}")
            if plan.agent_metadata.quality_score:
                print(f"  Quality Score: {plan.agent_metadata.quality_score:.2f}")
            if plan.agent_metadata.refinement_iterations:
                print(f"  Refinement Iterations: {plan.agent_metadata.refinement_iterations}")
        
        # Daily plans analysis
        total_activities = sum(len(day.activities) for day in plan.daily_plans)
        print(f"\n📅 Daily Plans Analysis:")
        print(f"  Total Activities: {total_activities}")
        for i, daily_plan in enumerate(plan.daily_plans, 1):
            print(f"  Day {i} ({daily_plan.date.strftime('%Y-%m-%d')}):")
            print(f"    Activities: {len(daily_plan.activities)}")
            print(f"    Daily Cost: ${daily_plan.total_cost:.2f}")
            
            # Show sample activity
            if daily_plan.activities:
                activity = daily_plan.activities[0]
                print(f"    Sample: {activity.name} at {activity.location} (${activity.estimated_cost:.2f})")
        
        results.add_result("Structured Plan Creation", True, f"Created plan with {total_activities} activities")
        return plan
        
    except Exception as e:
        print(f"❌ Structured plan creation failed: {e}")
        import traceback
        traceback.print_exc()
        results.add_result("Structured Plan Creation", False, str(e))
        return None


async def test_agent_status():
    """Test the agent status endpoint"""
    print("\n🧪 Testing Agent Status Endpoint")
    print("=" * 60)
    
    try:
        status = await get_agent_status()
        
        # Validation
        assert "agent_info" in status, "Should have agent info"
        assert "refinement_config" in status, "Should have refinement config"
        assert "quality_dimensions" in status, "Should have quality dimensions"
        
        print("✅ Agent Status Retrieved:")
        print(f"📊 Agent Information:")
        print(f"  Name: {status['agent_info']['name']}")
        print(f"  Capabilities: {len(status['agent_info']['capabilities'])}")
        print(f"  Available Tools: {len(status['agent_info']['tools'])}")
        print(f"  System Status: {status['system_status']}")
        
        print(f"\n🔧 Refinement Configuration:")
        print(f"  Enabled: {status['refinement_config']['enabled']}")
        print(f"  Quality Threshold: {status['refinement_config']['quality_threshold']}")
        print(f"  Max Iterations: {status['refinement_config']['max_iterations']}")
        
        print(f"\n🎯 Quality Dimensions:")
        for dim, weight in status['quality_dimensions'].items():
            print(f"  {dim}: {weight:.2f}")
        
        results.add_result("Agent Status", True, f"Retrieved status with {len(status['quality_dimensions'])} quality dimensions")
        return status
        
    except Exception as e:
        print(f"❌ Agent status failed: {e}")
        results.add_result("Agent Status", False, str(e))
        return None


async def test_refinement_configuration():
    """Test the refinement configuration endpoint"""
    print("\n🧪 Testing Refinement Configuration")
    print("=" * 60)
    
    config = RefinementConfig(
        enabled=True,
        quality_threshold=0.8,
        max_iterations=2
    )
    
    try:
        result = await configure_agent_refinement(config)
        
        # Validation
        assert "message" in result, "Should have confirmation message"
        assert "config" in result, "Should return config"
        
        print("✅ Refinement Configuration:")
        print(f"📋 Configuration Result:")
        print(f"  Message: {result['message']}")
        print(f"  Applied Config: {result['config']}")
        print(f"  Note: {result.get('note', 'N/A')}")
        
        results.add_result("Refinement Configuration", True, "Configuration applied successfully")
        return result
        
    except Exception as e:
        print(f"❌ Refinement configuration failed: {e}")
        results.add_result("Refinement Configuration", False, str(e))
        return None


async def test_conversation_continuity(session_id: str):
    """Test conversation continuity within a session"""
    if not session_id:
        print("\n⚠️ Skipping conversation continuity test - no session ID")
        results.add_result("Conversation Continuity", False, "No session ID available")
        return
    
    print(f"\n🧪 Testing Conversation Continuity")
    print("=" * 60)
    print(f"📱 Using Session: {session_id}")
    
    followup_message = ChatMessage(
        message="Actually, can you suggest some specific restaurants in Paris that would be perfect for our anniversary dinner? Price range $100-200 per person.",
        session_id=session_id,
        enable_refinement=True
    )
    
    try:
        response = await chat_with_agent(followup_message)
        
        # Validation
        assert response.success, "Follow-up should succeed"
        assert response.session_id == session_id, "Should maintain same session"
        
        print("✅ Follow-up Response:")
        print(f"📊 Continuity Analysis:")
        print(f"  Success: {response.success}")
        print(f"  Same Session: {response.session_id == session_id}")
        print(f"  Content Length: {len(response.content)}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Context Maintained: {'Paris' in response.content and 'anniversary' in response.content}")
        
        print(f"\n📝 Follow-up Content (first 200 chars):")
        print(f"{response.content[:200]}...")
        
        results.add_result("Conversation Continuity", True, "Session and context maintained")
        
    except Exception as e:
        print(f"❌ Conversation continuity test failed: {e}")
        results.add_result("Conversation Continuity", False, str(e))


async def test_full_integration():
    """Test the complete integration pipeline"""
    print("\n🧪 Testing Full Integration Pipeline")
    print("=" * 60)
    
    preferences = TravelPreferences(
        origin="Chicago",
        destination="Kyoto",
        start_date=datetime.now() + timedelta(days=60),
        end_date=datetime.now() + timedelta(days=67),
        budget=Budget(total=4500, currency="USD"),
        trip_style=TripStyle.CULTURAL,
        travelers=[Traveler(age=35, type=TravelerType.ADULT)],
        interests=["temples", "gardens", "tea ceremony", "traditional crafts"],
        goals=["spiritual experience", "cultural immersion", "photography"],
        additional_notes="Solo traveler seeking authentic Japanese cultural experiences"
    )
    
    try:
        print("🔄 Running full integration pipeline...")
        
        # Step 1: Preferences to message
        message = _preferences_to_message(preferences)
        print("✅ Step 1: Preferences converted to natural language")
        
        # Step 2: Create structured plan
        plan = await create_travel_plan(preferences)
        print("✅ Step 2: Structured plan created via agent")
        
        # Step 3: Validate complete pipeline
        assert plan.preferences.destination == preferences.destination, "Destination should match"
        assert len(plan.daily_plans) > 0, "Should have daily plans"
        assert plan.agent_metadata is not None, "Should have agent metadata"
        
        print(f"\n🎉 Full Integration Successful!")
        print(f"📊 Pipeline Results:")
        print(f"  Original: {preferences.destination} for {(preferences.end_date - preferences.start_date).days} days")
        print(f"  Generated Plan ID: {plan.id}")
        print(f"  Plan Status: {plan.status}")
        print(f"  Total Cost: ${plan.total_cost:.2f}")
        print(f"  Agent Confidence: {plan.agent_metadata.confidence:.2f}")
        print(f"  Quality Score: {plan.quality_score:.2f if plan.quality_score else 'N/A'}")
        
        results.add_result("Full Integration", True, "Complete pipeline working end-to-end")
        return plan
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        results.add_result("Full Integration", False, str(e))
        return None


async def run_comprehensive_test_suite():
    """Run the complete comprehensive test suite"""
    print("🚀 Starting Comprehensive API Test Suite")
    print("=" * 70)
    print("Testing all enhanced functionality:")
    print("• Agent response parsing with metadata")
    print("• Enhanced schema validation")
    print("• Translation layer functionality")
    print("• Chat endpoints with refinement")
    print("• Structured plan creation")
    print("• Agent status and configuration")
    print("• Conversation continuity")
    print("• Full integration pipeline")
    print()
    
    try:
        # Core parsing and translation tests
        await test_agent_response_parsing()
        await test_preferences_to_message_conversion()
        
        # API endpoint tests
        session_id = await test_chat_endpoint_with_refinement()
        await test_chat_endpoint_without_refinement()
        await test_structured_plan_creation()
        
        # Agent management tests
        await test_agent_status()
        await test_refinement_configuration()
        
        # Advanced functionality tests
        await test_conversation_continuity(session_id)
        await test_full_integration()
        
        # Print comprehensive results
        results.print_summary()
        
        # Final assessment
        if results.failed == 0:
            print(f"\n🎊 COMPREHENSIVE TEST SUITE: ALL SYSTEMS OPERATIONAL!")
            print(f"✅ All immediate action items successfully implemented")
            print(f"✅ Enhanced API endpoints fully functional")
            print(f"✅ Agent integration working perfectly")
            print(f"✅ Self-refinement capabilities validated")
        else:
            print(f"\n⚠️ Some tests failed - system needs attention")
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        results.add_result("Test Suite Execution", False, str(e))


if __name__ == "__main__":
    print("🧪 Comprehensive Travel Planning API Test Suite")
    print("=" * 70)
    print("This unified test suite validates:")
    print("1. Agent response → TravelPlan parser")
    print("2. Enhanced schema with agent metadata")  
    print("3. Improved translation layers")
    print("4. Chat endpoints with self-refinement")
    print("5. Structured plan creation and management")
    print("6. Agent status and configuration")
    print("7. Conversation continuity")
    print("8. Complete integration validation")
    print()
    
    asyncio.run(run_comprehensive_test_suite()) 