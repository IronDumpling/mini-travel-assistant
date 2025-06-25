#!/usr/bin/env python3
"""
Test script for the new agent-based API endpoints

This script tests the refactored travel_plans.py endpoints that use the TravelAgent
with self-refine capabilities.
"""

import asyncio
import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.api.endpoints.travel_plans import (
    ChatMessage, RefinementConfig, 
    chat_with_agent, configure_agent_refinement, get_agent_status
)
from app.models.schemas import TravelPreferences, Budget, Traveler, TravelerType, TripStyle
from datetime import datetime, timedelta


async def test_chat_endpoint():
    """Test the chat endpoint with self-refinement"""
    print("🧪 Testing Chat Endpoint with Self-Refinement")
    print("=" * 50)
    
    # Test message
    message = ChatMessage(
        message="I want to plan a romantic 5-day trip to Paris for my anniversary. Budget is around $4000 for 2 people. We love art, good food, and historic sites.",
        enable_refinement=True
    )
    
    print(f"Message: {message.message}")
    print(f"Refinement Enabled: {message.enable_refinement}")
    print()
    
    try:
        response = await chat_with_agent(message)
        
        print("✅ Chat Response Received:")
        print(f"Success: {response.success}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Session ID: {response.session_id}")
        print(f"Actions Taken: {len(response.actions_taken)}")
        print(f"Next Steps: {len(response.next_steps)}")
        
        if response.refinement_details:
            print(f"\n🔍 Refinement Details:")
            print(f"Final Iteration: {response.refinement_details['final_iteration']}")
            print(f"Quality Score: {response.refinement_details['final_quality_score']:.2f}")
            print(f"Status: {response.refinement_details['refinement_status']}")
        
        print(f"\n📝 Response Content (first 200 chars):")
        print(f"{response.content[:200]}...")
        
        if response.next_steps:
            print(f"\n📋 Next Steps:")
            for i, step in enumerate(response.next_steps[:3], 1):
                print(f"  {i}. {step}")
        
        return response.session_id
        
    except Exception as e:
        print(f"❌ Chat endpoint failed: {e}")
        return None


async def test_chat_without_refinement():
    """Test the chat endpoint without self-refinement for comparison"""
    print("\n🧪 Testing Chat Endpoint WITHOUT Self-Refinement")
    print("=" * 50)
    
    message = ChatMessage(
        message="Plan a quick weekend trip to New York City for 2 people, budget $2000.",
        enable_refinement=False
    )
    
    try:
        response = await chat_with_agent(message)
        
        print("✅ Chat Response (No Refinement):")
        print(f"Success: {response.success}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Refinement Details: {response.refinement_details}")
        print(f"Content Length: {len(response.content)}")
        
        return response
        
    except Exception as e:
        print(f"❌ Chat without refinement failed: {e}")
        return None


async def test_agent_status():
    """Test the agent status endpoint"""
    print("\n🧪 Testing Agent Status Endpoint")
    print("=" * 50)
    
    try:
        status = await get_agent_status()
        
        print("✅ Agent Status Retrieved:")
        print(f"Agent Name: {status['agent_info']['name']}")
        print(f"Capabilities: {len(status['agent_info']['capabilities'])}")
        print(f"Available Tools: {len(status['agent_info']['tools'])}")
        print(f"Refinement Enabled: {status['refinement_config']['enabled']}")
        print(f"Quality Threshold: {status['refinement_config']['quality_threshold']}")
        
        print(f"\n🎯 Quality Dimensions:")
        for dim, weight in status['quality_dimensions'].items():
            print(f"  {dim}: {weight:.2f}")
        
        return status
        
    except Exception as e:
        print(f"❌ Agent status failed: {e}")
        return None


async def test_refinement_config():
    """Test the refinement configuration endpoint"""
    print("\n🧪 Testing Refinement Configuration")
    print("=" * 50)
    
    config = RefinementConfig(
        enabled=True,
        quality_threshold=0.8,
        max_iterations=2
    )
    
    try:
        result = await configure_agent_refinement(config)
        
        print("✅ Refinement Configuration:")
        print(f"Message: {result['message']}")
        print(f"Config Applied: {result['config']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Refinement configuration failed: {e}")
        return None


async def test_structured_plan_creation():
    """Test the structured plan creation endpoint"""
    print("\n🧪 Testing Structured Plan Creation")
    print("=" * 50)
    
    # Import the endpoint function
    from app.api.endpoints.travel_plans import create_travel_plan
    
    # Create test preferences
    preferences = TravelPreferences(
        origin="Los Angeles",
        destination="Tokyo",
        start_date=datetime.now() + timedelta(days=30),
        end_date=datetime.now() + timedelta(days=37),
        budget=Budget(total=5000, currency="USD"),
        trip_style=TripStyle.CULTURAL,
        travelers=[
            Traveler(age=30, type=TravelerType.ADULT),
            Traveler(age=28, type=TravelerType.ADULT)
        ],
        interests=["temples", "food", "technology", "gardens"],
        goals=["cultural immersion", "photography", "authentic experiences"]
    )
    
    try:
        plan = await create_travel_plan(preferences)
        
        print("✅ Structured Plan Created:")
        print(f"Plan ID: {plan.id}")
        print(f"Destination: {plan.preferences.destination}")
        print(f"Duration: {(plan.preferences.end_date - plan.preferences.start_date).days} days")
        print(f"Budget: {plan.preferences.budget.total} {plan.preferences.budget.currency}")
        print(f"Status: {plan.status}")
        
        return plan
        
    except Exception as e:
        print(f"❌ Structured plan creation failed: {e}")
        return None


async def test_conversation_continuity(session_id: str):
    """Test conversation continuity within a session"""
    if not session_id:
        print("\n⚠️ Skipping conversation continuity test - no session ID")
        return
    
    print(f"\n🧪 Testing Conversation Continuity (Session: {session_id})")
    print("=" * 50)
    
    followup_message = ChatMessage(
        message="Actually, can you suggest some specific restaurants in Paris that would be perfect for our anniversary dinner?",
        session_id=session_id,
        enable_refinement=True
    )
    
    try:
        response = await chat_with_agent(followup_message)
        
        print("✅ Follow-up Response:")
        print(f"Success: {response.success}")
        print(f"Same Session: {response.session_id == session_id}")
        print(f"Content Length: {len(response.content)}")
        print(f"Confidence: {response.confidence:.2f}")
        
        print(f"\n📝 Follow-up Content (first 150 chars):")
        print(f"{response.content[:150]}...")
        
    except Exception as e:
        print(f"❌ Conversation continuity test failed: {e}")


async def run_all_tests():
    """Run all API endpoint tests"""
    print("🚀 Starting API Endpoint Tests")
    print("=" * 60)
    
    try:
        # Test basic functionality
        session_id = await test_chat_endpoint()
        await test_chat_without_refinement()
        await test_agent_status()
        await test_refinement_config()
        
        # Test structured endpoints
        await test_structured_plan_creation()
        
        # Test conversation continuity
        await test_conversation_continuity(session_id)
        
        print("\n🎉 All API Tests Completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 API Endpoint Test Suite")
    print("Testing the new agent-based travel planning endpoints")
    print()
    
    asyncio.run(run_all_tests()) 