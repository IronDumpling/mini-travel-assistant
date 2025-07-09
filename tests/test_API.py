#!/usr/bin/env python3
"""
Comprehensive test for OpenAI API integration with Travel Agent

This test verifies that the OpenAI integration is working correctly
with the travel agent system using the provided API key.
"""

import os
# Set the API key as early as possible
os.environ["OPENAI_API_KEY"] = "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"

import sys
import asyncio
import logging
import time
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.llm_service import LLMConfig, LLMServiceFactory, get_llm_service
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage, AgentResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Your OpenAI API key
# OPENAI_API_KEY = "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"


async def test_openai_api_connection():
    """Test basic OpenAI API connection"""
    
    print("🧪 Testing OpenAI API Connection")
    print("=" * 50)
    
    # Set the API key
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print(f"✅ API Key set: {os.environ.get('OPENAI_API_KEY', 'N/A')[:20]}...")
    print()
    
    # Test 1: Basic Configuration
    print("1️⃣ Testing LLM Configuration")
    config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",  # Using 3.5-turbo for cost efficiency
        api_key=os.environ.get('OPENAI_API_KEY'),
        temperature=0.7,
        max_tokens=150,  # Reasonable limit for testing
        retry_attempts=3,
        retry_delay=1.0
    )
    
    print(f"   Provider: {config.provider}")
    print(f"   Model: {config.model}")
    print(f"   Max Tokens: {config.max_tokens}")
    print(f"   Temperature: {config.temperature}")
    print()
    
    # Test 2: Service Creation
    print("2️⃣ Testing LLM Service Creation")
    try:
        service = LLMServiceFactory.create_service(config)
        print(f"   ✅ Service Type: {type(service).__name__}")
        print(f"   ✅ Model: {service.model}")
        print(f"   ✅ Client Initialized: {service.client is not None}")
        print()
    except Exception as e:
        print(f"   ❌ Service Creation Failed: {e}")
        return False
    
    # Test 3: Basic API Call
    print("3️⃣ Testing Basic API Call")
    messages = [
        {"role": "user", "content": "Hello! Can you help me plan a trip to Japan?"}
    ]
    
    try:
        print("   Making API call...")
        response = await service.chat_completion(messages)
        
        print("   ✅ API Call Successful!")
        print(f"   Content: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Finish Reason: {response.finish_reason}")
        
        if response.usage:
            print(f"   Total Tokens: {response.usage.get('total_tokens', 'unknown')}")
            print(f"   Prompt Tokens: {response.usage.get('prompt_tokens', 'unknown')}")
            print(f"   Completion Tokens: {response.usage.get('completion_tokens', 'unknown')}")
        
        if response.metadata:
            print(f"   Provider: {response.metadata.get('provider', 'unknown')}")
            print(f"   API Call: {response.metadata.get('api_call', False)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ API Call Failed: {e}")
        print("   This might be due to:")
        print("   - Invalid API key")
        print("   - Network connectivity issues")
        print("   - Rate limiting")
        print("   - Account billing issues")
        return False


async def test_travel_agent_integration():
    """Test Travel Agent with OpenAI integration"""
    
    print("🎯 Testing Travel Agent Integration")
    print("=" * 50)
    
    # Set the API key for the travel agent
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Test 1: Travel Agent Initialization
    print("1️⃣ Testing Travel Agent Initialization")
    try:
        travel_agent = TravelAgent()
        print(f"   ✅ Agent Name: {travel_agent.name}")
        print(f"   ✅ Description: {travel_agent.description}")
        print(f"   ✅ Status: {travel_agent.status}")
        print(f"   ✅ Capabilities: {travel_agent.get_capabilities()}")
        print(f"   ✅ Available Tools: {travel_agent.get_available_tools()}")
        print()
    except Exception as e:
        print(f"   ❌ Travel Agent Initialization Failed: {e}")
        return False
    
    # Test 2: Travel Agent Message Processing
    print("2️⃣ Testing Travel Agent Message Processing")
    
    # Create a test message
    test_message = AgentMessage(
        sender="user",
        receiver="travel_agent",
        content="I want to plan a 5-day trip to Tokyo, Japan. I have a budget of $2000 and I'm interested in culture and food. Can you help me create an itinerary?",
        message_type="text",
        metadata={"user_id": "test_user_001"}
    )
    
    try:
        print("   Processing travel request...")
        print(f"   Message: {test_message.content}")
        
        # Process the message
        response = await travel_agent.process_message(test_message)
        
        print("   ✅ Message Processing Successful!")
        print(f"   Success: {response.success}")
        print(f"   Content Length: {len(response.content)} characters")
        print(f"   Actions Taken: {response.actions_taken}")
        print(f"   Next Steps: {response.next_steps}")
        print(f"   Confidence: {response.confidence}")
        
        # Show a preview of the response
        if response.content:
            preview = response.content[:200] + "..." if len(response.content) > 200 else response.content
            print(f"   Response Preview: {preview}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Message Processing Failed: {e}")
        print("   This might be due to:")
        print("   - LLM service issues")
        print("   - Tool execution problems")
        print("   - Configuration errors")
        return False
    
    # Test 3: Travel Agent with Refinement
    print("3️⃣ Testing Travel Agent with Self-Refinement")
    
    refinement_message = AgentMessage(
        sender="user",
        receiver="travel_agent",
        content="I need a detailed travel plan for a family trip to Kyoto, Japan. We have 3 days, budget of $3000, and we want to see temples and traditional culture. Please include specific recommendations for hotels and restaurants.",
        message_type="text",
        metadata={"user_id": "test_user_002", "family_size": 4}
    )
    
    try:
        print("   Processing detailed travel request with refinement...")
        print(f"   Message: {refinement_message.content}")
        
        # Process with refinement enabled
        travel_agent.refine_enabled = True
        travel_agent.quality_threshold = 0.8
        
        response = await travel_agent.process_with_refinement(refinement_message)
        
        print("   ✅ Refinement Processing Successful!")
        print(f"   Success: {response.success}")
        print(f"   Content Length: {len(response.content)} characters")
        print(f"   Actions Taken: {response.actions_taken}")
        print(f"   Confidence: {response.confidence}")
        
        # Show refinement metadata
        if response.metadata:
            print(f"   Refinement Iteration: {response.metadata.get('refinement_iteration', 'N/A')}")
            print(f"   Quality Score: {response.metadata.get('quality_score', 'N/A')}")
            print(f"   Refinement Status: {response.metadata.get('refinement_status', 'N/A')}")
        
        # Show a preview of the refined response
        if response.content:
            preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
            print(f"   Refined Response Preview: {preview}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Refinement Processing Failed: {e}")
        return False


async def test_function_calling():
    """Test function calling capabilities"""
    
    print("🔧 Testing Function Calling")
    print("=" * 50)
    
    # Set the API key
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.environ.get('OPENAI_API_KEY'),
        temperature=0.7,
        max_tokens=100
    )
    
    service = LLMServiceFactory.create_service(config)
    
    # Define test functions
    functions = [
        {
            "name": "search_flights",
            "description": "Search for available flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city"},
                    "destination": {"type": "string", "description": "Arrival city"},
                    "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)"}
                },
                "required": ["origin", "destination", "date"]
            }
        },
        {
            "name": "search_hotels",
            "description": "Search for available hotels",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or area"},
                    "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                    "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                    "guests": {"type": "integer", "description": "Number of guests"}
                },
                "required": ["location", "check_in", "check_out"]
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "I need to find flights from New York to Tokyo for March 15, 2024, and hotels in Tokyo for March 15-20, 2024 for 2 guests."}
    ]
    
    try:
        print("   Testing function calling...")
        response = await service.function_call(messages, functions)
        
        print("   ✅ Function Call Test Successful!")
        print(f"   Content: {response.content}")
        print(f"   Function Called: {response.metadata.get('function_called', False)}")
        
        if response.metadata.get('function_called'):
            print(f"   Function Name: {response.metadata.get('function_name', 'N/A')}")
            print(f"   Function Args: {response.metadata.get('function_args', 'N/A')}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Function Call Test Failed: {e}")
        return False


async def test_error_handling():
    """Test error handling scenarios"""
    
    print("⚠️ Testing Error Handling")
    print("=" * 50)
    
    # Test 1: Invalid API Key
    print("1️⃣ Testing Invalid API Key Handling")
    
    invalid_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="invalid-key-12345",
        temperature=0.7,
        max_tokens=50
    )
    
    try:
        service = LLMServiceFactory.create_service(invalid_config)
        response = await service.chat_completion([{"role": "user", "content": "Hello"}])
        print("   ❌ Should have failed with invalid key")
        return False
    except Exception as e:
        print(f"   ✅ Correctly handled invalid API key: {str(e)[:100]}...")
    
    # Test 2: Invalid Model
    print("2️⃣ Testing Invalid Model Handling")
    
    invalid_model_config = LLMConfig(
        provider="openai",
        model="invalid-model-12345",
        api_key=os.environ.get('OPENAI_API_KEY'),
        temperature=0.7,
        max_tokens=50
    )
    
    try:
        service = LLMServiceFactory.create_service(invalid_model_config)
        response = await service.chat_completion([{"role": "user", "content": "Hello"}])
        print("   ❌ Should have failed with invalid model")
        return False
    except Exception as e:
        print(f"   ✅ Correctly handled invalid model: {str(e)[:100]}...")
    
    print()
    return True


async def test_performance_and_limits():
    """Test performance and rate limiting"""
    
    print("⚡ Testing Performance and Limits")
    print("=" * 50)
    
    # Set the API key
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.environ.get('OPENAI_API_KEY'),
        temperature=0.7,
        max_tokens=100
    )
    
    service = LLMServiceFactory.create_service(config)
    
    # Test response time
    print("1️⃣ Testing Response Time")
    
    start_time = time.time()
    try:
        response = await service.chat_completion([{"role": "user", "content": "Quick test"}])
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"   ✅ Response Time: {response_time:.2f} seconds")
        
        if response_time < 5.0:
            print("   ✅ Response time is acceptable (< 5 seconds)")
        else:
            print("   ⚠️ Response time is slow (> 5 seconds)")
        
        if response.usage:
            tokens_used = response.usage.get('total_tokens', 0)
            print(f"   Tokens Used: {tokens_used}")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False
    
    print()
    return True


async def main():
    """Main test function"""
    
    print("🚀 Starting OpenAI API Integration Tests with Travel Agent")
    print("=" * 70)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔑 Using API Key: {os.environ.get('OPENAI_API_KEY', 'N/A')[:20]}...")
    print()
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("OpenAI API Connection", test_openai_api_connection),
        ("Travel Agent Integration", test_travel_agent_integration),
        ("Function Calling", test_function_calling),
        ("Error Handling", test_error_handling),
        ("Performance and Limits", test_performance_and_limits)
    ]
    
    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            test_results[test_name] = result
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception")
            print(f"   Error: {e}")
            test_results[test_name] = False
        
        print()
        # Add a small delay between tests to be respectful to the API
        await asyncio.sleep(1)
    
    # Print summary
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print()
    print(f"🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your OpenAI API integration is working correctly.")
        print("✅ The travel agent is ready to use with your API key.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("🔧 You may need to:")
        print("   - Verify your API key is correct")
        print("   - Check your OpenAI account billing status")
        print("   - Ensure you have sufficient API credits")
        print("   - Check your network connectivity")
    
    print()
    print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc() 