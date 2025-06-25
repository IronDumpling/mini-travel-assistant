#!/usr/bin/env python3
"""
Test script for the self-refine loop implementation

This script demonstrates how the self-refine loop works in the Travel Agent.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage


async def test_self_refine_loop():
    """Test the self-refine loop functionality"""
    
    print("🧪 Testing Self-Refine Loop Implementation")
    print("=" * 50)
    
    # Create travel agent
    agent = TravelAgent()
    
    # Configure refinement settings for testing
    agent.configure_refinement(
        enabled=True,
        quality_threshold=0.7,  # Lower threshold for testing
        max_iterations=2        # Fewer iterations for testing
    )
    
    print(f"Agent: {agent.name}")
    print(f"Quality Threshold: {agent.quality_threshold}")
    print(f"Max Iterations: {agent.max_refine_iterations}")
    print(f"Refinement Enabled: {agent.refine_enabled}")
    print()
    
    # Test message - intentionally basic to trigger refinement
    test_message = AgentMessage(
        sender="user",
        receiver="travel_agent",
        content="I want to plan a family trip to Paris for 5 days with a budget of $3000. We have 2 kids and prefer cultural activities."
    )
    
    print("📨 Test Message:")
    print(f"Content: {test_message.content}")
    print()
    
    # Test with refinement enabled
    print("🔄 Processing with Self-Refinement...")
    response = await agent.plan_travel(test_message)
    
    print("\n📊 Response Analysis:")
    print(f"Success: {response.success}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Content Length: {len(response.content)} characters")
    print(f"Actions Taken: {len(response.actions_taken)}")
    print(f"Next Steps: {len(response.next_steps)}")
    print()
    
    # Print refinement metadata
    if "refinement_iteration" in response.metadata:
        print("🔍 Refinement Details:")
        print(f"Final Iteration: {response.metadata['refinement_iteration']}")
        print(f"Final Quality Score: {response.metadata['quality_score']:.2f}")
        print(f"Refinement Status: {response.metadata['refinement_status']}")
        
        if "refinement_history" in response.metadata:
            print("\nRefinement History:")
            for entry in response.metadata["refinement_history"]:
                print(f"  Iteration {entry['iteration']}: Score {entry['score']:.2f}")
                if entry["suggestions"]:
                    print(f"    Suggestions: {', '.join(entry['suggestions'][:2])}...")
        print()
    
    # Print travel-specific refinement info
    if response.metadata.get("travel_refined"):
        print("✈️ Travel-Specific Refinements:")
        improved_dims = response.metadata.get("quality_dimensions_improved", [])
        if improved_dims:
            print(f"Improved Dimensions: {', '.join(improved_dims)}")
        
        applied_improvements = response.metadata.get("improvement_applied", [])
        if applied_improvements:
            print("Applied Improvements:")
            for improvement in applied_improvements[:3]:  # Show first 3
                print(f"  • {improvement}")
        print()
    
    # Print sample of response content
    print("📝 Response Content (first 300 chars):")
    print(f"{response.content[:300]}...")
    print()
    
    if response.next_steps:
        print("📋 Next Steps:")
        for i, step in enumerate(response.next_steps[:3], 1):
            print(f"  {i}. {step}")
        print()
    
    # Test with refinement disabled for comparison
    print("🔄 Processing WITHOUT Self-Refinement (for comparison)...")
    agent.configure_refinement(enabled=False)
    response_no_refine = await agent.process_message(test_message)
    
    print(f"Without Refinement - Confidence: {response_no_refine.confidence:.2f}")
    print(f"Without Refinement - Content Length: {len(response_no_refine.content)}")
    print(f"Without Refinement - Next Steps: {len(response_no_refine.next_steps)}")
    print()
    
    # Compare results
    print("📈 Comparison:")
    confidence_improvement = response.confidence - response_no_refine.confidence
    content_improvement = len(response.content) - len(response_no_refine.content)
    steps_improvement = len(response.next_steps) - len(response_no_refine.next_steps)
    
    print(f"Confidence Improvement: {confidence_improvement:+.2f}")
    print(f"Content Length Improvement: {content_improvement:+d} characters")
    print(f"Next Steps Improvement: {steps_improvement:+d} steps")
    
    print("\n✅ Self-Refine Loop Test Completed!")


async def test_quality_dimensions():
    """Test the quality assessment dimensions"""
    
    print("\n🎯 Testing Quality Assessment Dimensions")
    print("=" * 50)
    
    agent = TravelAgent()
    
    # Test message
    test_message = AgentMessage(
        sender="user",
        receiver="travel_agent", 
        content="I need budget travel options for Tokyo"
    )
    
    # Create a mock response for testing
    from app.agents.base_agent import AgentResponse
    mock_response = AgentResponse(
        success=True,
        content="Here are some budget options for Tokyo.",
        actions_taken=["search_hotels"],
        next_steps=["Book accommodation"],
        confidence=0.7
    )
    
    # Test each quality dimension
    dimensions = agent.get_quality_dimensions()
    print("Quality Dimensions and Weights:")
    for dim, weight in dimensions.items():
        print(f"  {dim}: {weight:.2f}")
    print()
    
    print("Testing Dimension Assessment:")
    for dimension in dimensions.keys():
        score = await agent._assess_dimension(dimension, test_message, mock_response)
        print(f"  {dimension}: {score:.2f}")
    
    # Test full quality assessment
    quality_assessment = await agent._assess_response_quality(
        test_message, mock_response, 1
    )
    
    print(f"\nOverall Quality Assessment:")
    print(f"  Overall Score: {quality_assessment.overall_score:.2f}")
    print(f"  Meets Threshold: {quality_assessment.meets_threshold}")
    print(f"  Improvement Suggestions: {len(quality_assessment.improvement_suggestions)}")
    
    if quality_assessment.improvement_suggestions:
        print("  Sample Suggestions:")
        for suggestion in quality_assessment.improvement_suggestions[:2]:
            print(f"    • {suggestion}")


if __name__ == "__main__":
    print("🚀 Starting Self-Refine Loop Tests")
    print()
    
    try:
        asyncio.run(test_self_refine_loop())
        asyncio.run(test_quality_dimensions())
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 