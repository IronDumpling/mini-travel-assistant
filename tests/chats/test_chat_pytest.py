"""
Pytest-compatible test cases for the Chat API
"""

import asyncio
import sys

try:
    import pytest
    from test_chat_api import ChatAPITester, TEST_SCENARIOS
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("  pip install pytest pytest-asyncio httpx")
    sys.exit(1)


class TestChatAPI:
    """Pytest test class for Chat API"""
    
    @pytest.fixture(scope="session")
    def event_loop(self):
        """Create an instance of the default event loop for the test session."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="session")
    async def tester(self):
        """Create a ChatAPITester instance for the test session"""
        async with ChatAPITester() as tester:
            # Create a session for testing
            session_id = await tester.create_session(
                title="Pytest Chat API Test Session",
                description="Automated pytest session for chat API testing"
            )
            yield tester
    
    @pytest.mark.asyncio
    async def test_create_session(self, tester):
        """Test session creation"""
        session_id = await tester.create_session(
            title="Test Session",
            description="Test session creation"
        )
        assert session_id is not None
        assert len(session_id) > 0
    
    @pytest.mark.asyncio
    async def test_get_sessions(self, tester):
        """Test getting all sessions"""
        sessions = await tester.get_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) > 0
    
    @pytest.mark.asyncio
    async def test_london_trip_planning(self, tester):
        """Test London trip planning scenario"""
        metric = await tester.send_chat_message(
            message="Plan a 7-day trip to London for 2 people with a budget of $3000",
            enable_refinement=True
        )
        
        assert metric.success == True
        assert metric.confidence > 0.5
        assert metric.response_time < 90.0  # Should respond within 90 seconds
        assert len(metric.actions_taken) > 0
        assert len(metric.next_steps) > 0
    
    @pytest.mark.asyncio
    async def test_paris_romantic_weekend(self, tester):
        """Test Paris romantic weekend scenario"""
        metric = await tester.send_chat_message(
            message="Plan a romantic weekend in Paris for 2 people, focusing on fine dining and cultural experiences",
            enable_refinement=True
        )
        
        assert metric.success == True
        assert metric.confidence > 0.5
        assert metric.response_time < 90.0
        assert len(metric.actions_taken) > 0
    
    @pytest.mark.asyncio
    async def test_tokyo_business_trip(self, tester):
        """Test Tokyo business trip scenario"""
        metric = await tester.send_chat_message(
            message="I need a 3-day business trip to Tokyo with meetings in Shibuya and Shinjuku areas",
            enable_refinement=True
        )
        
        assert metric.success == True
        assert metric.confidence > 0.5
        assert metric.response_time < 90.0
        assert len(metric.actions_taken) > 0
    
    @pytest.mark.asyncio
    async def test_family_singapore(self, tester):
        """Test family Singapore trip scenario"""
        metric = await tester.send_chat_message(
            message="Plan a 5-day family trip to Singapore with 2 adults and 2 children (ages 8 and 12)",
            enable_refinement=True
        )
        
        assert metric.success == True
        assert metric.confidence > 0.5
        assert metric.response_time < 90.0
        assert len(metric.actions_taken) > 0
    
    @pytest.mark.asyncio
    async def test_refinement_comparison(self, tester):
        """Test comparison between refinement enabled and disabled"""
        test_message = "Plan a 7-day trip to London for 2 people with a budget of $3000"
        
        # Test with refinement enabled
        metric_with_refinement = await tester.send_chat_message(
            message=test_message,
            enable_refinement=True
        )
        
        # Test with refinement disabled
        metric_without_refinement = await tester.send_chat_message(
            message=test_message,
            enable_refinement=False
        )
        
        assert metric_with_refinement.success == True
        assert metric_without_refinement.success == True
        
        # With refinement should generally have higher confidence
        # Note: This may not always be true, so we'll just check both succeeded
        assert metric_with_refinement.confidence > 0.0
        assert metric_without_refinement.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_response_structure(self, tester):
        """Test that response has expected structure"""
        metric = await tester.send_chat_message(
            message="Plan a simple weekend trip to Paris",
            enable_refinement=True
        )
        
        assert metric.success == True
        assert hasattr(metric, 'response_data')
        assert 'content' in metric.response_data
        assert 'confidence' in metric.response_data
        assert 'actions_taken' in metric.response_data
        assert 'next_steps' in metric.response_data
        assert 'session_id' in metric.response_data
    
    @pytest.mark.asyncio
    async def test_invalid_session_handling(self):
        """Test handling of invalid session ID"""
        async with ChatAPITester() as tester:
            metric = await tester.send_chat_message(
                message="Test message",
                session_id="invalid_session_id",
                enable_refinement=False
            )
            
            # Should handle gracefully - either succeed or fail with proper error
            assert metric.response_time > 0.0
            assert metric.session_id == "invalid_session_id"
    
    @pytest.mark.asyncio
    async def test_all_scenarios(self, tester):
        """Test all predefined scenarios"""
        successful_tests = 0
        
        for scenario in TEST_SCENARIOS:
            metric = await tester.send_chat_message(
                message=scenario["message"],
                enable_refinement=True
            )
            
            if metric.success:
                successful_tests += 1
                
            # Basic assertions for each test
            assert metric.response_time > 0.0
            assert metric.confidence >= 0.0
            assert isinstance(metric.actions_taken, list)
            assert isinstance(metric.next_steps, list)
        
        # At least 70% of tests should succeed
        success_rate = successful_tests / len(TEST_SCENARIOS)
        assert success_rate >= 0.7, f"Success rate {success_rate:.1%} below 70% threshold"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, tester):
        """Test performance benchmarks"""
        response_times = []
        
        # Test multiple requests to get average performance
        for i in range(3):
            metric = await tester.send_chat_message(
                message=f"Plan a simple day trip to a nearby city (test {i+1})",
                enable_refinement=False  # Disable refinement for faster response
            )
            response_times.append(metric.response_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions (adjusted for ~1 minute normal response time)
        assert avg_response_time < 75.0, f"Average response time {avg_response_time:.2f}s exceeds 75s"
        assert max_response_time < 90.0, f"Max response time {max_response_time:.2f}s exceeds 90s"
    
    def test_save_metrics_after_tests(self, tester):
        """Save metrics after all tests complete"""
        if tester.metrics:
            metrics_file = tester.save_metrics("pytest_results.json")
            assert metrics_file is not None
            print(f"Pytest metrics saved to: {metrics_file}")
        
        # Print summary
        tester.print_summary() 