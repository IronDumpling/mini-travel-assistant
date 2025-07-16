"""
Test cases for the Chat API with comprehensive baseline metrics and evaluation
"""

import asyncio
import json
import time
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import httpx
from pathlib import Path


@dataclass
class TestMetrics:
    """Store test metrics for evaluation"""
    test_name: str
    request_data: Dict
    response_data: Dict
    response_time: float
    success: bool
    confidence: float
    actions_taken: List[str]
    next_steps: List[str]
    refinement_details: Optional[Dict] = None
    timestamp: str = None
    session_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ChatAPITester:
    """Test suite for Chat API with metrics recording"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # Configure client with longer timeout for chat API (normally takes ~1 minute)
        # Extra time for refinement-enabled requests which may take longer
        timeout = httpx.Timeout(120.0, connect=10.0)  # 120 seconds for chat responses
        self.client = httpx.AsyncClient(timeout=timeout)
        self.metrics: List[TestMetrics] = []
        self.session_id = None
        
    async def __aenter__(self):
        # Check if server is ready
        await self._check_server_ready()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.client.aclose()
        except Exception as e:
            print(f"Warning: Error closing client: {e}")
    
    async def _check_server_ready(self, max_attempts: int = 10) -> bool:
        """Check if the server is ready to accept requests"""
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(f"{self.base_url}/", timeout=5.0)
                if response.status_code in [200, 404]:  # 404 is also OK, means server is responding
                    print(f"Server is ready at {self.base_url}")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                print(f"Attempt {attempt + 1}/{max_attempts}: Server not ready - {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
        
        raise ConnectionError(f"Server not ready after {max_attempts} attempts. Please ensure the server is running at {self.base_url}")
    
    async def _safe_request(self, method: str, url: str, **kwargs):
        """Make a safe HTTP request with proper error handling"""
        try:
            response = await self.client.request(method, url, **kwargs)
            return response
        except httpx.TimeoutException:
            print(f"Request timeout for {method} {url}")
            raise
        except httpx.ConnectError:
            print(f"Connection error for {method} {url}")
            raise
        except Exception as e:
            print(f"Unexpected error for {method} {url}: {e}")
            raise
        
    async def create_session(self, title: str = None, description: str = None) -> str:
        """Create a new session and return session ID"""
        session_data = {}
        if title:
            session_data["title"] = title
        if description:
            session_data["description"] = description
            
        try:
            response = await self._safe_request(
                "POST",
                f"{self.base_url}/api/sessions",
                json=session_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = result["session_id"]
                print(f"Created session: {self.session_id}")
                return self.session_id
            else:
                error_msg = f"Failed to create session: HTTP {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            print(f"Error creating session: {e}")
            raise
    
    async def get_sessions(self) -> List[Dict]:
        """Get all sessions"""
        try:
            response = await self._safe_request("GET", f"{self.base_url}/api/sessions")
            if response.status_code == 200:
                return response.json()["sessions"]
            else:
                error_msg = f"Failed to get sessions: HTTP {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            print(f"Error getting sessions: {e}")
            raise
    
    async def send_chat_message(self, message: str, session_id: str = None, 
                               enable_refinement: bool = True) -> TestMetrics:
        """Send a chat message and record metrics"""
        if session_id is None:
            session_id = self.session_id
            
        request_data = {
            "message": message,
            "session_id": session_id,
            "enable_refinement": enable_refinement
        }
        
        start_time = time.time()
        
        try:
            response = await self._safe_request(
                "POST",
                f"{self.base_url}/api/chat",
                json=request_data
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                metrics = TestMetrics(
                    test_name=f"chat_message_{len(self.metrics) + 1}",
                    request_data=request_data,
                    response_data=response_data,
                    response_time=response_time,
                    success=response_data.get("success", False),
                    confidence=response_data.get("confidence", 0.0),
                    actions_taken=response_data.get("actions_taken", []),
                    next_steps=response_data.get("next_steps", []),
                    refinement_details=response_data.get("refinement_details"),
                    session_id=session_id
                )
                
                self.metrics.append(metrics)
                return metrics
            else:
                # Handle error response
                end_time = time.time()
                response_time = end_time - start_time
                
                metrics = TestMetrics(
                    test_name=f"chat_message_{len(self.metrics) + 1}",
                    request_data=request_data,
                    response_data={"error": response.text},
                    response_time=response_time,
                    success=False,
                    confidence=0.0,
                    actions_taken=[],
                    next_steps=[],
                    session_id=session_id
                )
                
                self.metrics.append(metrics)
                return metrics
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            metrics = TestMetrics(
                test_name=f"chat_message_{len(self.metrics) + 1}",
                request_data=request_data,
                response_data={"error": str(e)},
                response_time=response_time,
                success=False,
                confidence=0.0,
                actions_taken=[],
                next_steps=[],
                session_id=session_id
            )
            
            self.metrics.append(metrics)
            return metrics
    
    def save_metrics(self, filename: str = None):
        """Save all metrics to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_test_metrics_{timestamp}.json"
            
        # Create output directory if it doesn't exist
        output_dir = Path("tests/chats/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        # Convert metrics to serializable format
        metrics_data = []
        for metric in self.metrics:
            metric_dict = {
                "test_name": metric.test_name,
                "request_data": metric.request_data,
                "response_data": metric.response_data,
                "response_time": metric.response_time,
                "success": metric.success,
                "confidence": metric.confidence,
                "actions_taken": metric.actions_taken,
                "next_steps": metric.next_steps,
                "refinement_details": metric.refinement_details,
                "timestamp": metric.timestamp,
                "session_id": metric.session_id
            }
            metrics_data.append(metric_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of test results"""
        if not self.metrics:
            print("No metrics recorded yet")
            return
            
        print("\n" + "="*50)
        print("CHAT API TEST SUMMARY")
        print("="*50)
        
        total_tests = len(self.metrics)
        successful_tests = sum(1 for m in self.metrics if m.success)
        failed_tests = total_tests - successful_tests
        
        avg_response_time = sum(m.response_time for m in self.metrics) / total_tests
        avg_confidence = sum(m.confidence for m in self.metrics if m.success) / max(successful_tests, 1)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        print("\nTest Details:")
        for i, metric in enumerate(self.metrics, 1):
            status = "✅" if metric.success else "❌"
            print(f"{i}. {status} {metric.test_name} - {metric.response_time:.2f}s - Confidence: {metric.confidence:.2f}")
        
        print("="*50)


# Test scenarios
TEST_SCENARIOS = [
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
        "name": "German_7_days_trip",
        "message": "Plan a 7-day trip to Berlin and Munich for 2 people with a budget of $7000 in this summer",
        "expected_keywords": ["berlin", "munich", "summer", "$7000", "7 days", "2 people"]
    },
]


async def run_chat_tests():
    """Run all chat API tests"""
    print("Starting Chat API Tests...")
    
    try:
        async with ChatAPITester() as tester:
            # Create a new session for testing
            print("Creating test session...")
            session_id = await tester.create_session(
                title="Chat API Test Session",
                description="Automated testing session for chat API baseline metrics"
            )
            
            print(f"✅ Created test session: {session_id}")
            
            # Run tests with refinement disabled first (faster and more reliable)
            print("\n--- Running tests with refinement disabled ---")
            for i, scenario in enumerate(TEST_SCENARIOS, 1):
                print(f"[{i}/{len(TEST_SCENARIOS)}] Testing: {scenario['name']} (no refinement)")
                
                try:
                    metric = await tester.send_chat_message(
                        message=scenario["message"],
                        session_id=session_id,
                        enable_refinement=False
                    )
                    
                    status = "✅" if metric.success else "❌"
                    print(f"  {status} Response time: {metric.response_time:.2f}s")
                    print(f"  {status} Success: {metric.success}")
                    print(f"  {status} Confidence: {metric.confidence:.2f}")
                    
                    if not metric.success:
                        print(f"  Error: {metric.response_data.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"  ❌ Error testing {scenario['name']}: {e}")
                    traceback.print_exc()
                
                # Brief pause between tests
                await asyncio.sleep(1)
            
            # Run a subset of tests with refinement enabled for comparison
            print("\n--- Running tests with refinement enabled ---")
            comparison_scenarios = TEST_SCENARIOS[:3]  # First 3 scenarios
            
            for i, scenario in enumerate(comparison_scenarios, 1):
                print(f"[{i}/{len(comparison_scenarios)}] Testing: {scenario['name']} (with refinement)")
                
                try:
                    metric = await tester.send_chat_message(
                        message=scenario["message"],
                        session_id=session_id,
                        enable_refinement=True
                    )
                    
                    status = "✅" if metric.success else "❌"
                    print(f"  {status} Response time: {metric.response_time:.2f}s")
                    print(f"  {status} Success: {metric.success}")
                    print(f"  {status} Confidence: {metric.confidence:.2f}")
                    
                    if not metric.success:
                        print(f"  Error: {metric.response_data.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"  ❌ Error testing {scenario['name']} (no refinement): {e}")
                    traceback.print_exc()
                
                # Brief pause between tests
                await asyncio.sleep(1)
            
            # Print summary and save metrics
            tester.print_summary()
            metrics_file = tester.save_metrics()
            
            print(f"\n✅ Test completed. Metrics saved to: {metrics_file}")
            
            return tester.metrics
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_chat_tests()) 