"""
Chat API Tester with metrics and refinement loop tracking
Supports two testing modes:
1. Single-query tests: Different sessions with single questions each
2. Multi-query tests: Same session with multiple questions
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import httpx


@dataclass 
class RefinementLoopMetric:
    """Metrics for a single refinement loop iteration"""
    iteration: int
    confidence: float
    quality_scores: Dict[str, float]
    actions_taken: List[str]
    response_time: float
    improvements_made: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestMetric:
    """Test metrics with detailed refinement tracking"""
    test_id: str
    test_type: str  # "single_session" or "multi_session"
    session_id: str
    query_index: int  # For multi-session tests
    test_name: str
    request_data: Dict
    response_data: Dict
    
    # Timing metrics
    total_response_time: float
    refinement_enabled: bool
    
    # Refinement loop details
    refinement_loops: List[RefinementLoopMetric] = field(default_factory=list)
    total_loops: int = 0
    
    # Final results
    final_success: bool = False
    final_confidence: float = 0.0
    final_actions: List[str] = field(default_factory=list)
    final_next_steps: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class SessionTestSuite:
    """A collection of tests for a single session"""
    session_id: str
    session_title: str
    session_description: str
    queries: List[str]
    metrics: List[TestMetric] = field(default_factory=list)
    session_created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ChatAPITester:
    """Chat API tester with metrics and two testing modes"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: float = 300.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Test metrics storage
        self.single_session_metrics: List[TestMetric] = []
        self.multi_session_suites: List[SessionTestSuite] = []
        
        # Create timestamped results directory
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"tests/chats/results/{self.test_run_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Test run ID: {self.test_run_id}")
        print(f"Results directory: {self.results_dir}")
    
    async def __aenter__(self):
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
                if response.status_code in [200, 404]:
                    print(f"✅ Server ready at {self.base_url}")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                print(f"Attempt {attempt + 1}/{max_attempts}: Server not ready - {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
        
        raise ConnectionError(f"Server not ready after {max_attempts} attempts")
    
    async def _safe_request(self, method: str, url: str, **kwargs):
        """Make a safe HTTP request with error handling"""
        try:
            response = await self.client.request(method, url, **kwargs)
            return response
        except Exception as e:
            print(f"Request error for {method} {url}: {e}")
            raise
    
    async def create_session(self, title: str = None, description: str = None) -> str:
        """Create a new session"""
        session_data = {}
        if title:
            session_data["title"] = title
        if description:
            session_data["description"] = description
        
        response = await self._safe_request(
            "POST", f"{self.base_url}/api/sessions", json=session_data
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result["session_id"]
            print(f"Created session: {session_id}")
            return session_id
        else:
            raise Exception(f"Failed to create session: {response.status_code} - {response.text}")
    
    async def get_agent_status(self) -> Dict:
        """Get current agent status"""
        response = await self._safe_request("GET", f"{self.base_url}/api/agent/status")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get agent status: {response.status_code}")
    
    async def configure_agent(self, enabled: bool = True, fast_response_threshold: float = 0.75, 
                             quality_threshold: float = 0.9, max_iterations: int = 3):
        """Configure agent refinement settings with two-tier thresholds"""
        config = {
            "enabled": enabled,
            "fast_response_threshold": fast_response_threshold,
            "quality_threshold": quality_threshold,
            "max_iterations": max_iterations
        }
        
        response = await self._safe_request(
            "POST", f"{self.base_url}/api/agent/configure", json=config
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to configure agent: {response.status_code}")
    
    async def send_chat_message(self, message: str, session_id: str, 
                                       enable_refinement: bool = True,
                                       test_type: str = "single_session",
                                       query_index: int = 0) -> TestMetric:
        """Send chat message with metrics tracking"""
        
        test_id = str(uuid.uuid4())
        request_data = {
            "message": message,
            "session_id": session_id,
            "enable_refinement": enable_refinement
        }
        
        start_time = time.time()
        
        try:
            response = await self._safe_request(
                "POST", f"{self.base_url}/api/chat", json=request_data
            )
            
            end_time = time.time()
            total_response_time = end_time - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract refinement loop details if available
                refinement_loops = []
                refinement_details = response_data.get("refinement_details", {})
                
                if refinement_details and "iterations" in refinement_details:
                    for i, iteration in enumerate(refinement_details["iterations"]):
                        loop_metric = RefinementLoopMetric(
                            iteration=i + 1,
                            confidence=iteration.get("confidence", 0.0),
                            quality_scores=iteration.get("quality_scores", {}),
                            actions_taken=iteration.get("actions_taken", []),
                            response_time=iteration.get("response_time", 0.0),
                            improvements_made=iteration.get("improvements_made", [])
                        )
                        refinement_loops.append(loop_metric)
                
                metric = TestMetric(
                    test_id=test_id,
                    test_type=test_type,
                    session_id=session_id,
                    query_index=query_index,
                    test_name=f"{test_type}_query_{query_index}",
                    request_data=request_data,
                    response_data=response_data,
                    total_response_time=total_response_time,
                    refinement_enabled=enable_refinement,
                    refinement_loops=refinement_loops,
                    total_loops=len(refinement_loops),
                    final_success=response_data.get("success", False),
                    final_confidence=response_data.get("confidence", 0.0),
                    final_actions=response_data.get("actions_taken", []),
                    final_next_steps=response_data.get("next_steps", [])
                )
                
                return metric
            else:
                # Handle error response
                end_time = time.time()
                total_response_time = end_time - start_time
                
                metric = TestMetric(
                    test_id=test_id,
                    test_type=test_type,
                    session_id=session_id,
                    query_index=query_index,
                    test_name=f"{test_type}_query_{query_index}",
                    request_data=request_data,
                    response_data={"error": response.text},
                    total_response_time=total_response_time,
                    refinement_enabled=enable_refinement,
                    final_success=False,
                    error_message=response.text
                )
                
                return metric
                
        except Exception as e:
            end_time = time.time()
            total_response_time = end_time - start_time
            
            metric = TestMetric(
                test_id=test_id,
                test_type=test_type,
                session_id=session_id,
                query_index=query_index,
                test_name=f"{test_type}_query_{query_index}",
                request_data=request_data,
                response_data={"error": str(e)},
                total_response_time=total_response_time,
                refinement_enabled=enable_refinement,
                final_success=False,
                error_message=str(e)
            )
            
            return metric
    
    async def run_single_session_tests(self, test_scenarios: List[Dict], enable_refinement: bool = True) -> List[TestMetric]:
        """Run single-session tests: different sessions with single questions each"""
        print(f"\n🔄 Running Single-Session Tests ({'with' if enable_refinement else 'without'} refinement)")
        print("=" * 60)
        
        metrics = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"[{i}/{len(test_scenarios)}] Testing: {scenario['name']}")
            
            try:
                # Create a new session for each test
                session_id = await self.create_session(
                    title=f"Single Test: {scenario['name']}",
                    description=f"Single query test for {scenario['name']}"
                )
                
                # Send the message
                metric = await self.send_chat_message(
                    message=scenario["message"],
                    session_id=session_id,
                    enable_refinement=enable_refinement,
                    test_type="single_session",
                    query_index=0
                )
                
                metrics.append(metric)
                status = "✅" if metric.final_success else "❌"
                print(f"  {status} Session: {session_id}")
                print(f"  {status} Response time: {metric.total_response_time:.2f}s")
                print(f"  {status} Success: {metric.final_success}")
                print(f"  {status} Confidence: {metric.final_confidence:.2f}")
                print(f"  {status} Refinement loops: {metric.total_loops}")
                
                if not metric.final_success:
                    print(f"  ❌ Error: {metric.error_message}")
                
            except Exception as e:
                print(f"  ❌ Error in scenario {scenario['name']}: {e}")
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        self.single_session_metrics.extend(metrics)
        return metrics
    
    async def run_multi_session_tests(self, session_scenarios: List[Dict], enable_refinement: bool = True) -> List[SessionTestSuite]:
        """Run multi-session tests: same session with multiple questions"""
        print(f"\n🔄 Running Multi-Session Tests ({'with' if enable_refinement else 'without'} refinement)")
        print("=" * 60)
        
        test_suites = []
        
        for suite_config in session_scenarios:
            print(f"\n📋 Session Suite: {suite_config['title']}")
            print(f"   Queries: {len(suite_config['queries'])}")
            
            try:
                # Create a session for this suite
                session_id = await self.create_session(
                    title=suite_config["title"],
                    description=suite_config["description"]
                )
                
                suite = SessionTestSuite(
                    session_id=session_id,
                    session_title=suite_config["title"],
                    session_description=suite_config["description"],
                    queries=suite_config["queries"]
                )
                
                # Send multiple queries to the same session
                for query_idx, query in enumerate(suite_config["queries"]):
                    print(f"  [{query_idx + 1}/{len(suite_config['queries'])}] Query: {query[:50]}...")
                    
                    metric = await self.send_chat_message(
                        message=query,
                        session_id=session_id,
                        enable_refinement=enable_refinement,
                        test_type="multi_session",
                        query_index=query_idx
                    )
                    
                    suite.metrics.append(metric)
                    
                    status = "✅" if metric.final_success else "❌"
                    print(f"    {status} Response time: {metric.total_response_time:.2f}s")
                    print(f"    {status} Confidence: {metric.final_confidence:.2f}")
                    print(f"    {status} Refinement loops: {metric.total_loops}")
                    
                    # Pause between queries in the same session
                    await asyncio.sleep(0.5)
                
                test_suites.append(suite)
                
                # Session summary
                successful_queries = sum(1 for m in suite.metrics if m.final_success)
                print(f"  📊 Session Summary: {successful_queries}/{len(suite.metrics)} successful")
                
            except Exception as e:
                print(f"  ❌ Error in session suite {suite_config['title']}: {e}")
            
            # Pause between session suites
            await asyncio.sleep(2)
        
        self.multi_session_suites.extend(test_suites)
        return test_suites
    
    def save_results(self):
        """Save all test results in the timestamped directory"""
        
        # Save single-session results
        single_session_file = self.results_dir / "single_session_tests.json"
        single_session_data = []
        for metric in self.single_session_metrics:
            # Convert dataclass to dict, handling nested dataclasses
            metric_dict = {
                "test_id": metric.test_id,
                "test_type": metric.test_type,
                "session_id": metric.session_id,
                "query_index": metric.query_index,
                "test_name": metric.test_name,
                "request_data": metric.request_data,
                "response_data": metric.response_data,
                "total_response_time": metric.total_response_time,
                "refinement_enabled": metric.refinement_enabled,
                "refinement_loops": [
                    {
                        "iteration": loop.iteration,
                        "confidence": loop.confidence,
                        "quality_scores": loop.quality_scores,
                        "actions_taken": loop.actions_taken,
                        "response_time": loop.response_time,
                        "improvements_made": loop.improvements_made,
                        "timestamp": loop.timestamp
                    } for loop in metric.refinement_loops
                ],
                "total_loops": metric.total_loops,
                "final_success": metric.final_success,
                "final_confidence": metric.final_confidence,
                "final_actions": metric.final_actions,
                "final_next_steps": metric.final_next_steps,
                "timestamp": metric.timestamp,
                "error_message": metric.error_message
            }
            single_session_data.append(metric_dict)
        
        with open(single_session_file, 'w', encoding='utf-8') as f:
            json.dump(single_session_data, f, indent=2, ensure_ascii=False)
        
        # Save multi-session results
        multi_session_file = self.results_dir / "multi_session_tests.json"
        multi_session_data = []
        for suite in self.multi_session_suites:
            suite_dict = {
                "session_id": suite.session_id,
                "session_title": suite.session_title,
                "session_description": suite.session_description,
                "queries": suite.queries,
                "session_created_at": suite.session_created_at,
                "metrics": []
            }
            
            for metric in suite.metrics:
                metric_dict = {
                    "test_id": metric.test_id,
                    "test_type": metric.test_type,
                    "session_id": metric.session_id,
                    "query_index": metric.query_index,
                    "test_name": metric.test_name,
                    "request_data": metric.request_data,
                    "response_data": metric.response_data,
                    "total_response_time": metric.total_response_time,
                    "refinement_enabled": metric.refinement_enabled,
                    "refinement_loops": [
                        {
                            "iteration": loop.iteration,
                            "confidence": loop.confidence,
                            "quality_scores": loop.quality_scores,
                            "actions_taken": loop.actions_taken,
                            "response_time": loop.response_time,
                            "improvements_made": loop.improvements_made,
                            "timestamp": loop.timestamp
                        } for loop in metric.refinement_loops
                    ],
                    "total_loops": metric.total_loops,
                    "final_success": metric.final_success,
                    "final_confidence": metric.final_confidence,
                    "final_actions": metric.final_actions,
                    "final_next_steps": metric.final_next_steps,
                    "timestamp": metric.timestamp,
                    "error_message": metric.error_message
                }
                suite_dict["metrics"].append(metric_dict)
            
            multi_session_data.append(suite_dict)
        
        with open(multi_session_file, 'w', encoding='utf-8') as f:
            json.dump(multi_session_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = self.results_dir / "test_summary.json"
        summary = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "single_session_tests": {
                "total": len(self.single_session_metrics),
                "successful": sum(1 for m in self.single_session_metrics if m.final_success),
                "with_refinement": sum(1 for m in self.single_session_metrics if m.refinement_enabled),
                "total_refinement_loops": sum(m.total_loops for m in self.single_session_metrics)
            },
            "multi_session_tests": {
                "total_suites": len(self.multi_session_suites),
                "total_queries": sum(len(suite.metrics) for suite in self.multi_session_suites),
                "successful_queries": sum(sum(1 for m in suite.metrics if m.final_success) for suite in self.multi_session_suites),
                "total_refinement_loops": sum(sum(m.total_loops for m in suite.metrics) for suite in self.multi_session_suites)
            },
            "files": {
                "single_session": "single_session_tests.json",
                "multi_session": "multi_session_tests.json",
                "summary": "test_summary.json"
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {self.results_dir}")
        print(f"   - Single-session tests: {single_session_file}")
        print(f"   - Multi-session tests: {multi_session_file}")
        print(f"   - Summary: {summary_file}")
        
        return {
            "results_dir": str(self.results_dir),
            "single_session_file": str(single_session_file),
            "multi_session_file": str(multi_session_file),
            "summary_file": str(summary_file)
        }
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("ADVANCED CHAT API TEST SUMMARY")
        print("=" * 80)
        print(f"Test Run ID: {self.test_run_id}")
        print(f"Results Directory: {self.results_dir}")
        
        # Single-session summary
        single_total = len(self.single_session_metrics)
        single_success = sum(1 for m in self.single_session_metrics if m.final_success)
        single_with_refinement = sum(1 for m in self.single_session_metrics if m.refinement_enabled)
        single_total_loops = sum(m.total_loops for m in self.single_session_metrics)
        
        print(f"\n📊 SINGLE-SESSION TESTS:")
        print(f"   Total tests: {single_total}")
        print(f"   Successful: {single_success} ({single_success/max(single_total,1)*100:.1f}%)")
        print(f"   With refinement: {single_with_refinement}")
        print(f"   Total refinement loops: {single_total_loops}")
        if single_total > 0:
            avg_time = sum(m.total_response_time for m in self.single_session_metrics) / single_total
            avg_confidence = sum(m.final_confidence for m in self.single_session_metrics if m.final_success) / max(single_success, 1)
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Average confidence: {avg_confidence:.2f}")
        
        # Multi-session summary
        multi_suites = len(self.multi_session_suites)
        multi_total_queries = sum(len(suite.metrics) for suite in self.multi_session_suites)
        multi_successful_queries = sum(sum(1 for m in suite.metrics if m.final_success) for suite in self.multi_session_suites)
        multi_total_loops = sum(sum(m.total_loops for m in suite.metrics) for suite in self.multi_session_suites)
        
        print(f"\n📊 MULTI-SESSION TESTS:")
        print(f"   Session suites: {multi_suites}")
        print(f"   Total queries: {multi_total_queries}")
        print(f"   Successful queries: {multi_successful_queries} ({multi_successful_queries/max(multi_total_queries,1)*100:.1f}%)")
        print(f"   Total refinement loops: {multi_total_loops}")
        
        if multi_total_queries > 0:
            all_multi_metrics = [m for suite in self.multi_session_suites for m in suite.metrics]
            avg_time = sum(m.total_response_time for m in all_multi_metrics) / multi_total_queries
            successful_multi_metrics = [m for m in all_multi_metrics if m.final_success]
            avg_confidence = sum(m.final_confidence for m in successful_multi_metrics) / max(len(successful_multi_metrics), 1)
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Average confidence: {avg_confidence:.2f}")
        
        print("\n" + "=" * 80) 