"""
Comprehensive Test Runner for Chat API Tests
Supports both refinement and non-refinement testing modes
"""

import asyncio
import sys
import argparse
import traceback

try:
    from test_chat_api import ChatAPITester, TEST_SCENARIOS
    from metrics_analyzer import analyze_latest_metrics
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure you're running this from the tests/chats directory")
    sys.exit(1)


async def run_chat_tests(enable_refinement=False, timeout_seconds=300.0):
    """Run chat API tests with configurable refinement and timeout"""
    refinement_mode = "with" if enable_refinement else "without"
    print(f"Starting Chat API Tests ({refinement_mode} refinement, {timeout_seconds}s timeout)...")
    
    try:
        async with ChatAPITester(timeout_seconds=timeout_seconds) as tester:
            # Create a new session for testing
            print("Creating test session...")
            session_id = await tester.create_session(
                title=f"Chat API Test Session ({refinement_mode} refinement)",
                description=f"Automated testing session for chat API baseline metrics ({refinement_mode} refinement)"
            )
            
            print(f"✅ Created test session: {session_id}")
            
            # Run all tests with specified refinement setting
            print(f"\n--- Running tests {refinement_mode} refinement ---")
            for i, scenario in enumerate(TEST_SCENARIOS, 1):
                print(f"[{i}/{len(TEST_SCENARIOS)}] Testing: {scenario['name']}")
                
                try:
                    metric = await tester.send_chat_message(
                        message=scenario["message"],
                        session_id=session_id,
                        enable_refinement=enable_refinement
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
            
            # If refinement is enabled, also run a few tests without refinement for comparison
            if enable_refinement:
                print("\n--- Running comparison tests without refinement ---")
                comparison_scenarios = TEST_SCENARIOS[:3]  # First 3 scenarios
                
                for i, scenario in enumerate(comparison_scenarios, 1):
                    print(f"[{i}/{len(comparison_scenarios)}] Testing: {scenario['name']} (no refinement)")
                    
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
                        print(f"  ❌ Error testing {scenario['name']} (no refinement): {e}")
                        traceback.print_exc()
                    
                    # Brief pause between tests
                    await asyncio.sleep(1)
            
            # Print summary and save metrics
            tester.print_summary()
            
            # Save with appropriate filename
            filename = f"{'refinement' if enable_refinement else 'no_refinement'}_test_results.json"
            metrics_file = tester.save_metrics(filename)
            
            print(f"\n✅ Test completed. Metrics saved to: {metrics_file}")
            
            return tester.metrics
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Run Chat API Tests")
    parser.add_argument("--refinement", action="store_true", 
                       help="Enable self-refinement during testing (slower, may timeout)")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze existing metrics without running new tests")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Run tests but skip analysis")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=float, default=300.0,
                       help="Timeout in seconds for API requests (default: 300.0 = 5 minutes)")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print("Analyzing existing metrics...")
        try:
            analyze_latest_metrics()
        except Exception as e:
            print(f"Error analyzing metrics: {e}")
            traceback.print_exc()
        return
    
    # Run the tests
    refinement_mode = "with" if args.refinement else "without"
    print("Starting Chat API Tests...")
    print(f"Base URL: {args.base_url}")
    print(f"Timeout: {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Refinement: {'Enabled' if args.refinement else 'Disabled'}")
    print("-" * 50)
    print("📋 Prerequisites:")
    print("   1. API server must be running at the specified URL")
    print("   2. Required dependencies should be installed (httpx, matplotlib, etc.)")
    print("   3. Server should be fully initialized with knowledge base")
    print()
    
    if args.refinement:
        print("⚠️  Refinement mode enabled - tests will be slower and may timeout")
        print("💡 Use without --refinement for faster, more reliable testing")
    else:
        print("ℹ️  Refinement mode disabled - tests will be faster and more reliable")
        print("💡 Use --refinement to test self-refinement functionality")
    print()
    
    try:
        # Run async tests
        asyncio.run(run_chat_tests(enable_refinement=args.refinement, timeout_seconds=args.timeout))
        
        if not args.no_analysis:
            print("\nAnalyzing results...")
            try:
                analyze_latest_metrics()
            except Exception as e:
                print(f"Error analyzing results: {e}")
                traceback.print_exc()
            
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        print("💡 Please ensure the API server is running:")
        print("   python -m uvicorn app.main:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 