"""
Test Runner for Chat API Testing
Supports both single-session and multi-session testing modes
"""

import asyncio
import sys
import argparse
import traceback
from pathlib import Path

try:
    from chat_tester import ChatAPITester
    from test_scenarios import SINGLE_QUERY_SCENARIOS, MULTI_QUERY_SCENARIOS, REFINEMENT_TEST_CONFIG
    from metrics_analyzer import analyze_latest_results
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure you're running this from the tests/chats directory")
    sys.exit(1)


async def run_tests(
    test_mode: str = "both",
    enable_refinement: bool = False, 
    timeout_seconds: float = 300.0,
    base_url: str = "http://localhost:8000"
):
    """Run chat API tests with configurable mode and refinement"""
    
    refinement_mode = "with" if enable_refinement else "without"
    print(f"🚀 Starting Advanced Chat API Tests")
    print(f"   Mode: {test_mode}")
    print(f"   Refinement: {refinement_mode}")
    print(f"   Timeout: {timeout_seconds}s")
    print(f"   Base URL: {base_url}")
    print("=" * 60)
    
    try:
        async with ChatAPITester(base_url=base_url, timeout_seconds=timeout_seconds) as tester:
            
            # Configure agent if refinement is enabled
            if enable_refinement:
                print("🔧 Configuring agent for refinement...")
                try:
                    agent_config = await tester.configure_agent(
                        enabled=True,
                        fast_response_threshold=0.75,  # For LLM enhancement decision
                        quality_threshold=0.9,  # For refinement loop iteration
                        max_iterations=3
                    )
                    print(f"   ✅ Agent configured: {agent_config.get('message', 'Success')}")
                except Exception as e:
                    print(f"   ⚠️  Agent configuration failed: {e}")
                    print("   Continuing with default settings...")
            
            # Get initial agent status
            try:
                status = await tester.get_agent_status()
                print(f"📊 Agent Status: {status.get('system_status', 'unknown')}")
                print(f"   Refinement enabled: {status.get('refinement_config', {}).get('enabled', False)}")
                print(f"   Available tools: {len(status.get('available_tools', []))}")
            except Exception as e:
                print(f"⚠️  Could not get agent status: {e}")
            
            print()
            
            # Run tests based on mode
            if test_mode in ["single", "both"]:
                print("🎯 Running Single-Session Tests...")
                single_metrics = await tester.run_single_session_tests(
                    test_scenarios=SINGLE_QUERY_SCENARIOS,
                    enable_refinement=enable_refinement
                )
                print(f"   Completed {len(single_metrics)} single-session tests")
            
            if test_mode in ["multi", "both"]:
                print("🎯 Running Multi-Session Tests...")
                multi_suites = await tester.run_multi_session_tests(
                    session_scenarios=MULTI_QUERY_SCENARIOS,
                    enable_refinement=enable_refinement
                )
                print(f"   Completed {len(multi_suites)} multi-session test suites")
            
            # Save results and print summary
            print("\n💾 Saving results...")
            file_paths = tester.save_results()
            
            tester.print_summary()
            
            print(f"\n📁 Results saved to:")
            for key, path in file_paths.items():
                print(f"   {key}: {path}")
            
            return tester
            
    except Exception as e:
        print(f"❌ Error running advanced tests: {e}")
        traceback.print_exc()
        raise


async def run_refinement_comparison_tests(
    timeout_seconds: float = 300.0,
    base_url: str = "http://localhost:8000"
):
    """Run comparison tests with and without refinement"""
    
    print("🔄 Running Refinement Comparison Tests")
    print("   Testing both with and without refinement for comparison")
    print("=" * 60)
    
    # Use a smaller subset for comparison
    subset_single = SINGLE_QUERY_SCENARIOS[:4]  # First 4 scenarios
    subset_multi = MULTI_QUERY_SCENARIOS[:2]    # First 2 multi scenarios
    
    try:
        async with ChatAPITester(base_url=base_url, timeout_seconds=timeout_seconds) as tester:
            
            # Test without refinement first (faster)
            print("🎯 Phase 1: Tests WITHOUT refinement")
            
            single_metrics_no_ref = await tester.run_single_session_tests(
                test_scenarios=subset_single,
                enable_refinement=False
            )
            
            multi_suites_no_ref = await tester.run_multi_session_tests(
                session_scenarios=subset_multi,
                enable_refinement=False
            )
            
            print(f"   Phase 1 Complete: {len(single_metrics_no_ref)} single + {len(multi_suites_no_ref)} multi suites")
            
            # Brief pause between phases
            await asyncio.sleep(2)
            
            # Test with refinement
            print("\n🎯 Phase 2: Tests WITH refinement")
            
            single_metrics_ref = await tester.run_single_session_tests(
                test_scenarios=subset_single,
                enable_refinement=True
            )
            
            multi_suites_ref = await tester.run_multi_session_tests(
                session_scenarios=subset_multi,
                enable_refinement=True
            )
            
            print(f"   Phase 2 Complete: {len(single_metrics_ref)} single + {len(multi_suites_ref)} multi suites")
            
            # Save results and summary
            file_paths = tester.save_results()
            tester.print_summary()
            
            print(f"\n📊 Comparison Results Available in: {file_paths['results_dir']}")
            
            return tester
            
    except Exception as e:
        print(f"❌ Error running comparison tests: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Advanced Chat API Test Runner")
    
    # Test mode options
    parser.add_argument("--mode", choices=["single", "multi", "both", "comparison"], 
                       default="both",
                       help="Test mode: single-session only, multi-session only, both, or comparison")
    
    # Refinement options
    parser.add_argument("--refinement", action="store_true", 
                       help="Enable self-refinement during testing")
    parser.add_argument("--no-refinement", action="store_true",
                       help="Explicitly disable refinement (overrides --refinement)")
    
    # Analysis options  
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze existing results without running new tests")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Run tests but skip analysis")
    
    # Configuration options
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL for the API")
    parser.add_argument("--timeout", type=float, default=300.0,
                       help="Timeout in seconds for API requests")
    
    args = parser.parse_args()
    
    # Handle analyze-only mode
    if args.analyze_only:
        print("📊 Analyzing existing enhanced test results...")
        try:
            analyze_latest_results()
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
            traceback.print_exc()
        return
    
    # Determine refinement setting
    enable_refinement = args.refinement and not args.no_refinement
    
    # Print configuration
    print("🔧 ADVANCED CHAT API TEST CONFIGURATION")
    print("=" * 50)
    print(f"Test Mode: {args.mode}")
    print(f"Refinement: {'Enabled' if enable_refinement else 'Disabled'}")
    print(f"Base URL: {args.base_url}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.1f} minutes)")
    print(f"Analysis: {'Enabled' if not args.no_analysis else 'Disabled'}")
    print()
    
    print("📋 Prerequisites:")
    print("   1. API server running at specified URL")
    print("   2. Server fully initialized with knowledge base")
    print("   3. Required dependencies installed")
    print()
    
    if enable_refinement:
        print("⚠️  Refinement enabled - tests will be slower but more comprehensive")
    else:
        print("ℹ️  Refinement disabled - tests will be faster")
    
    if args.mode == "comparison":
        print("🔄 Comparison mode - will test both with and without refinement")
    
    print()
    
    try:
        # Run the appropriate test mode
        if args.mode == "comparison":
            asyncio.run(run_refinement_comparison_tests(
                timeout_seconds=args.timeout,
                base_url=args.base_url
            ))
        else:
            asyncio.run(run_tests(
                test_mode=args.mode,
                enable_refinement=enable_refinement,
                timeout_seconds=args.timeout,
                base_url=args.base_url
            ))
        
        # Run analysis unless disabled
        if not args.no_analysis:
            print("\n📊 Analyzing results...")
            try:
                analyze_latest_results()
            except Exception as e:
                print(f"⚠️  Analysis failed: {e}")
                traceback.print_exc()
        
        print("\n✅ Testing completed successfully!")
        
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