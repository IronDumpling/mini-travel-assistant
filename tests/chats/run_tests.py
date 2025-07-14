"""
Test Runner for Chat API Tests
"""

import asyncio
import sys
import argparse
import traceback

try:
    from test_chat_api import run_chat_tests
    from metrics_analyzer import analyze_latest_metrics
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure you're running this from the tests/chats directory")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Chat API Tests")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze existing metrics without running new tests")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Run tests but skip analysis")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL for the API (default: http://localhost:8000)")
    
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
    print("Starting Chat API Tests...")
    print(f"Base URL: {args.base_url}")
    print("-" * 50)
    print("📋 Prerequisites:")
    print("   1. API server must be running at the specified URL")
    print("   2. Required dependencies should be installed (httpx, matplotlib, etc.)")
    print("   3. Server should be fully initialized with knowledge base")
    print()
    
    try:
        # Run async tests
        asyncio.run(run_chat_tests())
        
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