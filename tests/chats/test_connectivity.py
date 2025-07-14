"""
Connectivity test script to check if the Chat API server is working
Tests basic server connectivity, sessions API, and chat API functionality
"""

import asyncio
import httpx
from datetime import datetime


async def test_server_connection():
    """Test basic server connection"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing server connection...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test root endpoint
            response = await client.get(f"{base_url}/")
            print(f"✅ Server is responding: HTTP {response.status_code}")
            
            # Test docs endpoint
            response = await client.get(f"{base_url}/docs")
            print(f"✅ API docs available: HTTP {response.status_code}")
            
            return True
            
        except httpx.ConnectError:
            print("❌ Cannot connect to server")
            print("💡 Please start the server with:")
            print("   python -m uvicorn app.main:app --reload")
            return False
        except Exception as e:
            print(f"❌ Error testing connection: {e}")
            return False


async def test_sessions_api():
    """Test the sessions API"""
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing sessions API...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test creating a session
            session_data = {
                "title": "Connectivity Test Session",
                "description": "Testing if sessions API works"
            }
            
            response = await client.post(f"{base_url}/api/sessions", json=session_data)
            
            if response.status_code == 200:
                result = response.json()
                session_id = result["session_id"]
                print(f"✅ Created session: {session_id}")
                
                # Test getting sessions
                response = await client.get(f"{base_url}/api/sessions")
                if response.status_code == 200:
                    sessions = response.json()["sessions"]
                    print(f"✅ Retrieved {len(sessions)} sessions")
                    return session_id
                else:
                    print(f"❌ Failed to get sessions: HTTP {response.status_code}")
                    return None
            else:
                print(f"❌ Failed to create session: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing sessions API: {e}")
            return None


async def test_chat_api(session_id):
    """Test the chat API with a simple message"""
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing chat API...")
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            # Test simple chat message
            chat_data = {
                "message": "Hello, can you help me plan a simple day trip to Paris?",
                "session_id": session_id,
                "enable_refinement": False  # Keep it simple for connectivity test
            }
            
            print("📤 Sending chat message...")
            start_time = datetime.now()
            
            response = await client.post(f"{base_url}/api/chat", json=chat_data)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Chat API working!")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
                print(f"   Content length: {len(result.get('content', ''))}")
                
                # Show first 200 characters of response
                content = result.get('content', '')
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Response preview: {preview}")
                
                return True
            else:
                print(f"❌ Chat API failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print("❌ Chat API timeout (>90s) - server may be overloaded")
            print("💡 Normal response time is ~1 minute, but this exceeded the limit")
            return False
        except Exception as e:
            print(f"❌ Error testing chat API: {e}")
            return False


async def main():
    """Main connectivity test function"""
    print("🚀 Chat API Connectivity Test")
    print("=" * 40)
    
    # Test server connection
    if not await test_server_connection():
        print("\n❌ Server connection failed - cannot continue")
        return
    
    # Test sessions API
    session_id = await test_sessions_api()
    if not session_id:
        print("\n❌ Sessions API failed - cannot continue")
        return
    
    # Test chat API
    chat_success = await test_chat_api(session_id)
    
    print("\n" + "=" * 40)
    print("📊 Connectivity Test Summary:")
    print("✅ Server Connection: OK")
    print("✅ Sessions API: OK")
    print(f"{'✅' if chat_success else '❌'} Chat API: {'OK' if chat_success else 'FAILED'}")
    
    if chat_success:
        print("\n🎉 All connectivity tests passed! Your Chat API is working correctly.")
        print("💡 You can now run the full test suite with:")
        print("   python run_tests.py                    # Reliable tests without refinement")
        print("   python run_tests.py --refinement       # Full tests with refinement (slower)")
    else:
        print("\n⚠️  Basic connectivity tests passed but chat API had issues.")
        print("💡 Check the server logs for more details.")


if __name__ == "__main__":
    asyncio.run(main()) 