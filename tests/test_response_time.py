"""
Chat API Request Tests

Test cases:
1. Sequential messages in the same session
2. Varied length requests in separate sessions
"""

import requests
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatAPITester:
    """Test class for chat API requests"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"

    def send_chat_message(
        self,
        message: str,
        session_id: str | None = None,
        enable_refinement: bool = True,
    ) -> Dict[str, Any]:
        """Send a single chat message and return the response"""

        payload = {"message": message, "enable_refinement": enable_refinement}

        if session_id:
            payload["session_id"] = session_id

        logger.info(f"Sending message: {message[:50]}...")
        if session_id:
            logger.info(f"Session ID: {session_id}")

        # Start timing
        start_time = time.time()

        try:
            response = requests.post(self.chat_endpoint, json=payload, timeout=300)

            # Calculate response time
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                response_content = data.get("content", "")
                response_length = len(response_content)

                logger.info(
                    f"✅ Success - Status: {data.get('success')}, Confidence: {data.get('confidence')}"
                )
                logger.info(f"⏱️ Response time: {response_time:.2f} seconds")
                logger.info(f"📏 Response length: {response_length} characters")
                logger.info(f"Response: {response_content[:100]}...")

                # Add timing and length info to response
                data["response_time"] = response_time
                data["response_length"] = response_length
                data["message_length"] = len(message)

                return data
            else:
                logger.error(f"❌ HTTP Error {response.status_code}: {response.text}")
                logger.info(f"⏱️ Time taken: {response_time:.2f} seconds")
                return {
                    "error": f"HTTP {response.status_code}",
                    "details": response.text,
                    "response_time": response_time,
                }

        except requests.exceptions.ConnectionError:
            response_time = time.time() - start_time
            logger.error("❌ Connection Error: Make sure the server is running")
            logger.info(f"⏱️ Time taken: {response_time:.2f} seconds")
            return {"error": "Connection failed", "response_time": response_time}
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            logger.error("❌ Timeout: Request took too long")
            logger.info(f"⏱️ Time taken: {response_time:.2f} seconds")
            return {"error": "Request timeout", "response_time": response_time}
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"❌ Unexpected error: {e}")
            logger.info(f"⏱️ Time taken: {response_time:.2f} seconds")
            return {"error": str(e), "response_time": response_time}

    def test_sequential_messages_same_session(self):
        """Test Case 1: Sequential messages in the same session"""

        logger.info("=" * 60)
        logger.info("TEST CASE 1: Sequential Messages in Same Session")
        logger.info("=" * 60)

        # First message - establish session
        logger.info("📝 Message 1: Initial request")
        response1 = self.send_chat_message(
            "Hello! I'm planning a trip to Japan next month. Can you help me?",
            enable_refinement=True,
        )

        if "error" in response1:
            logger.error("❌ Test failed - couldn't establish session")
            return

        session_id = response1.get("session_id")
        logger.info(f"📋 Session established: {session_id}")

        # Wait a moment between requests
        time.sleep(1)

        # Second message - follow up in same session
        logger.info("📝 Message 2: Follow-up question")
        response2 = self.send_chat_message(
            "What are the best cities to visit in Japan for a first-time traveler?",
            session_id=session_id,
            enable_refinement=True,
        )

        if "error" in response2:
            logger.error("❌ Test failed - follow-up message failed")
            return

        # Wait a moment between requests
        time.sleep(1)

        # Third message - specific request in same session
        logger.info("📝 Message 3: Specific request")
        response3 = self.send_chat_message(
            "I'm particularly interested in Tokyo and Kyoto. What should I know about transportation between these cities?",
            session_id=session_id,
            enable_refinement=False,  # Test without refinement
        )

        if "error" in response3:
            logger.error("❌ Test failed - specific request failed")
            return

        # Fourth message - final question in same session
        logger.info("📝 Message 4: Final question")
        response4 = self.send_chat_message(
            "What's the best time of year to visit Japan?",
            session_id=session_id,
            enable_refinement=True,
        )

        if "error" in response4:
            logger.error("❌ Test failed - final message failed")
            return

        logger.info("✅ Test Case 1 completed successfully!")
        logger.info(f"Session maintained across {4} messages")
        logger.info(f"Final session ID: {session_id}")

        # Calculate timing statistics
        responses = [response1, response2, response3, response4]
        successful_responses = [r for r in responses if "error" not in r]

        if successful_responses:
            total_time = sum(r.get("response_time", 0) for r in successful_responses)
            avg_time = total_time / len(successful_responses)
            total_response_length = sum(
                r.get("response_length", 0) for r in successful_responses
            )
            avg_response_length = total_response_length / len(successful_responses)

            logger.info("📊 Test Case 1 Statistics:")
            logger.info(f"   Total response time: {total_time:.2f} seconds")
            logger.info(f"   Average response time: {avg_time:.2f} seconds")
            logger.info(f"   Total response length: {total_response_length} characters")
            logger.info(
                f"   Average response length: {avg_response_length:.0f} characters"
            )
            logger.info(
                f"   Messages per second: {len(successful_responses) / total_time:.2f}"
            )

    def test_varied_length_separate_sessions(self):
        """Test Case 2: Varied length requests in separate sessions"""

        logger.info("=" * 60)
        logger.info("TEST CASE 2: Varied Length Requests in Separate Sessions")
        logger.info("=" * 60)

        # Test cases with different message lengths
        test_messages = [
            {"message": "Hi", "description": "Very short message"},
            {"message": "I want to travel to Europe", "description": "Short message"},
            {
                "message": "I'm planning a 3-week backpacking trip through Europe starting in Paris, then going to Amsterdam, Berlin, Prague, Vienna, Budapest, and ending in Rome. I'm interested in art, history, and local food. My budget is around $3000 and I prefer hostels. What's the best route and what should I know about transportation between these cities?",
                "description": "Long detailed message",
            },
            {"message": "Tell me about Paris", "description": "Medium message"},
            {
                "message": "I need help with my travel itinerary for a family vacation to Disney World in Orlando, Florida. We have 2 adults and 3 children (ages 8, 10, and 12). We're planning to stay for 5 days and want to visit all 4 parks. We're staying at a Disney resort hotel. What's the best way to plan our days? Should we get the park hopper pass? What restaurants should we book in advance? And what are some tips for managing the crowds and making the most of our time?",
                "description": "Very long complex message",
            },
            {"message": "Thanks!", "description": "Very short closing message"},
        ]

        session_responses = []

        for i, test_case in enumerate(test_messages, 1):
            logger.info(f"📝 Message {i}: {test_case['description']}")
            logger.info(f"Length: {len(test_case['message'])} characters")

            # Send message in new session (no session_id)
            response = self.send_chat_message(
                test_case["message"], enable_refinement=True
            )

            if "error" in response:
                logger.error(f"❌ Message {i} failed: {response['error']}")
                continue

            session_id = response.get("session_id")
            session_responses.append(
                {
                    "message_number": i,
                    "description": test_case["description"],
                    "message_length": len(test_case["message"]),
                    "session_id": session_id,
                    "success": response.get("success"),
                    "confidence": response.get("confidence"),
                    "response_length": len(response.get("content", "")),
                    "response_time": response.get("response_time", 0),
                }
            )

            logger.info(f"✅ Message {i} completed - Session: {session_id}")
            logger.info(
                f"Response length: {len(response.get('content', ''))} characters"
            )

            # Wait between requests
            time.sleep(1)

        # Summary
        logger.info("=" * 60)
        logger.info("TEST CASE 2 SUMMARY")
        logger.info("=" * 60)

        successful_messages = [r for r in session_responses if r["success"]]
        failed_messages = [r for r in session_responses if not r["success"]]

        logger.info(f"Total messages sent: {len(test_messages)}")
        logger.info(f"Successful messages: {len(successful_messages)}")
        logger.info(f"Failed messages: {len(failed_messages)}")

        if successful_messages:
            avg_confidence = sum(r["confidence"] for r in successful_messages) / len(
                successful_messages
            )
            avg_response_length = sum(
                r["response_length"] for r in successful_messages
            ) / len(successful_messages)
            total_response_time = sum(
                r.get("response_time", 0) for r in successful_messages
            )
            avg_response_time = total_response_time / len(successful_messages)
            total_response_chars = sum(
                r["response_length"] for r in successful_messages
            )

            logger.info(f"Average confidence: {avg_confidence:.2f}")
            logger.info(
                f"Average response length: {avg_response_length:.0f} characters"
            )
            logger.info(f"Total response time: {total_response_time:.2f} seconds")
            logger.info(f"Average response time: {avg_response_time:.2f} seconds")
            logger.info(f"Total response characters: {total_response_chars}")
            if total_response_time > 0:
                logger.info(
                    f"Characters per second: {total_response_chars / total_response_time:.0f}"
                )
            else:
                logger.info("Characters per second: N/A (no response time recorded)")

        # Show unique session IDs (should be different for each message)
        unique_sessions = set(
            r["session_id"] for r in session_responses if r["session_id"]
        )
        logger.info(f"Unique sessions created: {len(unique_sessions)}")

        if len(unique_sessions) == len(test_messages):
            logger.info("✅ All messages created separate sessions as expected")
        else:
            logger.warning(
                f"⚠️ Expected {len(test_messages)} unique sessions, got {len(unique_sessions)}"
            )

    def run_all_tests(self):
        """Run both test cases"""

        logger.info("🚀 Starting Chat API Request Tests")
        logger.info(f"Target URL: {self.chat_endpoint}")
        logger.info("=" * 60)

        # Test Case 1: Sequential messages in same session
        self.test_sequential_messages_same_session()

        logger.info("\n" + "=" * 60)

        # Test Case 2: Varied length requests in separate sessions
        self.test_varied_length_separate_sessions()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 All tests completed!")

        # Final overall summary
        logger.info("\n" + "=" * 60)
        logger.info("📈 OVERALL TEST SUMMARY")
        logger.info("=" * 60)


def main():
    """Main function to run the tests"""

    # Create tester instance
    tester = ChatAPITester()

    # Run all tests
    tester.run_all_tests()


if __name__ == "__main__":
    main()
