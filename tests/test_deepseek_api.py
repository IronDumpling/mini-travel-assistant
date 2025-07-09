"""
Minimal DeepSeek API Test
Makes only ONE API call to avoid exceeding credits
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from app.core.llm_service import LLMConfig, LLMServiceFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deepseek_api():
    """
    Test DeepSeek API with single request and minimal tokens
    """
    print("🧪 DeepSeek API Test (Single Request)")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found!")
        print("Please set it in your .env file:")
        print("DEEPSEEK_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # Create configuration with absolute minimal tokens
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
            temperature=0.1,  # Lower temperature for more predictable short responses
            max_tokens=1  # Absolute minimum - just 1 token
        )
        
        service = LLMServiceFactory.create_service(config)
        print(f"✅ DeepSeek service created with model: {service.model}")
        
        # Single test message - absolute minimum tokens
        messages = [
            {"role": "user", "content": "Hi"}  # Shortest possible prompt
        ]
        
        print("🔄 Making ONE API call (absolute minimal tokens)...")
        response = await service.chat_completion(messages)
        
        print("✅ API call successful!")
        print(f"   Response: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Tokens used: {response.usage['total_tokens'] if response.usage else 'unknown'}")
        print(f"   Provider: {response.metadata['provider']}")
        
        # Verify it's a real API call
        if response.metadata.get('api_call', False):
            print("✅ Confirmed: Real API call made successfully!")
        else:
            print("⚠️  Note: This appears to be a mock response")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "Insufficient Balance" in error_msg or "402" in error_msg:
            print(f"❌ {error_msg}")
            print("\n💡 Solutions:")
            print("1. Add credits to your DeepSeek account at https://platform.deepseek.com/")
            print("2. Use a different API provider (OpenAI, Claude)")
            print("3. Test with mock responses for development")
        else:
            print(f"❌ Error: {error_msg}")
        
        return False

def show_credit_tips():
    """Show tips for managing API credits"""
    print("\n💡 Credit Management Tips:")
    print("=" * 30)
    print("1. Use absolute minimal tokens for testing:")
    print("   - Set max_tokens=1 for tests")
    print("   - Use shortest possible prompts ('Hi')")
    print()
    print("2. Monitor your usage:")
    print("   - Check your DeepSeek account balance")
    print("   - Track token usage in responses")
    print()
    print("3. Alternative testing approaches:")
    print("   - Use mock responses for development")
    print("   - Test with free tier providers")
    print("   - Use local models if available")

async def main():
    """Main test function"""
    print("🚀 Single Request DeepSeek Test")
    print("=" * 50)
    
    success = await test_deepseek_api()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("   Your DeepSeek API integration is working.")
        print("   You can now use it in your travel assistant.")
    else:
        print("\n💥 Test failed!")
        show_credit_tips()

if __name__ == "__main__":
    asyncio.run(main()) 