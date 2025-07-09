#!/usr/bin/env python3
"""
Simple API Key Setup
Helps you set up your DeepSeek API key
"""

import os
from pathlib import Path

def setup_api_key():
    """Set up API key interactively"""
    print("🔑 DeepSeek API Key Setup")
    print("=" * 30)
    
    # Get new API key from user
    api_key = input("Enter your new DeepSeek API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return False
    
    # Create .env content
    env_content = f"""# DeepSeek API Configuration
DEEPSEEK_API_KEY={api_key}
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.7
"""
    
    # Write to .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created/updated successfully!")
        print(f"   API key: {api_key[:8]}...{api_key[-4:]}")
        print("   File location: .env")
        print("   ⚠️  Remember: .env is in .gitignore, so it won't be committed to git")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def main():
    """Main function"""
    success = setup_api_key()
    
    if success:
        print("\n🎉 Setup completed!")
        print("You can now run the test:")
        print("python -m tests.test_deepseek_api")
    else:
        print("\n💥 Setup failed!")
        print("Please try again or set the API key manually.")

if __name__ == "__main__":
    main() 