#!/usr/bin/env python3
"""
Environment Setup Script for AI Travel Planning Agent

This script helps you set up the required environment variables for the DeepSeek API.
It only appends DeepSeek-specific parameters to the .env file, without modifying or removing any existing lines.
"""

import os

def setup_environment():
    """Set up environment variables for the application"""
    
    print("🔧 Setting up environment variables for AI Travel Planning Agent")
    print("=" * 60)
    
    # Your DeepSeek API key
    deepseek_api_key = "sk-d6f66ddb3a174cb3b57367e97207e1fe"
    
    # Define DeepSeek-specific environment variables
    deepseek_vars = {
        "LLM_PROVIDER": "deepseek",
        "LLM_MODEL": "deepseek-chat", 
        "LLM_API_KEY": deepseek_api_key,
        "DEEPSEEK_API_KEY": deepseek_api_key,
        "LLM_TEMPERATURE": "0.7",
        "LLM_MAX_TOKENS": "4000"
    }
    
    # Set environment variables for current session
    for key, value in deepseek_vars.items():
        os.environ[key] = value
    
    # Always append DeepSeek params to the .env file
    env_file_path = ".env"
    with open(env_file_path, 'a', encoding='utf-8') as f:
        f.write("\n# DeepSeek API Configuration\n")
        for key, value in deepseek_vars.items():
            f.write(f"{key}={value}\n")
    print(f"✅ Appended DeepSeek variables to .env file.")
    
    print()
    print("📋 Configuration Summary:")
    print(f"   LLM Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"   LLM Model: {os.environ.get('LLM_MODEL')}")
    print(f"   API Key: {os.environ.get('LLM_API_KEY')[:20]}...")
    print(f"   Temperature: {os.environ.get('LLM_TEMPERATURE')}")
    print(f"   Max Tokens: {os.environ.get('LLM_MAX_TOKENS')}")
    print()
    print("🚀 You can now run the application!")
    print("   Command: python -m uvicorn app.main:app --reload")
    print()
    print("📝 Note: Environment variables are set for this session and appended to .env file.")

if __name__ == "__main__":
    setup_environment() 