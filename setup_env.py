#!/usr/bin/env python3
"""
Environment Setup Script for AI Travel Planning Agent

This script helps you set up the required environment variables for the DeepSeek API.
"""

import os

def setup_environment():
    """Set up environment variables for the application"""
    
    print("🔧 Setting up environment variables for AI Travel Planning Agent")
    print("=" * 60)
    
    # Your DeepSeek API key
    deepseek_api_key = "sk-76bbf9b4105240459a733da25d415f47"
    
    # Set environment variables (DeepSeek is now the default)
    os.environ["LLM_PROVIDER"] = "deepseek"  # This is now the default
    os.environ["LLM_MODEL"] = "deepseek-chat"
    os.environ["LLM_API_KEY"] = deepseek_api_key
    os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
    os.environ["LLM_TEMPERATURE"] = "0.7"
    os.environ["LLM_MAX_TOKENS"] = "4000"
    
    # Set other required environment variables
    os.environ["CHROMA_DB_PATH"] = "./data/chroma_db"
    os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["RAG_TOP_K"] = "5"
    os.environ["RAG_SIMILARITY_THRESHOLD"] = "0.7"
    os.environ["DATABASE_URL"] = "sqlite:///./data/travel_agent.db"
    os.environ["HOST"] = "0.0.0.0"
    os.environ["PORT"] = "8000"
    os.environ["DEBUG"] = "True"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["MOCK_MODE"] = "false"
    
    print("✅ Environment variables set successfully!")
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
    print("📝 Note: These environment variables are set for this session only.")
    print("   For permanent setup, create a .env file in the project root with:")
    print("   LLM_PROVIDER=deepseek")
    print("   LLM_MODEL=deepseek-chat")
    print("   LLM_API_KEY=sk-76bbf9b4105240459a733da25d415f47")
    print("   DEEPSEEK_API_KEY=sk-76bbf9b4105240459a733da25d415f47")

if __name__ == "__main__":
    setup_environment() 