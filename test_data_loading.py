#!/usr/bin/env python3
"""
Test script to debug data loading issues
"""

import asyncio
import sys
from pathlib import Path
import json

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

REQUIRED_FIELDS = ["id", "title", "content", "category"]

def check_json_fields(file_path):
    """Check if a JSON file has all required fields and print warnings if not."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # If the file is a list, check each item
        if isinstance(data, list):
            for i, item in enumerate(data):
                missing = [field for field in REQUIRED_FIELDS if field not in item or not item[field]]
                if missing:
                    print(f"⚠️  {file_path} [item {i}] missing fields: {missing}")
        # If the file is a dict, check directly
        elif isinstance(data, dict):
            # If it has 'knowledge_items', check each
            if 'knowledge_items' in data and isinstance(data['knowledge_items'], list):
                for i, item in enumerate(data['knowledge_items']):
                    missing = [field for field in REQUIRED_FIELDS if field not in item or not item[field]]
                    if missing:
                        print(f"⚠️  {file_path} [knowledge_items item {i}] missing fields: {missing}")
            else:
                missing = [field for field in REQUIRED_FIELDS if field not in data or not data[field]]
                if missing:
                    print(f"⚠️  {file_path} missing fields: {missing}")
        else:
            print(f"⚠️  {file_path} has unexpected JSON structure: {type(data)}")
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

async def test_data_loading():
    """Test the data loading process"""
    print("🔍 Testing Data Loading Process")
    print("=" * 50)
    
    # Test 1: Check if files exist and validate fields
    print("\n1. Checking file existence and required fields...")
    documents_dir = Path("app/knowledge/documents")
    
    if not documents_dir.exists():
        print(f"❌ Documents directory not found: {documents_dir}")
        return
    
    print(f"✅ Documents directory found: {documents_dir}")
    
    # Count files in each subdirectory
    total_files = 0
    for file_path in documents_dir.rglob("*.json"):
        total_files += 1
        print(f"   📄 Found: {file_path}")
        check_json_fields(file_path)
    
    print(f"\n📊 Total JSON files found: {total_files}")
    
    # Test 2: Test data loader
    print("\n2. Testing data loader...")
    try:
        from app.core.data_loader import TravelDataLoader
        
        loader = TravelDataLoader()
        knowledge_items = await loader.load_all_data()
        
        print(f"✅ Data loader loaded {len(knowledge_items)} knowledge items")
        
        # Show all loaded items
        for i, item in enumerate(knowledge_items):
            print(f"   {i+1:2d}. {item.id} - {item.title}")
        
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test knowledge base initialization
    print("\n3. Testing knowledge base initialization...")
    try:
        from app.core.knowledge_base import get_knowledge_base
        
        kb = await get_knowledge_base()
        stats = kb.get_knowledge_stats()
        
        print(f"✅ Knowledge base stats:")
        print(f"   Total items: {stats['total_knowledge_items']}")
        print(f"   Items by category: {stats['items_by_category']}")
        print(f"   Vector store stats: {stats['vector_store_stats']}")
        
    except Exception as e:
        print(f"❌ Knowledge base error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check specific file loading
    print("\n4. Testing specific file loading...")
    test_files = [
        "app/knowledge/documents/destinations/asia/tokyo_attractions.json",
        "app/knowledge/documents/destinations/asia/kyoto_attractions.json",
        "app/knowledge/documents/practical/visa/japan_visa.json",
        "app/knowledge/documents/transportation/metro/japan_metro.json"
    ]
    
    for test_file in test_files:
        file_path = Path(test_file)
        if file_path.exists():
            print(f"✅ File exists: {test_file}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   📄 JSON valid, keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
            except Exception as e:
                print(f"   ❌ JSON error: {e}")
        else:
            print(f"❌ File missing: {test_file}")

if __name__ == "__main__":
    asyncio.run(test_data_loading()) 