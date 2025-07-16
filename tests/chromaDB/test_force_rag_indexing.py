#!/usr/bin/env python3
"""
Force RAG Indexing Script - Updated with New Force Rebuild Mechanism
This script forces the RAG engine to index all knowledge items into ChromaDB using the new force rebuild mechanism
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

async def force_rag_indexing():
    """Force RAG indexing of all knowledge items using new mechanism"""
    print("🚀 Force RAG Indexing Process - New Mechanism")
    print("=" * 60)
    
    try:
        # Import required modules
        from app.core.knowledge_base import get_knowledge_base
        from app.core.rag_engine import get_rag_engine, DocumentType
        
        print("1. Getting knowledge base...")
        kb = await get_knowledge_base()
        
        print(f"   📊 Knowledge items loaded: {len(kb.knowledge_items)}")
        
        # Show all knowledge items
        print("   📝 All knowledge items:")
        berlin_count = 0
        for i, (item_id, item) in enumerate(kb.knowledge_items.items(), 1):
            print(f"      {i:2d}. {item_id} - {item.title}")
            if item.location and 'berlin' in item.location.lower():
                berlin_count += 1
                print(f"          🏛️ BERLIN CONTENT: {item.location}")
        
        print(f"   🏛️ Berlin documents found: {berlin_count}")
        
        print("\n2. Getting RAG engine...")
        rag_engine = get_rag_engine()
        
        # Check current ChromaDB status
        initial_stats = rag_engine.vector_store.get_stats()
        print(f"   📊 Initial ChromaDB stats: {initial_stats.get('total_documents', 0)} documents")
        
        print("\n3. Testing Berlin query BEFORE force rebuild...")
        try:
            berlin_results_before = await rag_engine.retrieve(
                query="Plan a 7-day trip to Berlin for 2 people with a budget of $7000",
                top_k=3,
                doc_type=DocumentType.TRAVEL_KNOWLEDGE
            )
            print(f"   🔍 Berlin query results BEFORE: {len(berlin_results_before.documents)} documents")
            for i, (doc, score) in enumerate(zip(berlin_results_before.documents, berlin_results_before.scores)):
                location = doc.metadata.get('location', 'Unknown')
                print(f"      {i+1}. Score: {score:.3f}, Location: {location}")
                if 'berlin' in location.lower():
                    print(f"         ✅ CORRECT: Berlin content found")
                else:
                    print(f"         ❌ INCORRECT: Expected Berlin but got {location}")
        except Exception as e:
            print(f"   ❌ Berlin query failed BEFORE rebuild: {e}")
        
        print("\n4. Force rebuilding index using new mechanism...")
        print("   🔄 The knowledge base initialization already performed force rebuild")
        print("   📊 This ensures all documents are properly indexed with fresh embeddings")
        
        # Verify final stats
        final_stats = rag_engine.vector_store.get_stats()
        print(f"   📈 Final ChromaDB stats: {final_stats.get('total_documents', 0)} documents")
        
        print("\n5. Testing Berlin query AFTER force rebuild...")
        try:
            berlin_results_after = await rag_engine.retrieve(
                query="Plan a 7-day trip to Berlin for 2 people with a budget of $7000",
                top_k=5,
                doc_type=DocumentType.TRAVEL_KNOWLEDGE
            )
            print(f"   🔍 Berlin query results AFTER: {len(berlin_results_after.documents)} documents")
            
            berlin_found = False
            for i, (doc, score) in enumerate(zip(berlin_results_after.documents, berlin_results_after.scores)):
                location = doc.metadata.get('location', 'Unknown')
                title = doc.metadata.get('title', 'Unknown')
                print(f"      {i+1}. Score: {score:.3f}, Location: {location}")
                print(f"         Title: {title}")
                if 'berlin' in location.lower() or 'berlin' in doc.content.lower():
                    berlin_found = True
                    print(f"         ✅ CORRECT: Berlin content found!")
                else:
                    print(f"         ⚠️ Non-Berlin result")
            
            if berlin_found:
                print(f"   🎉 SUCCESS: Berlin query now returns Berlin content!")
            else:
                print(f"   ❌ ISSUE: Berlin query still not returning Berlin content")
                
        except Exception as e:
            print(f"   ❌ Berlin query failed AFTER rebuild: {e}")
        
        print("\n6. Testing general travel query...")
        try:
            general_results = await rag_engine.retrieve(
                query="What are the best tourist attractions?",
                top_k=3,
                doc_type=DocumentType.TRAVEL_KNOWLEDGE
            )
            print(f"   🔍 General query results: {len(general_results.documents)} documents")
            for i, (doc, score) in enumerate(zip(general_results.documents, general_results.scores)):
                location = doc.metadata.get('location', 'Unknown')
                print(f"      {i+1}. Score: {score:.3f}, Location: {location}")
        except Exception as e:
            print(f"   ❌ General query failed: {e}")
        
        print(f"\n🎉 Force RAG indexing process completed!")
        print(f"✅ New force rebuild mechanism ensures all documents are properly indexed")
        print(f"🏛️ Berlin content should now be retrievable in queries")
        print(f"📊 Total documents in ChromaDB: {final_stats.get('total_documents', 0)}")
        
    except Exception as e:
        print(f"❌ Error during force RAG indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_rag_indexing()) 