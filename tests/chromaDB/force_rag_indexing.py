#!/usr/bin/env python3
"""
Force RAG Indexing Script
This script forces the RAG engine to index all knowledge items into ChromaDB
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

async def force_rag_indexing():
    """Force RAG indexing of all knowledge items"""
    print("🚀 Force RAG Indexing Process")
    print("=" * 50)
    
    try:
        # Import required modules
        from app.core.knowledge_base import get_knowledge_base
        from app.core.rag_engine import get_rag_engine, Document, DocumentType
        
        print("1. Getting knowledge base...")
        kb = await get_knowledge_base()
        
        print(f"   📊 Knowledge items loaded: {len(kb.knowledge_items)}")
        
        # Show all knowledge items
        print("   📝 All knowledge items:")
        for i, (item_id, item) in enumerate(kb.knowledge_items.items(), 1):
            print(f"      {i:2d}. {item_id} - {item.title}")
        
        print("\n2. Getting RAG engine...")
        rag_engine = get_rag_engine()
        
        print("3. Converting knowledge items to documents...")
        documents = []
        for knowledge in kb.knowledge_items.values():
            doc = Document(
                id=knowledge.id,
                content=f"{knowledge.title}\n\n{knowledge.content}",
                metadata={
                    "category": knowledge.category,
                    "location": knowledge.location or "",
                    "tags": ",".join(knowledge.tags),
                    "language": knowledge.language,
                    "title": knowledge.title,
                    "parent_id": knowledge.id
                },
                doc_type=DocumentType.TRAVEL_KNOWLEDGE
            )
            documents.append(doc)
        
        print(f"   📄 Created {len(documents)} documents for indexing")
        
        print("4. Indexing documents into RAG engine...")
        success = await rag_engine.index_documents(documents)
        
        if success:
            print("✅ RAG indexing completed successfully!")
        else:
            print("❌ RAG indexing failed!")
            return
        
        print("5. Checking ChromaDB stats...")
        vector_stats = rag_engine.vector_store.get_stats()
        print(f"   📊 ChromaDB documents: {vector_stats.get('total_documents', 0)}")
        
        print("\n🎉 Force RAG indexing completed!")
        print("All knowledge items should now be indexed in ChromaDB.")
        
    except Exception as e:
        print(f"❌ Error during RAG indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_rag_indexing()) 