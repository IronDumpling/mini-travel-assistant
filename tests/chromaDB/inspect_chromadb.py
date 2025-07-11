#!/usr/bin/env python3
"""
ChromaDB Inspection Utility
A simple script to inspect what's stored in ChromaDB
"""

import chromadb
from pathlib import Path
import json
from typing import Dict, Any, List
import sys
import os

def inspect_chromadb(db_path: str = "./data/chroma_db"):
    """Inspect ChromaDB contents"""
    
    if not os.path.exists(db_path):
        print(f"❌ ChromaDB path not found: {db_path}")
        return
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Get all collections
        collections = client.list_collections()
        
        print(f"🔍 ChromaDB Inspection Report")
        print(f"📁 Database Path: {db_path}")
        print(f"📊 Total Collections: {len(collections)}")
        
        # Calculate total documents across all collections
        total_docs = sum(collection.count() for collection in collections)
        print(f"📄 Total Documents: {total_docs}")
        print("=" * 60)
        
        if not collections:
            print("❌ No collections found in ChromaDB")
            return
        
        for collection in collections:
            print(f"\n📚 Collection: {collection.name}")
            print(f"   📄 Document Count: {collection.count()}")
            
            # Get collection metadata
            try:
                metadata = collection.metadata
                if metadata:
                    print(f"   🔧 Metadata: {metadata}")
            except Exception as e:
                print(f"   ⚠️  Could not retrieve metadata: {e}")
            
            # Get all documents
            try:
                if collection.count() > 0:
                    # Get all documents
                    results = collection.get(
                        limit=collection.count(),
                        include=['documents', 'metadatas', 'embeddings']
                    )
                    
                    print(f"   📝 All Documents ({collection.count()}):")
                    if results['ids'] and results['documents'] and results['metadatas']:
                        for i, (doc_id, doc_content, metadata) in enumerate(zip(
                            results['ids'],
                            results['documents'],
                            results['metadatas']
                        )):
                            print(f"      {i+1:2d}. ID: {doc_id}")
                            # Print chunk title and parent guide if available
                            title = metadata.get('title', '[No title]') if metadata else '[No title]'
                            parent = metadata.get('parent_id', None) if metadata else None
                            if parent:
                                print(f"           Title: {title}")
                                print(f"           Parent Guide: {parent}")
                            else:
                                print(f"           Title: {title}")
                            if metadata and 'category' in metadata:
                                print(f"           Category: {metadata['category']}")
                            if metadata and 'location' in metadata:
                                print(f"           Location: {metadata['location']}")
                            print(f"           Content Preview: {doc_content[:80]}...")
                            print()
                        
                        # Show document types if available
                        if results['metadatas']:
                            doc_types = {}
                            for meta in results['metadatas']:
                                if meta and 'doc_type' in meta:
                                    doc_type = meta['doc_type']
                                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                            
                            if doc_types:
                                print(f"   📋 Document Types Distribution:")
                                for doc_type, count in doc_types.items():
                                    print(f"      - {doc_type}: {count}")
                
            except Exception as e:
                print(f"   ⚠️  Could not retrieve sample documents: {e}")
            
            print("-" * 40)
        
        # Show overall statistics
        print(f"\n📈 Overall Statistics:")
        total_docs = sum(collection.count() for collection in collections)
        print(f"   Total Documents: {total_docs}")
        
        # Show document types across all collections
        all_doc_types = {}
        for collection in collections:
            try:
                if collection.count() > 0:
                    results = collection.get(
                        limit=collection.count(),
                        include=['metadatas']
                    )
                    
                    if results and results['metadatas']:
                        for metadata in results['metadatas']:
                            if metadata and 'doc_type' in metadata:
                                doc_type = metadata['doc_type']
                                all_doc_types[doc_type] = all_doc_types.get(doc_type, 0) + 1
                            
            except Exception as e:
                print(f"   ⚠️  Could not analyze collection {collection.name}: {e}")
        
        if all_doc_types:
            print(f"   Document Types Distribution:")
            for doc_type, count in sorted(all_doc_types.items()):
                percentage = (count / total_docs) * 100
                print(f"      - {doc_type}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error inspecting ChromaDB: {e}")

def search_chromadb(query: str, db_path: str = "./data/chroma_db", collection_name: str = "travel_knowledge", top_k: int = 5):
    """Search ChromaDB with a query"""
    
    if not os.path.exists(db_path):
        print(f"❌ ChromaDB path not found: {db_path}")
        return
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            print(f"❌ Collection '{collection_name}' not found: {e}")
            return
        
        print(f"🔍 Searching ChromaDB for: '{query}'")
        print(f"📚 Collection: {collection_name}")
        print(f"📊 Total Documents: {collection.count()}")
        print("=" * 60)
        
        # Perform text search
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("❌ No results found")
            return
        
        print(f"📝 Found {len(results['documents'][0])} results:")
        print()
        
        if results['documents'] and results['documents'][0]:
            for i, (doc_content, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'] else [],
                results['distances'][0] if results['distances'] else []
            )):
                similarity = 1 - distance
                print(f"{i+1}. Similarity: {similarity:.3f}")
                print(f"   Content: {doc_content[:200]}...")
                if metadata:
                    print(f"   Metadata: {metadata}")
                print()
        
    except Exception as e:
        print(f"❌ Error searching ChromaDB: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_chromadb.py inspect [db_path]")
        print("  python inspect_chromadb.py search <query> [collection_name] [top_k]")
        print()
        print("Examples:")
        print("  python inspect_chromadb.py inspect")
        print("  python inspect_chromadb.py inspect ./data/chroma_db")
        print("  python inspect_chromadb.py search 'Tokyo attractions'")
        print("  python inspect_chromadb.py search 'visa requirements' travel_knowledge 10")
        return
    
    command = sys.argv[1].lower()
    
    if command == "inspect":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "./data/chroma_db"
        inspect_chromadb(db_path)
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("❌ Please provide a search query")
            return
        
        query = sys.argv[2]
        collection_name = sys.argv[3] if len(sys.argv) > 3 else "travel_knowledge"
        top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        
        search_chromadb(query, collection_name=collection_name, top_k=top_k)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: inspect, search")

if __name__ == "__main__":
    main() 