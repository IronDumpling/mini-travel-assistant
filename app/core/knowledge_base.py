"""
Knowledge Base Module - Intelligent Travel Knowledge Management

Handles initialization, loading, and management of travel knowledge base with
smart data loading and RAG-enhanced search capabilities.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from datetime import datetime
from pydantic import BaseModel

from app.core.rag_engine import Document, DocumentType, get_rag_engine
from app.core.data_loader import TravelDataLoader
from app.core.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class KnowledgeCategory(BaseModel):
    """Knowledge category structure"""
    id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class KnowledgeSource(BaseModel):
    """Knowledge source information"""
    id: str
    name: str
    url: Optional[str] = None
    last_updated: Optional[str] = None
    reliability_score: float = 1.0
    language: str = "en"


class TravelKnowledge(BaseModel):
    """Travel knowledge data structure"""
    id: str
    title: str
    content: str
    category: str
    location: Optional[str] = None
    tags: List[str] = []
    source: Optional[KnowledgeSource] = None
    language: str = "en"
    last_updated: Optional[str] = None


class KnowledgeBase:
    """Intelligent travel knowledge base manager"""
    
    def __init__(self, knowledge_dir: str = "app/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.categories: Dict[str, KnowledgeCategory] = {}
        self.knowledge_items: Dict[str, TravelKnowledge] = {}
        self.rag_engine = get_rag_engine()
        self.data_loader = TravelDataLoader(knowledge_dir)
        self._initialized = False
        self.has_document_changes = False  # 🔧 Track document changes for indexing decisions
        
        logger.info(f"Knowledge base initialized for directory: {self.knowledge_dir}")
    
    async def initialize(self):
        """Initialize knowledge base"""
        if self._initialized:
            return
            
        logger.info("Starting knowledge base initialization...")
        
        try:
            # 1. Load category configuration
            await self._load_categories()
            
            # 2. Load knowledge data
            await self._load_knowledge_data()
            
            # 3. 🔧 FORCE REBUILD RAG INDEX - 强制重新构建索引
            # 移除原来的条件检查，直接强制重建索引以确保数据一致性
            logger.info("🔄 FORCE REBUILDING RAG INDEX - 强制重新构建索引...")
            logger.info("  - This ensures all documents are properly indexed")
            logger.info("  - Previous index will be updated/replaced")
            await self._build_index(force_rebuild=True)
            
            self._initialized = True
            logger.info(f"Knowledge base ready with {len(self.knowledge_items)} items")
            
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            raise
    
    async def _load_categories(self):
        """Load knowledge categories from configuration"""
        try:
            categories_file = self.knowledge_dir / "categories.yaml"
            if categories_file.exists():
                with open(categories_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if 'categories' in config:
                    for cat_data in config['categories']:
                        category = KnowledgeCategory(**cat_data)
                        self.categories[category.id] = category
                        
                        # Load subcategories if present
                        if 'subcategories' in cat_data:
                            for subcat_data in cat_data['subcategories']:
                                subcat_data['parent_id'] = category.id
                                subcategory = KnowledgeCategory(**subcat_data)
                                self.categories[subcategory.id] = subcategory
                                
                    logger.info(f"Loaded {len(self.categories)} categories from configuration")
                else:
                    logger.warning("No categories found in configuration file")
                    await self._create_default_categories()
            else:
                logger.info("Creating default categories...")
                await self._create_default_categories()
                
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            await self._create_default_categories()
    
    async def _create_default_categories(self):
        """Create default knowledge categories"""
        default_categories = [
            {
                "id": "destinations",
                "name": "Destination Information",
                "description": "Travel destinations, attractions, and cultural information"
            },
            {
                "id": "transportation",
                "name": "Transportation",
                "description": "Flights, trains, buses, and other transportation options"
            },
            {
                "id": "accommodation",
                "name": "Accommodation",
                "description": "Hotels, hostels, and other lodging options"
            },
            {
                "id": "activities",
                "name": "Activities & Experiences",
                "description": "Tourist activities, tours, and experiences"
            },
            {
                "id": "practical",
                "name": "Practical Information",
                "description": "Visas, currency, weather, safety, and other practical info"
            }
        ]
        
        for cat_data in default_categories:
            category = KnowledgeCategory(**cat_data)
            self.categories[category.id] = category
            
        logger.info(f"Created {len(self.categories)} default categories")
    
    async def _load_knowledge_data(self):
        """Load knowledge data from files"""
        try:
            # 🔧 DEBUG: Log knowledge loading start
            logger.info(f"📚 STARTING KNOWLEDGE DATA LOADING...")
            logger.info(f"  - Data loader: {self.data_loader}")
            logger.info(f"  - Knowledge directory: {self.knowledge_dir}")
            
            # 🔧 检测文档变化并决定是否需要强制重建
            changes, new_version_info = self.data_loader.detect_changes()
            self.has_document_changes = changes["has_changes"]
            
            if self.has_document_changes:
                logger.info(f"📋 DOCUMENT CHANGES DETECTED - 将强制重建索引")
                logger.info(f"  - Added files: {len(changes['files_added'])}")
                logger.info(f"  - Modified files: {len(changes['files_modified'])}")
                logger.info(f"  - Deleted files: {len(changes['files_deleted'])}")
            else:
                logger.info(f"✅ NO DOCUMENT CHANGES - 但仍将执行强制重建以确保一致性")
            
            # Load from files using data loader
            knowledge_items = await self.data_loader.load_all_data()
            
            # 🔧 DEBUG: Log loaded items summary
            logger.info(f"📊 KNOWLEDGE LOADING RESULTS:")
            logger.info(f"  - Total items loaded: {len(knowledge_items)}")
            
            # Process loaded knowledge
            self.knowledge_items.clear()
            processed_count = 0
            berlin_count = 0
            
            for knowledge in knowledge_items:
                self.knowledge_items[knowledge.id] = knowledge
                processed_count += 1
                
                # 🔧 DEBUG: Log each knowledge item
                logger.debug(f"📖 PROCESSED KNOWLEDGE {processed_count}: {knowledge.id}")
                logger.debug(f"  - Title: {knowledge.title}")
                logger.debug(f"  - Location: {knowledge.location}")
                logger.debug(f"  - Category: {knowledge.category}")
                logger.debug(f"  - Content length: {len(knowledge.content)}")
                logger.debug(f"  - Tags: {knowledge.tags}")
                
                # 🔧 DEBUG: Track Berlin content specifically
                if (knowledge.location and 'berlin' in knowledge.location.lower()) or 'berlin' in knowledge.title.lower():
                    berlin_count += 1
                    logger.info(f"🏛️ BERLIN KNOWLEDGE LOADED: {knowledge.id}")
                    logger.info(f"  - Title: {knowledge.title}")
                    logger.info(f"  - Location: {knowledge.location}")
                    logger.info(f"  - Content preview: {knowledge.content[:200]}...")
            
            # 🔧 DEBUG: Log final knowledge stats
            logger.info(f"✅ KNOWLEDGE PROCESSING COMPLETE:")
            logger.info(f"  - Total processed: {processed_count}")
            logger.info(f"  - Berlin items found: {berlin_count}")
            logger.info(f"  - Items in memory: {len(self.knowledge_items)}")
            logger.info(f"  - Has document changes: {self.has_document_changes}")
            
            # Log appropriate message based on what was loaded
            if self.knowledge_items:
                logger.info(f"Loaded {len(self.knowledge_items)} knowledge items from files")
                
                # 🔧 DEBUG: Log knowledge by category
                category_stats = {}
                location_stats = {}
                for knowledge in self.knowledge_items.values():
                    category_stats[knowledge.category] = category_stats.get(knowledge.category, 0) + 1
                    if knowledge.location:
                        location_stats[knowledge.location] = location_stats.get(knowledge.location, 0) + 1
                
                logger.info(f"📊 KNOWLEDGE BREAKDOWN:")
                logger.info(f"  - By category: {dict(sorted(category_stats.items()))}")
                logger.info(f"  - By location: {dict(sorted(location_stats.items()))}")
                
                if 'Berlin' not in location_stats:
                    logger.warning(f"⚠️ NO BERLIN KNOWLEDGE FOUND IN PROCESSED ITEMS!")
                
            else:
                logger.warning("No knowledge files found in the documents directory")
                logger.info("Please ensure knowledge files are present in app/knowledge/documents/")
            
        except Exception as e:
            logger.error(f"❌ FAILED to load knowledge data: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            # Don't fall back to hardcoded data - let the system handle empty knowledge gracefully
            logger.warning("Knowledge base will operate with empty knowledge set")
            self.has_document_changes = False  # 加载失败时设为False
    
    async def _build_index(self, force_rebuild: bool = False):
        """Build vector index for all knowledge items"""
        try:
            # 🔧 DEBUG: Log indexing start
            logger.info(f"🔧 STARTING KNOWLEDGE INDEX BUILDING...")
            logger.info(f"  - Knowledge items to index: {len(self.knowledge_items)}")
            logger.info(f"  - Force rebuild: {force_rebuild}")
            
            if not self.knowledge_items:
                logger.warning("No knowledge items to index")
                return
            
            # 🔧 FORCE REBUILD: 清空现有索引
            if force_rebuild:
                logger.info(f"🗑️ CLEARING EXISTING INDEX (Force Rebuild)...")
                try:
                    # 获取当前所有文档ID
                    current_stats = self.rag_engine.vector_store.get_stats()
                    current_doc_count = current_stats.get("total_documents", 0)
                    logger.info(f"  - Current documents in vector store: {current_doc_count}")
                    
                    if current_doc_count > 0:
                        # 清空集合以强制重新索引
                        await self._clear_travel_knowledge_index()
                        logger.info(f"✅ CLEARED EXISTING INDEX")
                    else:
                        logger.info(f"  - No existing documents to clear")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clear existing index: {e}")
                    logger.info(f"  - Continuing with indexing (will update existing docs)")
            
            # Convert knowledge items to documents
            documents = []
            berlin_docs = 0
            
            for i, knowledge in enumerate(self.knowledge_items.values()):
                # 🔧 DEBUG: Log document creation
                logger.debug(f"🏗️ CREATING DOCUMENT {i+1}: {knowledge.id}")
                
                doc = Document(
                    id=knowledge.id,
                    content=f"{knowledge.title}\n\n{knowledge.content}",
                    metadata={
                        "category": knowledge.category,
                        "location": knowledge.location or "",
                        "tags": ",".join(knowledge.tags),
                        "language": knowledge.language,
                        "title": knowledge.title
                    },
                    doc_type=DocumentType.TRAVEL_KNOWLEDGE  # 🔧 Add proper document type
                )
                documents.append(doc)
                
                # 🔧 DEBUG: Track Berlin documents
                if 'berlin' in doc.content.lower() or 'berlin' in str(doc.metadata).lower():
                    berlin_docs += 1
                    logger.info(f"🏛️ BERLIN DOCUMENT CREATED FOR INDEXING: {doc.id}")
                    logger.info(f"  - Title: {knowledge.title}")
                    logger.info(f"  - Location metadata: {doc.metadata.get('location')}")
                    logger.info(f"  - Content length: {len(doc.content)}")
                    logger.info(f"  - Content preview: {doc.content[:200]}...")
                
                logger.debug(f"  - Document created with {len(doc.content)} chars")
                logger.debug(f"  - Metadata: {doc.metadata}")
            
            # 🔧 DEBUG: Log documents prepared for indexing
            logger.info(f"📋 DOCUMENTS PREPARED FOR INDEXING:")
            logger.info(f"  - Total documents: {len(documents)}")
            logger.info(f"  - Berlin documents: {berlin_docs}")
            logger.info(f"  - RAG engine: {self.rag_engine}")
            
            # Index documents in RAG engine
            logger.info(f"🚀 STARTING RAG INDEXING...")
            success = await self.rag_engine.index_documents(documents)
            
            if success:
                logger.info(f"✅ Successfully indexed {len(documents)} knowledge items")
                logger.info(f"  - Berlin documents indexed: {berlin_docs}")
                
                # 🔧 DEBUG: Verify indexing worked by checking vector store stats
                vector_stats = self.rag_engine.vector_store.get_stats()
                logger.info(f"📊 POST-INDEXING VECTOR STORE STATS:")
                logger.info(f"  - Total documents in vector store: {vector_stats.get('total_documents', 'Unknown')}")
                
            else:
                logger.error("❌ Failed to index knowledge items")
                
        except Exception as e:
            logger.error(f"❌ Failed to build knowledge index: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            raise
    
    async def _clear_travel_knowledge_index(self):
        """Clear all travel knowledge documents from the vector store"""
        try:
            logger.info(f"🧹 CLEARING TRAVEL KNOWLEDGE FROM VECTOR STORE...")
            
            # 使用ChromaDB的delete方法删除所有旅行知识文档
            # 通过doc_type过滤来只删除旅行知识文档，保留工具文档
            await self.rag_engine.vector_store.clear_documents_by_type(DocumentType.TRAVEL_KNOWLEDGE)
            
            logger.info(f"✅ TRAVEL KNOWLEDGE INDEX CLEARED")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear travel knowledge index: {e}")
            raise
    
    async def add_knowledge(self, knowledge: TravelKnowledge) -> bool:
        """Add new knowledge item"""
        try:
            # 🔧 DEBUG: Log knowledge addition
            logger.info(f"➕ ADDING NEW KNOWLEDGE ITEM: {knowledge.id}")
            logger.info(f"  - Title: {knowledge.title}")
            logger.info(f"  - Location: {knowledge.location}")
            logger.info(f"  - Category: {knowledge.category}")
            logger.info(f"  - Content length: {len(knowledge.content)}")
            
            if len(knowledge.content.strip()) < 10:
                raise ValueError("Content too short")
            
            # Add to memory
            self.knowledge_items[knowledge.id] = knowledge
            logger.debug(f"  - Added to memory: {knowledge.id}")
            
            # Create document and index
            doc = Document(
                id=knowledge.id,
                content=f"{knowledge.title}\n\n{knowledge.content}",
                metadata={
                    "category": knowledge.category,
                    "location": knowledge.location or "",
                    "tags": ",".join(knowledge.tags),
                    "language": knowledge.language,
                    "title": knowledge.title
                },
                doc_type=DocumentType.TRAVEL_KNOWLEDGE  # 🔧 Add proper document type
            )
            
            # 🔧 DEBUG: Log document creation for indexing
            logger.debug(f"  - Document created for indexing: {doc.id}")
            logger.debug(f"  - Document metadata: {doc.metadata}")
            
            # Index in RAG engine
            logger.debug(f"  - Indexing in RAG engine...")
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                logger.info(f"✅ Added knowledge item: {knowledge.id}")
                return True
            else:
                del self.knowledge_items[knowledge.id]
                logger.error(f"❌ Failed to index knowledge item: {knowledge.id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to add knowledge item: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            return False
    
    async def update_knowledge(self, knowledge_id: str, updated_knowledge: TravelKnowledge) -> bool:
        """Update existing knowledge item"""
        try:
            if knowledge_id not in self.knowledge_items:
                logger.error(f"Knowledge item not found: {knowledge_id}")
                return False
            
            # Update in memory
            self.knowledge_items[knowledge_id] = updated_knowledge
            
            # Update in vector store
            doc = Document(
                id=updated_knowledge.id,
                content=f"{updated_knowledge.title}\n\n{updated_knowledge.content}",
                metadata={
                    "category": updated_knowledge.category,
                    "location": updated_knowledge.location or "",
                    "tags": ",".join(updated_knowledge.tags),
                    "language": updated_knowledge.language,
                    "title": updated_knowledge.title
                },
                doc_type=DocumentType.TRAVEL_KNOWLEDGE  # 🔧 Add proper document type
            )
            
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                logger.info(f"Updated knowledge item: {knowledge_id}")
                return True
            else:
                logger.error(f"Failed to update knowledge item: {knowledge_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update knowledge item: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge item"""
        try:
            if knowledge_id not in self.knowledge_items:
                logger.error(f"Knowledge item not found: {knowledge_id}")
                return False
            
            # Remove from memory
            del self.knowledge_items[knowledge_id]
            
            # Remove from vector store
            success = await self.rag_engine.vector_store.delete_documents([knowledge_id])
            
            if success:
                logger.info(f"Deleted knowledge item: {knowledge_id}")
                return True
            else:
                logger.error(f"Failed to delete knowledge item: {knowledge_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete knowledge item: {e}")
            return False
    
    async def search_knowledge(
        self, 
        query: str, 
        category: Optional[str] = None,
        location: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search knowledge using RAG engine with semantic search"""
        try:
            # 🔧 DEBUG: Log search operation start
            logger.info(f"🔍 KNOWLEDGE SEARCH OPERATION:")
            logger.info(f"  - Query: '{query}'")
            logger.info(f"  - Category filter: {category}")
            logger.info(f"  - Location filter: {location}")
            logger.info(f"  - Top K: {top_k}")
            logger.info(f"  - Min score: {min_score}")
            logger.info(f"  - Available knowledge items: {len(self.knowledge_items)}")
            
            # 🔧 DEBUG: Check for Berlin in query
            if 'berlin' in query.lower():
                logger.info(f"🏛️ BERLIN SEARCH QUERY DETECTED!")
                # Check if we have Berlin knowledge
                berlin_knowledge = [k for k in self.knowledge_items.values() 
                                  if k.location and 'berlin' in k.location.lower()]
                logger.info(f"  - Berlin knowledge items available: {len(berlin_knowledge)}")
                for bk in berlin_knowledge:
                    logger.info(f"    - {bk.id}: {bk.title}")
            
            # Build filter metadata
            filter_metadata = {}
            if category:
                filter_metadata["category"] = category
                logger.info(f"  - Added category filter: {category}")
            if location:
                filter_metadata["location"] = location
                logger.info(f"  - Added location filter: {location}")
            
            # 🔧 DEBUG: Log RAG engine call
            logger.info(f"🧠 CALLING RAG ENGINE RETRIEVE...")
            logger.info(f"  - Filter metadata: {filter_metadata}")
            logger.info(f"  - Doc type filter: {DocumentType.TRAVEL_KNOWLEDGE}")
            
            # Use RAG engine for semantic search with travel knowledge filter
            result = await self.rag_engine.retrieve(
                query=query,
                top_k=top_k * 2,  # Get more candidates for filtering
                filter_metadata=filter_metadata,
                doc_type=DocumentType.TRAVEL_KNOWLEDGE  # 🔧 Filter for travel knowledge only
            )
            
            # 🔧 DEBUG: Log RAG engine results
            logger.info(f"📊 RAG ENGINE RESULTS:")
            logger.info(f"  - Documents returned: {len(result.documents)}")
            logger.info(f"  - Scores: {result.scores}")
            logger.info(f"  - Query: {result.query}")
            logger.info(f"  - Total results: {result.total_results}")
            
            for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
                logger.info(f"  📄 RAG RESULT {i+1}:")
                logger.info(f"    - Score: {score:.4f}")
                logger.info(f"    - Document ID: {doc.id}")
                logger.info(f"    - Metadata: {doc.metadata}")
                logger.info(f"    - Content preview: {doc.content[:100]}...")
                
                # 🔧 DEBUG: Check for Berlin content mismatch
                if 'berlin' in query.lower() and 'berlin' not in doc.content.lower():
                    logger.warning(f"    🚨 BERLIN QUERY MISMATCH!")
                    logger.warning(f"    - Query has 'berlin' but result doesn't")
                    logger.warning(f"    - Result location: {doc.metadata.get('location', 'Unknown')}")
            
            # Process and filter results
            knowledge_results = []
            for doc, score in zip(result.documents, result.scores):
                if score < min_score:
                    logger.debug(f"  - Skipping doc {doc.id} due to low score: {score:.4f}")
                    continue
                    
                # Find the original knowledge item
                knowledge = self.knowledge_items.get(doc.id)
                if knowledge:
                    knowledge_results.append({
                        "knowledge": knowledge,
                        "relevance_score": score,
                        "highlights": self._extract_highlights(doc.content, query)
                    })
                    logger.debug(f"  - Added knowledge result: {knowledge.id} (score: {score:.4f})")
                else:
                    logger.warning(f"  - Knowledge item not found for doc: {doc.id}")
            
            # Sort by relevance score and limit results
            knowledge_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            knowledge_results = knowledge_results[:top_k]
            
            # 🔧 DEBUG: Log final search results
            logger.info(f"✅ KNOWLEDGE SEARCH COMPLETE:")
            logger.info(f"  - Final results: {len(knowledge_results)}")
            logger.info(f"  - Query: '{query}'")
            
            for i, result in enumerate(knowledge_results):
                knowledge = result["knowledge"]
                score = result["relevance_score"]
                logger.info(f"  📖 FINAL RESULT {i+1}:")
                logger.info(f"    - Knowledge ID: {knowledge.id}")
                logger.info(f"    - Title: {knowledge.title}")
                logger.info(f"    - Location: {knowledge.location}")
                logger.info(f"    - Score: {score:.4f}")
                
                # 🔧 DEBUG: Final Berlin analysis
                if 'berlin' in query.lower():
                    if knowledge.location and 'berlin' in knowledge.location.lower():
                        logger.info(f"    ✅ CORRECT BERLIN RESULT!")
                    else:
                        logger.error(f"    🚨 INCORRECT RESULT FOR BERLIN QUERY!")
                        logger.error(f"    - Expected Berlin but got: {knowledge.location}")
            
            logger.info(f"Found {len(knowledge_results)} relevant knowledge items for query: '{query}'")
            return knowledge_results
            
        except Exception as e:
            logger.error(f"❌ Knowledge search failed: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            return []
    
    def _extract_highlights(self, content: str, query: str) -> List[str]:
        """Extract relevant text snippets containing query terms"""
        # Simple highlight extraction (can be enhanced with better NLP)
        highlights = []
        paragraphs = content.split("\n\n")
        query_terms = query.lower().split()
        
        for para in paragraphs:
            if any(term in para.lower() for term in query_terms):
                highlights.append(para.strip())
        
        return highlights[:3]  # Return top 3 most relevant snippets
    
    def get_categories(self) -> List[KnowledgeCategory]:
        """Get all knowledge categories"""
        return list(self.categories.values())
    
    def get_knowledge_by_category(self, category_id: str) -> List[TravelKnowledge]:
        """Get knowledge items by category"""
        return [
            knowledge for knowledge in self.knowledge_items.values()
            if knowledge.category == category_id
        ]
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        # 🔧 DEBUG: Log stats generation
        logger.debug(f"📊 GENERATING KNOWLEDGE BASE STATS...")
        
        stats = {
            "total_knowledge_items": len(self.knowledge_items),
            "total_categories": len(self.categories),
            "items_by_category": {},
            "items_by_language": {},
            "vector_store_stats": self.rag_engine.vector_store.get_stats(),
            "data_loader_stats": self.data_loader.get_data_stats(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Count by category and language
        for knowledge in self.knowledge_items.values():
            stats["items_by_category"][knowledge.category] = stats["items_by_category"].get(knowledge.category, 0) + 1
            stats["items_by_language"][knowledge.language] = stats["items_by_language"].get(knowledge.language, 0) + 1
        
        # 🔧 DEBUG: Log Berlin content in stats
        berlin_items = [k for k in self.knowledge_items.values() 
                       if k.location and 'berlin' in k.location.lower()]
        if berlin_items:
            logger.debug(f"📊 BERLIN CONTENT IN STATS: {len(berlin_items)} items")
            for item in berlin_items:
                logger.debug(f"  - {item.id}: {item.title}")
        else:
            logger.warning(f"📊 NO BERLIN CONTENT FOUND IN STATS!")
        
        return stats


# Global knowledge base instance
knowledge_base: Optional[KnowledgeBase] = None


async def get_knowledge_base() -> KnowledgeBase:
    """Get knowledge base instance (singleton pattern)"""
    global knowledge_base
    if knowledge_base is None:
        knowledge_base = KnowledgeBase()
        await knowledge_base.initialize()
    return knowledge_base 