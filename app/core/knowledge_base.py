"""
Knowledge Base Module - Intelligent Travel Knowledge Management

Handles initialization, loading, and management of travel knowledge base with
smart data loading and RAG-enhanced search capabilities.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging
from datetime import datetime
from pydantic import BaseModel

from app.core.rag_engine import Document, DocumentType, get_rag_engine
from app.core.data_loader import TravelDataLoader

# Set up logging
logger = logging.getLogger(__name__)


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
            
            # 3. Build RAG index if needed
            if not self.rag_engine.vector_store.get_stats().get("total_documents", 0):
                await self._build_index()
            
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
            # Load from files using data loader
            knowledge_items = await self.data_loader.load_all_data()
            
            # Process loaded knowledge
            self.knowledge_items.clear()
            for knowledge in knowledge_items:
                self.knowledge_items[knowledge.id] = knowledge
            
            # Load default knowledge if no files found
            if not self.knowledge_items:
                logger.info("No knowledge files found, loading default knowledge...")
                await self._load_default_travel_knowledge()
            
            logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge data: {e}")
            await self._load_default_travel_knowledge()
    
    async def _load_default_travel_knowledge(self):
        """Load default travel knowledge data"""
        default_knowledge = [
            {
                "id": "paris_eiffel_tower",
                "title": "Eiffel Tower - Paris",
                "content": """The Eiffel Tower is an iconic landmark in Paris, France. Built in 1889, it stands 324 meters tall and offers spectacular views of the city.

**Visiting Information:**
- Opening Hours: 9:00 AM - 11:00 PM (extended hours in summer)
- Ticket Prices: Adults €29.40 (top floor), €18.10 (second floor)
- Best Time to Visit: Early morning or late evening to avoid crowds
- Location: Champ de Mars, 5 Avenue Anatole France, 75007 Paris

**Tips:**
- Book tickets online in advance to skip lines
- Visit during sunset for the most beautiful views
- Take the stairs to the second floor for a discount
- Security checks can cause delays, arrive early""",
                "category": "destinations",
                "location": "Paris, France",
                "tags": ["Paris", "Eiffel Tower", "Landmark", "France", "Europe"],
                "language": "en"
            },
            {
                "id": "japan_tourist_visa",
                "title": "Japan Tourist Visa Requirements",
                "content": """Information for obtaining a tourist visa to Japan.

**Required Documents:**
1. **Passport**: Valid for at least 6 months, with 2 blank pages
2. **Visa Application Form**: Completed and signed
3. **Photo**: 2-inch color photo on white background
4. **Employment Certificate**: Company letterhead with position and salary
5. **Bank Statements**: Last 6 months showing sufficient funds
6. **Travel Itinerary**: Detailed travel plan with dates and locations
7. **Hotel Reservations**: Confirmed bookings for entire stay
8. **Flight Reservations**: Round-trip flight booking confirmation

**Application Process:**
1. Prepare documents (3-5 business days)
2. Submit application at consulate or authorized agency
3. Processing time: 5-7 business days
4. Collect passport with visa

**Important Notes:**
- Apply 1 month in advance for regular season
- Apply 2 months in advance for peak seasons (cherry blossom, autumn)
- First-time applicants recommended to use authorized travel agencies""",
                "category": "practical",
                "location": "Japan",
                "tags": ["Japan", "Visa", "Tourism", "Application", "Requirements"],
                "language": "en"
            }
        ]
        
        for knowledge_data in default_knowledge:
            knowledge = TravelKnowledge(**knowledge_data)
            self.knowledge_items[knowledge.id] = knowledge
            
        logger.info(f"Created {len(default_knowledge)} default knowledge items")
    
    async def _build_index(self):
        """Build vector index for all knowledge items"""
        try:
            if not self.knowledge_items:
                logger.warning("No knowledge items to index")
                return
            
            # Convert knowledge items to documents
            documents = []
            for knowledge in self.knowledge_items.values():
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
            
            # Index documents in RAG engine
            success = await self.rag_engine.index_documents(documents)
            
            if success:
                logger.info(f"Successfully indexed {len(documents)} knowledge items")
            else:
                logger.error("Failed to index knowledge items")
                
        except Exception as e:
            logger.error(f"Failed to build knowledge index: {e}")
            raise
    
    async def add_knowledge(self, knowledge: TravelKnowledge) -> bool:
        """Add new knowledge item"""
        try:
            if len(knowledge.content.strip()) < 10:
                raise ValueError("Content too short")
            
            # Add to memory
            self.knowledge_items[knowledge.id] = knowledge
            
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
            
            # Index in RAG engine
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                logger.info(f"Added knowledge item: {knowledge.id}")
                return True
            else:
                del self.knowledge_items[knowledge.id]
                logger.error(f"Failed to index knowledge item: {knowledge.id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")
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
            # Build filter metadata
            filter_metadata = {}
            if category:
                filter_metadata["category"] = category
            if location:
                filter_metadata["location"] = location
            
            # Use RAG engine for semantic search with travel knowledge filter
            result = await self.rag_engine.retrieve(
                query=query,
                top_k=top_k * 2,  # Get more candidates for filtering
                filter_metadata=filter_metadata,
                doc_type=DocumentType.TRAVEL_KNOWLEDGE  # 🔧 Filter for travel knowledge only
            )
            
            # Process and filter results
            knowledge_results = []
            for doc, score in zip(result.documents, result.scores):
                if score < min_score:
                    continue
                    
                # Find the original knowledge item
                knowledge = self.knowledge_items.get(doc.id)
                if knowledge:
                    knowledge_results.append({
                        "knowledge": knowledge,
                        "relevance_score": score,
                        "highlights": self._extract_highlights(doc.content, query)
                    })
            
            # Sort by relevance score and limit results
            knowledge_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            knowledge_results = knowledge_results[:top_k]
            
            logger.info(f"Found {len(knowledge_results)} relevant knowledge items for query: '{query}'")
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
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