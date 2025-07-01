"""
Knowledge Base Module - Intelligent Travel Knowledge Management

Handles initialization, loading, and management of travel knowledge base with
smart data loading, version control, and persistence optimization.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
import logging
import time
from datetime import datetime
from pydantic import BaseModel

from app.core.rag_engine import Document, RAGEngine, get_rag_engine
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
    language: str = "zh"


class TravelKnowledge(BaseModel):
    """Travel knowledge data structure"""
    id: str
    title: str
    content: str
    category: str
    location: Optional[str] = None
    tags: List[str] = []
    source: Optional[KnowledgeSource] = None
    language: str = "zh"
    last_updated: Optional[str] = None


class KnowledgeBase:
    """Intelligent travel knowledge base manager"""
    
    def __init__(self, knowledge_dir: str = "app/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.categories: Dict[str, KnowledgeCategory] = {}
        self.knowledge_items: Dict[str, TravelKnowledge] = {}
        self.rag_engine = get_rag_engine()
        self.data_loader = TravelDataLoader(knowledge_dir)
        
        # Version and persistence management
        self.data_version_file = Path("./data/knowledge_version.json")
        self.current_version = self._load_version_info()
        
        logger.info(f"Knowledge base initialized for directory: {self.knowledge_dir}")
    
    async def initialize(self):
        """Smart initialization with version control and incremental loading"""
        logger.info("🔄 Starting knowledge base initialization...")
        
        try:
            # 1. Load category configuration
            await self._load_categories()
            
            # 2. Check if data needs to be reloaded
            should_reload = await self._should_reload_data()
            
            if should_reload:
                logger.info("📥 Loading fresh data from disk...")
                # Load all data from files
                await self._load_knowledge_data()
                
                # Rebuild vector index
                await self._build_index()
                
                # Update version information
                self._update_version_info()
            else:
                logger.info("✅ Using existing indexed data (ChromaDB persistent storage)")
                # Only load data structures in memory, skip indexing
                await self._load_knowledge_data_memory_only()
            
            logger.info(f"📚 Knowledge base ready with {len(self.knowledge_items)} items")
            
        except Exception as e:
            logger.error(f"❌ Knowledge base initialization failed: {e}")
            raise
    
    async def _should_reload_data(self) -> bool:
        """Determine if data should be reloaded based on version and changes"""
        try:
            # 1. Check if this is the first run
            stats = self.rag_engine.vector_store.get_stats()
            if stats.get("total_documents", 0) == 0:
                logger.info("First run detected - will load all data")
                return True
            
            # 2. Check file modification times
            latest_file_time = self.data_loader.get_latest_modification_time()
            last_update_time = self.current_version.get("last_update", 0)
            
            if latest_file_time > last_update_time:
                logger.info("File changes detected - will reload data")
                return True
            
            # 3. Check data version hash
            current_hash = self.data_loader.calculate_data_version()
            stored_hash = self.current_version.get("data_hash", "")
            
            if current_hash != stored_hash:
                logger.info("Data version changed - will reload data")
                return True
            
            # 4. Force reload if no version info
            if not self.current_version:
                logger.info("No version info found - will reload data")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking data version, will reload: {e}")
            return True
    
    async def _load_categories(self):
        """Load knowledge categories from configuration"""
        try:
            categories_file = self.knowledge_dir / "categories.yaml"
            if categories_file.exists():
                logger.info("Loading categories from configuration file...")
                await self._load_categories_from_file(categories_file)
            else:
                logger.info("Creating default categories...")
                await self._create_default_categories()
                
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            await self._create_default_categories()
    
    async def _load_categories_from_file(self, categories_file: Path):
        """Load categories from YAML configuration file"""
        try:
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
                
        except Exception as e:
            logger.error(f"Error loading categories from file: {e}")
            raise
    
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
        """Load all knowledge data from files"""
        try:
            # Load from files using data loader
            knowledge_items = await self.data_loader.load_all_data()
            
            # Process loaded knowledge
            self.knowledge_items.clear()
            for knowledge in knowledge_items:
                self.knowledge_items[knowledge.id] = knowledge
            
            # Load default knowledge if no files found
            if not self.knowledge_items:
                logger.info("No knowledge files found, creating default knowledge...")
                await self._load_default_travel_knowledge()
            
            logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge data: {e}")
            # Fallback to default knowledge
            await self._load_default_travel_knowledge()
    
    async def _load_knowledge_data_memory_only(self):
        """Load knowledge data structures without rebuilding index"""
        try:
            # This is a lightweight version that only loads data structures
            # without rebuilding the vector index (assuming it's already built)
            knowledge_items = await self.data_loader.load_all_data()
            
            self.knowledge_items.clear()
            for knowledge in knowledge_items:
                self.knowledge_items[knowledge.id] = knowledge
            
            if not self.knowledge_items:
                await self._load_default_travel_knowledge()
                
            logger.info(f"Loaded {len(self.knowledge_items)} knowledge items (memory only)")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge data in memory: {e}")
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
                "content": """Information for obtaining a tourist visa to Japan for Chinese citizens.

**Required Documents:**
1. **Passport**: Valid for at least 6 months, with 2 blank pages
2. **Visa Application Form**: Completed and signed
3. **Photo**: 2-inch color photo on white background
4. **Employment Certificate**: Company letterhead with position and salary
5. **Bank Statements**: Last 6 months showing sufficient funds
6. **Property Documents**: Real estate certificate or purchase contract
7. **Travel Itinerary**: Detailed travel plan with dates and locations
8. **Hotel Reservations**: Confirmed bookings for entire stay
9. **Flight Reservations**: Round-trip flight booking confirmation

**Application Process:**
1. Prepare documents (3-5 business days)
2. Submit application at consulate or authorized agency
3. Processing time: 5-7 business days
4. Collect passport with visa

**Fees:**
- Single entry: ¥200 (~$30)
- Multiple entry (3 years): ¥400 (~$60)
- Multiple entry (5 years): ¥700 (~$100)

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
                        "tags": knowledge.tags,  # Will be converted to string in RAG engine
                        "language": knowledge.language,
                        "title": knowledge.title
                    }
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
        """Add new knowledge item with validation and indexing"""
        try:
            # 1. Validate knowledge quality
            if len(knowledge.content.strip()) < 10:
                raise ValueError("Content too short")
            
            # 2. Add to memory
            self.knowledge_items[knowledge.id] = knowledge
            
            # 3. Create document and index
            doc = Document(
                id=knowledge.id,
                content=f"{knowledge.title}\n\n{knowledge.content}",
                metadata={
                    "category": knowledge.category,
                    "location": knowledge.location or "",
                    "tags": knowledge.tags,  # Will be converted to string in RAG engine
                    "language": knowledge.language,
                    "title": knowledge.title
                }
            )
            
            # 4. Index in RAG engine
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                logger.info(f"Added knowledge item: {knowledge.id}")
                # Update version after successful addition
                self._update_version_info()
                return True
            else:
                # Rollback if indexing failed
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
                    "tags": updated_knowledge.tags,  # Will be converted to string in RAG engine
                    "language": updated_knowledge.language,
                    "title": updated_knowledge.title
                }
            )
            
            success = await self.rag_engine.index_documents([doc])
            
            if success:
                logger.info(f"Updated knowledge item: {knowledge_id}")
                self._update_version_info()
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
                self._update_version_info()
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
        top_k: int = 5
    ) -> List[TravelKnowledge]:
        """Search knowledge using RAG engine"""
        try:
            # Build filter metadata
            filter_metadata = {}
            if category:
                filter_metadata["category"] = category
            if location:
                filter_metadata["location"] = location
            
            # Use RAG engine to retrieve relevant documents
            result = await self.rag_engine.retrieve(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Convert retrieved documents back to knowledge objects
            knowledge_results = []
            for doc in result.documents:
                # Try to find the original knowledge item
                for knowledge in self.knowledge_items.values():
                    if knowledge.id == doc.id or doc.id.startswith(f"{knowledge.id}_chunk_"):
                        if knowledge not in knowledge_results:
                            knowledge_results.append(knowledge)
                        break
            
            logger.info(f"Found {len(knowledge_results)} knowledge items for query: '{query}'")
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
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
            "data_loader_stats": self.data_loader.get_data_stats()
        }
        
        # Count by category
        for knowledge in self.knowledge_items.values():
            category = knowledge.category
            stats["items_by_category"][category] = stats["items_by_category"].get(category, 0) + 1
            
            language = knowledge.language
            stats["items_by_language"][language] = stats["items_by_language"].get(language, 0) + 1
        
        return stats
    
    def _load_version_info(self) -> Dict[str, Any]:
        """Load version information from file"""
        try:
            if self.data_version_file.exists():
                with open(self.data_version_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load version info: {e}")
        return {}
    
    def _update_version_info(self):
        """Update version information"""
        try:
            version_info = {
                "last_update": time.time(),
                "data_hash": self.data_loader.calculate_data_version(),
                "total_items": len(self.knowledge_items),
                "updated_at": datetime.now().isoformat()
            }
            
            # Ensure directory exists
            self.data_version_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.data_version_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, indent=2)
                
            self.current_version = version_info
            logger.debug("Version information updated")
            
        except Exception as e:
            logger.error(f"Failed to update version info: {e}")


# Global knowledge base instance
knowledge_base: Optional[KnowledgeBase] = None


async def get_knowledge_base() -> KnowledgeBase:
    """Get knowledge base instance (singleton pattern)"""
    global knowledge_base
    if knowledge_base is None:
        knowledge_base = KnowledgeBase()
        await knowledge_base.initialize()
    return knowledge_base 