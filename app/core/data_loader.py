"""
Data Loader Module - Travel Knowledge Data Loading

Handles loading travel knowledge from various sources including JSON, YAML, 
and other file formats. Supports schema validation and data preprocessing.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import hashlib
import time
from pydantic import BaseModel, ValidationError
from app.core.logging_config import get_logger

# Import will be done locally to avoid circular imports

# Set up logging
logger = get_logger(__name__)


class DataLoadStats(BaseModel):
    """Statistics for data loading operations"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_knowledge_items: int = 0
    load_time: float = 0.0
    errors: List[str] = []


class TravelDataLoader:
    """Travel knowledge data loader with validation and preprocessing"""
    
    def __init__(self, knowledge_dir: str = "app/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.documents_dir = self.knowledge_dir / "documents"
        self.schema_file = self.knowledge_dir / "schemas" / "knowledge_schema.json"
        
        # Supported file formats
        self.supported_formats = {'.json', '.yaml', '.yml'}
        
        logger.info(f"Data loader initialized for directory: {self.knowledge_dir}")
    
    async def load_all_data(self) -> List[Any]:
        """Load all travel knowledge data from the documents directory"""
        start_time = time.time()
        stats = DataLoadStats()
        all_knowledge = []
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory does not exist: {self.documents_dir}")
            return []
        
        # Recursively find all supported files
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                stats.total_files += 1
                try:
                    knowledge_items = await self._load_file(file_path)
                    all_knowledge.extend(knowledge_items)
                    stats.successful_files += 1
                    stats.total_knowledge_items += len(knowledge_items)
                    logger.debug(f"Loaded {len(knowledge_items)} items from {file_path}")
                except Exception as e:
                    stats.failed_files += 1
                    error_msg = f"Failed to load {file_path}: {str(e)}"
                    stats.errors.append(error_msg)
                    logger.error(error_msg)
        
        # Calculate load time
        stats.load_time = time.time() - start_time
        
        # Log summary
        logger.info(f"Data loading completed in {stats.load_time:.2f}s")
        logger.info(f"Files: {stats.successful_files}/{stats.total_files} successful")
        logger.info(f"Total knowledge items loaded: {stats.total_knowledge_items}")
        
        if stats.failed_files > 0:
            logger.warning(f"Failed to load {stats.failed_files} files")
            for error in stats.errors:
                logger.warning(f"  - {error}")
        
        return all_knowledge
    
    async def _load_file(self, file_path: Path) -> List[Any]:
        """Load knowledge from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Parse based on file format
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Process data structure
            return await self._process_data(data, file_path)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"File loading error: {e}")
    
    async def _process_data(self, data: Any, file_path: Path) -> List[Any]:
        """Process loaded data into TravelKnowledge objects"""
        knowledge_items = []
        
        if isinstance(data, list):
            # File contains multiple knowledge items
            for i, item_data in enumerate(data):
                try:
                    knowledge = await self._create_knowledge_item(item_data, file_path, i)
                    if knowledge:
                        knowledge_items.append(knowledge)
                except Exception as e:
                    logger.error(f"Failed to process item {i} in {file_path}: {e}")
                    
        elif isinstance(data, dict):
            # File contains single knowledge item or structured data
            if 'knowledge_items' in data:
                # Structured format with metadata
                for i, item_data in enumerate(data['knowledge_items']):
                    try:
                        knowledge = await self._create_knowledge_item(item_data, file_path, i)
                        if knowledge:
                            knowledge_items.append(knowledge)
                    except Exception as e:
                        logger.error(f"Failed to process item {i} in {file_path}: {e}")
            else:
                # Single knowledge item
                try:
                    knowledge = await self._create_knowledge_item(data, file_path, 0)
                    if knowledge:
                        knowledge_items.append(knowledge)
                except Exception as e:
                    logger.error(f"Failed to process single item in {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported data structure in {file_path}")
        
        return knowledge_items
    
    async def _create_knowledge_item(
        self, 
        data: Dict[str, Any], 
        file_path: Path, 
        item_index: int
    ) -> Optional[Any]:
        """Create a TravelKnowledge object from raw data"""
        try:
            # Validate required fields
            required_fields = ['id', 'title', 'content', 'category']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Auto-generate ID if not provided or invalid
            if not data.get('id') or not isinstance(data['id'], str):
                data['id'] = self._generate_id(data, file_path, item_index)
            
            # Set default values
            data.setdefault('language', 'zh')
            data.setdefault('tags', [])
            data.setdefault('last_updated', datetime.now().isoformat())
            
            # Process source information
            if 'source' in data and isinstance(data['source'], dict):
                from app.core.knowledge_base import KnowledgeSource
                data['source'] = KnowledgeSource(**data['source'])
            
            # Create TravelKnowledge object
            from app.core.knowledge_base import TravelKnowledge
            knowledge = TravelKnowledge(**data)
            
            # Validate knowledge item
            await self._validate_knowledge_item(knowledge)
            
            return knowledge
            
        except ValidationError as e:
            logger.error(f"Validation error for item in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating knowledge item from {file_path}: {e}")
            return None
    
    def _generate_id(self, data: Dict[str, Any], file_path: Path, item_index: int) -> str:
        """Generate a unique ID for knowledge item"""
        # Use file path and content to create consistent ID
        content_hash = hashlib.md5(
            f"{file_path.stem}_{item_index}_{data.get('title', '')}".encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{file_path.stem}_{content_hash}"
    
    async def _validate_knowledge_item(self, knowledge: Any) -> None:
        """Validate knowledge item content and structure"""
        # Basic validation
        if len(knowledge.content.strip()) < 10:
            raise ValueError("Content too short (minimum 10 characters)")
        
        if len(knowledge.title.strip()) < 2:
            raise ValueError("Title too short (minimum 2 characters)")
            
        # Category validation
        valid_categories = {
            'destinations', 'transportation', 'accommodation', 
            'activities', 'practical'
        }
        if knowledge.category not in valid_categories:
            logger.warning(f"Unknown category: {knowledge.category}")
    
    async def load_from_directory(self, directory: Path) -> List[Any]:
        """Load knowledge from a specific directory"""
        knowledge_items = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    items = await self._load_file(file_path)
                    knowledge_items.extend(items)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return knowledge_items
    
    async def load_single_file(self, file_path: Path) -> List[Any]:
        """Load knowledge from a single file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return await self._load_file(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats)
    
    def calculate_data_version(self) -> str:
        """Calculate version hash for all data files"""
        hash_md5 = hashlib.md5()
        
        # Get all supported files and sort for consistent hashing
        files = []
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                files.append(file_path)
        
        files.sort(key=lambda x: str(x))
        
        # Hash file contents
        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    hash_md5.update(f.read())
            except Exception as e:
                logger.warning(f"Could not hash file {file_path}: {e}")
                continue
        
        return hash_md5.hexdigest()
    
    def get_latest_modification_time(self) -> float:
        """Get the latest modification time of all data files"""
        latest_time = 0
        
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    file_time = file_path.stat().st_mtime
                    latest_time = max(latest_time, file_time)
                except Exception as e:
                    logger.warning(f"Could not get modification time for {file_path}: {e}")
                    continue
        
        return latest_time
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the data directory"""
        stats = {
            "total_files": 0,
            "files_by_format": {},
            "total_size": 0,
            "latest_modification": 0
        }
        
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                stats["total_files"] += 1
                
                # Count by format
                format_key = file_path.suffix.lower()
                stats["files_by_format"][format_key] = stats["files_by_format"].get(format_key, 0) + 1
                
                # Total size
                try:
                    stats["total_size"] += file_path.stat().st_size
                    stats["latest_modification"] = max(
                        stats["latest_modification"], 
                        file_path.stat().st_mtime
                    )
                except Exception:
                    continue
        
        return stats 