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
        
        # 🔧 Version and change detection
        self.cache_dir = self.knowledge_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.version_file = self.cache_dir / "data_version.json"
        self.last_known_version = self._load_version_info()
        
        logger.info(f"Data loader initialized for directory: {self.knowledge_dir}")
        logger.info(f"  - Cache directory: {self.cache_dir}")
        logger.info(f"  - Last known version: {self.last_known_version.get('version_hash', 'None')[:8]}...")
    
    def _load_version_info(self) -> Dict[str, Any]:
        """Load version information from cache"""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    version_info = json.load(f)
                    logger.debug(f"📋 LOADED VERSION INFO: {version_info.get('version_hash', 'Unknown')[:8]}...")
                    return version_info
        except Exception as e:
            logger.warning(f"Failed to load version info: {e}")
        
        return {
            "version_hash": "",
            "last_update": "",
            "files_count": 0,
            "files_info": {}
        }
    
    def _save_version_info(self, version_info: Dict[str, Any]) -> None:
        """Save version information to cache"""
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 SAVED VERSION INFO: {version_info.get('version_hash', 'Unknown')[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save version info: {e}")
    
    def detect_changes(self) -> Dict[str, Any]:
        """检测文档目录的变化"""
        logger.info(f"🔍 DETECTING CHANGES IN DOCUMENTS...")
        
        current_files_info = {}
        current_version_hash = hashlib.md5()
        
        # 扫描所有支持的文件
        for file_path in sorted(self.documents_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    stat = file_path.stat()
                    file_info = {
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "path": str(file_path.relative_to(self.documents_dir))
                    }
                    
                    # 计算文件内容hash
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        file_info["hash"] = file_hash
                    
                    current_files_info[str(file_path)] = file_info
                    current_version_hash.update(file_hash.encode())
                    
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
        
        current_version = current_version_hash.hexdigest()
        
        # 比较版本
        old_version = self.last_known_version.get("version_hash", "")
        old_files_info = self.last_known_version.get("files_info", {})
        
        changes = {
            "has_changes": current_version != old_version,
            "old_version": old_version,
            "new_version": current_version,
            "files_added": [],
            "files_modified": [],
            "files_deleted": [],
            "total_files": len(current_files_info)
        }
        
        # 检测具体变化
        if changes["has_changes"]:
            logger.info(f"📋 CHANGES DETECTED!")
            logger.info(f"  - Old version: {old_version[:8]}...")
            logger.info(f"  - New version: {current_version[:8]}...")
            
            # 找出新增和修改的文件
            for file_path, file_info in current_files_info.items():
                if file_path not in old_files_info:
                    changes["files_added"].append(file_path)
                    logger.info(f"  📄 ADDED: {file_info['path']}")
                elif file_info["hash"] != old_files_info[file_path].get("hash", ""):
                    changes["files_modified"].append(file_path)
                    logger.info(f"  ✏️ MODIFIED: {file_info['path']}")
            
            # 找出删除的文件
            for file_path in old_files_info:
                if file_path not in current_files_info:
                    changes["files_deleted"].append(file_path)
                    logger.info(f"  🗑️ DELETED: {old_files_info[file_path]['path']}")
            
            logger.info(f"📊 CHANGE SUMMARY:")
            logger.info(f"  - Added: {len(changes['files_added'])}")
            logger.info(f"  - Modified: {len(changes['files_modified'])}")
            logger.info(f"  - Deleted: {len(changes['files_deleted'])}")
        else:
            logger.info(f"✅ NO CHANGES DETECTED")
            logger.info(f"  - Current version: {current_version[:8]}...")
        
        # 更新版本信息
        new_version_info = {
            "version_hash": current_version,
            "last_update": datetime.now().isoformat(),
            "files_count": len(current_files_info),
            "files_info": current_files_info
        }
        
        return changes, new_version_info
    
    async def load_all_data(self) -> List[Any]:
        """Load all travel knowledge data from the documents directory"""
        start_time = time.time()
        stats = DataLoadStats()
        all_knowledge = []
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory does not exist: {self.documents_dir}")
            return []
        
        # 🔧 检测文档变化
        changes, new_version_info = self.detect_changes()
        
        # 🔧 DEBUG: Log directory structure scan
        logger.info(f"🔍 STARTING DOCUMENT SCAN: {self.documents_dir}")
        logger.info(f"  - Force reload due to changes: {changes['has_changes']}")
        
        # Recursively find all supported files
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                stats.total_files += 1
                
                # 🔧 DEBUG: Log each file being processed
                logger.info(f"📄 PROCESSING FILE: {file_path}")
                logger.info(f"  - File size: {file_path.stat().st_size} bytes")
                logger.info(f"  - File extension: {file_path.suffix}")
                
                try:
                    knowledge_items = await self._load_file(file_path)
                    all_knowledge.extend(knowledge_items)
                    stats.successful_files += 1
                    stats.total_knowledge_items += len(knowledge_items)
                    
                    # 🔧 DEBUG: Log successful loading details
                    logger.info(f"✅ LOADED {len(knowledge_items)} items from {file_path.name}")
                    for i, item in enumerate(knowledge_items):
                        if hasattr(item, 'id') and hasattr(item, 'title'):
                            logger.info(f"  - Item {i+1}: ID='{item.id}', Title='{item.title}'")
                            if hasattr(item, 'location'):
                                logger.info(f"    Location: {item.location}")
                            if hasattr(item, 'category'):
                                logger.info(f"    Category: {item.category}")
                    
                    # 🔧 DEBUG: Special attention to Berlin content
                    for item in knowledge_items:
                        if hasattr(item, 'location') and item.location and 'berlin' in item.location.lower():
                            logger.info(f"🏛️ BERLIN CONTENT FOUND: {item.id}")
                            logger.info(f"  - Title: {item.title}")
                            logger.info(f"  - Content preview: {item.content[:200]}...")
                        elif hasattr(item, 'title') and 'berlin' in item.title.lower():
                            logger.info(f"🏛️ BERLIN CONTENT FOUND (by title): {item.id}")
                            logger.info(f"  - Title: {item.title}")
                        elif hasattr(item, 'content') and 'berlin' in item.content.lower():
                            logger.info(f"🏛️ BERLIN CONTENT FOUND (in content): {item.id}")
                            logger.info(f"  - Title: {item.title}")
                    
                except Exception as e:
                    stats.failed_files += 1
                    error_msg = f"Failed to load {file_path}: {str(e)}"
                    stats.errors.append(error_msg)
                    
                    # 🔧 DEBUG: Log loading failures
                    logger.error(f"❌ FAILED TO LOAD: {file_path}")
                    logger.error(f"  - Error: {str(e)}")
                    logger.error(f"  - Error type: {type(e).__name__}")
        
        # Calculate load time
        stats.load_time = time.time() - start_time
        
        # 🔧 保存新的版本信息
        if changes["has_changes"]:
            logger.info(f"💾 SAVING NEW VERSION INFO...")
            self._save_version_info(new_version_info)
            self.last_known_version = new_version_info
            logger.info(f"  - New version saved: {new_version_info['version_hash'][:8]}...")
        
        # 🔧 DEBUG: Comprehensive loading summary
        logger.info(f"📊 DOCUMENT LOADING SUMMARY:")
        logger.info(f"  - Total files scanned: {stats.total_files}")
        logger.info(f"  - Successfully loaded: {stats.successful_files}")
        logger.info(f"  - Failed to load: {stats.failed_files}")
        logger.info(f"  - Total knowledge items: {stats.total_knowledge_items}")
        logger.info(f"  - Load time: {stats.load_time:.2f}s")
        logger.info(f"  - Version changes: {changes['has_changes']}")
        
        # 🔧 DEBUG: Check for specific destinations
        destinations_found = set()
        for item in all_knowledge:
            if hasattr(item, 'location') and item.location:
                destinations_found.add(item.location)
        
        logger.info(f"🌍 DESTINATIONS FOUND: {sorted(destinations_found)}")
        if 'Berlin' not in destinations_found:
            logger.warning(f"⚠️ BERLIN NOT FOUND in loaded destinations!")
        
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
            # 🔧 DEBUG: Log file reading attempt
            logger.debug(f"📖 READING FILE: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Parse based on file format
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # 🔧 DEBUG: Log parsed data structure
            logger.debug(f"📋 PARSED DATA STRUCTURE:")
            logger.debug(f"  - Data type: {type(data)}")
            if isinstance(data, dict):
                logger.debug(f"  - Dict keys: {list(data.keys())}")
                if 'title' in data:
                    logger.debug(f"  - Title: {data['title']}")
                if 'location' in data:
                    logger.debug(f"  - Location: {data['location']}")
            elif isinstance(data, list):
                logger.debug(f"  - List length: {len(data)}")
            
            # Process data structure
            return await self._process_data(data, file_path)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path}: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            logger.error(f"File reading error in {file_path}: {e}")
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
            # 🔧 DEBUG: Log knowledge item creation
            logger.debug(f"🏗️ CREATING KNOWLEDGE ITEM: {file_path}[{item_index}]")
            logger.debug(f"  - Raw data keys: {list(data.keys())}")
            
            # Validate required fields
            required_fields = ['id', 'title', 'content', 'category']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                logger.warning(f"Missing required fields in {file_path}[{item_index}]: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # 🔧 DEBUG: Log field values
            logger.debug(f"  - ID: {data.get('id', 'MISSING')}")
            logger.debug(f"  - Title: {data.get('title', 'MISSING')}")
            logger.debug(f"  - Category: {data.get('category', 'MISSING')}")
            logger.debug(f"  - Location: {data.get('location', 'MISSING')}")
            logger.debug(f"  - Content length: {len(str(data.get('content', '')))}")
            
            # Auto-generate ID if not provided or invalid
            if not data.get('id') or not isinstance(data['id'], str):
                data['id'] = self._generate_id(data, file_path, item_index)
                logger.debug(f"  - Generated ID: {data['id']}")
            
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
            
            # 🔧 DEBUG: Log created knowledge object
            logger.debug(f"✅ KNOWLEDGE ITEM CREATED: {knowledge.id}")
            logger.debug(f"  - Final title: {knowledge.title}")
            logger.debug(f"  - Final location: {knowledge.location}")
            logger.debug(f"  - Final category: {knowledge.category}")
            logger.debug(f"  - Tags: {knowledge.tags}")
            
            # Validate knowledge item
            await self._validate_knowledge_item(knowledge)
            
            return knowledge
            
        except ValidationError as e:
            logger.error(f"Validation error for item in {file_path}[{item_index}]: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating knowledge item from {file_path}[{item_index}]: {e}")
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