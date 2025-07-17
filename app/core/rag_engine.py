"""
RAG Engine Module - Optimized Retrieval-Augmented Generation Engine

Implements the core RAG functionality with ChromaDB persistence, 
document chunking, embedding, and semantic search capabilities.
Optimized for different use cases: travel knowledge, conversation memory, and tool selection.
"""

from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import asyncio
from datetime import datetime
import tiktoken
from enum import Enum
from app.core.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class DocumentType(Enum):
    """Document type enumeration for better organization"""
    TRAVEL_KNOWLEDGE = "travel_knowledge"
    CONVERSATION_TURN = "conversation_turn"
    TOOL_KNOWLEDGE = "tool_knowledge"
    GENERAL = "general"


class Document(BaseModel):
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    doc_type: DocumentType = DocumentType.GENERAL


class RetrievalResult(BaseModel):
    """Retrieval result structure"""
    documents: List[Document]
    scores: List[float]
    query: str
    total_results: int


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to vectors"""
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    """SentenceTransformer embedding model implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding model"""
        if not self._initialized:
            try:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer(self.model_name)
                )
                self._initialized = True
                logger.info(f"Embedding model '{self.model_name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to vectors"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Run encoding in thread pool for better performance
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: self.model.encode(texts, convert_to_numpy=True)
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise


class ChromaVectorStore:
    """ChromaDB vector store with persistence and multi-collection support"""
    
    def __init__(self, collection_name: str = "travel_knowledge"):
        self.collection_name = collection_name
        
        # Configure persistent storage
        db_path = Path("./data/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Get or create collection with optimized settings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",      # Use cosine similarity
                "hnsw:M": 16,                # HNSW parameter optimization
                "hnsw:construction_ef": 200, # Build-time search parameter
                "hnsw:search_ef": 100        # Query-time search parameter
            }
        )
        
        logger.info(f"ChromaDB initialized at {db_path}")
        logger.info(f"Collection '{collection_name}' loaded with {self.collection.count()} documents")
    
    async def add_documents(self, documents: List[Document], embedding_model: BaseEmbeddingModel) -> bool:
        """Add documents to vector database using provided embedding model"""
        try:
            if not documents:
                return True
            
            # Check for existing documents and separate new vs existing
            existing_ids = set()
            try:
                # Get all existing document IDs in the collection
                existing_result = self.collection.get(include=[])
                if existing_result and existing_result.get('ids'):
                    existing_ids = set(existing_result['ids'])
            except Exception as e:
                logger.warning(f"Could not check existing documents: {e}")
                # Continue with update/upsert approach
            
            # Separate documents into new and existing
            new_documents = []
            existing_documents = []
            
            for doc in documents:
                if doc.id in existing_ids:
                    existing_documents.append(doc)
                else:
                    new_documents.append(doc)
            
            # Process new documents
            if new_documents:
                ids = [doc.id for doc in new_documents]
                contents = [doc.content for doc in new_documents]
                
                # Convert list values in metadata to strings for ChromaDB compatibility
                metadatas = []
                for doc in new_documents:
                    processed_metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, list):
                            # Convert list to comma-separated string
                            processed_metadata[key] = ", ".join(str(v) for v in value)
                        elif value is not None:
                            processed_metadata[key] = str(value)
                    # Add document type to metadata
                    processed_metadata["doc_type"] = doc.doc_type.value
                    metadatas.append(processed_metadata)
                
                # Generate embeddings using provided embedding model
                embeddings = await embedding_model.encode(contents)
                
                # Add new documents
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(new_documents)} new documents to collection")
            
            # Process existing documents (update)
            if existing_documents:
                # Delete existing documents first
                existing_ids_to_update = [doc.id for doc in existing_documents]
                self.collection.delete(ids=existing_ids_to_update)
                
                # Re-add updated documents
                ids = [doc.id for doc in existing_documents]
                contents = [doc.content for doc in existing_documents]
                
                # Convert list values in metadata to strings for ChromaDB compatibility
                metadatas = []
                for doc in existing_documents:
                    processed_metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, list):
                            # Convert list to comma-separated string
                            processed_metadata[key] = ", ".join(str(v) for v in value)
                        elif value is not None:
                            processed_metadata[key] = str(value)
                    # Add document type to metadata
                    processed_metadata["doc_type"] = doc.doc_type.value
                    metadatas.append(processed_metadata)
                
                # Generate embeddings using provided embedding model
                embeddings = await embedding_model.encode(contents)
                
                # Add updated documents
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Updated {len(existing_documents)} existing documents in collection")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FAILED to add documents to vector store: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform vector similarity search"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if filter_metadata:
                for key, value in filter_metadata.items():
                    where_clause[key] = {"$eq": value}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            logger.debug(f"Vector search returned {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0} results")
            
            # Process results
            documents_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity_score = 1 - distance
                    
                    document = Document(
                        id=f"retrieved_{i}",
                        content=doc_text,
                        metadata=metadata or {}
                    )
                    
                    documents_with_scores.append((document, similarity_score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"❌ SEARCH FAILED: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector database"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def clear_documents_by_type(self, doc_type: 'DocumentType') -> bool:
        """Clear all documents of a specific type from vector database"""
        try:
            logger.info(f"🗑️ CLEARING DOCUMENTS BY TYPE: {doc_type.value}")
            
            try:
                # Query documents by type
                result = self.collection.get(
                    where={"doc_type": {"$eq": doc_type.value}},
                    include=['metadatas']
                )
                
                if result['ids']:
                    logger.info(f"  - Found {len(result['ids'])} documents of type {doc_type.value}")
                    self.collection.delete(ids=result['ids'])
                    logger.info(f"✅ CLEARED {len(result['ids'])} documents of type {doc_type.value}")
                else:
                    logger.info(f"  - No documents found with type {doc_type.value}")
                
                return True
                
            except Exception as e:
                # ✅ SAFE: Fail gracefully instead of destructive fallback
                logger.error(f"❌ Failed to clear documents by type {doc_type.value}: {e}")
                logger.error(f"  - Operation aborted to prevent data loss")
                return False  # Return failure instead of destroying all data
                
        except Exception as e:
            logger.error(f"❌ FAILED to clear documents by type {doc_type.value}: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


class RAGEngine:
    """Main RAG retrieval engine with optimized multi-use-case support"""
    
    def __init__(
        self, 
        embedding_model: BaseEmbeddingModel = None,
        vector_store: ChromaVectorStore = None
    ):
        self.embedding_model = embedding_model or SentenceTransformerModel()
        self.vector_store = vector_store or ChromaVectorStore()
        self._embedding_initialized = False
        logger.info("RAG Engine initialized")
    
    async def _ensure_embedding_initialized(self):
        """Ensure embedding model is initialized (lazy initialization)"""
        if not self._embedding_initialized:
            await self.embedding_model.initialize()
            self._embedding_initialized = True
    
    async def index_documents(self, documents: List[Document]) -> bool:
        """Index documents with chunking and embedding"""
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return True
            
            # Ensure embedding model is initialized
            await self._ensure_embedding_initialized()
            
            # 1. Chunk documents for better retrieval
            chunked_docs = self._chunk_documents(documents)
            logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
            
            # 2. Store documents in vector database using the shared embedding model
            success = await self.vector_store.add_documents(chunked_docs, self.embedding_model)
            
            if success:
                logger.info(f"Successfully indexed {len(chunked_docs)} document chunks")
            else:
                logger.error("Failed to index documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return False
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        doc_type: Optional[DocumentType] = None,
        structured_intent: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant documents with enhanced multi-destination and intent support"""
        try:
            # Ensure embedding model is initialized
            await self._ensure_embedding_initialized()
            
            # Extract destinations from structured intent (preferred) or query parsing
            target_destinations = []
            
            if structured_intent:
                # Extract destinations from LLM-analyzed structured intent
                destination_info = structured_intent.get("destination", {})
                
                if isinstance(destination_info, dict):
                    primary_dest = destination_info.get("primary")
                    secondary_dests = destination_info.get("secondary", [])
                    
                    if primary_dest and primary_dest != "Unknown":
                        target_destinations.append(primary_dest)
                    
                    if isinstance(secondary_dests, list):
                        for dest in secondary_dests:
                            if dest and dest not in target_destinations:
                                target_destinations.append(dest)
                
            else:
                # Fallback: Use query parsing for location detection
                target_destinations = self._detect_query_locations(query)
            
            # Smart routing: Choose retrieval strategy based on destination count
            if len(target_destinations) >= 2:
                # Multi-destination query - use enhanced multi-destination retrieval
                return await self.retrieve_multi_destination(
                    query=query,
                    destinations=target_destinations,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    doc_type=doc_type
                )
                
            elif len(target_destinations) == 1:
                # Single destination query - use enhanced single destination retrieval
                target_location = target_destinations[0]
                
            else:
                # No specific destinations detected - use general retrieval
                target_location = None
            
            # 1. Encode query to vector
            query_embedding = await self.embedding_model.encode([query])
            
            # 2. Add document type filter if specified
            if doc_type:
                if filter_metadata is None:
                    filter_metadata = {}
                filter_metadata["doc_type"] = doc_type.value
            
            # 3. Search for similar documents (get more candidates for filtering)
            extended_top_k = min(top_k * 2, 15)  # Get more candidates for smart filtering
            search_results = await self.vector_store.search(
                query_embedding=query_embedding[0],
                top_k=extended_top_k,
                filter_metadata=filter_metadata
            )
            
            # 4. Apply smart filtering based on destination detection
            if target_location:
                filtered_results = self._apply_location_smart_filtering(
                    search_results, target_location, top_k, query
                )
            else:
                # For non-location queries, use similarity threshold
                filtered_results = self._apply_similarity_filtering(
                    search_results, top_k, min_similarity=0.3
                )
            
            # 5. Process and rank results
            documents = []
            scores = []
            
            for i, (doc, score) in enumerate(filtered_results):
                documents.append(doc)
                scores.append(score)
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                total_results=len(documents)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                query=query,
                total_results=0
            )
    
    async def retrieve_by_type(
        self, 
        query: str, 
        doc_type: DocumentType,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Convenience method to retrieve documents by type"""
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            doc_type=doc_type
        )
    
    async def retrieve_and_generate(
        self, 
        query: str, 
        context_template: str = None,
        **llm_kwargs
    ) -> str:
        """Complete RAG pipeline: retrieve and generate answer"""
        try:
            # 1. Retrieve relevant documents
            retrieval_result = await self.retrieve(query, top_k=5)
            
            # 2. Build context from retrieved documents
            context = self._compress_context(retrieval_result.documents)
            
            # 3. Build prompt template using prompt manager
            if not context_template:
                from app.core.prompt_manager import prompt_manager, PromptType
                context_template = prompt_manager.get_prompt(PromptType.RAG_GENERATION)
            
            prompt = context_template.format(context=context, query=query)
            
            # 4. Call LLM service to generate answer
            from app.core.llm_service import get_llm_service
            llm_service = get_llm_service()
            
            # TODO: Replace with actual LLM service call when available
            try:
                response = await llm_service.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    **llm_kwargs
                )
                return response.content
            except Exception as e:
                # Fallback: Generate response based on retrieved context
                logger.warning(f"LLM service unavailable, using fallback response: {e}")
                
                if context:
                    # Create a structured response based on retrieved knowledge
                    fallback_response = f"""Based on the available travel information, here's what I found:

{context}

This information should help answer your question about: {query}

Would you like me to search for more specific information or help you with travel planning?"""
                    return fallback_response
                else:
                    return f"I understand you're asking about: {query}. While I don't have specific information available right now, I'd be happy to help you with travel planning. Could you provide more details about what you're looking for?"
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Intelligent document chunking with overlap"""
        chunked_docs = []
        
        for doc_idx, doc in enumerate(documents):
            # Split content into paragraphs
            paragraphs = doc.content.split('\n\n')
            
            current_chunk = ""
            chunk_id = 0
            max_chunk_size = 1000  # characters
            overlap_size = 200     # characters for overlap
            
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if adding this paragraph would exceed max size
                if len(current_chunk + paragraph) < max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip():
                        chunk_doc = Document(
                            id=f"{doc.id}_chunk_{chunk_id}",
                            content=current_chunk.strip(),
                            metadata={
                                **doc.metadata, 
                                "chunk_id": chunk_id,
                                "parent_id": doc.id,
                                "chunk_type": "text"
                            },
                            doc_type=doc.doc_type
                        )
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap_size:
                        overlap_text = current_chunk[-overlap_size:].split('\n\n')[-1]
                        current_chunk = overlap_text + "\n\n" + paragraph + "\n\n"
                    else:
                        current_chunk = paragraph + "\n\n"
            
            # Save final chunk
            if current_chunk.strip():
                final_chunk = Document(
                    id=f"{doc.id}_chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    metadata={
                        **doc.metadata, 
                        "chunk_id": chunk_id,
                        "parent_id": doc.id,
                        "chunk_type": "text"
                    },
                    doc_type=doc.doc_type
                )
                chunked_docs.append(final_chunk)
        
        return chunked_docs
    
    def _compress_context(self, documents: List[Document], max_tokens: int = 4000) -> str:
        """Compress context to fit within token limits"""
        if not documents:
            return ""
        
        try:
            # Initialize tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")
            
            compressed_context = ""
            total_tokens = 0
            
            for i, doc in enumerate(documents):
                # Calculate tokens for this document
                doc_content = f"\n\n=== Knowledge {i+1} ===\n{doc.content}"
                doc_tokens = len(encoding.encode(doc_content))
                
                # Check if we can add this document
                if total_tokens + doc_tokens <= max_tokens:
                    compressed_context += doc_content
                    total_tokens += doc_tokens
                else:
                    # Try to add partial content if space allows
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 100:  # Minimum useful content
                        # Truncate content to fit
                        truncated_tokens = encoding.encode(doc.content)[:remaining_tokens-50]
                        truncated_content = encoding.decode(truncated_tokens)
                        compressed_context += f"\n\n=== Knowledge {i+1} (truncated) ===\n{truncated_content}..."
                    break
            
            return compressed_context
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            # Fallback: simple truncation
            context = "\n\n".join([f"=== Knowledge {i+1} ===\n{doc.content}" 
                                  for i, doc in enumerate(documents)])
            return context[:max_tokens * 4]  # Rough character approximation

    def _detect_query_location(self, query: str) -> Optional[str]:
        """Detect if query is asking about a specific location"""
        query_lower = query.lower()
        
        # Known locations mapping
        location_keywords = {
            'berlin': 'Berlin',
            'munich': 'Munich',
            'tokyo': 'Tokyo', 
            'japan': 'Tokyo',
            'london': 'London',
            'paris': 'Paris',
            'rome': 'Rome',
            'amsterdam': 'Amsterdam',
            'prague': 'Prague',
            'vienna': 'Vienna',
            'barcelona': 'Barcelona',
            'budapest': 'Budapest',
            'beijing': 'Beijing',
            'shanghai': 'Shanghai',
            'seoul': 'Seoul',
            'singapore': 'Singapore',
            'kyoto': 'Kyoto',
            'china': 'China'
        }
        
        for keyword, location in location_keywords.items():
            if keyword in query_lower:
                return location
        
        return None
    
    def _detect_query_locations(self, query: str) -> List[str]:
        """Detect multiple locations in query (enhanced multi-destination support)"""
        query_lower = query.lower()
        detected_locations = []
        
        # Known locations mapping with aliases
        location_keywords = {
            'berlin': 'Berlin',
            'munich': 'Munich',
            'münchen': 'Munich',
            'tokyo': 'Tokyo', 
            'japan': 'Tokyo',
            'london': 'London',
            'paris': 'Paris',
            'rome': 'Rome',
            'amsterdam': 'Amsterdam',
            'prague': 'Prague',
            'vienna': 'Vienna',
            'barcelona': 'Barcelona',
            'budapest': 'Budapest',
            'beijing': 'Beijing',
            'shanghai': 'Shanghai',
            'seoul': 'Seoul',
            'singapore': 'Singapore',
            'kyoto': 'Kyoto',
            'china': 'China'
        }
        
        # Find ALL matching locations, not just the first one
        for keyword, location in location_keywords.items():
            if keyword in query_lower and location not in detected_locations:
                detected_locations.append(location)
        
        return detected_locations
    
    async def retrieve_multi_destination(
        self, 
        query: str,
        destinations: List[str],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        doc_type: Optional[DocumentType] = None
    ) -> RetrievalResult:
        """Enhanced retrieve method for multi-destination queries"""
        try:
            # Ensure embedding model is initialized
            await self._ensure_embedding_initialized()
            
            # Encode query to vector once
            query_embedding = await self.embedding_model.encode([query])
            query_vector = query_embedding[0]
            
            all_results = []
            destination_results = {}
            
            # Retrieve for each destination separately
            for destination in destinations:
                # Build destination-specific filter
                dest_filter = filter_metadata.copy() if filter_metadata else {}
                
                # Add document type filter if specified
                if doc_type:
                    dest_filter["doc_type"] = doc_type.value
                
                # Perform search for this specific destination
                search_results = await self.vector_store.search(
                    query_embedding=query_vector,
                    top_k=top_k * 2,  # Get more candidates for filtering
                    filter_metadata=dest_filter
                )
                
                # Apply destination-specific filtering
                filtered_results = self._apply_destination_smart_filtering(
                    search_results, destination, top_k, query
                )
                
                destination_results[destination] = filtered_results
                all_results.extend(filtered_results)
            
            # Sort all results by score and take top_k overall
            all_results.sort(key=lambda x: x[1], reverse=True)
            final_results = all_results[:top_k]
            
            # Extract documents and scores
            documents = [doc for doc, score in final_results]
            scores = [score for doc, score in final_results]
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                total_results=len(documents)
            )
            
        except Exception as e:
            logger.error(f"Multi-destination retrieval failed: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                query=query,
                total_results=0
            )
    
    def _apply_destination_smart_filtering(
        self, 
        search_results: List[Tuple], 
        target_destination: str,
        requested_top_k: int,
        query: str
    ) -> List[Tuple]:
        """Apply smart filtering for a specific destination"""
        
        destination_matched = []
        high_quality_others = []
        
        for doc, score in search_results:
            doc_location = doc.metadata.get('location', '').lower()
            doc_content = doc.content.lower()
            target_lower = target_destination.lower()
            
            # Check if document matches the target destination
            is_destination_match = (target_lower in doc_location or 
                                  target_lower in doc_content)
            
            if is_destination_match:
                destination_matched.append((doc, score))
            elif score > 0.6:  # High quality non-destination match
                high_quality_others.append((doc, score))
        
        # Smart decision for destination-specific results
        if len(destination_matched) >= 2:
            # If we have 2+ good destination matches, return mostly those
            final_results = destination_matched[:requested_top_k]
        elif len(destination_matched) == 1:
            # If only 1 destination match, include some high-quality others
            combined = destination_matched + high_quality_others[:requested_top_k-1]
            final_results = combined[:requested_top_k]
        else:
            # No destination matches, use high-quality results but fewer
            final_results = high_quality_others[:max(1, requested_top_k//2)]
        
        return final_results
    
    def _apply_location_smart_filtering(
        self, 
        search_results: List[Tuple], 
        target_location: str,
        requested_top_k: int,
        query: str
    ) -> List[Tuple]:
        """Apply smart filtering for location-specific queries"""
        
        location_matched = []
        high_quality_others = []
        
        for doc, score in search_results:
            doc_location = doc.metadata.get('location', '').lower()
            doc_content = doc.content.lower()
            target_lower = target_location.lower()
            
            # Check if document matches the target location
            is_location_match = (target_lower in doc_location or 
                               target_lower in doc_content)
            
            if is_location_match:
                location_matched.append((doc, score))
            elif score > 0.6:  # High quality non-location match
                high_quality_others.append((doc, score))
        
        # Smart decision on how many to return
        if len(location_matched) >= 2:
            # If we have 2+ good location matches, return mostly those
            final_results = location_matched[:requested_top_k]
        elif len(location_matched) == 1:
            # If only 1 location match, include some high-quality others
            combined = location_matched + high_quality_others[:requested_top_k-1]
            final_results = combined[:requested_top_k]
        else:
            # No location matches, use high-quality results but fewer
            final_results = high_quality_others[:max(2, requested_top_k//2)]
        
        return final_results
    
    def _apply_similarity_filtering(
        self, 
        search_results: List[Tuple], 
        requested_top_k: int,
        min_similarity: float = 0.3
    ) -> List[Tuple]:
        """Apply similarity threshold filtering for non-location queries"""
        
        filtered_results = []
        
        for doc, score in search_results:
            if score >= min_similarity:
                filtered_results.append((doc, score))
                
                if len(filtered_results) >= requested_top_k:
                    break
        
        return filtered_results


# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get RAG engine instance (singleton pattern)"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine 