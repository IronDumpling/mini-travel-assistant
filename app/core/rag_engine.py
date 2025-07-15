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
            
            # 🔧 DEBUG: Log document addition process
            logger.info(f"🗂️ ADDING DOCUMENTS TO VECTOR STORE:")
            logger.info(f"  - Collection: {self.collection_name}")
            logger.info(f"  - Documents to add: {len(documents)}")
            
            # Prepare data for batch insertion
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            
            # 🔧 DEBUG: Log document details
            for i, doc in enumerate(documents):
                logger.info(f"📄 DOC {i+1}: {doc.id}")
                logger.info(f"  - Content length: {len(doc.content)}")
                logger.info(f"  - Content preview: {doc.content[:100]}...")
                logger.info(f"  - Metadata: {doc.metadata}")
                logger.info(f"  - Doc type: {doc.doc_type}")
                
                # 🔧 DEBUG: Special attention to Berlin content
                if 'berlin' in doc.content.lower() or 'berlin' in str(doc.metadata).lower():
                    logger.info(f"🏛️ BERLIN DOCUMENT DETECTED: {doc.id}")
                    logger.info(f"  - Title in metadata: {doc.metadata.get('title', 'N/A')}")
                    logger.info(f"  - Location in metadata: {doc.metadata.get('location', 'N/A')}")
            
            # Convert list values in metadata to strings for ChromaDB compatibility
            metadatas = []
            for doc in documents:
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
            
            # 🔧 DEBUG: Log embedding generation start
            logger.info(f"🧠 GENERATING EMBEDDINGS...")
            logger.info(f"  - Embedding model: {embedding_model.model_name if hasattr(embedding_model, 'model_name') else 'Unknown'}")
            logger.info(f"  - Number of texts: {len(contents)}")
            
            # Generate embeddings using provided embedding model
            embeddings = await embedding_model.encode(contents)
            
            # 🔧 DEBUG: Log embedding results
            logger.info(f"✅ EMBEDDINGS GENERATED:")
            logger.info(f"  - Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
            logger.info(f"  - Total embeddings: {len(embeddings)}")
            
            # Log sample embedding values for Berlin content
            for i, doc in enumerate(documents):
                if 'berlin' in doc.content.lower():
                    logger.info(f"🏛️ BERLIN EMBEDDING SAMPLE: {doc.id}")
                    logger.info(f"  - First 5 dims: {embeddings[i][:5]}")
                    logger.info(f"  - Embedding magnitude: {sum(x*x for x in embeddings[i])**0.5:.4f}")
            
            # Check for existing documents
            try:
                existing_docs = self.collection.get(ids=ids)
                existing_ids = set(existing_docs['ids'] if existing_docs['ids'] else [])
                logger.info(f"🔍 EXISTING DOCUMENTS CHECK: {len(existing_ids)} found")
            except Exception:
                existing_ids = set()
                logger.info(f"🔍 NO EXISTING DOCUMENTS FOUND")
            
            # Separate new and existing documents
            new_docs = []
            new_embeddings = []
            new_metadatas = []
            new_ids = []
            
            update_docs = []
            update_embeddings = []
            update_metadatas = []
            update_ids = []
            
            for i, doc_id in enumerate(ids):
                if doc_id in existing_ids:
                    update_ids.append(doc_id)
                    update_docs.append(contents[i])
                    update_embeddings.append(embeddings[i])
                    update_metadatas.append(metadatas[i])
                else:
                    new_ids.append(doc_id)
                    new_docs.append(contents[i])
                    new_embeddings.append(embeddings[i])
                    new_metadatas.append(metadatas[i])
            
            # 🔧 DEBUG: Log document categorization
            logger.info(f"📊 DOCUMENT CATEGORIZATION:")
            logger.info(f"  - New documents: {len(new_docs)}")
            logger.info(f"  - Documents to update: {len(update_docs)}")
            
            # Add new documents
            if new_docs:
                self.collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    embeddings=new_embeddings,
                    metadatas=new_metadatas
                )
                logger.info(f"✅ ADDED {len(new_docs)} new documents to collection")
                
                # 🔧 DEBUG: Log new Berlin documents specifically
                for i, doc_id in enumerate(new_ids):
                    if 'berlin' in new_docs[i].lower():
                        logger.info(f"🏛️ NEW BERLIN DOC ADDED: {doc_id}")
            
            # Update existing documents
            if update_docs:
                self.collection.update(
                    ids=update_ids,
                    documents=update_docs,
                    embeddings=update_embeddings,
                    metadatas=update_metadatas
                )
                logger.info(f"✅ UPDATED {len(update_docs)} existing documents")
                
                # 🔧 DEBUG: Log updated Berlin documents specifically
                for i, doc_id in enumerate(update_ids):
                    if 'berlin' in update_docs[i].lower():
                        logger.info(f"🏛️ BERLIN DOC UPDATED: {doc_id}")
            
            # 🔧 DEBUG: Log final collection stats
            final_count = self.collection.count()
            logger.info(f"📈 FINAL COLLECTION STATS:")
            logger.info(f"  - Total documents in collection: {final_count}")
            
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
            # 🔧 DEBUG: Log search operation start
            logger.info(f"🔍 VECTOR SEARCH OPERATION:")
            logger.info(f"  - Collection: {self.collection_name}")
            logger.info(f"  - Query embedding dims: {len(query_embedding)}")
            logger.info(f"  - Query embedding sample: {query_embedding[:5]}")
            logger.info(f"  - Top K: {top_k}")
            logger.info(f"  - Filter metadata: {filter_metadata}")
            
            # Build where clause for filtering
            where_clause = {}
            if filter_metadata:
                for key, value in filter_metadata.items():
                    where_clause[key] = {"$eq": value}
                logger.info(f"  - Where clause: {where_clause}")
            
            # 🔧 DEBUG: Log collection status before search
            collection_count = self.collection.count()
            logger.info(f"  - Total documents in collection: {collection_count}")
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 🔧 DEBUG: Log raw search results
            logger.info(f"📊 RAW SEARCH RESULTS:")
            if results['documents'] and results['documents'][0]:
                logger.info(f"  - Results found: {len(results['documents'][0])}")
                logger.info(f"  - Distances: {results['distances'][0] if results['distances'] else 'N/A'}")
                
                # Log details of each result
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    similarity_score = 1 - distance
                    logger.info(f"  📄 RESULT {i+1}:")
                    logger.info(f"    - Distance: {distance:.4f}")
                    logger.info(f"    - Similarity: {similarity_score:.4f}")
                    logger.info(f"    - Metadata: {metadata}")
                    logger.info(f"    - Content preview: {doc_text[:100]}...")
                    
                    # 🔧 DEBUG: Check if result contains Berlin content
                    if 'berlin' in doc_text.lower():
                        logger.info(f"    🏛️ BERLIN CONTENT IN RESULT!")
                    else:
                        logger.warning(f"    ⚠️ NON-BERLIN CONTENT: {metadata.get('location', 'Unknown location')}")
            else:
                logger.warning(f"  - NO RESULTS FOUND!")
            
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
            
            # 🔧 DEBUG: Log final processed results
            logger.info(f"✅ PROCESSED SEARCH RESULTS:")
            logger.info(f"  - Documents returned: {len(documents_with_scores)}")
            for i, (doc, score) in enumerate(documents_with_scores):
                logger.info(f"  📄 FINAL RESULT {i+1}: Score={score:.4f}, Location={doc.metadata.get('location', 'Unknown')}")
            
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
        doc_type: Optional[DocumentType] = None
    ) -> RetrievalResult:
        """Retrieve relevant documents based on query with optional type filtering"""
        try:
            # 🔧 DEBUG: Log retrieval operation start
            logger.info(f"🎯 RETRIEVAL OPERATION START:")
            logger.info(f"  - Query: '{query}'")
            logger.info(f"  - Top K: {top_k}")
            logger.info(f"  - Filter metadata: {filter_metadata}")
            logger.info(f"  - Doc type filter: {doc_type}")
            
            # Ensure embedding model is initialized
            await self._ensure_embedding_initialized()
            
            # 🔧 DEBUG: Log query analysis
            if 'berlin' in query.lower():
                logger.info(f"🏛️ BERLIN QUERY DETECTED!")
            else:
                logger.info(f"🌍 NON-BERLIN QUERY: Checking for other locations...")
                for location in ['tokyo', 'japan', 'london', 'paris', 'rome']:
                    if location in query.lower():
                        logger.info(f"  - Found {location.upper()} in query")
            
            # 1. Encode query to vector
            logger.info(f"🧠 ENCODING QUERY TO VECTOR...")
            query_embedding = await self.embedding_model.encode([query])
            
            # 🔧 DEBUG: Log query embedding details
            logger.info(f"✅ QUERY EMBEDDING GENERATED:")
            logger.info(f"  - Embedding dims: {len(query_embedding[0])}")
            logger.info(f"  - First 5 dims: {query_embedding[0][:5]}")
            logger.info(f"  - Embedding magnitude: {sum(x*x for x in query_embedding[0])**0.5:.4f}")
            
            # 2. Add document type filter if specified
            if doc_type:
                if filter_metadata is None:
                    filter_metadata = {}
                filter_metadata["doc_type"] = doc_type.value
                logger.info(f"🔧 ADDED DOC TYPE FILTER: {doc_type.value}")
            
            # 3. Search for similar documents
            logger.info(f"🔍 PERFORMING VECTOR SEARCH...")
            search_results = await self.vector_store.search(
                query_embedding=query_embedding[0],
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # 4. Process and rank results
            documents = []
            scores = []
            
            # 🔧 DEBUG: Log result processing
            logger.info(f"📊 PROCESSING SEARCH RESULTS:")
            logger.info(f"  - Raw results count: {len(search_results)}")
            
            for i, (doc, score) in enumerate(search_results):
                documents.append(doc)
                scores.append(score)
                
                # 🔧 DEBUG: Log each processed result
                logger.info(f"  📄 PROCESSING RESULT {i+1}:")
                logger.info(f"    - Score: {score:.4f}")
                logger.info(f"    - Document ID: {doc.id}")
                logger.info(f"    - Metadata location: {doc.metadata.get('location', 'Unknown')}")
                logger.info(f"    - Content preview: {doc.content[:100]}...")
                
                # 🔧 DEBUG: Score analysis
                if score < 0.3:
                    logger.warning(f"    ⚠️ LOW SIMILARITY SCORE: {score:.4f}")
                elif score < 0.5:
                    logger.warning(f"    📉 MEDIUM SIMILARITY SCORE: {score:.4f}")
                else:
                    logger.info(f"    ✅ GOOD SIMILARITY SCORE: {score:.4f}")
                
                # 🔧 DEBUG: Content-query mismatch analysis
                if 'berlin' in query.lower() and 'berlin' not in doc.content.lower():
                    logger.warning(f"    🚨 QUERY-RESULT MISMATCH!")
                    logger.warning(f"    - Query contains 'berlin' but result doesn't")
                    logger.warning(f"    - Result location: {doc.metadata.get('location', 'Unknown')}")
                    logger.warning(f"    - Result title: {doc.metadata.get('title', 'Unknown')}")
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                total_results=len(documents)
            )
            
            # 🔧 DEBUG: Log final retrieval summary
            logger.info(f"🎯 RETRIEVAL OPERATION COMPLETE:")
            logger.info(f"  - Query: '{query[:50]}...'")
            logger.info(f"  - Documents retrieved: {len(documents)}")
            logger.info(f"  - Score range: {min(scores) if scores else 0:.4f} - {max(scores) if scores else 0:.4f}")
            
            # 🔧 DEBUG: Final Berlin-specific analysis
            if 'berlin' in query.lower():
                berlin_results = sum(1 for doc in documents if 'berlin' in doc.content.lower())
                logger.info(f"🏛️ BERLIN QUERY ANALYSIS:")
                logger.info(f"  - Berlin results found: {berlin_results}/{len(documents)}")
                if berlin_results == 0:
                    logger.error(f"🚨 BERLIN QUERY RETURNED NO BERLIN CONTENT!")
                    logger.error(f"  - This indicates a serious RAG retrieval issue")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ RETRIEVAL FAILED: {e}")
            logger.error(f"  - Error type: {type(e).__name__}")
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
        # 🔧 DEBUG: Log chunking operation start
        logger.info(f"✂️ DOCUMENT CHUNKING OPERATION:")
        logger.info(f"  - Input documents: {len(documents)}")
        
        chunked_docs = []
        
        for doc_idx, doc in enumerate(documents):
            # 🔧 DEBUG: Log document being chunked
            logger.debug(f"📄 CHUNKING DOC {doc_idx+1}: {doc.id}")
            logger.debug(f"  - Original content length: {len(doc.content)}")
            logger.debug(f"  - Doc type: {doc.doc_type}")
            
            # Split content into paragraphs
            paragraphs = doc.content.split('\n\n')
            logger.debug(f"  - Paragraphs found: {len(paragraphs)}")
            
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
                        
                        # 🔧 DEBUG: Log chunk creation
                        logger.debug(f"  ✂️ CREATED CHUNK {chunk_id}: {len(current_chunk.strip())} chars")
                        if 'berlin' in current_chunk.lower():
                            logger.info(f"  🏛️ BERLIN CHUNK CREATED: {chunk_doc.id}")
                            logger.info(f"    - Content preview: {current_chunk.strip()[:100]}...")
                        
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
                
                # 🔧 DEBUG: Log final chunk
                logger.debug(f"  ✂️ FINAL CHUNK {chunk_id}: {len(current_chunk.strip())} chars")
                if 'berlin' in current_chunk.lower():
                    logger.info(f"  🏛️ BERLIN FINAL CHUNK: {final_chunk.id}")
        
        # 🔧 DEBUG: Log chunking summary
        logger.info(f"✅ CHUNKING COMPLETE:")
        logger.info(f"  - Input documents: {len(documents)}")
        logger.info(f"  - Output chunks: {len(chunked_docs)}")
        logger.info(f"  - Avg chunks per doc: {len(chunked_docs)/len(documents):.1f}")
        
        # Count Berlin chunks
        berlin_chunks = sum(1 for chunk in chunked_docs if 'berlin' in chunk.content.lower())
        if berlin_chunks > 0:
            logger.info(f"  🏛️ Berlin chunks created: {berlin_chunks}")
        
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


# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get RAG engine instance (singleton pattern)"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine 