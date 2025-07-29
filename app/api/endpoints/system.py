"""
System Endpoints - Health checks and system status
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from app.tools.base_tool import tool_registry
from app.agents.base_agent import agent_manager
from datetime import datetime
from app.tools.flight_search import FlightSearchInput, FlightSearchOutput

router = APIRouter()

@router.get("/")
async def root():
    """Root path - system status overview"""
    try:
        # Get status of each component
        tool_status = tool_registry.get_registry_status()
        agent_status = agent_manager.get_system_status()
        
        return {
            "message": "Welcome to AI Travel Planning Agent v2.0",
            "status": "Running",
            "architecture": "Six-layer AI Agent architecture",
            "components": {
                "tools": {
                    "total": tool_status["total_tools"],
                    "categories": len(tool_status.get("categories", {})),
                    "active": len([tool for tool in tool_status.get("tools", {}).values() 
                                 if hasattr(tool, 'get') and tool.get("status") != "error"])
                },
                "agents": {
                    "total": agent_status["total_agents"],
                    "active": len([agent for agent in agent_status.get("agents", {}).values() 
                                 if hasattr(agent, 'get') and agent.get("status") != "stopped"])
                },
                "knowledge_base": "Loaded",
                "memory_system": "Active"
            },
            "capabilities": [
                "Intelligent travel planning",
                "Multi-tool coordination",
                "Retrieval-augmented generation",
                "Conversation memory management",
                "Self-learning and improvement"
            ]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.0.0"
    }

@router.get("/system/status")
async def system_status():
    """Detailed system status"""
    try:
        from app.memory.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        return {
            "tools": tool_registry.get_registry_status(),
            "agents": agent_manager.get_system_status(),
            "memory": {
                "conversation_memory": "active",
                "session_manager": "active",
                "session_stats": session_manager.get_session_stats()
            },
            "system_info": {
                "version": "2.0.0",
                "architecture": "Six-layer AI Agent architecture",
                "uptime": "active"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve system status: {str(e)}"}
        )

@router.get("/system/chromadb")
async def chromadb_status():
    """Get ChromaDB status and statistics"""
    try:
        from app.core.rag_engine import get_rag_engine
        rag_engine = get_rag_engine()
        
        # Get vector store stats
        vector_stats = rag_engine.vector_store.get_stats()
        
        # Get collection information
        client = rag_engine.vector_store.client
        collections = client.list_collections()
        
        collection_info = []
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.metadata or {}
                collection_info.append({
                    "name": collection.name,
                    "document_count": count,
                    "metadata": metadata
                })
            except Exception as e:
                collection_info.append({
                    "name": collection.name,
                    "error": str(e)
                })
        
        return {
            "chromadb_path": "./data/chroma_db",
            "total_collections": len(collections),
            "collections": collection_info,
            "vector_store_stats": vector_stats
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve ChromaDB status: {str(e)}"}
        )

@router.get("/system/chromadb/search")
async def search_chromadb(query: str, collection_name: str = "travel_knowledge", top_k: int = 5):
    """Search ChromaDB with a query"""
    try:
        from app.core.rag_engine import get_rag_engine
        rag_engine = get_rag_engine()
        
        # Get collection
        try:
            collection = rag_engine.vector_store.client.get_collection(name=collection_name)
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"error": f"Collection '{collection_name}' not found: {str(e)}"}
            )
        
        # Perform text search
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return {
                "query": query,
                "collection": collection_name,
                "results": [],
                "total_found": 0
            }
        
        # Process results
        processed_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_content, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'] else [],
                results['distances'][0] if results['distances'] else []
            )):
                similarity = 1 - distance
                processed_results.append({
                    "rank": i + 1,
                    "similarity_score": round(similarity, 3),
                    "content_preview": doc_content[:300] + "..." if len(doc_content) > 300 else doc_content,
                    "metadata": metadata or {}
                })
        
        return {
            "query": query,
            "collection": collection_name,
            "results": processed_results,
            "total_found": len(processed_results)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to search ChromaDB: {str(e)}"}
        ) 

@router.post("/api/tools/flight_search", response_model=FlightSearchOutput)
async def run_flight_search(request: Request):
    """Directly invoke the flight_search tool with input payload."""
    try:
        payload = await request.json()
        tool = tool_registry.get_tool("flight_search")
        if not tool:
            raise HTTPException(status_code=404, detail="flight_search tool not found")
        input_data = FlightSearchInput(**payload)
        # Use a dummy context for now
        from app.tools.base_tool import ToolExecutionContext
        context = ToolExecutionContext(request_id="api_flight_search")
        result = await tool.execute(input_data, context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flight search failed: {str(e)}") 