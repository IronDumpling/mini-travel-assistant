"""
Tool Executor Module - Tool Executor with RAG-Enhanced Tool Selection

Implements intelligent tool selection using RAG (Retrieval-Augmented Generation)
to improve tool selection accuracy based on semantic understanding of user requests.
"""

from typing import Dict, List, Any, Optional, Callable, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from app.tools.base_tool import BaseTool, ToolInput, ToolOutput, ToolExecutionContext, tool_registry
from app.core.llm_service import get_llm_service
from app.core.rag_engine import get_rag_engine, Document
import logging

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Tool call request"""
    tool_name: str
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ToolChain(BaseModel):
    """Tool execution chain"""
    calls: List[ToolCall]
    strategy: str = "sequential"  # sequential, parallel, conditional
    conditions: Optional[Dict[str, Any]] = None


class ExecutionResult(BaseModel):
    """Execution result"""
    success: bool
    results: Dict[str, ToolOutput]
    execution_time: float
    error: Optional[str] = None


class ToolSelector:
    """Enhanced tool selector with RAG-powered tool selection"""
    
    def __init__(self):
        # TODO: Initialize LLM service when available
        try:
            self.llm_service = get_llm_service()
        except Exception as e:
            # For now, work without LLM service
            self.llm_service = None
        
        # Initialize RAG engine for tool selection
        self.rag_engine = get_rag_engine()
        self._tool_knowledge_initialized = False
    
    async def initialize_tool_knowledge(self):
        """Initialize tool knowledge base for RAG-powered selection"""
        if self._tool_knowledge_initialized:
            return
        
        try:
            # Create tool knowledge documents
            tool_knowledge_docs = self._create_tool_knowledge_documents()
            
            # Index tool knowledge in RAG engine
            success = await self.rag_engine.index_documents(tool_knowledge_docs)
            
            if success:
                logger.info(f"Tool knowledge base initialized with {len(tool_knowledge_docs)} documents")
                self._tool_knowledge_initialized = True
            else:
                logger.error("Failed to initialize tool knowledge base")
        
        except Exception as e:
            logger.error(f"Error initializing tool knowledge: {e}")
    
    def _create_tool_knowledge_documents(self) -> List[Document]:
        """Create knowledge documents for each tool"""
        tool_docs = []
        
        # Flight Search Tool Knowledge
        flight_doc = Document(
            id="tool_flight_search",
            content="""
Flight Search Tool

Description:
- Search for flight information and prices
- Compare options from different airlines
- Provide flight time and price recommendations

Use Cases:
- User inquiries about tickets, flights, and air travel
- Finding flights for specific routes
- Comparing flight prices and times
- Planning transportation methods

Keywords:
flight, airplane, flights, tickets, flying, fly, airline, aviation, ticket, booking, air, air travel

Example Queries:
- "I want to find flights from Beijing to Tokyo"
- "Help me check ticket prices to Paris"
- "Are there any cheap flights to New York"
- "What flight options are available next Tuesday"
            """,
            metadata={
                "tool_name": "flight_search",
                "category": "transportation",
                "priority": "high"
            }
        )
        
        # Hotel Search Tool Knowledge
        hotel_doc = Document(
            id="tool_hotel_search",
            content="""
Hotel Search Tool

Description:
- Search for hotels and accommodation options
- Compare prices and facilities
- Provide accommodation recommendations and reviews

Use Cases:
- User inquiries about accommodation and hotels
- Finding hotels in specific areas
- Comparing hotel prices and facilities
- Planning accommodation arrangements

Keywords:
hotel, accommodation, stay, lodging, booking, room, inn, resort, reservation, place, motel, hostel

Example Queries:
- "What are good hotels in Tokyo"
- "Help me find affordable accommodation"
- "Hotel recommendations near city center"
- "What five-star hotels are available"
            """,
            metadata={
                "tool_name": "hotel_search",
                "category": "accommodation",
                "priority": "high"
            }
        )
        
        # Attraction Search Tool Knowledge
        attraction_doc = Document(
            id="tool_attraction_search",
            content="""
Attraction Search Tool

Description:
- Search for tourist attractions and activities
- Provide attraction information and recommendations
- Plan tourist routes and itineraries

Use Cases:
- User inquiries about attractions, tourism, and activities
- Learning about local famous attractions
- Planning travel itineraries and routes
- Finding specific types of activities

Keywords:
attraction, sightseeing, tourism, visit, tour, activity, explore, experience, museum, park, landmark, sight

Example Queries:
- "What are fun places to visit in Kyoto"
- "Recommend must-visit attractions"
- "What special activities are available locally"
- "Which places are good for taking photos"
            """,
            metadata={
                "tool_name": "attraction_search",
                "category": "entertainment",
                "priority": "medium"
            }
        )
        
        tool_docs.extend([flight_doc, hotel_doc, attraction_doc])
        
        return tool_docs
    
    async def select_tools(
        self, 
        user_request: str, 
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Enhanced tool selection using RAG for semantic understanding"""
        
        # Initialize tool knowledge if not done
        if not self._tool_knowledge_initialized:
            await self.initialize_tool_knowledge()
        
        try:
            # Use RAG to find relevant tools based on user request
            relevant_tools = await self._rag_based_tool_selection(user_request, available_tools, context)
            
            # If RAG selection fails, fall back to keyword-based selection
            if not relevant_tools:
                logger.warning("RAG tool selection failed, using keyword-based fallback")
                relevant_tools = await self._keyword_based_tool_selection(user_request, available_tools, context)
            
            return relevant_tools
            
        except Exception as e:
            logger.error(f"Tool selection error: {e}")
            # Fall back to keyword-based selection
            return await self._keyword_based_tool_selection(user_request, available_tools, context)
    
    async def _rag_based_tool_selection(
        self, 
        user_request: str, 
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Use RAG to select tools based on semantic similarity"""
        
        # Query the tool knowledge base
        retrieval_result = await self.rag_engine.retrieve(
            query=user_request,
            top_k=3,
            filter_metadata=None
        )
        
        selected_tools = []
        tool_scores = {}
        
        # Analyze retrieved documents to select tools
        for i, doc in enumerate(retrieval_result.documents):
            tool_name = doc.metadata.get("tool_name")
            if tool_name and tool_name in available_tools:
                # Calculate relevance score (higher score = more relevant)
                relevance_score = retrieval_result.scores[i] if i < len(retrieval_result.scores) else 0.0
                
                # Consider priority from metadata
                priority = doc.metadata.get("priority", "medium")
                priority_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}.get(priority, 0.0)
                
                final_score = relevance_score + priority_boost
                tool_scores[tool_name] = final_score
        
        # Select tools based on scores (threshold = 0.3)
        score_threshold = 0.3
        for tool_name, score in tool_scores.items():
            if score >= score_threshold:
                selected_tools.append(tool_name)
        
        # Sort by relevance score (descending)
        selected_tools.sort(key=lambda x: tool_scores.get(x, 0), reverse=True)
        
        logger.info(f"RAG tool selection: {selected_tools} (scores: {tool_scores})")
        
        return selected_tools
    
    async def _keyword_based_tool_selection(
        self, 
        user_request: str, 
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Fallback keyword-based tool selection (original implementation)"""
        
        user_request_lower = user_request.lower()
        selected_tools = []
        
        # Flight search keywords
        if any(word in user_request_lower for word in ["flight", "airplane", "flights", "tickets", "fly", "flying"]):
            if "flight_search" in available_tools:
                selected_tools.append("flight_search")
        
        # Hotel search keywords
        if any(word in user_request_lower for word in ["hotel", "accommodation", "stay", "lodging", "booking"]):
            if "hotel_search" in available_tools:
                selected_tools.append("hotel_search")
        
        # Attraction search keywords
        if any(word in user_request_lower for word in ["attraction", "sightseeing", "tourism", "visit", "tour", "activity", "explore"]):
            if "attraction_search" in available_tools:
                selected_tools.append("attraction_search")
        
        # Travel planning keywords - select all relevant tools
        if any(word in user_request_lower for word in ["plan", "schedule", "arrange", "organize", "trip", "travel", "journey"]):
            for tool in ["flight_search", "hotel_search", "attraction_search"]:
                if tool in available_tools and tool not in selected_tools:
                    selected_tools.append(tool)
        
        # If no specific tools were selected, default to attraction search for general queries
        if not selected_tools and available_tools:
            if "attraction_search" in available_tools:
                selected_tools.append("attraction_search")
        
        logger.info(f"Keyword tool selection: {selected_tools}")
        
        return selected_tools

    async def create_tool_chain(
        self, 
        user_request: str,
        selected_tools: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolChain:
        """Create tool execution chain"""
        # TODO: Implement tool chain creation logic with LLM
        # 1. Analyze tool dependencies
        # 2. Determine execution order
        # 3. Set execution strategy
        # 4. Generate tool call parameters
        
        # For now, implement basic tool chain creation
        calls = []
        
        # Create tool calls with basic parameters
        for tool_name in selected_tools:
            # Extract basic parameters from user request and context
            input_data = self._extract_tool_parameters(tool_name, user_request, context)
            
            call = ToolCall(
                tool_name=tool_name,
                input_data=input_data,
                context=context
            )
            calls.append(call)
        
        # Determine execution strategy
        # For travel planning, parallel execution is often better
        strategy = "parallel" if len(calls) > 1 else "sequential"
        
        # TODO: Use LLM to create more sophisticated tool chains
        # try:
        #     llm_prompt = f"""
        #     Based on the user request: "{user_request}"
        #     Selected tools: {selected_tools}
        #     Context: {context or {}}
        #     
        #     Create a tool execution chain including:
        #     1. Optimal execution order
        #     2. Parameters for each tool
        #     3. Execution strategy (sequential/parallel)
        #     4. Any conditional logic needed
        #     
        #     Return a structured plan.
        #     """
        #     response = await self.llm_service.chat_completion([
        #         {"role": "user", "content": llm_prompt}
        #     ])
        #     # Parse LLM response and create tool chain
        # except Exception as e:
        #     # Use the fallback chain created above
        #     pass
        
        return ToolChain(
            calls=calls,
            strategy=strategy,
            conditions=None
        )
    
    def _extract_tool_parameters(
        self, 
        tool_name: str, 
        user_request: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract parameters for a specific tool from user request"""
        user_request_lower = user_request.lower()
        params = {}
        
        # Extract destination
        destinations = ["tokyo", "kyoto", "osaka", "paris", "london", "new york", "beijing", "shanghai"]
        for dest in destinations:
            if dest in user_request_lower:
                params["destination"] = dest
                break
        
        # Extract dates information
        if any(word in user_request_lower for word in ["day", "days", "date", "dates"]):
            import re
            days_match = re.search(r'(\d+)\s*day', user_request_lower)
            if days_match:
                days = int(days_match.group(1))
                params["duration_days"] = days
        
        # Extract budget information
        if any(word in user_request_lower for word in ["budget", "cost", "price", "cheap", "expensive"]):
            params["budget_conscious"] = True
        
        # Tool-specific parameters
        if tool_name == "flight_search":
            params.update({
                "origin": params.get("origin", "unknown"),
                "destination": params.get("destination", "unknown"),
                "departure_date": params.get("departure_date", "flexible"),
                "return_date": params.get("return_date", "flexible"),
                "passengers": params.get("passengers", 1)
            })
        elif tool_name == "hotel_search":
            params.update({
                "destination": params.get("destination", "unknown"),
                "check_in": params.get("check_in", "flexible"),
                "check_out": params.get("check_out", "flexible"),
                "guests": params.get("guests", 1),
                "room_type": params.get("room_type", "standard")
            })
        elif tool_name == "attraction_search":
            params.update({
                "destination": params.get("destination", "unknown"),
                "interests": params.get("interests", ["general"]),
                "duration": params.get("duration_days", 1)
            })
        
        # Add context information if available
        if context:
            params.update({
                "user_context": context,
                "session_id": context.get("session_id")
            })
        
        return params

    def _create_tool_input(self, tool: BaseTool, input_data: Dict[str, Any]) -> ToolInput:
        """Create tool input object"""
        # TODO: Create input object based on tool input schema
        # For now, create a basic ToolInput object
        return ToolInput(
            data=input_data,
            metadata=input_data.get("user_context", {}),
            validation_required=True
        )

    async def _execute_conditional(
        self, 
        calls: List[ToolCall],
        context: Optional[ToolExecutionContext]
    ) -> Dict[str, ToolOutput]:
        """Execute tools conditionally"""
        # TODO: Implement conditional execution logic
        # 1. Determine whether to execute tools based on conditions
        # 2. Support if-else logic
        # 3. Support loop execution
        
        # For now, fall back to sequential execution
        return await self._execute_sequential(calls, context)


class ToolExecutor:
    """Tool executor"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.tool_selector = ToolSelector()
    
    async def execute_tool(
        self, 
        tool_name: str, 
        input_data: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None
    ) -> ToolOutput:
        """Execute a single tool"""
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return ToolOutput(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Convert input data to tool input object
        tool_input = self._create_tool_input(tool, input_data)
        
        return await tool.execute(tool_input, context)
    
    async def execute_chain(
        self, 
        chain: ToolChain,
        context: Optional[ToolExecutionContext] = None
    ) -> ExecutionResult:
        """Execute tool chain"""
        start_time = asyncio.get_event_loop().time()
        results = {}
        
        try:
            if chain.strategy == "sequential":
                results = await self._execute_sequential(chain.calls, context)
            elif chain.strategy == "parallel":
                results = await self._execute_parallel(chain.calls, context)
            elif chain.strategy == "conditional":
                results = await self._execute_conditional(chain.calls, context)
            else:
                raise ValueError(f"Unsupported execution strategy: {chain.strategy}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ExecutionResult(
                success=True,
                results=results,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ExecutionResult(
                success=False,
                results=results,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_sequential(
        self, 
        calls: List[ToolCall],
        context: Optional[ToolExecutionContext]
    ) -> Dict[str, ToolOutput]:
        """Execute tools sequentially"""
        results = {}
        
        for call in calls:
            result = await self.execute_tool(
                call.tool_name,
                call.input_data,
                context
            )
            results[call.tool_name] = result
            
            # If the tool execution fails, stop subsequent execution
            if not result.success:
                break
        
        return results
    
    async def _execute_parallel(
        self, 
        calls: List[ToolCall],
        context: Optional[ToolExecutionContext]
    ) -> Dict[str, ToolOutput]:
        """Execute tools in parallel"""
        tasks = []
        
        for call in calls:
            task = self.execute_tool(
                call.tool_name,
                call.input_data,
                context
            )
            tasks.append((call.tool_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for tool_name, task in tasks:
            result = await task
            results[tool_name] = result
        
        return results
    
    async def auto_execute(
        self, 
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Automatically select and execute tools"""
        # TODO: Implement automatic tool execution process
        # 1. Analyze user request
        # 2. Select appropriate tools
        # 3. Create execution chain
        # 4. Execute tool chain
        # 5. Return results
        
        try:
            # Get available tool list
            available_tools = tool_registry.list_tools()
            
            # Select tools
            selected_tools = await self.tool_selector.select_tools(
                user_request, 
                available_tools, 
                context
            )
            
            # Create tool chain
            tool_chain = await self.tool_selector.create_tool_chain(
                user_request,
                selected_tools,
                context
            )
            
            # Execute tool chain
            execution_context = ToolExecutionContext(
                request_id=f"auto_{int(asyncio.get_event_loop().time())}",
                metadata=context or {}
            )
            
            return await self.execute_chain(tool_chain, execution_context)
            
        except Exception as e:
            # Return error result if auto execution fails
            return ExecutionResult(
                success=False,
                results={},
                execution_time=0.0,
                error=f"Auto execution failed: {str(e)}"
            )


# Global tool executor instance
tool_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get tool executor instance"""
    global tool_executor
    if tool_executor is None:
        tool_executor = ToolExecutor()
    return tool_executor 