"""
Tool Executor Module - Tool Executor with RAG-Enhanced Tool Selection

Implements intelligent tool selection using RAG (Retrieval-Augmented Generation)
to improve tool selection accuracy based on semantic understanding of user requests.
"""

from typing import Dict, List, Any, Optional
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
    
    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get tool metadata for analysis and optimization"""
        
        # Define tool metadata for cost and performance analysis
        tool_metadata = {
            "flight_search": {
                "api_cost": 0.6,  # Relative cost score (0-1, lower is more expensive)
                "avg_execution_time": 3.0,  # Average execution time in seconds
                "rate_limit": 1000,  # Daily rate limit
                "failure_rate": 0.05,  # Historical failure rate
                "data_freshness": "real-time",  # Data freshness
                "dependencies": [],  # Tool dependencies
                "resource_usage": "medium"  # Resource usage level
            },
            "hotel_search": {
                "api_cost": 0.5,
                "avg_execution_time": 2.5,
                "rate_limit": 2000,
                "failure_rate": 0.03,
                "data_freshness": "real-time",
                "dependencies": [],
                "resource_usage": "medium"
            },
            "attraction_search": {
                "api_cost": 0.8,  # Lower cost (Google Places API)
                "avg_execution_time": 2.0,
                "rate_limit": 5000,
                "failure_rate": 0.02,
                "data_freshness": "real-time",
                "dependencies": [],
                "resource_usage": "low"
            }
        }
        
        return tool_metadata.get(tool_name, {
            "api_cost": 0.5,
            "avg_execution_time": 2.0,
            "rate_limit": 1000,
            "failure_rate": 0.05,
            "data_freshness": "unknown",
            "dependencies": [],
            "resource_usage": "medium"
        })
    
    def analyze_tool_dependencies(self, tools: List[str]) -> Dict[str, Any]:
        """Analyze tool dependencies for execution optimization"""
        
        dependencies = {
            "has_dependencies": False,
            "has_strong_dependencies": False,
            "dependency_chain": [],
            "parallel_groups": [],
            "sequential_requirements": []
        }
        
        # Define dependency rules
        dependency_rules = {
            ("flight_search", "hotel_search"): "weak",    # Weak dependency: results can cross-reference
            ("hotel_search", "attraction_search"): "weak", # Weak dependency: hotel location affects attraction choice
            ("flight_search", "attraction_search"): "none"  # No dependency: can run in parallel
        }
        
        # Analyze dependencies
        for i, tool1 in enumerate(tools):
            for j, tool2 in enumerate(tools):
                if i != j:
                    dep_type = dependency_rules.get((tool1, tool2), "none")
                    if dep_type != "none":
                        dependencies["has_dependencies"] = True
                        dependencies["dependency_chain"].append({
                            "from": tool1,
                            "to": tool2,
                            "type": dep_type
                        })
                        if dep_type == "strong":
                            dependencies["has_strong_dependencies"] = True
                            dependencies["sequential_requirements"].append((tool1, tool2))
        
        # Group tools for parallel execution
        if not dependencies["has_strong_dependencies"]:
            # All tools can run in parallel if no strong dependencies
            if len(tools) > 1:
                dependencies["parallel_groups"] = [tools]
        
        return dependencies
    
    def estimate_resource_cost(self, tools: List[str], parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate resource cost for tool execution"""
        
        cost_analysis = {
            "total_api_cost": 0.0,
            "estimated_execution_time": 0.0,
            "rate_limit_concerns": [],
            "resource_usage": "low",
            "cost_breakdown": {}
        }
        
        total_cost = 0.0
        max_execution_time = 0.0
        resource_levels = []
        
        for tool_name in tools:
            tool_metadata = self.get_tool_metadata(tool_name)
            tool_params = parameters.get(tool_name, {})
            
            # Calculate API cost
            base_cost = tool_metadata["api_cost"]
            param_multiplier = self._calculate_parameter_cost_multiplier(tool_params)
            tool_cost = base_cost * param_multiplier
            
            # Calculate execution time
            execution_time = tool_metadata["avg_execution_time"]
            
            # Track resource usage
            resource_levels.append(tool_metadata["resource_usage"])
            
            # Check rate limits
            if tool_metadata["rate_limit"] < 1000:
                cost_analysis["rate_limit_concerns"].append(tool_name)
            
            cost_analysis["cost_breakdown"][tool_name] = {
                "api_cost": tool_cost,
                "execution_time": execution_time,
                "resource_usage": tool_metadata["resource_usage"]
            }
            
            total_cost += tool_cost
            max_execution_time = max(max_execution_time, execution_time)
        
        cost_analysis["total_api_cost"] = total_cost
        cost_analysis["estimated_execution_time"] = max_execution_time
        cost_analysis["resource_usage"] = self._aggregate_resource_usage(resource_levels)
        
        return cost_analysis
    
    def _calculate_parameter_cost_multiplier(self, parameters: Dict[str, Any]) -> float:
        """Calculate cost multiplier based on parameters"""
        
        multiplier = 1.0
        
        # More results = higher cost
        if "max_results" in parameters:
            max_results = parameters["max_results"]
            if max_results > 10:
                multiplier += 0.2
            elif max_results > 20:
                multiplier += 0.5
        
        # Photo requests = higher cost
        if parameters.get("include_photos", False):
            multiplier += 0.3
        
        # Broader search radius = higher cost
        if "radius_meters" in parameters:
            radius = parameters["radius_meters"]
            if radius > 10000:
                multiplier += 0.2
            elif radius > 50000:
                multiplier += 0.5
        
        return multiplier
    
    def _aggregate_resource_usage(self, resource_levels: List[str]) -> str:
        """Aggregate resource usage levels"""
        
        if not resource_levels:
            return "low"
        
        # Count usage levels
        usage_counts = {"low": 0, "medium": 0, "high": 0}
        for level in resource_levels:
            usage_counts[level] = usage_counts.get(level, 0) + 1
        
        # Determine overall usage
        if usage_counts["high"] > 0:
            return "high"
        elif usage_counts["medium"] > 1:
            return "high"
        elif usage_counts["medium"] > 0:
            return "medium"
        else:
            return "low"
    
    async def _execute_with_optimization(
        self, 
        chain: ToolChain,
        context: Optional[ToolExecutionContext] = None
    ) -> ExecutionResult:
        """Execute with optimization including retry logic and resource management"""
        
        start_time = asyncio.get_event_loop().time()
        results = {}
        
        try:
            # Analyze dependencies and resource requirements
            dependencies = self.analyze_tool_dependencies([call.tool_name for call in chain.calls])
            
            # Get tool parameters for cost analysis
            tool_params = {}
            for call in chain.calls:
                tool_params[call.tool_name] = call.input_data
            
            cost_analysis = self.estimate_resource_cost(
                [call.tool_name for call in chain.calls], 
                tool_params
            )
            
            logger.info(f"Executing tool chain with estimated cost: {cost_analysis['total_api_cost']:.2f}")
            
            # Execute based on optimized strategy
            if chain.strategy == "sequential":
                results = await self._execute_sequential_with_retry(chain.calls, context)
            elif chain.strategy == "parallel":
                results = await self._execute_parallel_with_retry(chain.calls, context)
            else:
                raise ValueError(f"Unsupported execution strategy: {chain.strategy}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ExecutionResult(
                success=True,
                results=results,
                execution_time=execution_time,
                metadata={
                    "cost_analysis": cost_analysis,
                    "dependencies": dependencies,
                    "optimization_applied": True
                }
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ExecutionResult(
                success=False,
                results=results,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_sequential_with_retry(
        self, 
        calls: List[ToolCall],
        context: Optional[ToolExecutionContext],
        max_retries: int = 2
    ) -> Dict[str, ToolOutput]:
        """Execute tools sequentially with retry logic"""
        
        results = {}
        
        for call in calls:
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    result = await self.execute_tool(
                        call.tool_name,
                        call.input_data,
                        context
                    )
                    results[call.tool_name] = result
                    
                    # If successful, break retry loop
                    if result.success:
                        break
                    else:
                        last_error = result.error
                        retry_count += 1
                        if retry_count <= max_retries:
                            await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                
                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(1.0 * retry_count)
            
            # If all retries failed, record the failure
            if call.tool_name not in results or not results[call.tool_name].success:
                results[call.tool_name] = ToolOutput(
                    success=False,
                    error=f"Failed after {max_retries} retries: {last_error}"
                )
                # Stop execution on critical failure
                break
        
        return results
    
    async def _execute_parallel_with_retry(
        self, 
        calls: List[ToolCall],
        context: Optional[ToolExecutionContext],
        max_retries: int = 2
    ) -> Dict[str, ToolOutput]:
        """Execute tools in parallel with retry logic"""
        
        async def execute_with_retry(call: ToolCall) -> ToolOutput:
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    result = await self.execute_tool(
                        call.tool_name,
                        call.input_data,
                        context
                    )
                    
                    if result.success:
                        return result
                    else:
                        last_error = result.error
                        retry_count += 1
                        if retry_count <= max_retries:
                            await asyncio.sleep(1.0 * retry_count)
                
                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(1.0 * retry_count)
            
            return ToolOutput(
                success=False,
                error=f"Failed after {max_retries} retries: {last_error}"
            )
        
        # Execute all tasks in parallel
        tasks = [execute_with_retry(call) for call in calls]
        task_results = await asyncio.gather(*tasks)
        
        # Combine results
        results = {}
        for call, result in zip(calls, task_results):
            results[call.tool_name] = result
        
        return results
    
    def _create_tool_input(self, tool: BaseTool, input_data: Dict[str, Any]) -> ToolInput:
        """Create tool input object with enhanced validation"""
        
        # Enhanced input creation with better validation
        try:
            # Try to use tool's input schema if available
            if hasattr(tool, 'get_input_schema'):
                schema = tool.get_input_schema()
                # Validate input against schema (simplified)
                validated_data = self._validate_input_data(input_data, schema)
            else:
                validated_data = input_data
            
            return ToolInput(
                data=validated_data,
                metadata=input_data.get("user_context", {}),
                validation_required=True
            )
            
        except Exception as e:
            logger.warning(f"Input validation failed: {e}, using original data")
            return ToolInput(
                data=input_data,
                metadata=input_data.get("user_context", {}),
                validation_required=True
            )
    
    def _validate_input_data(self, input_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against schema (simplified implementation)"""
        
        # This is a simplified validation - in production you'd use a proper schema validator
        validated_data = input_data.copy()
        
        # Basic validation checks
        if "required" in schema:
            for required_field in schema["required"]:
                if required_field not in validated_data:
                    logger.warning(f"Missing required field: {required_field}")
        
        return validated_data


# Global tool executor instance
tool_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get tool executor instance"""
    global tool_executor
    if tool_executor is None:
        tool_executor = ToolExecutor()
    return tool_executor 