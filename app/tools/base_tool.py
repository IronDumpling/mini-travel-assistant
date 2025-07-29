"""
Base Tool Module - Tool Base Class

TODO: Implement the following features
1. Unified tool interface definition  
2. Standardized tool input and output  
3. Tool metadata management  
4. Error handling and retry mechanisms  
5. Tool execution logging  
6. Tool performance monitoring  
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import time
from datetime import datetime
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ToolStatus(str, Enum):
    """Tool status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ToolMetadata(BaseModel):
    """Tool metadata"""
    name: str
    description: str
    version: str = "1.0.0"
    category: str
    tags: List[str] = []
    author: Optional[str] = None
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # Maximum number of calls per minute
    timeout: int = 30  # Timeout in seconds


class ToolInput(BaseModel):
    """Tool input base class"""
    pass


class ToolOutput(BaseModel):
    """Tool output base class"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolExecutionContext(BaseModel):
    """Tool execution context"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseTool(ABC):
    """Tool base class"""
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.status = ToolStatus.IDLE
        self.last_execution_time = None
        self.execution_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def _execute(self, input_data: ToolInput, context: ToolExecutionContext) -> ToolOutput:
        """Execute the tool logic (must be implemented by subclasses)"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema (must be implemented by subclasses)"""
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema (must be implemented by subclasses)"""
        pass
    
    async def execute(
        self, 
        input_data: ToolInput, 
        context: Optional[ToolExecutionContext] = None
    ) -> ToolOutput:
        """Execute the tool with enhanced error handling and monitoring"""
        if context is None:
            context = ToolExecutionContext(request_id=f"req_{int(time.time())}")
        
        start_time = time.time()
        self.status = ToolStatus.RUNNING
        
        logger.debug(f"Starting execution of {self.metadata.name} tool")
        
        try:
            # Check rate limit
            await self._check_rate_limit()
            
            # Validate input
            await self._validate_input(input_data)

            # Execute the tool logic with timeout
            result = await asyncio.wait_for(
                self._execute(input_data, context),
                timeout=self.metadata.timeout
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.status = ToolStatus.SUCCESS
            self.last_execution_time = datetime.utcnow()
            self.execution_count += 1
            
            logger.info(f"Tool {self.metadata.name} executed successfully in {execution_time:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            self.status = ToolStatus.TIMEOUT
            self.error_count += 1
            execution_time = time.time() - start_time
            
            logger.warning(f"Tool {self.metadata.name} timed out after {self.metadata.timeout}s")
            
            # Try to provide a useful response even on timeout
            fallback_result = await self._generate_timeout_fallback(input_data, context)
            fallback_result.execution_time = execution_time
            fallback_result.error = f"Tool execution timeout ({self.metadata.timeout} seconds)"
            
            return fallback_result
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            self.error_count += 1
            execution_time = time.time() - start_time
            
            logger.error(f"Tool {self.metadata.name} failed after {execution_time:.2f}s: {e}")
            
            # Try to provide a useful response even on error
            fallback_result = await self._generate_error_fallback(input_data, context, str(e))
            fallback_result.execution_time = execution_time
            fallback_result.error = str(e)
            
            return fallback_result

    async def _generate_timeout_fallback(
        self, input_data: ToolInput, context: ToolExecutionContext
    ) -> ToolOutput:
        """Generate a fallback response when tool times out"""
        return ToolOutput(
            success=False,
            data={
                "message": f"The {self.metadata.name} service is currently slow. Please try again later.",
                "fallback_advice": f"You can manually search for {self.metadata.category} information using alternative sources.",
                "tool_name": self.metadata.name
            },
            error=f"Timeout after {self.metadata.timeout} seconds"
        )

    async def _generate_error_fallback(
        self, input_data: ToolInput, context: ToolExecutionContext, error_msg: str
    ) -> ToolOutput:
        """Generate a fallback response when tool encounters an error"""
        return ToolOutput(
            success=False,
            data={
                "message": f"The {self.metadata.name} service encountered an issue.",
                "fallback_advice": f"Please check your input parameters or try again later.",
                "tool_name": self.metadata.name,
                "error_type": "execution_error"
            },
            error=error_msg
        )
    
    async def _check_rate_limit(self):
        """Check rate limit"""
        # TODO: Implement rate limit logic
        pass
    
    async def _validate_input(self, input_data: ToolInput):
        """Validate input data"""
        # TODO: Implement input validation logic
        pass
    

    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status information"""
        return {
            "name": self.metadata.name,
            "status": self.status,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1),
            "last_execution_time": self.last_execution_time
        }


class ToolRegistry:
    """Tool registry"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: BaseTool):
        """Register tool"""
        self._tools[tool.metadata.name] = tool
        
        # Organize tools by category
        category = tool.metadata.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.metadata.name)
        
        logger.info(f"Registered tool: {tool.metadata.name} (category: {category})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool"""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names]
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata"""
        tool = self._tools.get(name)
        return tool.metadata if tool else None
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "total_tools": len(self._tools),
            "categories": {
                category: len(tools) 
                for category, tools in self._categories.items()
            },
            "tools": {
                name: tool.get_status() 
                for name, tool in self._tools.items()
            }
        }


# Global tool registry
tool_registry = ToolRegistry() 