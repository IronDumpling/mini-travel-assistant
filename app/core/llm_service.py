"""
LLM Service Module - Core LLM Service Wrapper

This module provides a unified interface for different LLM providers including:
1. OpenAI API integration with proper error handling and retries
2. Claude API integration (TODO)
3. Prompt template management
4. Standardized response formatting
5. Cost and usage monitoring
6. Streaming response support
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel
import openai
import anthropic
import httpx
import json
import logging
import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """LLM Response Standard Format"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: str
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMConfig(BaseModel):
    """LLM Configuration"""
    provider: str = "deepseek"  # deepseek, openai, claude, etc.
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    retry_attempts: int = 3
    retry_delay: float = 1.0


class BaseLLMService(ABC):
    """Base LLM Service Interface"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Chat Completion Interface"""
        pass
    
    @abstractmethod
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """Function Call Interface"""
        pass


class OpenAIService(BaseLLMService):
    """OpenAI Service Implementation with real API integration"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with proper configuration"""
        try:
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            # Initialize OpenAI client
            self.client = openai.AsyncOpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {self.model}")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Real OpenAI Chat Completion with retry logic"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Prepare parameters for OpenAI API
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making OpenAI API call with model: {self.model}")
            
            # Make the API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Prepare usage data
            usage_data = None
            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            logger.info(f"OpenAI API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}")
            
            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={
                    "provider": "openai",
                    "api_call": True,
                    "model_used": response.model
                }
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"OpenAI API timeout: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"OpenAI API connection error: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI API call: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """Real OpenAI Function Call with retry logic"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Prepare parameters for OpenAI function calling
            params = {
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": func} for func in functions],
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making OpenAI function call with model: {self.model}")
            
            # Make the API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            # Handle function call response
            if message.tool_calls:
                # Function was called
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                content = f"Function '{function_name}' called with arguments: {function_args}"
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "provider": "openai",
                        "api_call": True,
                        "function_called": True,
                        "function_name": function_name,
                        "function_args": function_args
                    }
                )
            else:
                # No function call, regular response
                content = message.content or ""
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "provider": "openai",
                        "api_call": True,
                        "function_called": False
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in OpenAI function call: {e}")
            raise
    
    def _extract_usage(self, usage) -> Optional[Dict[str, Any]]:
        """Extract usage data from OpenAI response"""
        if usage:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        return None


class ClaudeService(BaseLLMService):
    """Claude Service Implementation with real API integration"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Claude client with proper configuration"""
        try:
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required")
            
            # Initialize Anthropic client
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info(f"Claude client initialized with model: {self.model}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError))
    )
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Real Claude Chat Completion with retry logic"""
        if not self.client:
            raise RuntimeError("Claude client not initialized")
        
        try:
            # Convert OpenAI format messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Prepare parameters for Claude API
            params = {
                "model": self.model,
                "messages": claude_messages,
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making Claude API call with model: {self.model}")
            
            # Make the API call
            response = await self.client.messages.create(**params)
            
            # Extract response data
            content = response.content[0].text if response.content else ""
            
            # Prepare usage data
            usage_data = None
            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            
            logger.info(f"Claude API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}")
            
            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=response.stop_reason,
                metadata={
                    "provider": "claude",
                    "api_call": True,
                    "model_used": response.model
                }
            )
            
        except anthropic.RateLimitError as e:
            logger.warning(f"Claude rate limit exceeded: {e}")
            raise
        except anthropic.APITimeoutError as e:
            logger.warning(f"Claude API timeout: {e}")
            raise
        except anthropic.APIConnectionError as e:
            logger.warning(f"Claude API connection error: {e}")
            raise
        except anthropic.AuthenticationError as e:
            logger.error(f"Claude authentication failed: {e}")
            raise
        except anthropic.BadRequestError as e:
            logger.error(f"Claude bad request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Claude API call: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError))
    )
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """Real Claude Function Call with retry logic"""
        if not self.client:
            raise RuntimeError("Claude client not initialized")
        
        try:
            # Convert OpenAI format messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Convert functions to Claude tools format
            tools = []
            for func in functions:
                tool = {
                    "type": "function",
                    "function": func
                }
                tools.append(tool)
            
            # Prepare parameters for Claude API
            params = {
                "model": self.model,
                "messages": claude_messages,
                "tools": tools,
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making Claude function call with model: {self.model}")
            
            # Make the API call
            response = await self.client.messages.create(**params)
            
            # Handle function call response
            if response.content and response.content[0].type == "tool_use":
                # Function was called
                tool_use = response.content[0]
                function_name = tool_use.name
                function_args = tool_use.input
                
                content = f"Function '{function_name}' called with arguments: {function_args}"
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=response.stop_reason,
                    metadata={
                        "provider": "claude",
                        "api_call": True,
                        "function_called": True,
                        "function_name": function_name,
                        "function_args": function_args
                    }
                )
            else:
                # No function call, regular response
                content = response.content[0].text if response.content else ""
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=response.stop_reason,
                    metadata={
                        "provider": "claude",
                        "api_call": True,
                        "function_called": False
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in Claude function call: {e}")
            raise
    
    def _convert_messages_to_claude_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI format messages to Claude format"""
        claude_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Claude uses "user" and "assistant" roles (same as OpenAI)
            claude_messages.append({
                "role": role,
                "content": content
            })
        
        return claude_messages
    
    def _extract_usage(self, usage) -> Optional[Dict[str, Any]]:
        """Extract usage data from Claude response"""
        if usage:
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens
            }
        return None


class DeepSeekService(BaseLLMService):
    """DeepSeek Service Implementation with real API integration"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize DeepSeek client with proper configuration"""
        try:
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key is required")
            
            # Initialize OpenAI client with DeepSeek base URL
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info(f"DeepSeek client initialized with model: {self.model}")
                
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Real DeepSeek Chat Completion with retry logic"""
        if not self.client:
            raise RuntimeError("DeepSeek client not initialized")
        
        try:
            # Prepare parameters for DeepSeek API
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making DeepSeek API call with model: {self.model}")
            
            # Make the API call using OpenAI client
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Prepare usage data
            usage_data = None
            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            logger.info(f"DeepSeek API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}")
            
            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={
                    "provider": "deepseek",
                    "api_call": True,
                    "model_used": response.model
                }
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"DeepSeek rate limit exceeded: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"DeepSeek API timeout: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"DeepSeek API connection error: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"DeepSeek authentication failed: {e}")
            raise
        except openai.BadRequestError as e:
            logger.error(f"DeepSeek bad request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek API call: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """Real DeepSeek Function Call with retry logic"""
        if not self.client:
            raise RuntimeError("DeepSeek client not initialized")
        
        try:
            # Prepare parameters for DeepSeek function calling
            params = {
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": func} for func in functions],
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                params["max_tokens"] = self.config.max_tokens
            
            # Add any additional kwargs
            params.update(kwargs)
            
            logger.debug(f"Making DeepSeek function call with model: {self.model}")
            
            # Make the API call using OpenAI client
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            # Handle function call response
            if message.tool_calls:
                # Function was called
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                content = f"Function '{function_name}' called with arguments: {function_args}"
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "provider": "deepseek",
                        "api_call": True,
                        "function_called": True,
                        "function_name": function_name,
                        "function_args": function_args
                    }
                )
            else:
                # No function call, regular response
                content = message.content or ""
                
                return LLMResponse(
                    content=content,
                    usage=self._extract_usage(response.usage),
                    model=response.model,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "provider": "deepseek",
                        "api_call": True,
                        "function_called": False
                    }
                )
                
        except openai.RateLimitError as e:
            logger.warning(f"DeepSeek rate limit exceeded: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"DeepSeek API timeout: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"DeepSeek API connection error: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"DeepSeek authentication failed: {e}")
            raise
        except openai.BadRequestError as e:
            logger.error(f"DeepSeek bad request: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in DeepSeek function call: {e}")
            raise
    
    def _extract_usage(self, usage) -> Optional[Dict[str, Any]]:
        """Extract usage data from DeepSeek response"""
        if usage:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        return None


class LLMServiceFactory:
    """LLM Service Factory for creating different LLM services"""
    
    @staticmethod
    def create_service(config: Optional[LLMConfig] = None) -> BaseLLMService:
        """Create LLM service based on configuration"""
        if config is None:
            config = LLMServiceFactory.get_default_config()
        
        if config.provider.lower() == "openai":
            return OpenAIService(config)
        elif config.provider.lower() == "claude":
            return ClaudeService(config)
        elif config.provider.lower() == "deepseek":
            return DeepSeekService(config)
        else:
            logger.warning(f"Unknown LLM provider: {config.provider}, defaulting to DeepSeek")
            config.provider = "deepseek"
            return DeepSeekService(config)
    
    @staticmethod
    def get_default_config() -> LLMConfig:
        """Get default LLM configuration from environment or defaults"""
        provider = os.getenv("LLM_PROVIDER", "deepseek")  # Changed default to deepseek
        
        # Get appropriate API key based on provider
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            if provider.lower() == "claude":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider.lower() == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
        
        # Get appropriate default model based on provider
        if provider.lower() == "claude":
            default_model = "claude-3-sonnet-20240229"
        elif provider.lower() == "deepseek":
            default_model = "deepseek-chat"  # or "deepseek-coder" for coding tasks
        else:
            default_model = "gpt-4"
        model = os.getenv("LLM_MODEL", default_model)
        
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        )
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available LLM providers"""
        return ["openai", "claude", "deepseek"]


# Global LLM Service Instance
llm_service: Optional[BaseLLMService] = None


def get_llm_service(config: Optional[LLMConfig] = None) -> BaseLLMService:
    """Get LLM Service Instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMServiceFactory.create_service(config)
        logger.info(f"Initialized LLM service: {llm_service.config.provider} ({llm_service.model})")
    return llm_service


def configure_llm_service(config: LLMConfig) -> BaseLLMService:
    """Configure and get LLM service with specific configuration"""
    global llm_service
    llm_service = LLMServiceFactory.create_service(config)
    logger.info(f"Configured LLM service: {config.provider} ({config.model})")
    return llm_service 