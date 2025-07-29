"""
LLM Service Module - Core LLM Service Wrapper

Enhanced with structured output capabilities and centralized prompt management.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel
import openai
import anthropic
import json
import os
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LLMStructuredOutputError(Exception):
    """Exception raised when structured output generation fails"""
    pass


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
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Chat Completion Interface"""
        pass

    @abstractmethod
    async def function_call(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Function Call Interface"""
        pass
    

    async def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        max_retries: int = 2,  # Reduced retries for performance
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate structured response with schema validation
        Optimized for performance with reduced retries and smarter error handling
        """

        # Add schema instruction to prompt
        schema_instruction = self._create_schema_instruction(response_schema)
        enhanced_messages = self._add_schema_instruction(messages, schema_instruction)

        # Set optimal parameters for structured output
        kwargs.setdefault('temperature', 0.1)  # Lower temperature for more consistent JSON
        kwargs.setdefault('max_tokens', 800)   # Reasonable limit for structured responses

        for attempt in range(max_retries):
            try:
                response = await self.chat_completion(enhanced_messages, **kwargs)

                # Extract and validate JSON
                parsed_response = self._extract_json_from_response(response.content)
                validated_response = self._validate_against_schema(
                    parsed_response, response_schema
                )

                return validated_response

            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    # Return best-effort fallback instead of raising error
                    logger.warning(f"Structured output failed, using fallback: {e}")
                    return self._create_fallback_response(response_schema, messages)

                # Add correction instruction for retry
                enhanced_messages = self._add_correction_instruction(
                    enhanced_messages, str(e)
                )

        # Should not reach here, but return fallback just in case
        return self._create_fallback_response(response_schema, messages)

    def _create_schema_instruction(self, schema: Dict[str, Any]) -> str:
        """Create schema instruction for structured output"""
        return f"""
        
        IMPORTANT: Please respond with a valid JSON object that matches this schema:
        {json.dumps(schema, indent=2)}
        
        Make sure to:
        1. Return ONLY valid JSON
        2. Include all required fields
        3. Use correct data types
        4. Follow the enum constraints where specified
        """

    def _add_schema_instruction(
        self, messages: List[Dict[str, str]], instruction: str
    ) -> List[Dict[str, str]]:
        """Add schema instruction to messages"""
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1].get("role") == "user":
            enhanced_messages[-1]["content"] += instruction
        else:
            enhanced_messages.append({"role": "user", "content": instruction})
        return enhanced_messages

    def _add_correction_instruction(
        self, messages: List[Dict[str, str]], error: str
    ) -> List[Dict[str, str]]:
        """Add correction instruction for retry"""
        correction = f"""
        
        The previous response had an error: {error}
        Please provide a corrected JSON response that follows the schema exactly.
        """
        return self._add_schema_instruction(messages, correction)

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to parse the entire content as JSON
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            cleaned_content = self._clean_json_string(content)
            return json.loads(cleaned_content)

    def _clean_json_string(self, content: str) -> str:
        """Clean up common JSON formatting issues"""
        # Remove markdown code blocks
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)

        # Find the JSON object
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return content.strip()

    def _create_fallback_response(
        self, response_schema: Dict[str, Any], messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create a fallback response when structured output fails"""
        fallback = {}
        
        # Extract properties from schema
        properties = response_schema.get("properties", {})
        
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")
            
            # Create sensible defaults based on field type
            if field_type == "string":
                fallback[field_name] = field_schema.get("default", "unknown")
            elif field_type == "number":
                fallback[field_name] = field_schema.get("default", 0.0)
            elif field_type == "integer":
                fallback[field_name] = field_schema.get("default", 0)
            elif field_type == "boolean":
                fallback[field_name] = field_schema.get("default", False)
            elif field_type == "array":
                fallback[field_name] = field_schema.get("default", [])
            elif field_type == "object":
                fallback[field_name] = field_schema.get("default", {})
            else:
                fallback[field_name] = None
        
        # Add some context-aware defaults for common travel fields
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "").lower()
                break
        
        # Intelligent defaults for travel-related fields
        if "intent_type" in fallback:
            if any(word in user_message for word in ["plan", "planning"]):
                fallback["intent_type"] = "planning"
            elif any(word in user_message for word in ["recommend", "suggest"]):
                fallback["intent_type"] = "recommendation"
            else:
                fallback["intent_type"] = "query"
        
        if "confidence_score" in fallback:
            fallback["confidence_score"] = 0.6  # Moderate confidence for fallback
            
        if "selected_tools" in fallback:
            fallback["selected_tools"] = ["attraction_search"]  # Safe default
            
        logger.info(f"Generated fallback response with {len(fallback)} fields")
        return fallback

    def _validate_against_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Basic schema validation"""
        required_fields = schema.get("required", [])

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Basic type checking for properties
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                self._validate_field(data[field], field_schema, field)

        return data

    def _validate_field(
        self, value: Any, field_schema: Dict[str, Any], field_name: str
    ):
        """Validate individual field against schema"""
        field_type = field_schema.get("type")

        if field_type == "string" and not isinstance(value, str):
            raise ValueError(f"Field {field_name} must be a string")
        elif field_type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Field {field_name} must be a number")
        elif field_type == "integer" and not isinstance(value, int):
            raise ValueError(f"Field {field_name} must be an integer")
        elif field_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Field {field_name} must be a boolean")
        elif field_type == "array" and not isinstance(value, list):
            raise ValueError(f"Field {field_name} must be an array")
        elif field_type == "object" and not isinstance(value, dict):
            raise ValueError(f"Field {field_name} must be an object")

        # Check enum constraints
        if "enum" in field_schema and value not in field_schema["enum"]:
            raise ValueError(
                f"Field {field_name} must be one of {field_schema['enum']}"
            )

        # Check numeric constraints
        if field_type in ["number", "integer"]:
            if "minimum" in field_schema and value < field_schema["minimum"]:
                raise ValueError(
                    f"Field {field_name} must be >= {field_schema['minimum']}"
                )
            if "maximum" in field_schema and value > field_schema["maximum"]:
                raise ValueError(
                    f"Field {field_name} must be <= {field_schema['maximum']}"
                )


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
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
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
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                f"OpenAI API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}"
            )

            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={
                    "provider": "openai",
                    "api_call": True,
                    "model_used": response.model,
                },
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
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def function_call(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]], **kwargs
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

                content = (
                    f"Function '{function_name}' called with arguments: {function_args}"
                )

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
                        "function_args": function_args,
                    },
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
                        "function_called": False,
                    },
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
                "total_tokens": usage.total_tokens,
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
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
            )
        ),
    )
    async def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
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
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }

            logger.info(
                f"Claude API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}"
            )

            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=response.stop_reason,
                metadata={
                    "provider": "claude",
                    "api_call": True,
                    "model_used": response.model,
                },
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
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
            )
        ),
    )
    async def function_call(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]], **kwargs
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
                tool = {"type": "function", "function": func}
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

                content = (
                    f"Function '{function_name}' called with arguments: {function_args}"
                )

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
                        "function_args": function_args,
                    },
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
                        "function_called": False,
                    },
                )

        except Exception as e:
            logger.error(f"Error in Claude function call: {e}")
            raise

    def _convert_messages_to_claude_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Convert OpenAI format messages to Claude format"""
        claude_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Claude uses "user" and "assistant" roles (same as OpenAI)
            claude_messages.append({"role": role, "content": content})

        return claude_messages

    def _extract_usage(self, usage) -> Optional[Dict[str, Any]]:
        """Extract usage data from Claude response"""
        if usage:
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
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
                api_key=api_key, base_url="https://api.deepseek.com"
            )
            logger.info(f"DeepSeek client initialized with model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
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
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                f"DeepSeek API call successful. Tokens used: {usage_data['total_tokens'] if usage_data else 'unknown'}"
            )

            return LLMResponse(
                content=content,
                usage=usage_data,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={
                    "provider": "deepseek",
                    "api_call": True,
                    "model_used": response.model,
                },
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
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def function_call(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]], **kwargs
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

                content = (
                    f"Function '{function_name}' called with arguments: {function_args}"
                )

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
                        "function_args": function_args,
                    },
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
                        "function_called": False,
                    },
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
                "total_tokens": usage.total_tokens,
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
            logger.warning(
                f"Unknown LLM provider: {config.provider}, defaulting to DeepSeek"
            )
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
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
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
        logger.info(
            f"Initialized LLM service: {llm_service.config.provider} ({llm_service.model})"
        )
    return llm_service


def configure_llm_service(config: LLMConfig) -> BaseLLMService:
    """Configure and get LLM service with specific configuration"""
    global llm_service
    llm_service = LLMServiceFactory.create_service(config)
    logger.info(f"Configured LLM service: {config.provider} ({config.model})")
    return llm_service
