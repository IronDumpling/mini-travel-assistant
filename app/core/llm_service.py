"""
LLM Service Module - Core LLM Service Wrapper

Enhanced with structured output capabilities and centralized prompt management.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel
import openai
import logging
import os
import json
import re

logger = logging.getLogger(__name__)


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
    provider: str = "openai"  # openai, claude, etc.
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    mock_mode: bool = True  # Default to mock mode for development


class BaseLLMService(ABC):
    """Base LLM Service Interface"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self.mock_mode = config.mock_mode
    
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
    
    async def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response with schema validation"""
        
        # Add schema instruction to prompt
        schema_instruction = self._create_schema_instruction(response_schema)
        enhanced_messages = self._add_schema_instruction(messages, schema_instruction)
        
        for attempt in range(max_retries):
            try:
                response = await self.chat_completion(enhanced_messages, **kwargs)
                
                # Extract and validate JSON
                parsed_response = self._extract_json_from_response(response.content)
                validated_response = self._validate_against_schema(parsed_response, response_schema)
                
                return validated_response
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise LLMStructuredOutputError(f"Failed to get valid structured output: {e}")
                
                # Add correction instruction for retry
                enhanced_messages = self._add_correction_instruction(enhanced_messages, str(e))
        
        raise LLMStructuredOutputError("Max retries exceeded for structured output")
    
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
    
    def _add_schema_instruction(self, messages: List[Dict[str, str]], instruction: str) -> List[Dict[str, str]]:
        """Add schema instruction to messages"""
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1].get("role") == "user":
            enhanced_messages[-1]["content"] += instruction
        else:
            enhanced_messages.append({"role": "user", "content": instruction})
        return enhanced_messages
    
    def _add_correction_instruction(self, messages: List[Dict[str, str]], error: str) -> List[Dict[str, str]]:
        """Add correction instruction for retry"""
        correction = f"""
        
        The previous response had an error: {error}
        Please provide a corrected JSON response that follows the schema exactly.
        """
        return self._add_schema_instruction(messages, correction)
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
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
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Find the JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return content.strip()
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], field_name: str):
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
            raise ValueError(f"Field {field_name} must be one of {field_schema['enum']}")
        
        # Check numeric constraints
        if field_type in ["number", "integer"]:
            if "minimum" in field_schema and value < field_schema["minimum"]:
                raise ValueError(f"Field {field_name} must be >= {field_schema['minimum']}")
            if "maximum" in field_schema and value > field_schema["maximum"]:
                raise ValueError(f"Field {field_name} must be <= {field_schema['maximum']}")


class OpenAIService(BaseLLMService):
    """OpenAI Service Implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None  # TODO: Initialize OpenAI Client
        # TODO: Initialize with real API key: openai.api_key = config.api_key
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        # TODO: Implement OpenAI Chat Completion
        if self.mock_mode:
            return await self._mock_chat_completion(messages, **kwargs)
        
        # TODO: Real OpenAI API implementation
        # response = await self.client.chat.completions.create(
        #     model=self.model,
        #     messages=messages,
        #     temperature=self.config.temperature,
        #     max_tokens=self.config.max_tokens,
        #     **kwargs
        # )
        # return LLMResponse(
        #     content=response.choices[0].message.content,
        #     usage=response.usage.dict() if response.usage else None,
        #     model=response.model,
        #     finish_reason=response.choices[0].finish_reason
        # )
        
        return await self._mock_chat_completion(messages, **kwargs)
    
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        # TODO: Implement OpenAI Function Call
        if self.mock_mode:
            return await self._mock_function_call(messages, functions, **kwargs)
        
        # TODO: Real function call implementation
        return await self._mock_function_call(messages, functions, **kwargs)
    
    async def _mock_chat_completion(
        self,
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Mock chat completion for development"""
        try:
            # Get the last user message
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            # Check if this is a structured output request
            if "JSON" in user_message and "schema" in user_message:
                mock_content = self._generate_mock_structured_response(user_message)
            else:
                # Simple contextual responses
                user_lower = user_message.lower()
                
                if any(word in user_lower for word in ["travel", "trip", "vacation"]):
                    mock_content = "I'm a travel planning assistant. I can help you plan your trip by finding destinations, accommodations, and activities. What would you like to know about your travel plans?"
                elif any(word in user_lower for word in ["hotel", "accommodation"]):
                    mock_content = "I can help you find hotels and accommodations. What destination are you considering and what are your preferences?"
                elif any(word in user_lower for word in ["flight", "airline"]):
                    mock_content = "I can assist you with flight information. Where are you planning to travel and when?"
                else:
                    mock_content = "I'm here to help with your travel planning needs. What would you like assistance with?"

            return LLMResponse(
                content=mock_content,
                usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                model=self.model,
                finish_reason="stop",
                metadata={"mock_response": True, "provider": "openai"}
            )
            
        except Exception as e:
            logger.error(f"Mock chat completion failed: {e}")
            return LLMResponse(
                content="I'm here to help with your travel planning. What can I assist you with?",
                usage={"total_tokens": 20},
                model=self.model,
                finish_reason="stop",
                metadata={"mock_response": True, "error": str(e)}
            )
    
    def _generate_mock_structured_response(self, user_message: str) -> str:
        """Generate mock structured JSON response"""
        
        # Extract the original message from the structured request
        original_message = ""
        message_lines = user_message.split('\n')
        for line in message_lines:
            if 'User Message:' in line:
                original_message = line.split('User Message:')[-1].strip().strip('"')
                break
        
        original_lower = original_message.lower()
        
        # Generate appropriate mock structured response based on context
        if "intent" in user_message.lower():
            # Mock intent analysis response
            if any(word in original_lower for word in ["plan", "trip", "travel"]):
                return json.dumps({
                    "intent_type": "planning",
                    "destination": {
                        "primary": "Tokyo" if "tokyo" in original_lower else "Unknown",
                        "secondary": [],
                        "region": "Asia" if "tokyo" in original_lower else "Unknown",
                        "confidence": 0.8
                    },
                    "travel_details": {
                        "duration": 7,
                        "travelers": 2,
                        "budget": {
                            "mentioned": "budget" in original_lower,
                            "amount": 4000,
                            "currency": "USD",
                            "level": "mid-range"
                        },
                        "dates": {
                            "departure": "flexible",
                            "return": "flexible",
                            "flexibility": "flexible"
                        }
                    },
                    "preferences": {
                        "travel_style": "mid-range",
                        "interests": ["culture", "food", "sightseeing"],
                        "accommodation_type": "hotel",
                        "transport_preference": "flight"
                    },
                    "sentiment": "positive",
                    "urgency": "medium",
                    "missing_info": ["specific_dates", "exact_budget"],
                    "key_requirements": ["destination_planning", "accommodation", "activities"],
                    "confidence_score": 0.75
                })
            elif any(word in original_lower for word in ["recommend", "suggest"]):
                return json.dumps({
                    "intent_type": "recommendation",
                    "destination": {
                        "primary": "Tokyo" if "tokyo" in original_lower else "Unknown",
                        "secondary": [],
                        "region": "Asia" if "tokyo" in original_lower else "Unknown",
                        "confidence": 0.9
                    },
                    "travel_details": {
                        "duration": 3,
                        "travelers": 1,
                        "budget": {
                            "mentioned": False,
                            "amount": 0,
                            "currency": "USD",
                            "level": "mid-range"
                        },
                        "dates": {
                            "departure": "unknown",
                            "return": "unknown",
                            "flexibility": "unknown"
                        }
                    },
                    "preferences": {
                        "travel_style": "mid-range",
                        "interests": ["attractions", "culture"],
                        "accommodation_type": "unknown",
                        "transport_preference": "unknown"
                    },
                    "sentiment": "positive",
                    "urgency": "low",
                    "missing_info": ["duration", "budget", "dates"],
                    "key_requirements": ["recommendations", "attractions"],
                    "confidence_score": 0.8
                })
            else:
                return json.dumps({
                    "intent_type": "query",
                    "destination": {
                        "primary": "Unknown",
                        "secondary": [],
                        "region": "Unknown",
                        "confidence": 0.5
                    },
                    "travel_details": {
                        "duration": 1,
                        "travelers": 1,
                        "budget": {
                            "mentioned": False,
                            "amount": 0,
                            "currency": "USD",
                            "level": "mid-range"
                        },
                        "dates": {
                            "departure": "unknown",
                            "return": "unknown",
                            "flexibility": "unknown"
                        }
                    },
                    "preferences": {
                        "travel_style": "mid-range",
                        "interests": [],
                        "accommodation_type": "unknown",
                        "transport_preference": "unknown"
                    },
                    "sentiment": "neutral",
                    "urgency": "low",
                    "missing_info": ["destination", "purpose"],
                    "key_requirements": ["information"],
                    "confidence_score": 0.6
                })
        
        elif "tool" in user_message.lower() and "selection" in user_message.lower():
            # Mock tool selection response
            return json.dumps({
                "selected_tools": ["attraction_search", "hotel_search"],
                "tool_priority": {
                    "attraction_search": 0.8,
                    "hotel_search": 0.7
                },
                "execution_strategy": "parallel",
                "reasoning": "Based on the user's request for recommendations and planning, attraction search provides relevant suggestions while hotel search supports accommodation needs.",
                "confidence": 0.75
            })
        
        elif "requirement" in user_message.lower():
            # Mock requirement extraction response
            return json.dumps({
                "budget_sensitivity": "medium",
                "time_sensitivity": "normal",
                "travel_style": "mid-range",
                "geographic_scope": "international",
                "tool_necessity_scores": {
                    "flight_search": 0.8,
                    "hotel_search": 0.7,
                    "attraction_search": 0.9
                },
                "preferences": {
                    "style": "cultural",
                    "activities": ["sightseeing", "food"]
                },
                "constraints": ["budget_limit", "time_limit"],
                "confidence": 0.7
            })
        
        # Default structured response
        return json.dumps({
            "response": "Mock structured response",
            "confidence": 0.6
        })
    
    async def _mock_function_call(
        self,
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """Mock function call for development"""
        # TODO: Implement mock function call logic
        return LLMResponse(
            content="Function calling is not yet implemented in mock mode.",
            usage={"total_tokens": 15},
            model=self.model,
            finish_reason="stop",
            metadata={"mock_response": True, "function_call": True}
        )


class ClaudeService(BaseLLMService):
    """Claude Service Implementation (TODO)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # TODO: Initialize Claude client
        logger.info("Claude service initialized in mock mode")
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        # TODO: Implement Claude Chat Completion
        return LLMResponse(
            content="Claude service is not yet implemented. Using mock response.",
            usage={"total_tokens": 20},
            model=self.model,
            finish_reason="stop",
            metadata={"mock_response": True, "provider": "claude"}
        )
    
    async def function_call(
        self, 
        messages: List[Dict[str, str]], 
        functions: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        # TODO: Implement Claude Function Call
        return LLMResponse(
            content="Claude function calling is not yet implemented.",
            usage={"total_tokens": 15},
            model=self.model,
            finish_reason="stop",
            metadata={"mock_response": True, "provider": "claude"}
        )


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
        else:
            logger.warning(f"Unknown LLM provider: {config.provider}, defaulting to OpenAI")
            config.provider = "openai"
            return OpenAIService(config)
    
    @staticmethod
    def get_default_config() -> LLMConfig:
        """Get default LLM configuration from environment or defaults"""
        return LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4"),
            api_key=os.getenv("LLM_API_KEY"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            mock_mode=os.getenv("LLM_MOCK_MODE", "true").lower() == "true"
        )
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available LLM providers"""
        return ["openai", "claude"]


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