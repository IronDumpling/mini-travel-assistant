"""
Prompt Manager - Centralized prompt template management system
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import json

class PromptType(Enum):
    """Prompt type enumeration"""
    INTENT_ANALYSIS = "intent_analysis"
    REQUIREMENT_EXTRACTION = "requirement_extraction"
    TOOL_SELECTION = "tool_selection"
    RESPONSE_GENERATION = "response_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    RESPONSE_REFINEMENT = "response_refinement"
    RAG_GENERATION = "rag_generation"
    FUNCTION_CALLING = "function_calling"

class PromptManager:
    """Centralized prompt template manager"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.schemas = self._initialize_schemas()
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize all prompt templates"""
        return {
            PromptType.INTENT_ANALYSIS.value: self._get_intent_analysis_template(),
            PromptType.REQUIREMENT_EXTRACTION.value: self._get_requirement_extraction_template(),
            PromptType.TOOL_SELECTION.value: self._get_tool_selection_template(),
            PromptType.RESPONSE_GENERATION.value: self._get_response_generation_template(),
            PromptType.QUALITY_ASSESSMENT.value: self._get_quality_assessment_template(),
            PromptType.RESPONSE_REFINEMENT.value: self._get_response_refinement_template(),
            PromptType.RAG_GENERATION.value: self._get_rag_generation_template(),
            PromptType.FUNCTION_CALLING.value: self._get_function_calling_template()
        }
    
    def _initialize_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize JSON schemas for structured outputs"""
        return {
            PromptType.INTENT_ANALYSIS.value: {
                "type": "object",
                "properties": {
                    "intent_type": {"type": "string", "enum": ["planning", "query", "recommendation", "modification", "booking", "complaint"]},
                    "destination": {
                        "type": "object",
                        "properties": {
                            "primary": {"type": "string"},
                            "secondary": {"type": "array", "items": {"type": "string"}},
                            "region": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "travel_details": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "integer", "minimum": 1},
                            "travelers": {"type": "integer", "minimum": 1},
                            "budget": {
                                "type": "object",
                                "properties": {
                                    "mentioned": {"type": "boolean"},
                                    "amount": {"type": "number"},
                                    "currency": {"type": "string"},
                                    "level": {"type": "string", "enum": ["budget", "mid-range", "luxury"]}
                                }
                            },
                            "dates": {
                                "type": "object",
                                "properties": {
                                    "departure": {"type": "string"},
                                    "return": {"type": "string"},
                                    "flexibility": {"type": "string", "enum": ["fixed", "flexible", "unknown"]}
                                }
                            }
                        }
                    },
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "travel_style": {"type": "string", "enum": ["luxury", "mid-range", "budget", "backpacking", "business", "family"]},
                            "interests": {"type": "array", "items": {"type": "string"}},
                            "accommodation_type": {"type": "string"},
                            "transport_preference": {"type": "string"}
                        }
                    },
                    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "excited", "worried"]},
                    "urgency": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "missing_info": {"type": "array", "items": {"type": "string"}},
                    "key_requirements": {"type": "array", "items": {"type": "string"}},
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["intent_type", "destination", "sentiment", "urgency", "confidence_score"]
            },
            
            PromptType.REQUIREMENT_EXTRACTION.value: {
                "type": "object",
                "properties": {
                    "budget_sensitivity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "time_sensitivity": {"type": "string", "enum": ["urgent", "normal", "flexible"]},
                    "travel_style": {"type": "string"},
                    "geographic_scope": {"type": "string"},
                    "tool_necessity_scores": {"type": "object"},
                    "preferences": {"type": "object"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["budget_sensitivity", "time_sensitivity", "confidence"]
            },
            
            PromptType.TOOL_SELECTION.value: {
                "type": "object",
                "properties": {
                    "selected_tools": {"type": "array", "items": {"type": "string"}},
                    "tool_priority": {"type": "object"},
                    "execution_strategy": {"type": "string", "enum": ["sequential", "parallel", "conditional"]},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["selected_tools", "execution_strategy", "reasoning", "confidence"]
            },
            
            PromptType.QUALITY_ASSESSMENT.value: {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "dimension_scores": {
                        "type": "object",
                        "properties": {
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                            "completeness": {"type": "number", "minimum": 0, "maximum": 1},
                            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                            "practicality": {"type": "number", "minimum": 0, "maximum": 1},
                            "personalization": {"type": "number", "minimum": 0, "maximum": 1},
                            "feasibility": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "improvement_suggestions": {"type": "array", "items": {"type": "string"}},
                    "missing_elements": {"type": "array", "items": {"type": "string"}},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "meets_threshold": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["overall_score", "dimension_scores", "meets_threshold"]
            },
            
            PromptType.RESPONSE_REFINEMENT.value: {
                "type": "object",
                "properties": {
                    "refined_content": {"type": "string"},
                    "refined_actions": {"type": "array", "items": {"type": "string"}},
                    "refined_next_steps": {"type": "array", "items": {"type": "string"}},
                    "confidence_boost": {"type": "number", "minimum": 0, "maximum": 0.5},
                    "applied_improvements": {"type": "array", "items": {"type": "string"}},
                    "refinement_notes": {"type": "string"}
                },
                "required": ["refined_content", "refined_actions", "refined_next_steps"]
            }
        }
    
    def _get_intent_analysis_template(self) -> str:
        """Intent analysis prompt template"""
        return """
        You are a world-class travel planning expert with deep understanding of user intentions.

        <role_definition>
        - Professional travel consultant with years of international travel planning experience
        - Expert in understanding travel habits and preferences across different cultures
        - Skilled at extracting explicit and implicit requirements from user messages
        </role_definition>

        <analysis_task>
        Please analyze the following user message and extract detailed travel intention information:

        User Message: "{user_message}"

        <analysis_dimensions>
        1. Intent Type Identification:
           - planning: Need to create a complete travel plan
           - query: Asking for specific information
           - recommendation: Seeking recommendation advice
           - modification: Modifying existing plans
           - booking: Booking related services
           - complaint: Complaint or issue feedback

        2. Destination Analysis:
           - Primary destination (explicitly mentioned)
           - Secondary destinations (implied or related)
           - Regional classification (Asia, Europe, etc.)
           - Confidence assessment

        3. Travel Details Extraction:
           - Trip duration
           - Number of travelers
           - Budget information (amount, currency, level)
           - Travel dates (fixed/flexible)

        4. Preference Identification:
           - Travel style (luxury, budget, backpacking, etc.)
           - Interests and hobbies
           - Accommodation preferences
           - Transportation preferences

        5. Emotion and Urgency:
           - User emotional state
           - Request urgency level
           - Missing critical information
           - Core requirements summary
        </analysis_dimensions>

        <output_requirements>
        Please return the analysis results in strict JSON format, ensuring all fields have values.
        For uncertain information, mark as "unknown" or use reasonable defaults.
        The confidence_score reflects your confidence in the overall analysis.

        JSON Format:
        {{
            "intent_type": "...",
            "destination": {{
                "primary": "...",
                "secondary": [...],
                "region": "...",
                "confidence": 0.0-1.0
            }},
            "travel_details": {{
                "duration": number,
                "travelers": number,
                "budget": {{
                    "mentioned": boolean,
                    "amount": number,
                    "currency": "...",
                    "level": "..."
                }},
                "dates": {{
                    "departure": "...",
                    "return": "...",
                    "flexibility": "..."
                }}
            }},
            "preferences": {{
                "travel_style": "...",
                "interests": [...],
                "accommodation_type": "...",
                "transport_preference": "..."
            }},
            "sentiment": "...",
            "urgency": "...",
            "missing_info": [...],
            "key_requirements": [...],
            "confidence_score": 0.0-1.0
        }}
        """
    
    def _get_requirement_extraction_template(self) -> str:
        """Requirement extraction prompt template"""
        return """
        You are a travel planning analyst specializing in detailed requirement extraction.

        <context>
        Extract comprehensive travel requirements from user intent and context.
        Consider budget sensitivity, time constraints, and implicit preferences.
        </context>

        <analysis_framework>
        1. BUDGET ANALYSIS
           - Stated budget vs implied budget level
           - Budget flexibility indicators
           - Price sensitivity markers

        2. TIME ANALYSIS
           - Fixed vs flexible dates
           - Duration preferences
           - Seasonal considerations

        3. PREFERENCE ANALYSIS
           - Travel style indicators
           - Activity preferences
           - Comfort level requirements

        4. CONSTRAINT ANALYSIS
           - Hard constraints (must-haves)
           - Soft constraints (nice-to-haves)
           - Deal-breakers

        5. TOOL NECESSITY ASSESSMENT
           - Flight search necessity (0-1 score)
           - Hotel search necessity (0-1 score)
           - Attraction search necessity (0-1 score)
        </analysis_framework>

        <input_data>
        User Message: "{user_message}"
        Intent Analysis: {intent_analysis}
        Context: {context}
        </input_data>

        Provide detailed requirement analysis in JSON format matching the schema.
        """
    
    def _get_tool_selection_template(self) -> str:
        """Tool selection prompt template"""
        return """
        You are an intelligent tool selection expert specialized in choosing the most appropriate tool combinations for travel planning tasks.

        <available_tools>
        1. flight_search: Search for flight information
           - Use cases: Need flight info, cross-city travel
           - Cost: Medium API call cost
           - Value: High, provides real-time flight data

        2. hotel_search: Search for hotel accommodations
           - Use cases: Need accommodation, overnight travel
           - Cost: Medium API call cost
           - Value: High, provides lodging options

        3. attraction_search: Search for attractions and activities
           - Use cases: Find attractions, plan activities
           - Cost: Low API call cost
           - Value: High, enriches travel experience
        </available_tools>

        <user_intent_analysis>
        {intent_analysis}
        </user_intent_analysis>

        <selection_criteria>
        1. Relevance: How well tools match user needs
        2. Necessity: Importance of tools for task completion
        3. Efficiency: Cost-benefit ratio
        4. User Experience: Coordination of tool combinations
        </selection_criteria>

        <selection_strategy>
        - If user needs complete travel planning, select multiple tools
        - If user only asks for specific information, select relevant tools
        - Consider tool dependencies and execution order
        - Balance information completeness with response speed
        </selection_strategy>

        Please analyze the user intent, select the most appropriate tool combination, and explain your reasoning.

        Return in JSON format:
        {{
            "selected_tools": ["tool1", "tool2", ...],
            "tool_priority": {{
                "tool1": 0.9,
                "tool2": 0.7,
                ...
            }},
            "execution_strategy": "sequential|parallel|conditional",
            "reasoning": "Detailed reasoning for tool selection...",
            "confidence": 0.0-1.0
        }}
        """
    
    def _get_response_generation_template(self) -> str:
        """Response generation prompt template"""
        return """
        You are a professional travel consultant generating comprehensive travel responses.

        <context>
        Generate a helpful, informative response based on the travel analysis and tool results.
        Maintain a professional yet friendly tone appropriate for travel planning.
        </context>

        <input_data>
        User Message: "{user_message}"
        Intent Analysis: {intent_analysis}
        Tool Results: {tool_results}
        Knowledge Context: {knowledge_context}
        </input_data>

        <response_guidelines>
        1. Address the user's specific intent and needs
        2. Incorporate tool results naturally into the response
        3. Provide actionable recommendations
        4. Include relevant context from knowledge base
        5. Maintain helpful and professional tone
        6. Structure information clearly with appropriate formatting
        </response_guidelines>

        Generate a comprehensive travel planning response.
        """
    
    def _get_quality_assessment_template(self) -> str:
        """Quality assessment prompt template"""
        return """
        You are a quality assessment expert for travel planning responses.

        <assessment_task>
        Evaluate the following travel planning response across multiple dimensions:

        Original User Request: "{original_message}"
        Agent Response: "{agent_response}"
        </assessment_task>

        <assessment_criteria>
        1. RELEVANCE (0-1): How well does the response address the user's specific request?
        2. COMPLETENESS (0-1): Does the response cover all important aspects?
        3. ACCURACY (0-1): Is the information provided factually correct?
        4. PRACTICALITY (0-1): Are the suggestions actionable and feasible?
        5. PERSONALIZATION (0-1): Is the response tailored to user preferences?
        6. FEASIBILITY (0-1): Are the recommendations realistic and achievable?
        </assessment_criteria>

        <quality_threshold>
        A response meets the quality threshold if:
        - Overall score >= 0.75
        - No dimension scores < 0.6
        - All critical travel planning elements are addressed
        </quality_threshold>

        For each dimension, provide:
        - Score (0.0 to 1.0)
        - Specific improvement suggestions
        - Missing elements that should be added
        - Strengths of the current response

        Provide detailed analysis in JSON format matching the schema.
        """
    
    def _get_response_refinement_template(self) -> str:
        """Response refinement prompt template"""
        return """
        You are a response refinement expert specializing in improving travel planning responses.

        <refinement_task>
        Improve the following travel response based on quality assessment feedback:

        Original Response: "{original_response}"
        Quality Assessment: {quality_assessment}
        Improvement Areas: {improvement_areas}
        </refinement_task>

        <refinement_guidelines>
        1. Address specific weaknesses identified in quality assessment
        2. Enhance personalization and relevance
        3. Add missing practical information
        4. Improve actionability with concrete next steps
        5. Maintain the original helpful tone
        6. Ensure all suggestions are feasible
        </refinement_guidelines>

        Generate an improved version of the travel response.
        """
    
    def _get_rag_generation_template(self) -> str:
        """RAG generation prompt template"""
        return """
        Based on the following knowledge, please answer the user's question:

        Knowledge:
        {context}

        User Question: {query}

        <response_guidelines>
        1. Use the provided knowledge to answer accurately
        2. If knowledge is insufficient, state what's missing
        3. Provide practical and actionable information
        4. Structure the response clearly
        5. Include relevant details from the knowledge base
        </response_guidelines>

        Please provide an accurate and helpful answer:
        """
    
    def _get_function_calling_template(self) -> str:
        """Function calling prompt template"""
        return """
        You are an AI assistant that can call functions to help users with travel planning.

        <available_functions>
        {functions}
        </available_functions>

        <user_message>
        {user_message}
        </user_message>

        <instructions>
        1. Analyze the user's message and determine which functions to call
        2. Extract appropriate parameters for each function
        3. Call functions in the optimal order
        4. Use function results to provide comprehensive responses
        </instructions>

        Please analyze the message and call the appropriate functions.
        """
    
    def get_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """Get and render prompt template"""
        template = self.templates.get(prompt_type.value)
        if not template:
            raise ValueError(f"Prompt template for {prompt_type.value} not found")
        
        return template.format(**kwargs)
    
    def get_schema(self, prompt_type: PromptType) -> Dict[str, Any]:
        """Get corresponding JSON schema"""
        return self.schemas.get(prompt_type.value, {})
    
    def validate_response(self, prompt_type: PromptType, response: Dict[str, Any]) -> bool:
        """Validate LLM response against expected schema"""
        schema = self.get_schema(prompt_type)
        if not schema:
            return True
        
        # Basic validation - can be enhanced with jsonschema library
        required_fields = schema.get("required", [])
        return all(field in response for field in required_fields)
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt types"""
        return list(self.templates.keys())

# Create global instance
prompt_manager = PromptManager() 