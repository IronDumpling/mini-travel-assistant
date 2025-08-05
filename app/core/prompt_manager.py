"""
Prompt Manager - Centralized prompt template management system
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Prompt type enumeration"""

    INTENT_ANALYSIS = "intent_analysis"
    REQUIREMENT_EXTRACTION = "requirement_extraction"
    TOOL_SELECTION = "tool_selection"
    RESPONSE_GENERATION = "response_generation"
    INFORMATION_FUSION = "information_fusion"
    QUALITY_ASSESSMENT = "quality_assessment"
    RESPONSE_REFINEMENT = "response_refinement"
    RAG_GENERATION = "rag_generation"
    FUNCTION_CALLING = "function_calling"
    PLAN_GENERATION = "plan_generation"
    PLAN_MODIFICATION = "plan_modification"
    EVENT_EXTRACTION = "event_extraction"


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
            PromptType.INFORMATION_FUSION.value: self._get_information_fusion_template(),
            PromptType.QUALITY_ASSESSMENT.value: self._get_quality_assessment_template(),
            PromptType.RESPONSE_REFINEMENT.value: self._get_response_refinement_template(),
            PromptType.RAG_GENERATION.value: self._get_rag_generation_template(),
            PromptType.FUNCTION_CALLING.value: self._get_function_calling_template(),
            PromptType.PLAN_GENERATION.value: self._get_plan_generation_template(),
            PromptType.PLAN_MODIFICATION.value: self._get_plan_modification_template(),
            PromptType.EVENT_EXTRACTION.value: self._get_event_extraction_template(),
        }

    def _initialize_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize JSON schemas for structured outputs"""
        return {
            PromptType.INTENT_ANALYSIS.value: {
                "type": "object",
                "properties": {
                    "intent_type": {
                        "type": "string",
                        "enum": [
                            "planning",
                            "query",
                            "recommendation",
                            "modification",
                            "booking",
                            "complaint",
                        ],
                    },
                    "destination": {
                        "type": "object",
                        "properties": {
                            "primary": {"type": "string"},
                            "secondary": {"type": "array", "items": {"type": "string"}},
                            "region": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
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
                                    "level": {
                                        "type": "string",
                                        "enum": ["budget", "mid-range", "luxury"],
                                    },
                                },
                            },
                            "dates": {
                                "type": "object",
                                "properties": {
                                    "departure": {"type": "string"},
                                    "return": {"type": "string"},
                                    "flexibility": {
                                        "type": "string",
                                        "enum": ["fixed", "flexible", "unknown"],
                                    },
                                },
                            },
                        },
                    },
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "travel_style": {
                                "type": "string",
                                "enum": [
                                    "luxury",
                                    "mid-range",
                                    "budget",
                                    "backpacking",
                                    "business",
                                    "family",
                                ],
                            },
                            "interests": {"type": "array", "items": {"type": "string"}},
                            "accommodation_type": {"type": "string"},
                            "transport_preference": {"type": "string"},
                        },
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": [
                            "positive",
                            "neutral",
                            "negative",
                            "excited",
                            "worried",
                        ],
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                    "missing_info": {"type": "array", "items": {"type": "string"}},
                    "key_requirements": {"type": "array", "items": {"type": "string"}},
                    "information_fusion_strategy": {
                        "type": "object",
                        "properties": {
                            "knowledge_priority": {
                                "type": "string",
                                "enum": ["very_high", "high", "medium", "low"],
                            },
                            "tool_priority": {
                                "type": "string",
                                "enum": ["very_high", "high", "medium", "low"],
                            },
                            "integration_approach": {
                                "type": "string",
                                "enum": ["knowledge_first", "tools_first", "balanced"],
                            },
                            "response_focus": {
                                "type": "string",
                                "enum": [
                                    "comprehensive_plan",
                                    "detailed_information",
                                    "curated_options",
                                    "actionable_steps",
                                    "specific_changes",
                                ],
                            },
                        },
                    },
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "intent_type",
                    "destination",
                    "sentiment",
                    "urgency",
                    "information_fusion_strategy",
                    "confidence_score",
                ],
            },
            PromptType.REQUIREMENT_EXTRACTION.value: {
                "type": "object",
                "properties": {
                    "budget_sensitivity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "time_sensitivity": {
                        "type": "string",
                        "enum": ["urgent", "normal", "flexible"],
                    },
                    "travel_style": {"type": "string"},
                    "geographic_scope": {"type": "string"},
                    "tool_necessity_scores": {"type": "object"},
                    "preferences": {"type": "object"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["budget_sensitivity", "time_sensitivity", "confidence"],
            },
            PromptType.TOOL_SELECTION.value: {
                "type": "object",
                "properties": {
                    "selected_tools": {"type": "array", "items": {"type": "string"}},
                    "tool_priority": {"type": "object"},
                    "execution_strategy": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "conditional"],
                    },
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "selected_tools",
                    "execution_strategy",
                    "reasoning",
                    "confidence",
                ],
            },
            PromptType.QUALITY_ASSESSMENT.value: {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "dimension_scores": {
                        "type": "object",
                        "properties": {
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                            "completeness": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                            "practicality": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "personalization": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "feasibility": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                    "improvement_suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_elements": {"type": "array", "items": {"type": "string"}},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "meets_threshold": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["overall_score", "dimension_scores", "meets_threshold"],
            },
            PromptType.RESPONSE_REFINEMENT.value: {
                "type": "object",
                "properties": {
                    "refined_content": {"type": "string"},
                    "refined_actions": {"type": "array", "items": {"type": "string"}},
                    "refined_next_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence_boost": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 0.5,
                    },
                    "applied_improvements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "refinement_notes": {"type": "string"},
                },
                "required": [
                    "refined_content",
                    "refined_actions",
                    "refined_next_steps",
                ],
            },
            PromptType.PLAN_GENERATION.value: {
                "type": "object",
                "properties": {
                    "natural_response": {"type": "string"},
                    "structured_plan": {
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string"},
                            "duration": {"type": "number"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "travelers": {"type": "number"},
                            "budget_estimate": {"type": "object"},
                            "metadata": {"type": "object"}
                        }
                    },
                    "plan_events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "event_type": {"type": "string"},
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"},
                                "location": {"type": "string"},
                                "coordinates": {"type": "object"},
                                "details": {"type": "object"}
                            },
                            "required": ["title", "event_type", "start_time", "end_time", "location"]
                        }
                    }
                },
                "required": ["natural_response", "structured_plan", "plan_events"]
            },
            PromptType.PLAN_MODIFICATION.value: {
                "type": "object",
                "properties": {
                    "new_events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "event_type": {"type": "string"},
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"},
                                "location": {"type": "string"},
                                "details": {"type": "object"}
                            },
                            "required": ["title", "event_type", "start_time", "end_time", "location"]
                        }
                    },
                    "updated_events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"}
                            },
                            "required": ["id"]
                        }
                    },
                    "deleted_event_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "plan_modifications": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "impact": {"type": "string"}
                        }
                    }
                },
                "required": ["new_events", "updated_events", "deleted_event_ids"]
            },
            PromptType.EVENT_EXTRACTION.value: {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "event_type": {"type": "string"},
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"},
                                "location": {"type": "string"}
                            },
                            "required": ["title", "event_type", "start_time", "end_time", "location"]
                        }
                    }
                },
                "required": ["events"]
            },
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

        6. Information Fusion Strategy:
           - What static knowledge would be most valuable for this request?
           - What dynamic/real-time data is essential?
           - How should information sources be prioritized?
           - What is the optimal response structure and focus?
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
            "information_fusion_strategy": {{
                "knowledge_priority": "high|medium|low",
                "tool_priority": "high|medium|low", 
                "integration_approach": "knowledge_first|tools_first|balanced",
                "response_focus": "comprehensive_plan|detailed_information|curated_options|actionable_steps|specific_changes"
            }},
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
        """Enhanced response generation prompt template with information fusion support"""
        return """
        You are a professional travel consultant generating comprehensive travel responses with intelligent information integration.

        <role_definition>
        Act as an expert travel planner who seamlessly combines authoritative knowledge with current data
        to provide comprehensive, actionable travel guidance.
        </role_definition>

        <input_data>
        User Message: "{user_message}"
        Intent Analysis: {intent_analysis}
        Tool Results: {tool_results}
        Knowledge Context: {knowledge_context}
        </input_data>

        <information_integration_guidelines>
        1. KNOWLEDGE CONTEXT (Static Authority):
           - Use for comprehensive destination details and background information
           - Leverage for cultural insights, historical context, and detailed descriptions
           - Provides authoritative foundation for recommendations

        2. TOOL RESULTS (Dynamic Currency):
           - Use for current prices, availability, and real-time data
           - Prioritize for actionable booking information and current options
           - Ensures recommendations are current and practical

        3. INTENT ANALYSIS (Smart Guidance):
           - Let user intent guide information prioritization and response structure
           - Address explicit needs directly and anticipate implicit requirements
           - Shape the response focus and level of detail
        </information_integration_guidelines>

        <response_requirements>
        1. Address the user's specific intent and needs directly
        2. Seamlessly integrate static knowledge with dynamic tool results
        3. Provide actionable recommendations based on current data
        4. Include rich contextual details from authoritative sources
        5. Maintain professional yet friendly tone appropriate for travel planning
        6. Structure information clearly with logical flow and appropriate formatting
        7. Ensure all information is accurate, consistent, and helpful
        </response_requirements>

        Generate a comprehensive, well-integrated travel planning response.
        """

    def _get_information_fusion_template(self) -> str:
        """Information fusion prompt template for intelligent multi-source integration"""
        return """
        You are an expert travel consultant creating comprehensive responses by intelligently fusing multiple information sources.
        
        <information_sources>
        Static Knowledge Context (Authoritative Details):
        {knowledge_context}
        
        Dynamic Tool Results (Current Data):
        {tool_results}
        
        User Intent Analysis (Smart Understanding):
        {intent_analysis}
        </information_sources>
        
        <fusion_strategy>
        INTELLIGENT INTEGRATION PRINCIPLES:
        
        1. INFORMATION HIERARCHY:
           - Use KNOWLEDGE CONTEXT for comprehensive details and background information
           - Use TOOL RESULTS for specific current recommendations and actionable data
           - Let INTENT ANALYSIS guide which information to emphasize
        
        2. CONFLICT RESOLUTION:
           - Tool results take precedence for current data (prices, availability)
           - Knowledge context takes precedence for cultural/historical information
           - Always favor more specific over general information
        
        3. NATURAL INTEGRATION:
           - Blend sources seamlessly without explicitly mentioning "sources"
           - Provide comprehensive yet digestible information
           - Ensure accuracy and consistency across all information
        </fusion_strategy>
        
        <user_request>
        {user_message}
        </user_request>
        
        <output_requirements>
        Create a well-structured, comprehensive travel response that:
        - Directly addresses the user's request
        - Intelligently combines static knowledge with dynamic data
        - Provides actionable recommendations based on current information
        - Includes rich details from authoritative sources where relevant
        - Maintains professional yet friendly tone
        </output_requirements>
        
        Generate the intelligently fused response:
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
        """
        Get and render prompt template with optimized performance
        Includes error handling and safe formatting
        """
        template = self.templates.get(prompt_type.value)
        if not template:
            # Return a basic fallback prompt instead of raising error
            return f"Please help with {prompt_type.value} task. {kwargs.get('user_message', '')}"

        try:
            # Safe template formatting with fallback values
            safe_kwargs = {}
            for key, value in kwargs.items():
                if value is None:
                    safe_kwargs[key] = "unknown"
                elif isinstance(value, dict):
                    safe_kwargs[key] = str(value)
                elif isinstance(value, list):
                    safe_kwargs[key] = ", ".join(str(item) for item in value)
                else:
                    safe_kwargs[key] = str(value)
            
            return template.format(**safe_kwargs)
            
        except KeyError as e:
            # Handle missing template variables gracefully
            logger.warning(f"Missing template variable {e} for {prompt_type.value}")
            return template.format_map(safe_kwargs)
        except Exception as e:
            # Fallback for any other formatting errors
            logger.error(f"Error formatting prompt {prompt_type.value}: {e}")
            return f"Please help with {prompt_type.value} task. {kwargs.get('user_message', '')}"

    def get_schema(self, prompt_type: PromptType) -> Dict[str, Any]:
        """Get corresponding JSON schema"""
        return self.schemas.get(prompt_type.value, {})

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt types"""
        return list(self.templates.keys())

    def _get_plan_generation_template(self) -> str:
        """Plan generation prompt template"""
        return """
        You are a professional travel planner generating a structured, detailed travel plan.
        
        <context>
        User Request: {user_message}
        Destination: {destination}
        Tool Results: {tool_results}
        Knowledge Context: {knowledge_context}
        Intent: {intent}
        </context>
        
        <requirements>
        Generate a complete travel plan with:
        1. Detailed daily itinerary with specific times
        2. Flight, hotel, and attraction events from tool results
        3. Precise event scheduling (hour-level precision)
        4. Practical information and recommendations
        5. Realistic timing and logistics
        </requirements>
        
        <output_format>
        Return JSON with three parts:
        1. "natural_response": User-friendly description of the travel plan
        2. "structured_plan": Complete plan metadata
        3. "plan_events": Detailed list of events with precise timing
        
        {{
            "natural_response": "Here's your detailed travel plan for [destination]...",
            "structured_plan": {{
                "destination": "Primary destination name",
                "duration": 7,
                "start_date": "2024-07-01",
                "end_date": "2024-07-07",
                "travelers": 2,
                "budget_estimate": {{"currency": "USD", "amount": 2500}},
                "metadata": {{
                    "travel_style": "moderate",
                    "season": "summer",
                    "generated_at": "2024-01-15T10:00:00Z"
                }}
            }},
            "plan_events": [
                {{
                    "id": "flight_outbound_001",
                    "title": "Flight to [Destination]",
                    "description": "Outbound flight details and travel instructions",
                    "event_type": "flight",
                    "start_time": "2024-07-01T08:00:00+00:00",
                    "end_time": "2024-07-01T14:00:00+00:00",
                    "location": "Departure Airport → Destination Airport",
                    "coordinates": {{"lat": 40.7128, "lng": -74.0060}},
                    "details": {{
                        "source": "flight_search",
                        "airline": "Airline name",
                        "flight_number": "AA123",
                        "price": {{"amount": 450, "currency": "USD"}},
                        "booking_info": {{"booking_url": "...", "confirmation": "..."}},
                        "recommendations": ["Arrive 2 hours early", "Check baggage restrictions"]
                    }}
                }},
                {{
                    "id": "hotel_checkin_001",
                    "title": "Hotel Check-in: [Hotel Name]",
                    "description": "Hotel accommodation details and check-in process",
                    "event_type": "hotel",
                    "start_time": "2024-07-01T15:00:00+00:00",
                    "end_time": "2024-07-04T11:00:00+00:00",
                    "location": "Hotel Address, City",
                    "coordinates": {{"lat": 40.7589, "lng": -73.9851}},
                    "details": {{
                        "source": "hotel_search",
                        "rating": 4.5,
                        "price_per_night": {{"amount": 180, "currency": "USD"}},
                        "amenities": ["WiFi", "Breakfast", "Gym"],
                        "booking_info": {{"booking_url": "...", "cancellation_policy": "..."}},
                        "recommendations": ["Request room with city view", "Join loyalty program"]
                    }}
                }},
                {{
                    "id": "attraction_visit_001",
                    "title": "Visit [Famous Attraction]",
                    "description": "Explore the famous landmark with guided tour",
                    "event_type": "attraction",
                    "start_time": "2024-07-02T09:00:00+00:00",
                    "end_time": "2024-07-02T12:00:00+00:00",
                    "location": "Attraction Address, City",
                    "coordinates": {{"lat": 40.7484, "lng": -73.9857}},
                    "details": {{
                        "source": "attraction_search",
                        "rating": 4.8,
                        "price": {{"amount": 25, "currency": "USD"}},
                        "opening_hours": "9:00 AM - 6:00 PM",
                        "booking_info": {{"advance_booking_required": true, "ticket_url": "..."}},
                        "recommendations": ["Book tickets in advance", "Bring camera", "Wear comfortable shoes"]
                    }}
                }}
            ]
        }}
        </output_format>
        
        <important_notes>
        1. Ensure all timestamps are in ISO 8601 format with timezone
        2. Make realistic time allocations (travel time, meal breaks, etc.)
        3. Use actual data from tool results when available
        4. Include practical recommendations and tips
        5. Coordinate events logically (check-in before activities, etc.)
        6. Generate unique IDs for each event
        7. Provide fallback events if tool results are insufficient
        </important_notes>
        """
    
    def _get_plan_aware_fusion_template(self) -> str:
        """Plan-aware fusion prompt template for integrating existing travel plans"""
        return """You are a travel planning assistant with access to an existing travel plan. Generate a helpful response that considers both the current plan and new information.

EXISTING TRAVEL PLAN CONTEXT:
Current plan events: {existing_events}
Identified gaps: {plan_gaps}
Last updated: {last_updated}

USER REQUEST: {user_message}

TRAVEL INTENT ANALYSIS:
{formatted_intent}

NEW INFORMATION FROM TOOLS:
{formatted_tools}

KNOWLEDGE BASE INSIGHTS:
{formatted_knowledge}

INSTRUCTIONS:
1. Acknowledge the user's existing plan when relevant
2. Identify any conflicts with existing events and suggest resolutions
3. Fill gaps in the current plan based on user's request
4. Suggest specific updates or additions to improve the plan
5. Maintain travel plan continuity and logical flow
6. Be specific about timing, locations, and practical details
7. If suggesting changes, explain why they improve the overall plan

RESPONSE REQUIREMENTS:
- Start by acknowledging relevant existing plan elements
- Integrate new findings with current plan
- Suggest specific, actionable updates
- Maintain helpful, travel-focused tone
- Provide practical, implementable advice

Generate a comprehensive response that helps the user optimize their travel plan:"""

    def _get_plan_modification_template(self) -> str:
        """Plan modification prompt template for analyzing changes to existing travel plans"""
        return """Analyze this travel planning conversation and determine what changes should be made to the existing travel plan.

EXISTING PLAN EVENTS:
{existing_events_context}

USER REQUEST: {user_message}
AGENT RESPONSE: {agent_response}

TASK: Determine what modifications should be made to the existing plan based on this conversation.

CRITICAL DELETION RULES:
- If user wants to REMOVE a city/destination, you MUST delete ALL related events:
  * ALL hotel events for that city
  * ALL activity/attraction events for that city
  * ALL connecting flights to/from that city
- Look for phrases like "remove", "delete", "skip", "don't go to", "take out", "exclude"
- When removing destinations from multi-city trips, be thorough in cleaning up ALL related events

Analyze for:
1. NEW events to add (flights, hotels, attractions, restaurants, activities)
2. UPDATES to existing events (time changes, location changes, details updates)
3. DELETIONS of existing events (if user wants to remove something)
   - Pay special attention to city/destination removals
   - Include ALL events related to removed destinations
4. PLAN METADATA changes (destination, dates, budget, etc.)

For each event, determine:
- Event type (flight/hotel/attraction/restaurant/meal/transportation/activity/meeting/free_time)
- Title and description
- Start and end times (use ISO format: YYYY-MM-DDTHH:MM:SS+00:00)
- Location
- Any specific details mentioned

MEAL EVENT TIMING GUIDELINES:
For meal events, use these time ranges based on meal type:
- BREAKFAST: 07:00-09:00 (1-2 hours)
- BRUNCH: 10:00-12:00 (1-2 hours)  
- LUNCH: 12:00-14:00 (1-2 hours)
- AFTERNOON TEA/SNACK: 15:00-16:00 (1 hour)
- DINNER: 18:00-21:00 (1-3 hours)
- LATE DINNER: 19:30-22:00 (1-3 hours)
- BAR/DRINKS: 17:00-23:00 (1-4 hours)

Examples of proper meal events:
- "Breakfast at Café Central" → 08:00-09:00
- "Traditional Lunch at Bistro" → 12:30-14:00  
- "Dinner at Fine Restaurant" → 19:00-21:30
- "Evening Drinks at Rooftop Bar" → 20:00-22:00

NEVER make meal events all-day unless specifically requested (e.g., food festivals).

When identifying events to delete:
- Check event titles for city names mentioned in removal requests
- Check event locations for city names
- Check event details for destination codes (LON, PAR, etc.)
- Look for hotel events that mention the removed city
- Look for activities/attractions in the removed city

Return ONLY a valid JSON response with this exact structure:
{{
    "new_events": [
        {{
            "title": "Event Title",
            "description": "Event description",
            "event_type": "flight|hotel|attraction|restaurant|meal|transportation|activity|meeting|free_time",
            "start_time": "2025-07-27T10:00:00+00:00",
            "end_time": "2025-07-27T16:00:00+00:00",
            "location": "Location name",
            "details": {{}}
        }}
    ],
    "updated_events": [
        {{
            "id": "existing_event_id",
            "title": "Updated Title",
            "start_time": "2025-07-27T11:00:00+00:00",
            "end_time": "2025-07-27T17:00:00+00:00"
        }}
    ],
    "deleted_event_ids": ["event_id_to_delete"],
    "plan_modifications": {{
        "reason": "Why these changes were made",
        "impact": "How this affects the overall plan"
    }}
}}

If no changes are needed, return empty arrays for each section.

IMPORTANT: Keep your response concise. If adding multiple meal events, limit to 3-5 events total to avoid token limits. Focus on the most important additions requested by the user."""

    def _get_event_extraction_template(self) -> str:
        """Event extraction prompt template for extracting calendar events from conversations"""
        return """Extract calendar events from this travel planning conversation.

User: {user_message}
Agent: {agent_response}

Extract any specific travel events mentioned (flights, hotels, attractions, restaurants, activities).
For each event, determine:
- Title
- Type (flight/hotel/attraction/restaurant/meal/transportation/activity/meeting/free_time)
- Start time (estimate if not explicit)
- Duration/end time
- Location
- Description

MEAL EVENT TIMING GUIDELINES:
For meal events, use these time ranges based on meal type:
- BREAKFAST: 07:00-09:00 (1-2 hours)
- BRUNCH: 10:00-12:00 (1-2 hours)  
- LUNCH: 12:00-14:00 (1-2 hours)
- AFTERNOON TEA/SNACK: 15:00-16:00 (1 hour)
- DINNER: 18:00-21:00 (1-3 hours)
- LATE DINNER: 19:30-22:00 (1-3 hours)
- BAR/DRINKS: 17:00-23:00 (1-4 hours)

NEVER make meal events all-day unless specifically requested (e.g., food festivals).

Return as JSON array of events. If no specific events are mentioned, return empty array."""


# Create global instance
prompt_manager = PromptManager()
