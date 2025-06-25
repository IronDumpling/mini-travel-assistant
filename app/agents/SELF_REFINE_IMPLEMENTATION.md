# Self-Refine Loop Implementation

## Overview

The self-refine loop is a critical component of the AI Travel Planning Agent that enables continuous quality improvement through iterative response refinement. This document provides a comprehensive guide to the implementation, usage, and customization of the self-refine system.

## Architecture Design Decision

### Placement: BaseAgent vs TravelAgent

**Decision: Implemented in BaseAgent with TravelAgent customization**

**Rationale:**
- **Framework Consistency**: Self-refinement is a core AI Agent capability that should be available to all agents
- **Future Scalability**: Other agents (BookingAgent, BudgetAgent, LocalExpertAgent) will inherit this capability
- **Customization Flexibility**: Each agent can override quality dimensions and refinement logic
- **Separation of Concerns**: BaseAgent provides the framework, specific agents customize the implementation

## Implementation Architecture

### Core Components

```python
# 1. Quality Assessment Framework
class QualityAssessment(BaseModel):
    overall_score: float          # 0.0 to 1.0
    dimension_scores: Dict[str, float]
    improvement_suggestions: List[str]
    meets_threshold: bool
    assessment_details: Dict[str, Any]

# 2. Self-Refine Loop Process
async def process_with_refinement(self, message: AgentMessage) -> AgentResponse:
    # Iterative refinement loop (max 3 iterations)
    # Quality assessment after each iteration
    # Early termination when threshold met
    # Comprehensive metadata tracking
```

### Quality Dimensions

#### Base Agent Dimensions (Default)
```python
{
    "relevance": 0.3,      # How relevant to user's request
    "completeness": 0.25,  # How complete the response is
    "accuracy": 0.25,      # How accurate the information
    "actionability": 0.2   # How actionable/useful
}
```

#### Travel Agent Dimensions (Customized)
```python
{
    "relevance": 0.25,          # Travel request relevance
    "completeness": 0.20,       # Travel information completeness
    "accuracy": 0.20,           # Travel information accuracy
    "actionability": 0.15,      # Travel recommendation actionability
    "personalization": 0.10,    # User preference personalization
    "feasibility": 0.10         # Travel plan feasibility
}
```

## Usage Examples

### Basic Usage

```python
from app.agents.travel_agent import TravelAgent
from app.agents.base_agent import AgentMessage

# Create agent with self-refinement enabled
agent = TravelAgent()

# Create message
message = AgentMessage(
    sender="user",
    receiver="travel_agent",
    content="Plan a family trip to Paris for 5 days, budget $3000"
)

# Process with self-refinement (default)
response = await agent.plan_travel(message)

# Check refinement results
if "refinement_iteration" in response.metadata:
    print(f"Refined {response.metadata['refinement_iteration']} times")
    print(f"Final quality score: {response.metadata['quality_score']:.2f}")
```

### Configuration Options

```python
# Configure refinement settings
agent.configure_refinement(
    enabled=True,              # Enable/disable refinement
    quality_threshold=0.8,     # Quality threshold (0.0-1.0)
    max_iterations=3           # Maximum refinement iterations
)

# Check current configuration
status = agent.get_status()
print(status["refinement_config"])
```

### API Usage

```bash
# Test via REST API
curl -X POST "http://localhost:8000/demo/self-refine" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need a romantic getaway to Italy for 7 days",
    "enable_refinement": true
  }'
```

## Implementation Details

### Self-Refine Loop Flow

```
1. Initial Response Generation
   ├── Call process_message() to generate initial response
   ├── Assess quality using quality dimensions
   └── Check if threshold met → Early termination if yes

2. Iterative Refinement (if needed)
   ├── Generate improvement suggestions
   ├── Apply refinement logic (_refine_response)
   ├── Re-assess quality
   └── Continue until threshold met or max iterations reached

3. Response Enhancement
   ├── Add refinement metadata
   ├── Track quality improvements
   └── Return enhanced response
```

### Quality Assessment Process

```python
async def _assess_response_quality(self, message, response, iteration):
    # 1. Get quality dimensions for this agent
    dimensions = self.get_quality_dimensions()
    
    # 2. Assess each dimension
    dimension_scores = {}
    for dimension, weight in dimensions.items():
        score = await self._assess_dimension(dimension, message, response)
        dimension_scores[dimension] = score
    
    # 3. Calculate weighted overall score
    overall_score = sum(score * weight for score, weight in zip(
        dimension_scores.values(), dimensions.values()
    )) / sum(dimensions.values())
    
    # 4. Generate improvement suggestions
    suggestions = []
    for dim, score in dimension_scores.items():
        if score < 0.6:  # Threshold for improvement
            suggestions.extend(await self._generate_improvement_suggestions(
                dim, message, response, score
            ))
    
    return QualityAssessment(...)
```

### Travel-Specific Enhancements

#### Personalization Assessment
```python
async def _assess_personalization(self, message, response):
    # Check for preference indicators in user message
    preference_indicators = [
        "budget", "prefer", "family", "luxury", "adventure", etc.
    ]
    
    # Calculate personalization ratio
    mentioned = count_indicators_in_message(message, preference_indicators)
    addressed = count_indicators_in_response(response, preference_indicators)
    
    return calculate_personalization_score(mentioned, addressed)
```

#### Feasibility Assessment
```python
async def _assess_feasibility(self, message, response):
    # Check for practical considerations
    practical_elements = [
        "time", "schedule", "transport", "booking", "weather", etc.
    ]
    
    # Score based on practical mentions and tool usage
    practical_score = count_practical_mentions(response, practical_elements)
    tool_usage_score = len(response.actions_taken) * 0.1
    
    return min(base_score + practical_score + tool_usage_score, 1.0)
```

## Customization Guide

### Creating Custom Quality Dimensions

```python
class CustomAgent(BaseAgent):
    def get_quality_dimensions(self) -> Dict[str, float]:
        return {
            "relevance": 0.3,
            "custom_dimension": 0.2,
            "another_dimension": 0.5
        }
    
    async def _assess_dimension(self, dimension, message, response):
        if dimension == "custom_dimension":
            return await self._assess_custom_dimension(message, response)
        return await super()._assess_dimension(dimension, message, response)
```

### Custom Refinement Logic

```python
async def _refine_response(self, message, response, quality_assessment):
    # Apply base refinements
    improved_response = await super()._refine_response(
        message, response, quality_assessment
    )
    
    # Apply custom refinements
    if quality_assessment.dimension_scores.get("custom_dimension", 1.0) < 0.6:
        improved_response.content += "\n\nCustom enhancement applied!"
    
    return improved_response
```

## Performance Considerations

### Optimization Strategies

1. **Early Termination**: Stop refinement when quality threshold is met
2. **Dimension Weighting**: Focus on most important quality aspects
3. **Caching**: Cache quality assessments for similar responses
4. **Parallel Assessment**: Assess dimensions concurrently when possible

### Performance Metrics

```python
# Track refinement performance
refinement_metrics = {
    "average_iterations": 1.8,
    "early_termination_rate": 0.65,
    "quality_improvement": 0.23,
    "processing_time_overhead": "15%"
}
```

## Testing and Validation

### Unit Tests

```python
# Test quality assessment
async def test_quality_assessment():
    agent = TravelAgent()
    message = create_test_message()
    response = create_test_response()
    
    assessment = await agent._assess_response_quality(message, response, 1)
    
    assert 0.0 <= assessment.overall_score <= 1.0
    assert len(assessment.dimension_scores) == 6  # Travel agent dimensions
    assert isinstance(assessment.improvement_suggestions, list)

# Test refinement loop
async def test_refinement_loop():
    agent = TravelAgent()
    agent.configure_refinement(quality_threshold=0.9, max_iterations=2)
    
    response = await agent.plan_travel(test_message)
    
    assert "refinement_iteration" in response.metadata
    assert response.metadata["quality_score"] > 0.5
```

### Integration Tests

```python
# Run the test script
python test_self_refine.py
```

### API Testing

```bash
# Test refinement enabled vs disabled
curl -X POST "http://localhost:8000/demo/self-refine" \
  -d '{"message": "Plan Tokyo trip", "enable_refinement": true}'

curl -X POST "http://localhost:8000/demo/self-refine" \
  -d '{"message": "Plan Tokyo trip", "enable_refinement": false}'
```

## Monitoring and Metrics

### Quality Tracking

```python
# Refinement metadata in response
{
    "refinement_iteration": 2,
    "quality_score": 0.82,
    "refinement_status": "completed_early",
    "refinement_history": [
        {"iteration": 1, "score": 0.65, "suggestions": [...]},
        {"iteration": 2, "score": 0.82, "suggestions": [...]}
    ],
    "travel_refined": true,
    "quality_dimensions_improved": ["personalization", "feasibility"]
}
```

### System Metrics

- **Refinement Rate**: Percentage of responses that undergo refinement
- **Quality Improvement**: Average quality score improvement
- **Processing Overhead**: Additional time for refinement process
- **Early Termination Rate**: Percentage of refinements that terminate early

## Future Enhancements

### Planned Improvements

1. **LLM-Based Assessment**: Use LLM for more sophisticated quality assessment
2. **Learning from Feedback**: Incorporate user feedback into quality scoring
3. **Adaptive Thresholds**: Dynamically adjust quality thresholds based on context
4. **Multi-Agent Refinement**: Collaborative refinement across multiple agents

### Integration Opportunities

- **RAG Enhancement**: Use refinement to improve knowledge retrieval
- **Tool Coordination**: Refine tool selection and execution strategies
- **Memory Integration**: Learn from refinement patterns for future responses
- **User Preference Learning**: Incorporate refinement outcomes into user profiles

## Conclusion

The self-refine loop implementation provides a robust framework for continuous quality improvement in AI agent responses. By implementing it in the BaseAgent class with travel-specific customizations in TravelAgent, we've created a scalable and flexible system that can be extended to other agent types while maintaining consistent quality standards.

The system successfully balances automation with customization, providing intelligent refinement capabilities while allowing for domain-specific quality assessments and improvement strategies. 