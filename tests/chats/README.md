# Chat API Testing Framework

This testing framework provides comprehensive testing capabilities for the Chat API with two distinct testing modes and detailed refinement loop analysis. The framework is designed to test the DeepSeek-powered travel agent's self-refinement capabilities and performance across different conversation scenarios.

## 🔧 Features

### Two Testing Modes

1. **Single-Session Tests**: Different sessions with single questions each
   - Simulates users creating multiple separate travel plans
   - Each test creates a new session with one query
   - Good for testing initial response quality

2. **Multi-Session Tests**: Same session with multiple questions  
   - Simulates users developing detailed travel plans through conversation
   - Multiple queries within the same session context
   - Tests conversation memory and progression

### Metrics

- **Refinement Loop Tracking**: Records each iteration of the self-refinement process
- **Confidence Progression**: Tracks how confidence changes through refinement loops
- **Performance Analysis**: Compares single vs multi-session performance
- **Session Progression**: Analyzes how responses improve within a conversation

### Timestamped Results

- Results are saved in `results/YYYYMMDD_HHMMSS/` directories
- Each test run gets its own directory with complete isolation
- Easy to compare different test runs over time

### Enhanced Visualizations

- **Confidence Progression**: Shows how confidence scores evolve through refinement loops
- **Response Time Distribution**: Analyzes response time patterns and performance
- **Quality Assessment Tracking**: Monitors the 6-dimension quality scoring system
- **Session Progression Analysis**: Tracks conversation development over multiple queries

## 📁 File Structure

```
tests/chats/
├── chat_tester.py                 # Core testing framework
├── test_scenarios.py              # Test scenario definitions  
├── metrics_analyzer.py            # Metrics analysis and visualization
├── test_runner.py                 # Main test runner
├── test_chat_pytest.py            # Pytest-based unit tests
├── test_connectivity.py           # API connectivity tests
├── README.md                      # This documentation
└── results/
    └── YYYYMMDD_HHMMSS/           # Timestamped results
        ├── single_session_tests.json
        ├── multi_session_tests.json
        ├── test_summary.json
        ├── enhanced_analysis_report.txt
        ├── confidence_vs_loops.png
        ├── response_time_distribution.png
        └── *.png                   # Additional visualization files
```

## 🚀 Usage

### Basic Usage

```bash
# Run both test types without refinement (fastest)
python test_runner.py

# Run both test types with refinement
python test_runner.py --refinement

# Run only single-session tests
python test_runner.py --mode single

# Run only multi-session tests  
python test_runner.py --mode multi

# Run comparison tests (both with and without refinement)
python test_runner.py --mode comparison
```

### Advanced Options

```bash
# Custom timeout (useful for refinement tests)
python test_runner.py --refinement --timeout 600

# Custom API URL
python test_runner.py --base-url http://localhost:8080

# Skip analysis after testing
python test_runner.py --no-analysis

# Only analyze existing results
python test_runner.py --analyze-only
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--mode {single,multi,both,comparison}` | Test mode selection |
| `--refinement` | Enable self-refinement |
| `--no-refinement` | Explicitly disable refinement |
| `--analyze-only` | Only analyze existing results |
| `--no-analysis` | Skip analysis after tests |
| `--base-url URL` | API base URL |
| `--timeout SECONDS` | Request timeout |

## 📊 Test Scenarios

### Single-Session Scenarios
- London 7-day budget trip
- Paris romantic weekend
- Tokyo business trip
- Singapore family vacation
- European backpacking
- Dubai luxury trip
- Barcelona solo travel
- Swiss Alps ski trip

### Multi-Session Scenarios
- **Tokyo Trip Planning**: 7 detailed questions about Tokyo travel
- **European Multi-City**: 7 questions about Paris/Amsterdam/Berlin/Prague/Vienna
- **Family Beach Vacation**: 7 questions about Caribbean family trip
- **Business Travel**: 7 questions about NYC/Chicago/LA business trip
- **Adventure Backpacking**: 7 questions about Southeast Asia adventure

## 📈 Metrics and Analysis

### Refinement Loop Metrics
- Number of loops per test
- Confidence improvement per loop
- Time cost per loop
- Success rate by loop count

### Test Type Comparison  
- Success rates: Single vs Multi-session
- Response times: Average and distribution
- Confidence scores: Progression and final values
- Refinement usage: Loop counts and effectiveness

### Session Progression Analysis
- How responses improve within multi-session conversations
- Confidence trends across queries
- Response time optimization within sessions

## 📋 Prerequisites

1. **API Server Running**: The Chat API server must be running at the specified URL (default: http://localhost:8000)
2. **Knowledge Base Initialized**: Server should have the travel knowledge base loaded with ChromaDB
3. **DeepSeek API Key**: Valid DeepSeek API key configured in environment
4. **RAG Engine**: ChromaDB vector database initialized with travel knowledge
5. **Dependencies**: Required Python packages for testing and visualization

```bash
# Install dependencies
pip install httpx matplotlib numpy pandas seaborn pytest pytest-asyncio
```

## 🎯 Example Workflows

### Performance Testing
```bash
# Quick performance test without refinement
python test_runner.py --mode single --no-refinement

# Comprehensive test with refinement
python test_runner.py --mode both --refinement --timeout 600
```

### Development Testing
```bash
# Test conversation flow
python test_runner.py --mode multi --no-refinement

# Compare refinement impact
python test_runner.py --mode comparison
```

### Analysis Only
```bash
# Re-analyze the latest results
python test_runner.py --analyze-only

# Or directly run the analyzer
python metrics_analyzer.py
```

## 🧪 Additional Test Types

### Unit Tests (`test_chat_pytest.py`)
- Pytest-based unit tests for individual API endpoints
- Validation of request/response schemas
- Error handling and edge case testing
- Integration with DeepSeek LLM service

### Connectivity Tests (`test_connectivity.py`)
- API server availability and health checks
- Session management functionality validation
- Knowledge base initialization verification
- RAG engine connectivity testing

## 📊 Understanding Results

### Single-Session Results (`single_session_tests.json`)
Each entry contains:
- `test_id`: Unique identifier
- `session_id`: Session used for the test
- `refinement_loops`: Detailed loop-by-loop metrics
- `final_confidence`: Final confidence score
- `total_response_time`: Complete response time
- `refinement_enabled`: Whether self-refinement was used

### Multi-Session Results (`multi_session_tests.json`)
Each session suite contains:
- `session_title`: Description of the session
- `queries`: List of all queries in the session
- `metrics`: Array of metrics for each query
- `session_progression`: Analysis of conversation development

### Enhanced Analysis Report (`enhanced_analysis_report.txt`)
Comprehensive analysis including:
- Overall summary statistics
- Test type performance comparison
- Refinement loop effectiveness analysis
- Session progression insights
- Quality dimension breakdown
- Performance metrics and trends

### Enhanced Visualizations
- `confidence_vs_loops.png`: Detailed confidence progression through refinement iterations
- `response_time_distribution.png`: Response time frequency analysis with statistical metrics
- `refinement_loop_distribution.png`: Distribution of refinement loop counts
- `test_type_comparison.png`: Performance comparison between single and multi-session tests
- `session_progression.png`: Multi-session conversation development charts

## 🔗 Integration with System Architecture

This testing framework is designed to work seamlessly with the complete Mini Travel Assistant architecture:

- **DeepSeek LLM Integration**: Tests validate the DeepSeek-powered agent responses
- **RAG Engine Testing**: Validates ChromaDB knowledge retrieval and semantic search
- **Session Management**: Tests the dual-system conversation storage with RAG indexing
- **Quality Assessment**: Validates the 6-dimension quality scoring system
- **Self-Refinement Loop**: Comprehensive testing of the iterative improvement process

## 🔍 Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure API server is running at the correct URL
2. **Timeout Errors**: Increase timeout for refinement tests (recommended: 600s+)
3. **Import Errors**: Ensure you're in the `tests/chats` directory
4. **Visualization Errors**: Install matplotlib/seaborn/numpy/pandas
5. **DeepSeek API Errors**: Verify API key is configured correctly
6. **ChromaDB Errors**: Ensure knowledge base is properly initialized

### Performance Tips

- Use `--no-refinement` for faster testing during development
- Use `--mode single` for quick API validation
- Use `--timeout 600` or higher for comprehensive refinement testing
- Use `--mode comparison` to evaluate refinement effectiveness
- Monitor system resources during multi-session tests with large datasets