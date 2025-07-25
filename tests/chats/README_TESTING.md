# Chat API Testing Framework

This testing framework provides comprehensive testing capabilities for the Chat API with two distinct testing modes and detailed refinement loop analysis.

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

## 📁 File Structure

```
tests/chats/
├── chat_tester.py                 # Core testing framework
├── test_scenarios.py              # Test scenario definitions  
├── metrics_analyzer.py            # Metrics analysis
├── test_runner.py                 # Main test runner
├── run_demo.py                    # Demo script
├── README_TESTING.md              # This documentation
└── results/
    └── YYYYMMDD_HHMMSS/           # Timestamped results
        ├── single_session_tests.json
        ├── multi_session_tests.json
        ├── test_summary.json
        ├── analysis_report.txt
        └── *.png                   # Visualization files
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

1. **API Server Running**: The Chat API server must be running at the specified URL
2. **Knowledge Base Initialized**: Server should have the travel knowledge base loaded
3. **Dependencies**: Required Python packages (httpx, matplotlib, numpy, pandas)

```bash
# Install dependencies
pip install httpx matplotlib numpy pandas seaborn
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

## 📊 Understanding Results

### Single-Session Results (`single_session_tests.json`)
Each entry contains:
- `test_id`: Unique identifier
- `session_id`: Session used for the test
- `refinement_loops`: Detailed loop-by-loop metrics
- `final_confidence`: Final confidence score
- `total_response_time`: Complete response time

### Multi-Session Results (`multi_session_tests.json`)
Each session suite contains:
- `session_title`: Description of the session
- `queries`: List of all queries in the session
- `metrics`: Array of metrics for each query

### Analysis Report (`analysis_report.txt`)
Comprehensive analysis including:
- Overall summary statistics
- Test type performance comparison
- Refinement loop effectiveness
- Session progression insights

### Visualizations
- `refinement_loop_distribution.png`: Distribution of loop counts
- `test_type_comparison.png`: Performance comparison chart
- `confidence_vs_loops.png`: Confidence improvement analysis
- `session_progression.png`: Multi-session progression charts
- `response_time_vs_loops.png`: Time cost analysis

## 🔍 Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure API server is running
2. **Timeout Errors**: Increase timeout for refinement tests
3. **Import Errors**: Ensure you're in the `tests/chats` directory
4. **Visualization Errors**: Install matplotlib/seaborn/numpy

### Performance Tips

- Use `--no-refinement` for faster testing during development
- Use `--mode single` for quick API validation
- Use `--timeout 600` or higher for comprehensive refinement testing
- Use `--mode comparison` to evaluate refinement effectiveness

## 🎯 Quick Start

Run a quick demo to see the framework in action:

```bash
# Run the demo script
python run_demo.py
```

This will demonstrate both single-session and multi-session testing capabilities with a small subset of scenarios. 