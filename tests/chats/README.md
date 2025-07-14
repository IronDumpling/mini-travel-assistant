# Chat API Tests

This directory contains comprehensive test cases for the Chat API, designed to establish baseline metrics and evaluate performance of the travel planning agent.

## Overview

The test suite includes:
- **Functional Tests**: Verify API endpoints work correctly
- **Performance Tests**: Measure response times and success rates
- **Refinement Comparison**: Compare behavior with/without self-refinement
- **Metrics Recording**: Capture detailed performance data
- **Analysis Tools**: Generate reports and visualizations

## Files

### Core Test Files
- `test_chat_api.py` - Main test implementation with metrics recording
- `test_chat_pytest.py` - Pytest-compatible test cases
- `run_tests.py` - Simple test runner script
- `simple_test.py` - Basic connectivity test (run this first)

### Analysis Tools
- `metrics_analyzer.py` - Analyze test results and generate reports
- `README.md` - This documentation

### Generated Results
- `results/` - Directory containing test metrics and reports
  - `chat_test_metrics_*.json` - Raw test data
  - `analysis_report.txt` - Generated analysis report
  - `*.png` - Visualization charts

## Prerequisites

Before running the tests, ensure you have:

1. **API Server Running**: Start the FastAPI server
   ```bash
   cd /path/to/project
   python -m uvicorn app.main:app --reload
   ```

2. **Required Dependencies**: Install test dependencies
   ```bash
   pip install httpx pytest pytest-asyncio pandas matplotlib seaborn
   ```

3. **Expected Response Times**: Chat API responses normally take ~1 minute due to AI processing and self-refinement

## Running Tests

### Option 1: Quick Connectivity Test (Recommended First)
```bash
# Test basic server connectivity and API functionality
python simple_test.py
```

### Option 2: Quick Test Run
```bash
# Run all tests and analysis
python run_tests.py

# Run tests without analysis
python run_tests.py --no-analysis

# Only analyze existing results
python run_tests.py --analyze-only

# Use different API URL
python run_tests.py --base-url http://localhost:8080
```

### Option 3: Direct Test Execution
```bash
# Test basic connectivity first
python simple_test.py

# Run main test suite
python test_chat_api.py

# Run pytest suite
pytest test_chat_pytest.py -v

# Analyze existing metrics
python metrics_analyzer.py
```

## Test Scenarios

The test suite includes these travel planning scenarios:

1. **London 7-Day Budget Trip**
   - Message: "Plan a 7-day trip to London for 2 people with a budget of $3000"
   - Tests: Budget planning, multi-day itinerary, group travel

2. **Paris Romantic Weekend**
   - Message: "Plan a romantic weekend in Paris for 2 people, focusing on fine dining and cultural experiences"
   - Tests: Theme-based planning, experience curation

3. **Tokyo Business Trip**
   - Message: "I need a 3-day business trip to Tokyo with meetings in Shibuya and Shinjuku areas"
   - Tests: Business travel, location-specific requirements

4. **Family Singapore Trip**
   - Message: "Plan a 5-day family trip to Singapore with 2 adults and 2 children (ages 8 and 12)"
   - Tests: Family travel, age-appropriate activities

5. **Europe Backpacking**
   - Message: "Create a 14-day backpacking itinerary across Europe for a student with a $1500 budget"
   - Tests: Budget constraints, multi-destination planning

6. **Dubai Luxury Trip**
   - Message: "Plan a luxury 4-day trip to Dubai for celebrating our anniversary, budget is flexible"
   - Tests: Luxury travel, special occasions

7. **Barcelona Solo Travel**
   - Message: "I'm planning a solo trip to Barcelona for 6 days, interested in art, architecture, and nightlife"
   - Tests: Solo travel, interest-based planning

8. **Swiss Alps Ski Trip**
   - Message: "Plan a 5-day ski trip to the Swiss Alps for 4 people in February"
   - Tests: Seasonal travel, activity-specific planning

## Metrics Captured

Each test captures:
- **Response Time**: How long the API takes to respond
- **Success Rate**: Whether the request completed successfully
- **Confidence Score**: Agent's confidence in the response
- **Actions Taken**: List of actions the agent performed
- **Next Steps**: Suggested follow-up actions
- **Refinement Details**: Information about self-refinement iterations
- **Request/Response Data**: Full API request and response

## Analysis Features

### Performance Analysis
- Average and median response times
- Success rate statistics
- Response time distribution
- Performance benchmarks (fast/medium/slow responses)

### Refinement Impact Analysis
- Comparison of responses with/without self-refinement
- Impact on response time and confidence
- Quality improvement metrics

### Scenario Analysis
- Performance breakdown by travel scenario
- Confidence levels for different types of requests
- Success rates by complexity

### Visualizations
- Response time distribution histograms
- Success rate comparisons
- Confidence vs response time scatter plots
- Performance trend analysis

## Understanding Results

### Success Rate Benchmarks
- **Excellent**: >95% success rate
- **Good**: 85-95% success rate
- **Acceptable**: 70-85% success rate
- **Poor**: <70% success rate

### Response Time Benchmarks
- **Fast**: <45 seconds
- **Medium**: 45-75 seconds  
- **Slow**: >75 seconds
- **Note**: Normal response time is ~1 minute due to AI processing

### Confidence Score Interpretation
- **High Confidence**: ≥0.8
- **Medium Confidence**: 0.6-0.8
- **Low Confidence**: <0.6

## Customization

### Adding New Test Scenarios
Edit `TEST_SCENARIOS` in `test_chat_api.py`:
```python
TEST_SCENARIOS.append({
    "name": "custom_scenario",
    "message": "Your custom travel request",
    "expected_keywords": ["keyword1", "keyword2"]
})
```

### Modifying Performance Thresholds
Update assertion values in `test_chat_pytest.py`:
```python
assert metric.response_time < 30.0  # Adjust time threshold
assert metric.confidence > 0.5      # Adjust confidence threshold
```

### Custom Analysis
Create custom analysis functions in `metrics_analyzer.py` or create new analysis scripts using the MetricsAnalyzer class.

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - **First, run the simple test**: `python simple_test.py`
   - Ensure API server is running: `python -m uvicorn app.main:app --reload`
   - Check the base URL is correct (default: http://localhost:8000)
   - Verify no firewall blocking localhost
   - Wait for server to fully initialize (check server logs)

2. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Ensure you're in the correct directory (`tests/chats/`)
   - Check Python path and virtual environment

3. **Test Failures**
   - Check API server logs for errors
   - Verify database/knowledge base is properly initialized
   - Ensure all required environment variables are set
   - Try running individual test scenarios first

4. **Server Crashes During Tests**
   - Server might be overloaded - reduce test concurrency
   - Check server memory usage
   - Verify all server dependencies are installed
   - Look for timeout issues in server logs

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View detailed test output:
```bash
pytest test_chat_pytest.py -v -s
```

## Continuous Integration

To use these tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Chat API Tests
  run: |
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    sleep 10  # Wait for server to start
    cd tests/chats
    python run_tests.py --no-analysis
    pytest test_chat_pytest.py -v
```

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Include appropriate assertions
3. Add metrics recording for new test types
4. Update documentation as needed
5. Test both success and failure scenarios

## Results Storage

All test results are stored in `tests/chats/results/` with timestamps, allowing you to:
- Track performance over time
- Compare different versions
- Analyze trends and improvements
- Generate historical reports

The JSON metrics files contain all raw data for detailed analysis and can be easily imported into other analysis tools. 