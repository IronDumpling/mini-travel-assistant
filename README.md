# AI Travel Planning Agent

An intelligent travel planning system that generates personalized itineraries using LLMs and real-time data from various travel services.

## Features

- Structured user input for travel preferences
- LLM-powered itinerary generation
- Real-time integration with travel services
- Interactive plan refinement
- Context-aware planning
- Self-improvement mechanism
- Optional visualization UI

## Project Structure

```
travel_planner/
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core functionality
│   ├── models/         # Data models
│   ├── services/       # Business logic
│   ├── tools/          # External tool integrations
│   └── utils/          # Utility functions
├── tests/              # Test files
├── .env.example        # Example environment variables
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

4. Run the development server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run tests with pytest:
```bash
pytest
```

## License

MIT 