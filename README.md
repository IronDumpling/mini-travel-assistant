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
├── .env                # Environment variables (create this file)
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Git Bash)
python -m venv venv
source venv/Scripts/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
Create a new file named `.env` in the project root and add the following content (replace with your actual values):
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
FLIGHT_SEARCH_API_KEY=your_flight_search_api_key_here
HOTEL_SEARCH_API_KEY=your_hotel_search_api_key_here
ATTRACTION_SEARCH_API_KEY=your_attraction_search_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/travel_planner

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
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

## Troubleshooting

1. If you get "Import could not be resolved" errors:
   - Make sure your virtual environment is activated
   - Try restarting your IDE
   - Verify that all dependencies are installed correctly

2. If you get "No such file or directory" errors:
   - Make sure you're in the correct directory
   - Check that all required files exist
   - Verify file paths are correct for your operating system

## License

MIT 