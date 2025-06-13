from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Travel Planning Agent",
    description="An intelligent travel planning system that generates personalized itineraries",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to AI Travel Planning Agent API"}

# Import and include routers
from app.api.endpoints import travel_plans, users, tools

app.include_router(travel_plans.router, prefix="/api/v1", tags=["travel-plans"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])
app.include_router(tools.router, prefix="/api/v1", tags=["tools"]) 