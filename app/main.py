import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from app.api.routes.properties import router as properties_router, get_model
from app.api.routes.feedback import router as feedback_router, get_feedback_system
from app.ml import initialize_model_from_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/recommendation_model.pkl")
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "appraisals_dataset.json")

# Create FastAPI app
app = FastAPI(
    title="Property Recommendation System",
    description="API for property recommendation with human feedback learning",
    version="1.0.0",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Include API routers
app.include_router(properties_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model and feedback system on startup
    """
    try:
        logger.info("Initializing model and feedback system...")
        model = get_model()
        feedback_system = get_feedback_system(model)
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")

@app.get("/", tags=["frontend"])
async def home(request: Request):
    """
    Render the home page
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Property Recommendation System"}
    )

@app.get("/dashboard", tags=["frontend"])
async def dashboard(request: Request):
    """
    Render the dashboard page
    """
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "title": "Feedback Dashboard"}
    )

@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "version": app.version,
        "service": app.title,
    }

# Run the application if executed directly
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    ) 