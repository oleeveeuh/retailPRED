"""
RetailPRED Backend API
FastAPI server for retail sales forecasting system
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

# Add app directory to Python path FIRST
app_path = Path(__file__).parent
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

# Import routes
from api.routes import router as api_router
from api.category_routes import router as category_router
from api.export import router as export_router
from api.scenario_routes import router as scenario_router
from api.training_metrics import router as training_metrics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RetailPRED API",
    description="""
    Retail Sales Forecasting System with ML model integration.

    ## Features
    * **Predictions**: Generate sales forecasts with SHAP explanations
    * **Data Management**: Refresh data from external sources
    * **Model Management**: Train, evaluate, and manage ML models
    * **Validation**: Track prediction accuracy over time
    * **Explainability**: Detailed SHAP value analysis
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)
app.include_router(category_router)
app.include_router(export_router, prefix="/api/export", tags=["export"])
app.include_router(scenario_router)
app.include_router(training_metrics_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RetailPRED API",
        "version": "2.0.0",
        "description": "Retail Sales Forecasting System",
        "documentation": "/docs",
        "endpoints": {
            "predictions": "/api/predict",
            "categories": "/api/categories/list",
            "category_predict": "/api/categories/predict",
            "data_refresh": "/api/refresh-data",
            "models": "/api/models",
            "history": "/api/predictions/history",
            "validation": "/api/predictions/validate",
            "shap": "/api/shap-explain",
            "training": "/api/train",
            "health": "/api/health"
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("RetailPRED API Starting...")
    logger.info("=" * 60)
    logger.info("Documentation available at: http://localhost:8000/docs")
    logger.info("ReDoc available at: http://localhost:8000/redoc")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("RetailPRED API Shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
