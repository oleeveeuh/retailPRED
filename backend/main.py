"""
RetailPRED Backend API
FastAPI server for retail sales forecasting system
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path

# Add app directory to Python path FIRST
app_path = Path(__file__).parent
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

from api.v2_forecasts import router as v2_forecast_router

LEGACY_API_ENABLED = os.getenv("RETAILPRED_ENABLE_LEGACY_API", "").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RetailPRED API",
    description="Evidence-backed retail forecasting with fail-closed model availability.",
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

app.include_router(v2_forecast_router)

if LEGACY_API_ENABLED:
    # Archival compatibility only. These routes contain v1 mutation and mock
    # behavior and are deliberately absent from the default application.
    from api.category_routes import router as category_router
    from api.export import router as export_router
    from api.routes import router as api_router
    from api.scenario_routes import router as scenario_router
    from api.training_metrics import router as training_metrics_router

    app.include_router(api_router)
    app.include_router(category_router)
    app.include_router(export_router, prefix="/api/export", tags=["legacy export"])
    app.include_router(scenario_router)
    app.include_router(training_metrics_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RetailPRED API",
        "version": "2.0.0",
        "description": "Evidence-backed retail forecasting service",
        "documentation": "/docs",
        "legacy_api_enabled": LEGACY_API_ENABLED,
        "endpoints": {
            "health": "/api/v2/health",
            "v2_models": "/api/v2/models",
            "v2_demo_catalog": "/api/v2/demo-catalog",
            "v2_forecasts": "/api/v2/forecasts",
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
    logger.info("Legacy API enabled: %s", LEGACY_API_ENABLED)


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
