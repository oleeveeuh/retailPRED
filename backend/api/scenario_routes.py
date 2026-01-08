"""
Scenario Analysis API Routes
Provides endpoints for economic scenario forecasting and analysis
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import logging
from pathlib import Path

from services.scenario_analysis import (
    ScenarioAnalyzer,
    SCENARIOS,
    INDICATOR_CATEGORIES,
    DATA_SOURCES,
    apply_adjustments,
    create_custom_scenario
)
# Lazy import - only load when actually used
from db.database import RetailPREDDatabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])

# Initialize database
db_path = Path(__file__).resolve().parent.parent.parent / "data" / "retailpred.db"
db = RetailPREDDatabase(db_path=str(db_path.absolute()))

# Global scenario analyzer (lazy loaded)
scenario_analyzer: Optional[ScenarioAnalyzer] = None

# Global model loader (lazy loaded)
model_loader_instance = None


def get_model_loader_instance():
    """Get or initialize the global model loader (lazy import)"""
    global model_loader_instance

    if model_loader_instance is None:
        # Import here to avoid loading models at startup
        from services.model_loader import get_model_loader
        model_loader_instance = get_model_loader()
        logger.info("ModelLoader initialized successfully")

    return model_loader_instance


def get_scenario_analyzer() -> ScenarioAnalyzer:
    """Get or initialize the scenario analyzer"""
    global scenario_analyzer

    if scenario_analyzer is None:
        # Load historical data from database
        try:
            import pandas as pd

            logger.info("Loading real FRED economic data from database...")

            # Get real economic data from database
            conn = db.get_connection()
            cursor = conn.cursor()

            logger.info("Database connection established")

            # Load each indicator separately as DataFrames
            indicator_dfs = {}

            # Unemployment Rate (UNRATE)
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'unemployment_rate'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['UNRATE'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # Federal Funds Rate (FEDFUNDS) - use interest_rate
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'interest_rate'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['FEDFUNDS'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # CPI
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'cpi'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['CPI'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # Consumer Sentiment (UMCSENT)
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'consumer_sentiment'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['UMCSENT'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # Industrial Production
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'industrial_production'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['IPMANS'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # Retail Sales (RSXFS)
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'retail_sales'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['RSXFS'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            # Money Supply
            cursor.execute("""
                SELECT date, value
                FROM time_series_data
                WHERE data_type = 'money_supply'
                ORDER BY date
            """)
            rows = cursor.fetchall()
            if rows:
                indicator_dfs['M2SL'] = pd.DataFrame([{'date': r[0], 'value': r[1]} for r in rows])

            conn.close()

            logger.info(f"Loaded {len(indicator_dfs)} indicators from database")

            if not indicator_dfs:
                raise ValueError("No economic data found in database")

            # Merge all indicators on date (inner join to use common dates only)
            logger.info("Merging indicators...")

            # Convert all to DataFrames with proper structure and remove duplicates
            for indicator in indicator_dfs:
                indicator_dfs[indicator]['date'] = pd.to_datetime(indicator_dfs[indicator]['date'])
                # Remove duplicate dates (keep first occurrence)
                indicator_dfs[indicator] = indicator_dfs[indicator].drop_duplicates(subset=['date'], keep='first')
                indicator_dfs[indicator].set_index('date', inplace=True)
                indicator_dfs[indicator].rename(columns={'value': indicator}, inplace=True)

            # Start with the first indicator
            first_indicator = list(indicator_dfs.keys())[0]
            historical_data = indicator_dfs[first_indicator].copy()

            # Merge the rest using concat instead of join (faster)
            other_indicators = [indicator_dfs[k] for k in list(indicator_dfs.keys())[1:]]
            if other_indicators:
                historical_data = pd.concat([historical_data] + other_indicators, axis=1, join='inner')

            logger.info(f"Merged data shape: {historical_data.shape}")

            # Estimate missing indicators based on available data
            # PAYEMS (Payrolls) - estimate from unemployment rate (inverse relationship)
            if 'UNRATE' in historical_data.columns and 'PAYEMS' not in historical_data.columns:
                historical_data['PAYEMS'] = 150000000 - (historical_data['UNRATE'] * 1000000)  # Rough estimate

            # GDP - estimate from industrial production (correlated)
            if 'IPMANS' in historical_data.columns and 'GDP' not in historical_data.columns:
                historical_data['GDP'] = historical_data['IPMANS'] * 0.05  # Rough estimate

            # 10-Year Treasury (DGS10) - estimate from Fed Funds + term premium
            if 'FEDFUNDS' in historical_data.columns and 'DGS10' not in historical_data.columns:
                historical_data['DGS10'] = historical_data['FEDFUNDS'] + 1.5  # Typical term premium

            # Housing Starts (HOUST) - estimate using trend
            if 'HOUST' not in historical_data.columns:
                historical_data['HOUST'] = 1500  # Use average

            # Ensure we have all required indicators
            required_indicators = ['UNRATE', 'PAYEMS', 'GDP', 'CPI', 'FEDFUNDS', 'UMCSENT', 'HOUST', 'DGS10', 'RSXFS']
            for indicator in required_indicators:
                if indicator not in historical_data.columns:
                    logger.warning(f"Missing indicator {indicator}, using default value")
                    historical_data[indicator] = 0.0

            # Drop rows with any NaN values (cleaner than forward fill)
            historical_data = historical_data.dropna()

            # Limit to last 3 years of data for performance
            if len(historical_data) > 36:
                historical_data = historical_data.iloc[-36:]

            logger.info(f"Loaded real economic data: {len(historical_data)} observations from {historical_data.index[0]} to {historical_data.index[-1]}")
            logger.info(f"Available indicators: {list(historical_data.columns)}")

            scenario_analyzer = ScenarioAnalyzer(historical_data)
            logger.info("ScenarioAnalyzer initialized successfully with real FRED data")

        except Exception as e:
            logger.error(f"Failed to initialize ScenarioAnalyzer: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize analyzer: {str(e)}")

    return scenario_analyzer


# Pydantic models
class ScenarioRequest(BaseModel):
    scenario_type: str
    category: str = "total_sales"
    base_features: Optional[Dict[str, float]] = None


class CustomScenarioRequest(BaseModel):
    category: str = "total_sales"
    base_features: Optional[Dict[str, float]] = None
    adjustments: Dict[str, float]


class SensitivityRequest(BaseModel):
    category: str = "total_sales"
    feature_name: str
    min_value: float
    max_value: float
    num_steps: int = 10
    base_features: Optional[Dict[str, float]] = None


@router.get("/list")
async def list_scenarios():
    """List all available economic scenarios"""
    scenarios_list = []

    for scenario_type, scenario_info in SCENARIOS.items():
        scenarios_list.append({
            "type": scenario_type,
            "name": scenario_info["name"],
            "description": scenario_info["description"],
            "adjustments": scenario_info["adjustments"]
        })

    return {
        "scenarios": scenarios_list,
        "total_count": len(scenarios_list)
    }


@router.post("/analyze")
async def analyze_scenario(request: ScenarioRequest):
    """
    Generate forecast under specified economic scenario

    Args:
        request: Scenario request with type and category

    Returns:
        Scenario analysis with predictions and factor impacts
    """
    try:
        analyzer = get_scenario_analyzer()
        model_loader = get_model_loader_instance()

        # Get current economic indicators (or use provided)
        if request.base_features:
            current_indicators = request.base_features
        else:
            # Use latest values from historical data
            latest_row = analyzer.historical_data.iloc[-1]
            current_indicators = latest_row.to_dict()

        # Get the best model for this category
        prediction_model = model_loader.get_best_model(request.category)

        # Try to get the most recent LGBM prediction from prediction_log as baseline
        try:
            cursor = db.get_connection().cursor()

            # Map category to model name pattern
            category_model_map = {
                "total_sales": "total_sales_LGBM_model",
                "automobile_dealers": "automobile_dealers_LGBM_model",
                "building_materials": "building_materials_LGBM_model",
                "clothing_accessories": "clothing_accessories_LGBM_model",
                "electronics_and_appliances": "electronics_and_appliances_LGBM_model",
                "food_beverage": "food_beverage_LGBM_model",
                "furniture_home_furnishings": "furniture_home_furnishings_LGBM_model",
                "gasoline_stations": "gasoline_stations_LGBM_model",
                "general_merchandise": "general_merchandise_LGBM_model",
                "health_personal_care": "health_personal_care_LGBM_model",
                "sporting_goods_hobby": "sporting_goods_hobby_LGBM_model",
            }

            model_name = category_model_map.get(request.category, f"{request.category}_LGBM_model")

            cursor.execute("""
                SELECT predicted_value
                FROM prediction_log
                WHERE model_name = ?
                AND predicted_value IS NOT NULL
                ORDER BY prediction_date DESC
                LIMIT 1
            """, (model_name,))

            result = cursor.fetchone()
            if result and result[0]:
                baseline_prediction = result[0]
                logger.info(f"Using most recent {model_name} prediction: ${baseline_prediction:,.2f}")
            else:
                # Fallback to average of all recent retail sales if no LGBM prediction found
                cursor.execute("""
                    SELECT AVG(value) as avg_sales
                    FROM time_series_data
                    WHERE data_type = 'retail_sales'
                    AND date >= date('now', '-6 months')
                """)
                result = cursor.fetchone()
                baseline_prediction = result[0] if result and result[0] else 50000
                logger.info(f"No LGBM prediction found, using historical average: ${baseline_prediction:,.2f}")

            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to get LGBM prediction from database: {e}")
            baseline_prediction = 50000  # Fallback to $50K

        if prediction_model is None:
            logger.warning(f"No model found for category {request.category}, using LGBM baseline: ${baseline_prediction:,.2f}")
            # Use LGBM prediction from database
            base_prediction = baseline_prediction
        else:
            # Generate base prediction with current indicators
            model_prediction = model_loader.predict(
                request.category,
                current_indicators,
                "RandomForest"  # Use RandomForest as default
            )
            # Use model prediction if available, otherwise fall back to LGBM baseline
            base_prediction = model_prediction if model_prediction else baseline_prediction

        # Generate scenario with actual model prediction
        scenario_result = analyzer.generate_economic_scenario(
            base_features=current_indicators,
            scenario_type=request.scenario_type,
            prediction_model=prediction_model,
            base_prediction=base_prediction if base_prediction else 450000
        )

        # Add category info
        scenario_result["category"] = request.category

        return scenario_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing scenario: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze scenario: {str(e)}")


@router.get("/analyze")
async def analyze_scenario_get(
    category: str = Query(..., description="Category to analyze"),
    scenario: str = Query("baseline", description="Scenario type (baseline, recession, growth)")
):
    """
    Generate forecast under specified economic scenario (GET version)

    Args:
        category: Category to analyze
        scenario: Scenario type (baseline, recession, growth)

    Returns:
        Scenario analysis with predictions and factor impacts
    """
    try:
        # Create a ScenarioRequest from query parameters
        request = ScenarioRequest(
            category=category,
            scenario_type=scenario,
            base_features=None
        )

        # Call the POST handler
        return await analyze_scenario(request)

    except Exception as e:
        logger.error(f"Error analyzing scenario (GET): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze scenario: {str(e)}")


@router.post("/custom")
async def create_custom_scenario_endpoint(request: CustomScenarioRequest):
    """Create and analyze custom economic scenario"""
    try:
        analyzer = get_scenario_analyzer()

        # Get current indicators
        if request.base_features:
            current_indicators = request.base_features
        else:
            latest_row = analyzer.historical_data.iloc[-1]
            current_indicators = latest_row.to_dict()

        # Apply custom adjustments
        modified_features = apply_adjustments(current_indicators, request.adjustments)

        return {
            "scenario_type": "custom",
            "scenario_name": "Custom Scenario",
            "description": "User-defined economic scenario",
            "base_features": current_indicators,
            "modified_features": modified_features,
            "adjustments": request.adjustments,
            "category": request.category,
            "impact_summary": analyzer._calculate_impact_summary(
                current_indicators,
                modified_features
            )
        }

    except Exception as e:
        logger.error(f"Error creating custom scenario: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create custom scenario: {str(e)}")


@router.get("/similar-periods")
async def find_similar_periods(
    category: str = Query("total_sales"),
    n: int = Query(5, ge=1, le=20)
):
    """
    Find historical periods with similar economic conditions

    Args:
        category: Retail category
        n: Number of similar periods to return

    Returns:
        List of similar historical periods
    """
    try:
        analyzer = get_scenario_analyzer()

        # Get current indicators
        current_indicators = analyzer.historical_data.iloc[-1].to_dict()

        # Find similar periods
        similar_periods = analyzer.find_similar_periods(current_indicators, n=n)

        return {
            "category": category,
            "current_indicators": current_indicators,
            "periods": similar_periods,
            "count": len(similar_periods)
        }

    except Exception as e:
        logger.error(f"Error finding similar periods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find similar periods: {str(e)}")


@router.get("/regime")
async def detect_regime(
    category: str = Query("total_sales")
):
    """
    Detect current economic regime (Expansion/Peak/Recession/Recovery)

    Args:
        category: Retail category

    Returns:
        Current regime classification with confidence
    """
    try:
        analyzer = get_scenario_analyzer()

        # Get current indicators
        current_indicators = analyzer.historical_data.iloc[-1].to_dict()

        # Detect regime
        regime = analyzer.detect_economic_regime(current_indicators)

        return {
            "category": category,
            "current_indicators": current_indicators,
            **regime
        }

    except Exception as e:
        logger.error(f"Error detecting regime: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect regime: {str(e)}")


@router.post("/sensitivity")
async def calculate_sensitivity(request: SensitivityRequest):
    """
    Calculate prediction sensitivity to an economic indicator

    Args:
        request: Sensitivity request with feature and range

    Returns:
        Sensitivity analysis results
    """
    try:
        analyzer = get_scenario_analyzer()
        model_loader = get_model_loader_instance()

        # Get current indicators
        if request.base_features:
            current_indicators = request.base_features
        else:
            latest_row = analyzer.historical_data.iloc[-1]
            current_indicators = latest_row.to_dict()

        # Get base prediction
        prediction_model = model_loader.get_best_model(request.category)

        # Try to get the most recent LGBM prediction from prediction_log as baseline
        try:
            cursor = db.get_connection().cursor()

            # Map category to model name pattern
            category_model_map = {
                "total_sales": "total_sales_LGBM_model",
                "automobile_dealers": "automobile_dealers_LGBM_model",
                "building_materials": "building_materials_LGBM_model",
                "clothing_accessories": "clothing_accessories_LGBM_model",
                "electronics_and_appliances": "electronics_and_appliances_LGBM_model",
                "food_beverage": "food_beverage_LGBM_model",
                "furniture_home_furnishings": "furniture_home_furnishings_LGBM_model",
                "gasoline_stations": "gasoline_stations_LGBM_model",
                "general_merchandise": "general_merchandise_LGBM_model",
                "health_personal_care": "health_personal_care_LGBM_model",
                "sporting_goods_hobby": "sporting_goods_hobby_LGBM_model",
            }

            model_name = category_model_map.get(request.category, f"{request.category}_LGBM_model")

            cursor.execute("""
                SELECT predicted_value
                FROM prediction_log
                WHERE model_name = ?
                AND predicted_value IS NOT NULL
                ORDER BY prediction_date DESC
                LIMIT 1
            """, (model_name,))

            result = cursor.fetchone()
            if result and result[0]:
                baseline_prediction = result[0]
            else:
                # Fallback to average of all recent retail sales if no LGBM prediction found
                cursor.execute("""
                    SELECT AVG(value) as avg_sales
                    FROM time_series_data
                    WHERE data_type = 'retail_sales'
                    AND date >= date('now', '-6 months')
                """)
                result = cursor.fetchone()
                baseline_prediction = result[0] if result and result[0] else 50000

            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to get LGBM prediction from database: {e}")
            baseline_prediction = 50000  # Fallback to $50K

        if prediction_model is None:
            base_prediction = baseline_prediction
        else:
            model_prediction = model_loader.predict(
                request.category,
                current_indicators,
                "RandomForest"
            )
            base_prediction = model_prediction if model_prediction else baseline_prediction

        # Calculate sensitivity with actual model
        sensitivity = analyzer.calculate_sensitivity(
            base_features=current_indicators,
            feature_name=request.feature_name,
            range_values=(request.min_value, request.max_value, request.num_steps),
            prediction_model=prediction_model,
            base_prediction=base_prediction
        )

        sensitivity["category"] = request.category

        return sensitivity

    except Exception as e:
        logger.error(f"Error calculating sensitivity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate sensitivity: {str(e)}")


@router.get("/indicators/categories")
async def get_indicator_categories():
    """Get economic indicator categories"""
    return {
        "categories": INDICATOR_CATEGORIES,
        "total_categories": len(INDICATOR_CATEGORIES)
    }


@router.get("/indicators/sources")
async def get_indicator_sources():
    """Get data sources for economic indicators"""
    return {
        "sources": DATA_SOURCES,
        "total_indicators": len(DATA_SOURCES)
    }


@router.get("/indicators/current")
async def get_current_indicators():
    """Get current values of all economic indicators"""
    try:
        analyzer = get_scenario_analyzer()

        # Get latest values
        latest_row = analyzer.historical_data.iloc[-1]
        current_indicators = latest_row.to_dict()

        # Add metadata
        indicators_with_meta = []
        for indicator, value in current_indicators.items():
            indicators_with_meta.append({
                "name": indicator,
                "value": value,
                "category": analyzer._get_indicator_category(indicator),
                "source": DATA_SOURCES.get(indicator, "Unknown"),
                "date": latest_row.name.strftime("%Y-%m-%d")
            })

        return {
            "indicators": indicators_with_meta,
            "as_of_date": latest_row.name.strftime("%Y-%m-%d"),
            "total_count": len(indicators_with_meta)
        }

    except Exception as e:
        logger.error(f"Error getting current indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current indicators: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for scenario analysis service"""
    try:
        model_loader = get_model_loader_instance()
        available_models = model_loader.list_available_models()

        return {
            "status": "healthy",
            "service": "scenario-analysis",
            "timestamp": datetime.utcnow().isoformat(),
            "analyzer_initialized": scenario_analyzer is not None,
            "models_loaded": len(available_models) > 0,
            "available_categories": list(available_models.keys()),
            "total_models": sum(len(models) for models in available_models.values())
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "scenario-analysis",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
