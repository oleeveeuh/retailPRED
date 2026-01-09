"""
Economic Context API Endpoints

These endpoints provide FRED economic indicators for interpreting predictions.
IMPORTANT: Economic data is NOT used for model predictions - only for post-hoc
explanation and context.

Models use only 74 time-series features from MRTS data and achieve
excellent performance (0.26-2.22% MAPE).
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
import logging

from services.economic_context import get_economic_context_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/context", tags=["economic-context"])


@router.get("/indicators/{date}")
async def get_economic_context(date: str) -> Dict:
    """
    Get economic context for a specific date.

    This provides FRED indicators to help explain predictions, but these
    indicators are NOT used in the actual model predictions.

    Args:
        date: Date string in format 'YYYY-MM-DD'

    Returns:
        Dictionary with economic indicators, regime classification, and anomalies

    Example:
        GET /api/context/indicators/2020-03-31

        Response:
        {
            "indicators": {
                "unemployment": 14.7,
                "consumer_confidence": 86.0,
                "fed_rate": 0.25,
                "cpi": 258.0,
                "industrial_production": 97.4
            },
            "regime": {
                "regime": "crisis",
                "confidence": "low",
                "trends": {
                    "unemployment": "rising",
                    "consumer_confidence": "falling"
                },
                "explanation": "Economic crisis: unemployment spike, confidence crash"
            },
            "anomalies": [
                {
                    "indicator": "unemployment",
                    "value": 14.7,
                    "z_score": 4.2,
                    "severity": "high",
                    "direction": "high"
                }
            ],
            "note": "Context only - not used for model predictions"
        }
    """
    try:
        # Validate date format
        datetime.strptime(date, '%Y-%m-%d')

        service = get_economic_context_service()

        indicators = service.get_indicators_for_date(date)
        regime = service.get_economic_regime(date)
        anomalies = service.detect_anomalies(date)

        return {
            'date': date,
            'indicators': indicators,
            'regime': regime,
            'anomalies': anomalies,
            'note': 'Context only - not used for model predictions'
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error fetching economic context for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_historical_anomalies(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    Get all historical economic anomalies for annotation and visualization.

    This returns dates where economic indicators were anomalous (deviating
    significantly from normal), which can help explain unusual predictions.

    Args:
        start_date: Start date in format 'YYYY-MM-DD' (default: 2 years ago)
        end_date: End date in format 'YYYY-MM-DD' (default: today)

    Returns:
        Dictionary with list of anomalous dates

    Example:
        GET /api/context/anomalies?start_date=2019-01-01&end_date=2021-12-31

        Response:
        {
            "anomalies": [
                {
                    "date": "2020-03-31",
                    "regime": "crisis",
                    "anomalies": ["unemployment_spike", "confidence_crash"],
                    "severity": "high",
                    "explanation": "Economic crisis: unemployment spike, confidence crash"
                },
                {
                    "date": "2020-04-30",
                    "regime": "crisis",
                    "anomalies": ["unemployment_spike"],
                    "severity": "high",
                    "explanation": "Economic crisis: unemployment spike"
                }
            ],
            "metadata": {
                "total_count": 2,
                "high_severity_count": 2,
                "note": "Historical anomalies for interpretation - not used in predictions"
            }
        }
    """
    try:
        # Validate dates if provided
        if start_date:
            datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            datetime.strptime(end_date, '%Y-%m-%d')

        service = get_economic_context_service()
        anomalies = service.get_historical_anomalies(start_date, end_date)

        # Count severities
        high_severity = sum(1 for a in anomalies if a['severity'] == 'high')

        return {
            'anomalies': anomalies,
            'metadata': {
                'total_count': len(anomalies),
                'high_severity_count': high_severity,
                'note': 'Historical anomalies for interpretation - not used in predictions'
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error fetching historical anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/{date}")
async def get_economic_regime(date: str) -> Dict:
    """
    Get economic regime classification for a specific date.

    Regime Classification:
    - Crisis: Unemployment spike >3% in 3mo OR confidence drop >20pts
    - Recession: Unemployment rising + confidence falling
    - Expansion: Unemployment falling + confidence rising
    - Normal: Stable indicators

    Args:
        date: Date string in format 'YYYY-MM-DD'

    Returns:
        Dictionary with regime classification and explanation

    Example:
        GET /api/context/regime/2020-03-31

        Response:
        {
            "date": "2020-03-31",
            "regime": "crisis",
            "confidence": "low",
            "indicators": {
                "unemployment": 14.7,
                "consumer_confidence": 86.0
            },
            "trends": {
                "unemployment": "rising",
                "consumer_confidence": "falling"
            },
            "explanation": "Economic crisis: unemployment spike (14.7% is 4.2σ above mean)"
        }
    """
    try:
        # Validate date format
        datetime.strptime(date, '%Y-%m-%d')

        service = get_economic_context_service()
        regime = service.get_economic_regime(date)

        return {
            'date': date,
            **regime
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error fetching economic regime for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(
    date: str,
    prediction_change: float,
    category: str = "retail"
) -> Dict:
    """
    Generate natural language explanation for anomalous predictions.

    This correlates unusual predictions with economic context to help
    stakeholders understand what's happening.

    Args:
        date: Date string in format 'YYYY-MM-DD'
        prediction_change: Percentage change in prediction (e.g., -15.2 for 15% drop)
        category: Retail category name

    Returns:
        Dictionary with explanation or null if no anomaly

    Example:
        POST /api/context/explain

        Body:
        {
            "date": "2020-03-31",
            "prediction_change": -15.2,
            "category": "Building Materials"
        }

        Response:
        {
            "date": "2020-03-31",
            "prediction_change": -15.2,
            "category": "Building Materials",
            "explanation": "Building materials sales dropped by 15.2% in 2020-03. Economic indicators show unprecedented economic shock: unemployment at 14.7% (high than normal), consumer confidence at 86.0 (low than normal). Model predicted this from sales patterns using 74 time-series features. Economic data confirms the crisis impact."
        }
    """
    try:
        # Validate date format
        datetime.strptime(date, '%Y-%m-%d')

        service = get_economic_context_service()
        explanation = service.explain_anomaly(date, prediction_change, category)

        if explanation is None:
            return {
                'date': date,
                'prediction_change': prediction_change,
                'category': category,
                'explanation': None,
                'note': 'No economic anomalies detected for this date'
            }

        return {
            'date': date,
            'prediction_change': prediction_change,
            'category': category,
            'explanation': explanation
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error explaining prediction for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_economic_summary() -> Dict:
    """
    Get summary of current economic conditions.

    Returns:
        Dictionary with latest economic indicators and regime

    Example:
        GET /api/context/summary

        Response:
        {
            "latest_date": "2025-12-31",
            "indicators": {
                "unemployment": 4.1,
                "consumer_confidence": 71.7,
                "fed_rate": 4.33
            },
            "regime": "normal",
            "recent_anomalies": [],
            "note": "Economic context for interpretation - not used for predictions"
        }
    """
    try:
        service = get_economic_context_service()

        # Get most recent date with FRED data
        conn = service._get_connection()
        query = """
        SELECT MAX(date) as max_date
        FROM time_series_data
        WHERE source = 'FRED'
        """
        result = pd.read_sql_query(query, conn)
        latest_date = result['max_date'].iloc[0]

        # Get current conditions
        indicators = service.get_indicators_for_date(latest_date)
        regime = service.get_economic_regime(latest_date)
        anomalies = service.detect_anomalies(latest_date)

        return {
            'latest_date': latest_date,
            'indicators': indicators,
            'regime': regime,
            'recent_anomalies': anomalies,
            'note': 'Economic context for interpretation - not used for predictions'
        }

    except Exception as e:
        logger.error(f"Error fetching economic summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
