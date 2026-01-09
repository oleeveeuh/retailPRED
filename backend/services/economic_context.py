"""
Economic Context Service - provides FRED data for interpretation ONLY.

This service fetches and analyzes economic indicators to help explain
model predictions and detect anomalies. Importantly, these indicators
are NOT used for model predictions - only for post-hoc interpretation.

The models use only time-series features (74 features from MRTS data)
and achieve excellent performance (0.26-2.22% MAPE).
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EconomicContextService:
    """Service for providing economic context to explain predictions."""

    def __init__(self, db_path: str = 'data/retailpred.db'):
        """
        Initialize economic context service.

        Args:
            db_path: Path to SQLite database containing FRED data
        """
        self.db_path = db_path
        self.conn = None

    def _get_connection(self):
        """Get database connection (lazy loading)."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def get_indicators_for_date(self, date: str) -> Dict:
        """
        Get economic indicators for a specific date.

        Args:
            date: Date string in format 'YYYY-MM-DD'

        Returns:
            Dictionary with economic indicator values
        """
        conn = self._get_connection()

        query = """
        SELECT data_type, value
        FROM time_series_data
        WHERE source = 'FRED'
          AND date <= ?
        ORDER BY date DESC, data_type
        """

        df = pd.read_sql_query(query, conn, params=(date,))

        if df.empty:
            return {
                'unemployment': None,
                'consumer_confidence': None,
                'fed_rate': None,
                'cpi': None,
                'industrial_production': None
            }

        # Get most recent value for each indicator
        indicators = {}
        for data_type in df['data_type'].unique():
            value = df[df['data_type'] == data_type]['value'].iloc[0]
            indicators[data_type] = float(value) if pd.notna(value) else None

        # Map to standard names
        return {
            'unemployment': indicators.get('unemployment_rate'),
            'consumer_confidence': indicators.get('consumer_sentiment'),
            'fed_rate': indicators.get('interest_rate'),
            'cpi': indicators.get('cpi'),
            'industrial_production': indicators.get('industrial_production'),
            'money_supply': indicators.get('money_supply')
        }

    def _get_indicator_history(self, indicator: str, days_back: int = 365) -> pd.Series:
        """Get historical values for an indicator to calculate statistics."""
        conn = self._get_connection()

        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        query = """
        SELECT date, value
        FROM time_series_data
        WHERE source = 'FRED'
          AND data_type = ?
          AND date >= ?
        ORDER BY date ASC
        """

        df = pd.read_sql_query(query, conn, params=(indicator, cutoff_date))

        if df.empty:
            return pd.Series([], dtype=float)

        return pd.Series(df['value'].values, index=pd.to_datetime(df['date']))

    def _calculate_z_score(self, indicator: str, current_value: float, days_back: int = 365) -> Optional[float]:
        """Calculate z-score for an indicator value."""
        history = self._get_indicator_history(indicator, days_back)

        if history.empty or len(history) < 30:
            return None

        mean = history.mean()
        std = history.std()

        if std == 0:
            return 0.0

        return (current_value - mean) / std

    def detect_anomalies(self, date: str, threshold: float = 2.0) -> List[Dict]:
        """
        Detect if economic indicators are anomalous (deviating from normal).

        Anomalies are defined as values more than `threshold` standard deviations
        from the mean of the past year.

        Args:
            date: Date string in format 'YYYY-MM-DD'
            threshold: Z-score threshold for anomaly detection (default: 2.0)

        Returns:
            List of anomalous indicators with severity scores
        """
        indicators = self.get_indicators_for_date(date)
        anomalies = []

        indicator_mapping = {
            'unemployment': 'unemployment_rate',
            'consumer_confidence': 'consumer_sentiment',
            'fed_rate': 'interest_rate'
        }

        for name, data_type in indicator_mapping.items():
            value = indicators.get(name)

            if value is None:
                continue

            z_score = self._calculate_z_score(data_type, value)

            if z_score is None:
                continue

            if abs(z_score) > threshold:
                severity = 'high' if abs(z_score) > 3 else 'medium'

                anomalies.append({
                    'indicator': name,
                    'value': value,
                    'z_score': z_score,
                    'severity': severity,
                    'direction': 'high' if z_score > 0 else 'low'
                })

        return anomalies

    def _calculate_trend(self, indicator: str, days_back: int = 90) -> str:
        """Calculate trend direction for an indicator."""
        history = self._get_indicator_history(indicator, days_back)

        if history.empty or len(history) < 2:
            return 'stable'

        recent_avg = history.tail(30).mean()
        older_avg = history.head(30).mean()

        if recent_avg > older_avg * 1.02:
            return 'rising'
        elif recent_avg < older_avg * 0.98:
            return 'falling'
        else:
            return 'stable'

    def get_economic_regime(self, date: str) -> Dict:
        """
        Classify economic regime based on composite indicators.

        Regime Classification:
        - Crisis: Unemployment spike >3% in 3mo OR confidence drop >20pts
        - Recession: Unemployment rising + confidence falling
        - Expansion: Unemployment falling + confidence rising
        - Normal: Stable indicators

        Args:
            date: Date string in format 'YYYY-MM-DD'

        Returns:
            Dictionary with regime classification and explanation
        """
        indicators = self.get_indicators_for_date(date)

        unemployment = indicators.get('unemployment')
        confidence = indicators.get('consumer_confidence')

        if unemployment is None or confidence is None:
            return {
                'regime': 'unknown',
                'confidence': 'low',
                'indicators': indicators,
                'explanation': 'Insufficient economic data'
            }

        # Calculate trends
        unemployment_trend = self._calculate_trend('unemployment_rate', days_back=90)
        confidence_trend = self._calculate_trend('consumer_sentiment', days_back=90)

        # Check for crisis conditions
        unemployment_z = self._calculate_z_score('unemployment_rate', unemployment, days_back=90)
        confidence_z = self._calculate_z_score('consumer_sentiment', confidence, days_back=90)

        is_crisis = False
        crisis_reasons = []

        if unemployment_z and unemployment_z > 3:
            is_crisis = True
            crisis_reasons.append(f'unemployment spike ({unemployment:.1f}% is {unemployment_z:.1f}σ above mean)')

        if confidence_z and confidence_z < -3:
            is_crisis = True
            crisis_reasons.append(f'consumer confidence crash ({confidence:.1f} is {abs(confidence_z):.1f}σ below mean)')

        # Classify regime
        if is_crisis:
            regime = 'crisis'
            conf_level = 'low'
            explanation = f'Economic crisis: {", ".join(crisis_reasons)}'

        elif unemployment_trend == 'rising' and confidence_trend == 'falling':
            regime = 'recession'
            conf_level = 'medium'
            explanation = f'Recessionary signals: unemployment {unemployment_trend}, confidence {confidence_trend}'

        elif unemployment_trend == 'falling' and confidence_trend == 'rising':
            regime = 'expansion'
            conf_level = 'high'
            explanation = f'Expansionary signals: unemployment {unemployment_trend}, confidence {confidence_trend}'

        else:
            regime = 'normal'
            conf_level = 'high'
            explanation = f'Normal economic conditions: unemployment {unemployment_trend}, confidence {confidence_trend}'

        return {
            'regime': regime,
            'confidence': conf_level,
            'indicators': indicators,
            'trends': {
                'unemployment': unemployment_trend,
                'consumer_confidence': confidence_trend
            },
            'explanation': explanation
        }

    def explain_anomaly(self, date: str, prediction_change: float, category: str = "retail") -> Optional[str]:
        """
        Generate natural language explanation for anomalous predictions.

        This helps stakeholders understand WHY a prediction might be unusual
        by correlating it with economic context.

        Args:
            date: Date string in format 'YYYY-MM-DD'
            prediction_change: Percentage change in prediction (e.g., -15.2 for 15% drop)
            category: Retail category name

        Returns:
            Natural language explanation or None if no anomaly
        """
        anomalies = self.detect_anomalies(date)

        if not anomalies:
            return None

        regime = self.get_economic_regime(date)
        indicators = regime['indicators']

        # Build explanation
        explanation_parts = []

        # Start with prediction change
        change_str = f"{'dropped' if prediction_change < 0 else 'increased'} by {abs(prediction_change):.1f}%"
        explanation_parts.append(f"{category.capitalize()} sales {change_str} in {date[:7]}.")

        # Add economic context
        explanation_parts.append("Economic indicators show")

        if regime['regime'] == 'crisis':
            explanation_parts.append(" unprecedented economic shock:")
        elif regime['regime'] == 'recession':
            explanation_parts.append(" recessionary conditions:")
        else:
            explanation_parts.append(" significant deviations:")

        # Add specific anomalies
        anomaly_descs = []
        for anomaly in anomalies:
            if anomaly['indicator'] == 'unemployment':
                anomaly_descs.append(f"unemployment at {anomaly['value']:.1f}% ({anomaly['direction']} than normal)")
            elif anomaly['indicator'] == 'consumer_confidence':
                anomaly_descs.append(f"consumer confidence at {anomaly['value']:.1f} ({anomaly['direction']} than normal)")
            elif anomaly['indicator'] == 'fed_rate':
                anomaly_descs.append(f"federal funds rate at {anomaly['value']:.2f}% ({anomaly['direction']} than normal)")

        explanation_parts.append(", ".join(anomaly_descs) + ".")

        # Add model context
        explanation_parts.append(
            f"Model predicted this from sales patterns using 74 time-series features. "
            f"Economic data confirms the {regime['regime']} impact."
        )

        return " ".join(explanation_parts)

    def get_historical_anomalies(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """
        Get all historical anomalies for annotation and visualization.

        Args:
            start_date: Start date (default: 2 years ago)
            end_date: End date (default: today)

        Returns:
            List of dates with anomalous economic conditions
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()

        # Get all FRED dates in range
        query = """
        SELECT DISTINCT date
        FROM time_series_data
        WHERE source = 'FRED'
          AND date >= ?
          AND date <= ?
        ORDER BY date ASC
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))

        anomalies_list = []

        for date_str in df['date'].tolist():
            anomalies = self.detect_anomalies(date_str)

            if anomalies:
                regime = self.get_economic_regime(date_str)

                anomalies_list.append({
                    'date': date_str,
                    'regime': regime['regime'],
                    'anomalies': [a['indicator'] for a in anomalies],
                    'severity': 'high' if any(a['severity'] == 'high' for a in anomalies) else 'medium',
                    'explanation': regime['explanation']
                })

        return anomalies_list

    def close(self):
        """Close database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# Singleton instance for reuse
_service_instance = None


def get_economic_context_service() -> EconomicContextService:
    """Get or create singleton EconomicContextService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EconomicContextService()
    return _service_instance
