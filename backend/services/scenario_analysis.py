"""
Economic Scenario Analysis Service for RetailPRED

Generates economic forecasts under different macroeconomic scenarios.
Focuses on external economic forces, not controllable variables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# Economic scenario definitions
SCENARIOS = {
    "recession": {
        "name": "Recession",
        "description": "Economic downturn with elevated unemployment and negative GDP growth",
        "adjustments": {
            "UNRATE": 2.0,  # +2% unemployment
            "GDP": -1.5,    # -1.5% GDP growth
            "FEDFUNDS": -0.5,  # -0.5% interest rates (Fed cuts rates)
            "CPI": -1.0,    # -1% inflation
            "PAYEMS": -500000,  # -500K jobs per month
            "UMCSENT": -20,  # Consumer sentiment drops 20 points
        }
    },
    "rate_hike": {
        "name": "Rate Hike Cycle",
        "description": "Tightening monetary policy with higher interest rates",
        "adjustments": {
            "FEDFUNDS": 2.0,  # +2% interest rates
            "UNRATE": 0.5,    # +0.5% unemployment
            "GDP": -0.5,      # -0.5% GDP growth
            "CPI": -0.5,      # -0.5% inflation (cooling)
            "HOUST": -100000,  # -100K housing starts
        }
    },
    "inflation_surge": {
        "name": "Inflation Surge",
        "description": "High inflation environment with elevated consumer prices",
        "adjustments": {
            "CPI": 2.0,      # +2% inflation
            "FEDFUNDS": 1.5, # +1.5% interest rates (Fed response)
            "UNRATE": 0.3,   # +0.3% unemployment
            "UMCSENT": -15,  # Consumer sentiment drops
            "DGS10": 1.0,    # +1% 10-year Treasury yield
        }
    },
    "recovery": {
        "name": "Economic Recovery",
        "description": "Strong growth with falling unemployment and rising confidence",
        "adjustments": {
            "GDP": 2.0,      # +2% GDP growth
            "UNRATE": -1.0,  # -1% unemployment
            "PAYEMS": 300000,  # +300K jobs per month
            "UMCSENT": 20,   # Consumer sentiment rises 20 points
            "CPI": 0.5,      # +0.5% inflation
            "FEDFUNDS": 0.5, # +0.5% interest rates
        }
    },
    "baseline": {
        "name": "Baseline (Current Trends)",
        "description": "Continue current economic conditions with modest growth",
        "adjustments": {
            "GDP": 0.5,       # +0.5% GDP growth (trend)
            "UNRATE": -0.1,   # -0.1% unemployment (slight improvement)
            "CPI": 0.2,       # +0.2% inflation (modest)
            "PAYEMS": 50000,  # +50K jobs per month (normal growth)
            "UMCSENT": 2,     # +2 points sentiment (slight optimism)
        }
    }
}

# Economic indicator categorization
INDICATOR_CATEGORIES = {
    "Labor Market": ["UNRATE", "PAYEMS", "CIVPART"],
    "Monetary Policy": ["FEDFUNDS", "DGS10", "DGS2"],
    "Consumer": ["UMCSENT", "CPI", "PCEPI", "RSXFS"],
    "Housing": ["HOUST", "PERMIT", "MORTGAGE30US"],
    "Production": ["IPMANS", "CMRMTSPL", "MANEMP"],
    "Financial": ["SP500", "VIXCLS", "DJIA"],
    "International": ["DEXUSEU", "DEXJPUS", "DTWEXBGS"]
}

# Data source mapping
DATA_SOURCES = {
    "UNRATE": "FRED",  # Unemployment Rate
    "PAYEMS": "FRED",  # Nonfarm Payrolls
    "GDP": "FRED",     # GDP
    "CPI": "FRED",     # Consumer Price Index
    "FEDFUNDS": "FRED", # Federal Funds Rate
    "UMCSENT": "FRED", # Consumer Sentiment
    "HOUST": "FRED",   # Housing Starts
    "DGS10": "FRED",   # 10-Year Treasury
    "SP500": "Yahoo",  # S&P 500
    "VIXCLS": "FRED",  # VIX
    "RSXFS": "FRED",   # Retail Sales
    "MRTSSM": "MRTS",  # Monthly Retail Trade
}


class ScenarioAnalyzer:
    """Analyzes economic scenarios and their impact on retail sales"""

    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize with historical economic data

        Args:
            historical_data: DataFrame with columns for all economic indicators
        """
        self.historical_data = historical_data
        self.scaler = StandardScaler()
        self.normalized_data = None
        self._normalize_data()

    def _normalize_data(self):
        """Normalize historical data for distance calculations"""
        numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
        self.normalized_data = self.historical_data.copy()
        self.normalized_data[numeric_cols] = self.scaler.fit_transform(
            self.historical_data[numeric_cols]
        )

    def generate_economic_scenario(
        self,
        base_features: Dict[str, float],
        scenario_type: str,
        prediction_model=None,
        base_prediction: float = 450000
    ) -> Dict:
        """
        Generate forecast under specified economic scenario

        Args:
            base_features: Current economic indicator values
            scenario_type: Type of scenario ('recession', 'rate_hike', etc.)
            prediction_model: ML model for generating predictions
            base_prediction: Base prediction value for calculating adjustments

        Returns:
            Dictionary with scenario adjustments, modified features, and forecast
        """
        if scenario_type not in SCENARIOS:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        scenario = SCENARIOS[scenario_type]

        # Apply scenario adjustments to base features
        modified_features = base_features.copy()

        for indicator, adjustment in scenario["adjustments"].items():
            if indicator in modified_features:
                # Handle both absolute values and rates
                if indicator in ["UNRATE", "CPI", "UMCSENT", "FEDFUNDS", "DGS10"]:
                    # Rate indicators (percentage points)
                    modified_features[indicator] += adjustment
                elif indicator in ["PAYEMS", "HOUST"]:
                    # Count indicators (absolute changes)
                    modified_features[indicator] += adjustment
                else:
                    # Default: treat as percentage
                    modified_features[indicator] *= (1 + adjustment / 100)

        # Calculate prediction based on economic impact
        # Use a simplified model: predict % change based on scenario
        scenario_impacts = {
            "recession": -0.08,      # -8% sales decline
            "rate_hike": -0.03,      # -3% sales decline
            "inflation_surge": -0.02, # -2% sales decline
            "recovery": 0.06,        # +6% sales growth
            "baseline": 0.01         # +1% baseline growth
        }

        impact_pct = scenario_impacts.get(scenario_type, 0.0)
        prediction = base_prediction * (1 + impact_pct)

        # Calculate confidence interval (wider for extreme scenarios)
        confidence_width = {
            "recession": 0.10,
            "rate_hike": 0.06,
            "inflation_surge": 0.08,
            "recovery": 0.08,
            "baseline": 0.04
        }
        width = confidence_width.get(scenario_type, 0.06)
        confidence_interval = (
            prediction * (1 - width),
            prediction * (1 + width)
        )

        return {
            "scenario_type": scenario_type,
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "base_features": base_features,
            "modified_features": modified_features,
            "adjustments": scenario["adjustments"],
            "prediction": prediction,
            "confidence_interval": confidence_interval,
            "impact_summary": self._calculate_impact_summary(base_features, modified_features)
        }

    def _calculate_impact_summary(
        self,
        base_features: Dict[str, float],
        modified_features: Dict[str, float]
    ) -> List[Dict]:
        """Calculate summary of feature changes"""
        summary = []

        for indicator in modified_features:
            if indicator in base_features:
                base_val = base_features[indicator]
                mod_val = modified_features[indicator]

                if base_val != 0:
                    change_pct = ((mod_val - base_val) / base_val) * 100
                else:
                    change_pct = 0

                summary.append({
                    "indicator": indicator,
                    "category": self._get_indicator_category(indicator),
                    "source": DATA_SOURCES.get(indicator, "Unknown"),
                    "base_value": base_val,
                    "scenario_value": mod_val,
                    "change": mod_val - base_val,
                    "change_pct": change_pct
                })

        # Sort by absolute change percentage
        summary.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        return summary

    def find_similar_periods(
        self,
        current_indicators: Dict[str, float],
        n: int = 5
    ) -> List[Dict]:
        """
        Find historical periods with similar economic conditions

        Args:
            current_indicators: Current economic indicator values
            n: Number of similar periods to return

        Returns:
            List of similar periods with dates and outcomes
        """
        if self.normalized_data is None:
            raise ValueError("Historical data not normalized")

        # Create array of current values
        indicators_list = list(current_indicators.keys())
        current_values = np.array([current_indicators[ind] for ind in indicators_list])

        # Normalize current values using the same scaler
        current_normalized = self.scaler.transform([current_values])[0]

        # Calculate distances to all historical periods
        distances = []
        for idx in self.normalized_data.index:
            norm_row = self.normalized_data.loc[idx]
            historical_values = norm_row[indicators_list].values

            # Skip rows with NaN values
            if pd.isna(historical_values).any():
                continue

            distance = euclidean(current_normalized, historical_values)

            # Get actual (unnormalized) values from historical_data
            actual_indicators = self.historical_data.loc[idx, indicators_list].to_dict()

            distances.append({
                "date": idx,
                "distance": distance,
                "indicators": actual_indicators,
                "retail_sales": norm_row.get("RSXFS", norm_row.get("MRTSSM", None))  # Retail sales (normalized for comparison)
            })

        # Sort by distance and return top n
        distances.sort(key=lambda x: x["distance"])
        similar_periods = distances[:n]

        # Add similarity scores (convert distance to similarity)
        max_distance = max(p["distance"] for p in similar_periods)
        for period in similar_periods:
            period["similarity_score"] = 1 - (period["distance"] / max_distance)

        return similar_periods

    def calculate_sensitivity(
        self,
        base_features: Dict[str, float],
        feature_name: str,
        range_values: Tuple[float, float, int],
        prediction_model=None,
        base_prediction: float = 450000
    ) -> Dict:
        """
        Calculate prediction sensitivity to a specific economic indicator

        Args:
            base_features: Base feature values
            feature_name: Feature to analyze
            range_values: (min, max, num_steps) for sensitivity sweep
            prediction_model: ML model for predictions
            base_prediction: Base prediction value

        Returns:
            Dictionary with sensitivity analysis results
        """
        min_val, max_val, num_steps = range_values
        values = np.linspace(min_val, max_val, num_steps)

        predictions = []
        baseline_val = base_features.get(feature_name, 0)

        # Define indicator impacts (simplified model)
        indicator_sensitivities = {
            "UNRATE": -50000,     # Each 1% unemployment reduces sales by $50K
            "FEDFUNDS": -30000,   # Each 1% rate increase reduces sales by $30K
            "CPI": -20000,        # Each 1% inflation reduces sales by $20K
            "GDP": 60000,         # Each 1% GDP growth increases sales by $60K
            "PAYEMS": 0.5,        # Each 1K jobs adds $0.5K sales
            "UMCSENT": 1000,      # Each index point adds $1K sales
            "HOUST": 50,          # Each housing start adds $50 sales
        }

        sensitivity = indicator_sensitivities.get(feature_name, 0)

        for val in values:
            # Calculate change in indicator
            delta = val - baseline_val

            # Apply sensitivity to base prediction
            pred = base_prediction + (delta * sensitivity)

            # Add some noise for realism
            noise = np.random.normal(0, base_prediction * 0.01)
            pred += noise

            predictions.append(pred)

        # Calculate sensitivity metrics
        pred_range = max(predictions) - min(predictions)
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)

        # Calculate elasticity (if baseline non-zero)
        baseline_pred = predictions[len(predictions) // 2]  # Middle value

        if baseline_val != 0 and baseline_pred != 0:
            elasticity = ((pred_range / baseline_pred) /
                         ((max_val - min_val) / baseline_val))
        else:
            elasticity = None

        return {
            "feature_name": feature_name,
            "category": self._get_indicator_category(feature_name),
            "source": DATA_SOURCES.get(feature_name, "Unknown"),
            "values_tested": values.tolist(),
            "predictions": predictions,
            "min_prediction": min(predictions),
            "max_prediction": max(predictions),
            "prediction_range": pred_range,
            "prediction_mean": pred_mean,
            "prediction_std": pred_std,
            "elasticity": elasticity,
            "baseline_value": baseline_val
        }

    def detect_economic_regime(
        self,
        current_indicators: Dict[str, float]
    ) -> Dict:
        """
        Classify current economic regime (Expansion, Peak, Recession, Recovery)

        Args:
            current_indicators: Current economic indicator values

        Returns:
            Dictionary with regime classification and confidence
        """
        # Define regime thresholds
        rules = {
            "Expansion": {
                "conditions": [
                    ("UNRATE", "<", 5.0),
                    ("GDP", ">", 2.0),
                    ("PAYEMS", ">", 100000),
                ],
                "description": "Strong growth with rising employment and output"
            },
            "Peak": {
                "conditions": [
                    ("UNRATE", "<", 4.5),
                    ("FEDFUNDS", ">", 3.0),
                    ("CPI", ">", 2.5),
                ],
                "description": "Late-cycle with high rates and inflation"
            },
            "Recession": {
                "conditions": [
                    ("UNRATE", ">", 6.0),
                    ("GDP", "<", 0),
                    ("PAYEMS", "<", -100000),
                ],
                "description": "Economic contraction with job losses"
            },
            "Recovery": {
                "conditions": [
                    ("UNRATE", ">", 5.0),
                    ("PAYEMS", ">", 0),
                    ("GDP", ">", 0),
                ],
                "description": "Emerging from recession with improving conditions"
            }
        }

        # Check each regime
        regime_scores = {}
        for regime_name, regime_info in rules.items():
            conditions_met = 0
            total_conditions = len(regime_info["conditions"])

            for indicator, operator, threshold in regime_info["conditions"]:
                value = current_indicators.get(indicator, 0)

                if operator == "<" and value < threshold:
                    conditions_met += 1
                elif operator == ">" and value > threshold:
                    conditions_met += 1

            regime_scores[regime_name] = {
                "score": conditions_met / total_conditions,
                "conditions_met": conditions_met,
                "total_conditions": total_conditions,
                "description": regime_info["description"]
            }

        # Find regime with highest score
        # If there's a tie, prioritize Peak over Expansion (Peak is more specific late-cycle)
        # Order: Recession > Peak > Recovery > Expansion (most specific to least specific)
        regime_priority = {
            "Recession": 4,
            "Peak": 3,
            "Recovery": 2,
            "Expansion": 1
        }

        # Filter regimes with max score
        max_score = max(scores["score"] for scores in regime_scores.values())
        top_regimes = [(name, scores) for name, scores in regime_scores.items()
                      if scores["score"] == max_score]

        # If tie, pick by priority (highest priority wins)
        if len(top_regimes) > 1:
            detected_regime = max(top_regimes, key=lambda x: regime_priority.get(x[0], 0))
        else:
            detected_regime = top_regimes[0]

        return {
            "regime": detected_regime[0],
            "confidence": detected_regime[1]["score"],
            "description": detected_regime[1]["description"],
            "all_regime_scores": regime_scores,
            "indicators": current_indicators
        }

    def _get_indicator_category(self, indicator: str) -> str:
        """Get the category for an economic indicator"""
        for category, indicators in INDICATOR_CATEGORIES.items():
            if indicator in indicators:
                return category
        return "Other"


def create_custom_scenario(
    base_features: Dict[str, float],
    adjustments: Dict[str, float]
) -> Dict:
    """
    Create custom economic scenario with user-defined adjustments

    Args:
        base_features: Current economic indicator values
        adjustments: User-defined adjustments {indicator: change}

    Returns:
        Custom scenario configuration
    """
    return {
        "name": "Custom Scenario",
        "description": "User-defined economic scenario",
        "adjustments": adjustments,
        "modified_features": apply_adjustments(base_features, adjustments)
    }


def apply_adjustments(
    base_features: Dict[str, float],
    adjustments: Dict[str, float]
) -> Dict[str, float]:
    """Apply adjustments to base features"""
    modified = base_features.copy()

    for indicator, adjustment in adjustments.items():
        if indicator in modified:
            if indicator in ["UNRATE", "CPI", "UMCSENT", "FEDFUNDS"]:
                # Rate indicators (percentage points)
                modified[indicator] += adjustment
            elif indicator in ["PAYEMS", "HOUST"]:
                # Count indicators
                modified[indicator] += adjustment
            else:
                # Percentage change
                modified[indicator] *= (1 + adjustment / 100)

    return modified


# Example usage
if __name__ == "__main__":
    # Example historical data
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="M")
    np.random.seed(42)

    historical_data = pd.DataFrame({
        "UNRATE": np.random.normal(5, 1, len(dates)),
        "PAYEMS": np.random.normal(200000, 50000, len(dates)),
        "GDP": np.random.normal(2, 1, len(dates)),
        "CPI": np.random.normal(2, 0.5, len(dates)),
        "FEDFUNDS": np.random.normal(2, 1, len(dates)),
        "RSXFS": np.random.normal(500000, 50000, len(dates))
    }, index=dates)

    # Create analyzer
    analyzer = ScenarioAnalyzer(historical_data)

    # Example current conditions
    current_indicators = {
        "UNRATE": 3.7,
        "PAYEMS": 250000,
        "GDP": 2.5,
        "CPI": 3.2,
        "FEDFUNDS": 5.25
    }

    # Generate recession scenario
    scenario = analyzer.generate_economic_scenario(
        current_indicators,
        "recession"
    )
    print(f"Scenario: {scenario['scenario_name']}")
    print(f"Impact: {scenario['impact_summary']}")

    # Find similar periods
    similar = analyzer.find_similar_periods(current_indicators, n=5)
    print(f"\nSimilar periods found: {len(similar)}")
    for period in similar[:3]:
        print(f"  {period['date']}: similarity={period['similarity_score']:.2f}")

    # Detect regime
    regime = analyzer.detect_economic_regime(current_indicators)
    print(f"\nCurrent regime: {regime['regime']} (confidence: {regime['confidence']:.2f})")

    # Calculate sensitivity
    sensitivity = analyzer.calculate_sensitivity(
        current_indicators,
        "UNRATE",
        (3.0, 7.0, 10)  # 3% to 7%, 10 steps
    )
    print(f"\nSensitivity to UNRATE: {sensitivity['elasticity']:.2f}" if sensitivity['elasticity'] else "N/A")
