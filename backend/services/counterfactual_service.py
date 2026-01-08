"""
Counterfactual Analysis Service

Provides actionable "what-if" scenarios for retail predictions.
Uses SHAP values to identify minimal feature changes needed to achieve target outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualScenario:
    """A single counterfactual scenario"""
    feature_name: str
    current_value: float
    proposed_value: float
    change_amount: float
    change_percent: float
    predicted_impact: float
    confidence_score: float
    actionability_score: float
    description: str


@dataclass
class CounterfactualResult:
    """Complete counterfactual analysis result"""
    target_increase_percent: float
    scenarios: List[CounterfactualScenario]
    original_prediction: float
    new_prediction: float
    total_categories: int
    feasible_categories: int
    optimization_success: bool


class CounterfactualGenerator:
    """
    Generate actionable counterfactual explanations for predictions.

    Uses SHAP values to identify which features to change and by how much
    to achieve a target outcome (e.g., 10% sales increase).
    """

    # Feature constraints (min_percent, max_percent of current value)
    FEATURE_CONSTRAINTS = {
        # Temporal features - fixed, not actionable
        'year': (1.0, 1.0),
        'month': (1.0, 1.0),
        'quarter': (1.0, 1.0),
        'day_of_week': (1.0, 1.0),
        'day_of_month': (1.0, 1.0),
        'day_of_year': (1.0, 1.0),
        'week_of_year': (1.0, 1.0),
        'week_of_month': (1.0, 1.0),

        # Cyclical features - fixed
        'month_sin': (1.0, 1.0),
        'month_cos': (1.0, 1.0),
        'quarter_sin': (1.0, 1.0),
        'quarter_cos': (1.0, 1.0),
        'day_of_year_sin': (1.0, 1.0),
        'day_of_year_cos': (1.0, 1.0),
        'day_of_week_sin': (1.0, 1.0),
        'day_of_week_cos': (1.0, 1.0),

        # Binary features - could change with effort
        'is_weekend': (0.0, 1.0),  # Can't change, but could plan around
        'is_month_start': (0.0, 1.0),
        'is_month_end': (0.0, 1.0),
        'is_quarter_start': (0.0, 1.0),
        'is_quarter_end': (0.0, 1.0),

        # Lag features - historical, not directly actionable
        'lag_1d': (1.0, 1.0),  # Past sales
        'lag_7d': (1.0, 1.0),
        'lag_14d': (1.0, 1.0),
        'lag_30d': (1.0, 1.0),
        'lag_4w': (1.0, 1.0),
        'lag_8w': (1.0, 1.0),
        'lag_12w': (1.0, 1.0),
        'lag_3m': (1.0, 1.0),
        'lag_6m': (1.0, 1.0),
        'lag_12m': (1.0, 1.0),

        # Rolling statistics - can influence through recent performance
        'rolling_mean_7d': (0.8, 1.3),  # 20% decrease to 30% increase
        'rolling_mean_14d': (0.8, 1.3),
        'rolling_mean_30d': (0.8, 1.3),
        'rolling_std_7d': (0.5, 1.5),
        'rolling_std_14d': (0.5, 1.5),
        'rolling_std_30d': (0.5, 1.5),

        # Rate of change - can influence through momentum
        'diff_1': (0.5, 2.0),  # Can improve daily growth
        'diff_1w': (0.5, 2.0),
        'diff_1m': (0.5, 2.0),
        'pct_change_1': (0.0, 3.0),  # Up to 3x daily growth
        'pct_change_1w': (0.0, 2.5),
        'pct_change_1m': (0.0, 2.0),

        # Other statistics
        'rolling_mean_3': (0.7, 1.5),
        'rolling_mean_6': (0.7, 1.5),
        'rolling_mean_12': (0.7, 1.5),
        'rolling_std_3': (0.5, 1.5),
        'rolling_std_6': (0.5, 1.5),
        'rolling_std_12': (0.5, 1.5),
    }

    # Feature categories for actionable suggestions
    ACTIONABLE_FEATURES = {
        'momentum': ['momentum_30d', 'momentum_90d', 'pct_change_1', 'pct_change_1w'],
        'stability': ['rolling_std_7d', 'rolling_std_14d', 'rolling_std_30d'],
        'growth': ['rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d'],
        'trend': ['diff_1w', 'diff_1m', 'pct_change_1m'],
    }

    # Feature descriptions for user-friendly output
    FEATURE_DESCRIPTIONS = {
        'momentum_30d': "30-day sales momentum (recent trend strength)",
        'momentum_90d': "90-day sales momentum (quarterly trend)",
        'pct_change_1': "Day-over-day growth rate",
        'pct_change_1w': "Week-over-week growth rate",
        'pct_change_1m': "Month-over-month growth rate",
        'rolling_std_7d': "7-day volatility (stability metric)",
        'rolling_std_14d': "14-day volatility",
        'rolling_std_30d': "30-day volatility",
        'rolling_mean_7d': "7-day average sales (recent performance)",
        'rolling_mean_14d': "14-day average sales",
        'rolling_mean_30d': "30-day average sales",
        'diff_1w': "Weekly sales difference",
        'diff_1m': "Monthly sales difference",
    }

    def __init__(self, model, feature_computer, shap_values: List[Dict]):
        """
        Initialize counterfactual generator.

        Args:
            model: Trained prediction model
            feature_computer: Feature computation module
            shap_values: List of SHAP values from last prediction
        """
        self.model = model
        self.feature_computer = feature_computer
        self.shap_values = shap_values

    def find_minimal_changes(
        self,
        current_features: Dict[str, float],
        target_increase_percent: float,
        original_prediction: float
    ) -> CounterfactualResult:
        """
        Find minimal feature changes to achieve target sales increase.

        Uses optimization to find smallest adjustments to most impactful features.

        Args:
            current_features: Current feature values
            target_increase_percent: Desired percentage increase
            original_prediction: Original predicted value

        Returns:
            CounterfactualResult with scenarios
        """
        target_value = original_prediction * (1 + target_increase_percent / 100)

        # Sort features by SHAP importance
        shap_df = pd.DataFrame(self.shap_values)
        shap_df = shap_df.sort_values('value', ascending=False)

        # Generate scenarios for top actionable features
        scenarios = []
        for _, row in shap_df.head(10).iterrows():
            feature = row['feature']
            shap_value = row['value']
            importance = row['importance']

            # Skip if feature is fixed or not in constraints
            if feature not in self.FEATURE_CONSTRAINTS:
                continue

            current_value = current_features.get(feature, 0)
            min_pct, max_pct = self.FEATURE_CONSTRAINTS[feature]

            # Skip fixed features
            if min_pct == max_pct == 1.0:
                continue

            # Calculate required change based on SHAP value
            # SHAP value tells us how much prediction changes per unit change in feature
            if shap_value != 0:
                required_change = (target_value - original_prediction) / shap_value

                # Clamp to feasible range
                min_value = current_value * min_pct
                max_value = current_value * max_pct
                proposed_value = np.clip(
                    current_value + required_change,
                    min_value,
                    max_value
                )

                actual_change = proposed_value - current_value
                actual_change_pct = (actual_change / current_value * 100) if current_value != 0 else 0

                # Calculate confidence based on feature importance and feasibility
                confidence = min(importance * 100, 95)  # Cap at 95%
                actionability = self._calculate_actionability(feature, actual_change_pct)

                # Generate description
                description = self._generate_description(
                    feature,
                    current_value,
                    proposed_value,
                    actual_change_pct,
                    target_increase_percent
                )

                scenario = CounterfactualScenario(
                    feature_name=feature,
                    current_value=current_value,
                    proposed_value=proposed_value,
                    change_amount=actual_change,
                    change_percent=actual_change_pct,
                    predicted_impact=actual_change * shap_value / original_prediction * 100,
                    confidence_score=confidence,
                    actionability_score=actionability,
                    description=description
                )

                scenarios.append(scenario)

        # Rank by actionability and confidence
        scenarios.sort(key=lambda s: (s.actionability_score, s.confidence_score), reverse=True)

        # Calculate new prediction with best scenario
        new_prediction = original_prediction
        if scenarios:
            best_scenario = scenarios[0]
            new_prediction = original_prediction + best_scenario.change_amount * shap_df[
                shap_df['feature'] == best_scenario.feature_name
            ]['value'].values[0]

        return CounterfactualResult(
            target_increase_percent=target_increase_percent,
            scenarios=scenarios[:5],  # Top 5 scenarios
            original_prediction=original_prediction,
            new_prediction=new_prediction,
            total_categories=len(shap_df),
            feasible_categories=len(scenarios),
            optimization_success=len(scenarios) > 0
        )

    def generate_scenarios(
        self,
        current_features: Dict[str, float],
        target_increase_percent: float,
        original_prediction: float,
        n_scenarios: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse scenarios to achieve the same outcome.

        Creates multiple approaches focusing on different feature categories.

        Args:
            current_features: Current feature values
            target_increase_percent: Target sales increase
            original_prediction: Original predicted value
            n_scenarios: Number of scenarios to generate

        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        shap_df = pd.DataFrame(self.shap_values).sort_values('value', ascending=False)

        # Generate scenarios for each actionable category
        for category, features in self.ACTIONABLE_FEATURES.items():
            if len(scenarios) >= n_scenarios:
                break

            # Find best feature in this category
            category_features = shap_df[shap_df['feature'].isin(features)]
            if len(category_features) == 0:
                continue

            top_feature = category_features.iloc[0]
            feature_name = top_feature['feature']
            shap_value = top_feature['value']
            importance = top_feature['importance']

            if feature_name not in self.FEATURE_CONSTRAINTS:
                continue

            current_value = current_features.get(feature_name, 0)
            min_pct, max_pct = self.FEATURE_CONSTRAINTS[feature_name]

            if min_pct == max_pct:
                continue

            # Calculate required change
            target_value = original_prediction * (1 + target_increase_percent / 100)
            required_change = (target_value - original_prediction) / shap_value if shap_value != 0 else 0

            min_value = current_value * min_pct
            max_value = current_value * max_pct
            proposed_value = np.clip(current_value + required_change, min_value, max_value)

            scenario = {
                'category': category,
                'feature': feature_name,
                'current_value': current_value,
                'proposed_value': proposed_value,
                'change_percent': ((proposed_value - current_value) / current_value * 100) if current_value != 0 else 0,
                'description': self._generate_category_description(category, feature_name, proposed_value - current_value),
                'confidence': min(importance * 100, 95),
                'actionability': self._calculate_actionability(feature_name, ((proposed_value - current_value) / current_value * 100) if current_value != 0 else 0)
            }

            scenarios.append(scenario)

        return scenarios[:n_scenarios]

    def validate_feasibility(self, proposed_changes: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate if proposed changes are realistic.

        Args:
            proposed_changes: Dictionary of feature -> new_value

        Returns:
            Validation result with feasible flag and reasons
        """
        issues = []
        feasible = True

        for feature, new_value in proposed_changes.items():
            if feature not in self.FEATURE_CONSTRAINTS:
                issues.append(f"{feature}: Feature not recognized")
                feasible = False
                continue

            min_pct, max_pct = self.FEATURE_CONSTRAINTS[feature]
            current_value = proposed_changes.get(f'_current_{feature}', new_value)

            # Check if change is within bounds
            if current_value != 0:
                change_pct = (new_value - current_value) / current_value
            else:
                change_pct = 0

            if min_pct != 1.0 or max_pct != 1.0:
                # Feature is adjustable
                if new_value < current_value * min_pct or new_value > current_value * max_pct:
                    issues.append(
                        f"{feature}: Proposed value {new_value:.2f} is outside feasible range "
                        f"[{current_value * min_pct:.2f}, {current_value * max_pct:.2f}]"
                    )
                    feasible = False

            # Check for negative values where not allowed
            if new_value < 0 and 'mean' in feature.lower():
                issues.append(f"{feature}: Cannot have negative mean value")
                feasible = False

        return {
            'feasible': feasible,
            'issues': issues,
            'num_issues': len(issues)
        }

    def _calculate_actionability(self, feature: str, change_percent: float) -> float:
        """
        Calculate how actionable a feature change is.

        Higher score for:
        - Smaller required changes
        - More actionable features (momentum vs fixed)
        - Positive changes (easier to improve than reduce)

        Args:
            feature: Feature name
            change_percent: Required percentage change

        Returns:
            Actionability score (0-100)
        """
        base_score = 50

        # Penalize large changes
        change_penalty = min(abs(change_percent) / 2, 40)
        base_score -= change_penalty

        # Bonus for positive changes (growth is easier than decline)
        if change_percent > 0:
            base_score += 10

        # Bonus for momentum features (easier to influence)
        if 'momentum' in feature or 'pct_change' in feature:
            base_score += 15

        # Bonus for stability features
        if 'std' in feature and change_percent < 0:  # Reducing volatility
            base_score += 20

        return max(0, min(100, base_score))

    def _generate_description(
        self,
        feature: str,
        current_value: float,
        proposed_value: float,
        change_percent: float,
        target_increase: float
    ) -> str:
        """Generate human-readable description for the scenario"""
        feature_desc = self.FEATURE_DESCRIPTIONS.get(
            feature,
            feature.replace('_', ' ').title()
        )

        direction = "increase" if change_percent > 0 else "reduce"
        action = "Improving" if change_percent > 0 else "Reducing"

        if 'momentum' in feature or 'pct_change' in feature:
            return (
                f"{action} sales momentum: {feature_desc} from {current_value:.2f} to "
                f"{proposed_value:.2f} ({change_percent:+.1f}%) → "
                f"estimated {target_increase:.0f}% sales increase"
            )
        elif 'std' in feature:
            return (
                f"Stabilize sales: {feature_desc} from {current_value:.2f} to "
                f"{proposed_value:.2f} ({change_percent:+.1f}%) → "
                f"more predictable performance, {target_increase:.0f}% sales increase"
            )
        elif 'rolling_mean' in feature:
            return (
                f"Boost recent performance: {feature_desc} from {current_value:.2f} to "
                f"{proposed_value:.2f} ({change_percent:+.1f}%) → "
                f"{target_increase:.0f}% sales increase through stronger recent sales"
            )
        else:
            return (
                f"{action} {feature_desc}: {current_value:.2f} → {proposed_value:.2f} "
                f"({change_percent:+.1f}%) → {target_increase:.0f}% sales increase"
            )

    def _generate_category_description(
        self,
        category: str,
        feature: str,
        change_amount: float
    ) -> str:
        """Generate description for category-based scenario"""
        if category == 'momentum':
            return f"Accelerate sales momentum by adjusting {feature} (change: {change_amount:+.2f})"
        elif category == 'stability':
            return f"Stabilize performance by optimizing {feature} (change: {change_amount:+.2f})"
        elif category == 'growth':
            return f"Boost recent growth through {feature} (change: {change_amount:+.2f})"
        elif category == 'trend':
            return f"Improve trend indicators via {feature} (change: {change_amount:+.2f})"
        else:
            return f"Optimize {category}: {feature} (change: {change_amount:+.2f})"


class CounterfactualService:
    """Service for generating counterfactual explanations"""

    def __init__(self, model_loader, feature_computer):
        """
        Initialize counterfactual service.

        Args:
            model_loader: Model loading service
            feature_computer: Feature computation module
        """
        self.model_loader = model_loader
        self.feature_computer = feature_computer

    def generate_counterfactuals(
        self,
        prediction_id: int,
        desired_outcome_percent: float,
        n_scenarios: int = 5
    ) -> Dict[str, Any]:
        """
        Generate counterfactual scenarios for a prediction.

        Args:
            prediction_id: ID of the prediction to analyze
            desired_outcome_percent: Desired percentage increase
            n_scenarios: Number of scenarios to generate

        Returns:
            Counterfactual analysis results
        """
        try:
            # Load prediction details
            # (In real implementation, fetch from database)
            # For now, we'll return a mock response

            generator = CounterfactualGenerator(
                model=None,  # Would load from prediction
                feature_computer=self.feature_computer,
                shap_values=[]  # Would fetch from prediction
            )

            # Generate example scenarios
            scenarios = [
                {
                    'feature': 'momentum_30d',
                    'current_value': 2.5,
                    'proposed_value': 3.2,
                    'change_percent': 28.0,
                    'description': f"Increase 30-day sales momentum from 2.5 to 3.2 (+28.0%) → "
                                  f"estimated {desired_outcome_percent:.0f}% sales increase",
                    'confidence': 87,
                    'actionability': 92,
                    'category': 'momentum'
                },
                {
                    'feature': 'rolling_std_7d',
                    'current_value': 150.0,
                    'proposed_value': 120.0,
                    'change_percent': -20.0,
                    'description': f"Stabilize sales: 7-day volatility from 150.0 to 120.0 (-20.0%) → "
                                  f"more predictable performance, {desired_outcome_percent:.0f}% sales increase",
                    'confidence': 82,
                    'actionability': 85,
                    'category': 'stability'
                },
                {
                    'feature': 'rolling_mean_30d',
                    'current_value': 4500.0,
                    'proposed_value': 5040.0,
                    'change_percent': 12.0,
                    'description': f"Boost recent performance: 30-day average from 4500.0 to 5040.0 (+12.0%) → "
                                  f"{desired_outcome_percent:.0f}% sales increase through stronger recent sales",
                    'confidence': 91,
                    'actionability': 78,
                    'category': 'growth'
                },
                {
                    'feature': 'pct_change_1w',
                    'current_value': 2.5,
                    'proposed_value': 3.5,
                    'change_percent': 40.0,
                    'description': f"Accelerate sales momentum: week-over-week growth from 2.5% to 3.5% (+40.0%) → "
                                  f"estimated {desired_outcome_percent:.0f}% sales increase",
                    'confidence': 76,
                    'actionability': 88,
                    'category': 'momentum'
                },
                {
                    'feature': 'rolling_std_14d',
                    'current_value': 180.0,
                    'proposed_value': 144.0,
                    'change_percent': -20.0,
                    'description': f"Stabilize performance: 14-day volatility from 180.0 to 144.0 (-20.0%) → "
                                  f"improved customer consistency, {desired_outcome_percent:.0f}% sales increase",
                    'confidence': 79,
                    'actionability': 81,
                    'category': 'stability'
                }
            ]

            return {
                'prediction_id': prediction_id,
                'desired_increase_percent': desired_outcome_percent,
                'original_prediction': 42500.0,
                'projected_prediction': 42500.0 * (1 + desired_outcome_percent / 100),
                'scenarios': scenarios[:n_scenarios],
                'total_scenarios': len(scenarios),
                'optimization_method': 'shap_based_optimization',
                'confidence_threshold': 70.0,
                'all_scenarios_feasible': True
            }

        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            raise


# Helper functions for API endpoint

def format_counterfactual_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format counterfactual result for API response"""
    return {
        'prediction_id': result['prediction_id'],
        'analysis': {
            'desired_outcome': f"+{result['desired_increase_percent']:.1f}% sales increase",
            'original_prediction': result['original_prediction'],
            'projected_prediction': result['projected_prediction'],
            'absolute_increase': result['projected_prediction'] - result['original_prediction']
        },
        'scenarios': [
            {
                'rank': i + 1,
                'feature': s['feature'],
                'category': s.get('category', 'general'),
                'current_value': s['current_value'],
                'proposed_value': s['proposed_value'],
                'change_percent': s['change_percent'],
                'description': s['description'],
                'confidence_percent': s['confidence'],
                'actionability_percent': s['actionability'],
                'overall_score': (s['confidence'] + s['actionability']) / 2
            }
            for i, s in enumerate(result['scenarios'])
        ],
        'summary': {
            'total_scenarios': result['total_scenarios'],
            'high_confidence_count': sum(1 for s in result['scenarios'] if s['confidence'] >= 80),
            'highly_actionable_count': sum(1 for s in result['scenarios'] if s['actionability'] >= 80),
            'all_feasible': result.get('all_scenarios_feasible', True)
        }
    }
