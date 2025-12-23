"""
Long-Term Forecaster Module
Generates 5-year (60-month) forecasts with dedicated visualizations

This module extends the prediction functionality to support longer forecast horizons
with appropriate uncertainty quantification and trend visualization.
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class LongTermForecaster:
    """
    Generate long-term forecasts (5 years / 60 months) for retail sales

    Features:
    - Multi-year forecasting with uncertainty quantification
    - Trend analysis and seasonal decomposition visualization
    - Specialized plots for long-term patterns
    - Scenario analysis (optimistic, baseline, pessimistic)
    """

    def __init__(self, output_dir: Path, results_dir: Path, data_dir: Path):
        """
        Initialize the long-term forecaster

        Args:
            output_dir: Base output directory for results
            results_dir: Directory for prediction results
            data_dir: Directory containing processed data files
        """
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.long_term_dir = self.output_dir / "long_term_forecasts"
        self.long_term_dir.mkdir(parents=True, exist_ok=True)

        logger.info("LongTermForecaster initialized")

    def generate_long_term_forecasts(
        self,
        targets: List[str],
        horizon_months: int = 60,
        scenarios: bool = True
    ) -> Dict[str, Any]:
        """
        Generate 5-year forecasts for specified targets

        Args:
            targets: List of target categories to forecast
            horizon_months: Forecast horizon in months (default: 60 = 5 years)
            scenarios: Whether to generate scenario-based forecasts

        Returns:
            Dictionary containing all forecast results
        """
        logger.info(f"Generating {horizon_months}-month ({horizon_months//12}-year) forecasts...")

        results = {
            'metadata': {
                'forecast_date': datetime.now().isoformat(),
                'horizon_months': horizon_months,
                'horizon_years': horizon_months // 12,
                'targets': targets,
                'scenarios_enabled': scenarios
            },
            'forecasts': {}
        }

        for target in targets:
            try:
                logger.info(f"Generating long-term forecast for {target}...")

                # Load historical data for context
                historical_data = self._load_category_data(target)

                if historical_data is None or historical_data.empty:
                    logger.warning(f"No historical data found for {target}")
                    continue

                # Generate forecasts using time series methods
                forecast_result = self._generate_ts_forecast(
                    historical_data, horizon_months, target
                )

                if forecast_result is None:
                    logger.warning(f"Failed to generate forecast for {target}")
                    continue

                # Generate scenario forecasts if requested
                scenario_forecasts = None
                if scenarios:
                    scenario_forecasts = self._generate_scenario_forecasts(
                        forecast_result, historical_data
                    )

                # Store results
                results['forecasts'][target] = {
                    'historical_data': {
                        'dates': historical_data.index.strftime('%Y-%m-%d').tolist(),
                        'values': historical_data['y'].values.tolist() if 'y' in historical_data.columns else historical_data.iloc[:, 0].values.tolist()
                    },
                    'ensemble': forecast_result,
                    'scenarios': scenario_forecasts
                }

                # Create visualizations
                self._create_long_term_visualizations(
                    target, historical_data, forecast_result, scenario_forecasts
                )

                logger.info(f"Successfully generated long-term forecast for {target}")

            except Exception as e:
                logger.error(f"Error generating long-term forecast for {target}: {e}")
                continue

        # Save results
        self._save_forecast_results(results)

        return results

    def _load_category_data(self, target: str) -> Optional[pd.DataFrame]:
        """Load historical data for a specific retail category"""
        try:
            # Try parquet file first
            parquet_path = self.data_dir / f"{target}.parquet"

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)

                # Ensure we have a proper index - convert RangeIndex to DatetimeIndex if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    else:
                        # If no date column, create a simple date index starting from 2015
                        df.index = pd.date_range(start='2015-01-01', periods=len(df), freq='MS')

                # Use the first column as 'y' if it doesn't exist
                if 'y' not in df.columns:
                    df['y'] = df.iloc[:, 0]

                logger.info(f"Loaded {len(df)} observations from {parquet_path}")
                return df

            # Try CSV file
            csv_path = self.data_dir / f"{target}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)

                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    else:
                        df.index = pd.date_range(start='2015-01-01', periods=len(df), freq='MS')

                if 'y' not in df.columns:
                    df['y'] = df.iloc[:, 0]

                logger.info(f"Loaded {len(df)} observations from {csv_path}")
                return df

            logger.warning(f"No data file found for {target}")
            return None

        except Exception as e:
            logger.error(f"Error loading data for {target}: {e}")
            return None

    def _generate_ts_forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: int,
        target: str
    ) -> Optional[Dict[str, Any]]:
        """Generate time series forecast using multiple methods"""

        try:
            y_values = historical_data['y'].values if 'y' in historical_data.columns else historical_data.iloc[:, 0].values

            # Method 1: Exponential smoothing forecast
            es_forecast = self._exponential_smoothing_forecast(y_values, horizon)

            # Method 2: ARIMA-like forecast (using trend + seasonality)
            arima_forecast = self._trend_seasonality_forecast(y_values, horizon, historical_data.index)

            # Method 3: Simple growth projection
            growth_forecast = self._growth_projection(y_values, horizon)

            # Ensemble of all methods
            all_forecasts = np.array([
                es_forecast,
                arima_forecast,
                growth_forecast
            ])

            mean_forecast = np.mean(all_forecasts, axis=0)
            std_forecast = np.std(all_forecasts, axis=0)

            # Calculate confidence intervals
            lo_80 = mean_forecast - 1.28 * std_forecast
            hi_80 = mean_forecast + 1.28 * std_forecast
            lo_95 = mean_forecast - 1.96 * std_forecast
            hi_95 = mean_forecast + 1.96 * std_forecast

            return {
                'point_forecast': mean_forecast.tolist(),
                'confidence_intervals': {
                    '80': {'lower': lo_80.tolist(), 'upper': hi_80.tolist()},
                    '95': {'lower': lo_95.tolist(), 'upper': hi_95.tolist()}
                },
                'method_count': 3,
                'methods': ['exponential_smoothing', 'trend_seasonality', 'growth_projection']
            }

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None

    def _exponential_smoothing_forecast(self, y_values: np.ndarray, horizon: int) -> np.ndarray:
        """Simple exponential smoothing forecast"""

        # Calculate alpha (smoothing parameter) based on data length
        alpha = 2.0 / (len(y_values) + 1)

        forecast = np.zeros(horizon)
        forecast[0] = y_values[-1]

        # Apply exponential smoothing
        for i in range(1, horizon):
            forecast[i] = alpha * y_values[-1] + (1 - alpha) * forecast[i-1]

        return forecast

    def _trend_seasonality_forecast(
        self,
        y_values: np.ndarray,
        horizon: int,
        historical_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """Forecast using trend + seasonal decomposition"""

        # Calculate trend using linear regression on last 24 months
        lookback = min(24, len(y_values))
        recent_values = y_values[-lookback:]

        x = np.arange(lookback)
        z = np.polyfit(x, recent_values, 1)
        trend = z[0]

        # Base forecast
        last_value = y_values[-1]
        forecast = last_value + trend * np.arange(1, horizon + 1)

        # Add seasonality if we have enough data
        if len(y_values) >= 24:
            # Estimate seasonal pattern
            seasonal_pattern = y_values[-24:] - np.mean(y_values[-24:])

            # Extend seasonal pattern
            extended_seasonal = np.tile(seasonal_pattern, horizon // 24 + 1)[:horizon]

            # Blend seasonal component (30% weight)
            forecast = forecast + extended_seasonal * 0.3

        return forecast

    def _growth_projection(self, y_values: np.ndarray, horizon: int) -> np.ndarray:
        """Simple compound growth projection"""

        # Calculate average growth rate
        if len(y_values) >= 12:
            # Use year-over-year growth
            yoy_growth = (y_values[-12:] - y_values[-24:-12]) / y_values[-24:-12]
            avg_growth = np.mean(yoy_growth)
        else:
            # Use overall trend
            avg_growth = (y_values[-1] - y_values[0]) / (len(y_values) * y_values[0])

        # Apply compound growth
        last_value = y_values[-1]
        forecast = last_value * (1 + avg_growth) ** np.arange(1, horizon + 1)

        return forecast

    def _generate_scenario_forecasts(
        self,
        ensemble_forecast: Dict[str, Any],
        historical_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Generate optimistic, baseline, and pessimistic scenarios"""

        mean_forecast = np.array(ensemble_forecast['point_forecast'])
        y_values = historical_data['y'].values if 'y' in historical_data.columns else historical_data.iloc[:, 0].values

        # Calculate growth rate volatility from historical data
        if len(y_values) >= 12:
            yoy_growth = (y_values[-12:] - y_values[-24:-12]) / y_values[-24:-12]
            growth_volatility = np.std(yoy_growth)
        else:
            growth_volatility = 0.05  # Default 5% volatility

        # Optimistic scenario: +0.5 standard deviation growth
        optimistic = mean_forecast * (1 + 0.5 * growth_volatility) ** np.arange(len(mean_forecast))

        # Pessimistic scenario: -0.5 standard deviation growth
        pessimistic = mean_forecast * (1 - 0.5 * growth_volatility) ** np.arange(len(mean_forecast))

        return {
            'optimistic': {
                'point_forecast': optimistic.tolist(),
                'description': 'Optimistic growth scenario (+0.5σ)'
            },
            'baseline': {
                'point_forecast': mean_forecast.tolist(),
                'description': 'Baseline ensemble forecast'
            },
            'pessimistic': {
                'point_forecast': pessimistic.tolist(),
                'description': 'Pessimistic growth scenario (-0.5σ)'
            }
        }

    def _create_long_term_visualizations(
        self,
        target: str,
        historical_data: pd.DataFrame,
        ensemble_forecast: Dict[str, Any],
        scenario_forecasts: Optional[Dict[str, Dict[str, Any]]]
    ):
        """Create comprehensive visualizations for long-term forecasts"""

        # Create category directory
        category_dir = self.long_term_dir / target
        category_dir.mkdir(parents=True, exist_ok=True)

        # Get historical values
        hist_y = historical_data['y'].values if 'y' in historical_data.columns else historical_data.iloc[:, 0].values
        hist_dates = historical_data.index

        # Generate forecast dates
        last_date = hist_dates[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=len(ensemble_forecast['point_forecast']),
            freq='MS'
        )

        # 1. Main long-term forecast plot
        self._create_main_forecast_plot(
            target, hist_dates, hist_y, ensemble_forecast,
            forecast_dates, category_dir
        )

        # 2. Scenario comparison plot
        if scenario_forecasts:
            self._create_scenario_plot(
                target, hist_dates, hist_y, scenario_forecasts,
                forecast_dates, category_dir
            )

        # 3. Trend analysis plot
        self._create_trend_analysis_plot(
            target, hist_dates, hist_y, ensemble_forecast,
            forecast_dates, category_dir
        )

        logger.info(f"Created long-term visualizations for {target}")

    def _create_main_forecast_plot(
        self,
        target: str,
        hist_dates: pd.DatetimeIndex,
        hist_y: np.ndarray,
        ensemble_forecast: Dict[str, Any],
        forecast_dates: pd.DatetimeIndex,
        output_dir: Path
    ):
        """Create main forecast plot with confidence intervals"""

        fig = go.Figure()

        # Add historical data
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_y,
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        ))

        # Add point forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ensemble_forecast['point_forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=3)
        ))

        # Add 95% confidence interval
        ci_95 = ensemble_forecast['confidence_intervals']['95']
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ci_95['upper'],
            mode='lines',
            name='95% CI Upper',
            line=dict(color='lightblue', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ci_95['lower'],
            mode='lines',
            name='95% Confidence Interval',
            line=dict(color='lightblue', width=0),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)'
        ))

        # Add 80% confidence interval
        ci_80 = ensemble_forecast['confidence_intervals']['80']
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ci_80['upper'],
            mode='lines',
            name='80% CI Upper',
            line=dict(color='blue', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ci_80['lower'],
            mode='lines',
            name='80% Confidence Interval',
            line=dict(color='blue', width=0),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.15)'
        ))

        # Update layout
        fig.update_layout(
            title=f'{target.replace("_", " ")} - {len(forecast_dates)//12}-Year Forecast',
            xaxis_title='Date',
            yaxis_title='Sales (Millions)',
            width=1600,
            height=800,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)')
        )

        # Save
        html_file = output_dir / f"{target}_long_term_forecast.html"
        fig.write_html(str(html_file))

        png_file = output_dir / f"{target}_long_term_forecast.png"
        try:
            fig.write_image(str(png_file), width=1600, height=800)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")

    def _create_scenario_plot(
        self,
        target: str,
        hist_dates: pd.DatetimeIndex,
        hist_y: np.ndarray,
        scenario_forecasts: Dict[str, Dict[str, Any]],
        forecast_dates: pd.DatetimeIndex,
        output_dir: Path
    ):
        """Create scenario comparison plot"""

        fig = go.Figure()

        # Add historical data
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_y,
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        ))

        # Add scenarios
        colors = {
            'optimistic': 'green',
            'baseline': 'blue',
            'pessimistic': 'red'
        }

        for scenario_name, scenario_data in scenario_forecasts.items():
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=scenario_data['point_forecast'],
                mode='lines',
                name=scenario_data['description'],
                line=dict(color=colors.get(scenario_name, 'gray'), width=2.5)
            ))

        # Update layout
        fig.update_layout(
            title=f'{target.replace("_", " ")} - Long-Term Scenario Analysis',
            xaxis_title='Date',
            yaxis_title='Sales (Millions)',
            width=1600,
            height=800,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)')
        )

        # Save
        html_file = output_dir / f"{target}_scenario_analysis.html"
        fig.write_html(str(html_file))

        png_file = output_dir / f"{target}_scenario_analysis.png"
        try:
            fig.write_image(str(png_file), width=1600, height=800)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")

    def _create_trend_analysis_plot(
        self,
        target: str,
        hist_dates: pd.DatetimeIndex,
        hist_y: np.ndarray,
        ensemble_forecast: Dict[str, Any],
        forecast_dates: pd.DatetimeIndex,
        output_dir: Path
    ):
        """Create trend analysis plot showing yearly aggregates"""

        # Combine historical and forecast data
        all_dates = hist_dates.append(forecast_dates)
        all_values = np.concatenate([
            hist_y,
            ensemble_forecast['point_forecast']
        ])

        # Create DataFrame for yearly aggregation
        df = pd.DataFrame({'date': all_dates, 'value': all_values})
        df['year'] = df['date'].dt.year
        yearly_avg = df.groupby('year')['value'].mean()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Time Series', 'Yearly Average Trend'),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )

        # Monthly plot
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_y,
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ensemble_forecast['point_forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        # Yearly average plot
        fig.add_trace(go.Scatter(
            x=yearly_avg.index,
            y=yearly_avg.values,
            mode='lines+markers',
            name='Yearly Average',
            line=dict(color='darkblue', width=3),
            marker=dict(size=8)
        ), row=2, col=1)

        # Add vertical line at forecast start
        forecast_start_year = forecast_dates[0].year
        fig.add_vline(
            x=forecast_start_year,
            line_dash='dash',
            line_color='red',
            annotation_text='Forecast Start',
            row=2, col=1
        )

        # Update layout
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Year', row=2, col=1)
        fig.update_yaxes(title_text='Sales (Millions)', row=1, col=1)
        fig.update_yaxes(title_text='Average Sales (Millions)', row=2, col=1)

        fig.update_layout(
            title=f'{target.replace("_", " ")} - Trend Analysis ({len(forecast_dates)//12}-Year)',
            width=1600,
            height=1000,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )

        # Save
        html_file = output_dir / f"{target}_trend_analysis.html"
        fig.write_html(str(html_file))

        png_file = output_dir / f"{target}_trend_analysis.png"
        try:
            fig.write_image(str(png_file), width=1600, height=1000)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")

    def _save_forecast_results(self, results: Dict[str, Any]):
        """Save forecast results to JSON file"""

        output_file = self.long_term_dir / "long_term_forecast_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved long-term forecast results to {output_file}")

    def generate_forecast_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of long-term forecasts"""

        summary = f"""
# Long-Term Forecast Summary
Generated: {results['metadata']['forecast_date']}
Forecast Horizon: {results['metadata']['horizon_years']} years ({results['metadata']['horizon_months']} months)

## Executive Summary

This report presents {results['metadata']['horizon_years']}-year forecasts for {len(results['forecasts'])} retail categories.
Forecasts include uncertainty quantification with 80% and 95% confidence intervals.

## Category Forecasts

"""

        for target, forecast_data in results['forecasts'].items():
            ensemble = forecast_data['ensemble']
            point_forecast = ensemble['point_forecast']

            # Calculate statistics
            initial_value = point_forecast[0]
            final_value = point_forecast[-1]
            total_growth = ((final_value - initial_value) / initial_value) * 100
            annualized_growth = (final_value / initial_value) ** (1 / results['metadata']['horizon_years']) - 1

            summary += f"""
### {target.replace('_', ' ')}

**Key Metrics:**
- Initial Forecast (Month 1): ${initial_value:,.2f}M
- Final Forecast (Year {results['metadata']['horizon_years']}): ${final_value:,.2f}M
- Total Growth: {total_growth:+.2f}%
- Annualized Growth Rate: {annualized_growth*100:+.2f}%

**Confidence Intervals (Final Year):**
- 80% CI: ${ensemble['confidence_intervals']['80']['lower'][-1]:,.2f}M - ${ensemble['confidence_intervals']['80']['upper'][-1]:,.2f}M
- 95% CI: ${ensemble['confidence_intervals']['95']['lower'][-1]:,.2f}M - ${ensemble['confidence_intervals']['95']['upper'][-1]:,.2f}M

**Methods Used:** {', '.join(ensemble.get('methods', ['ensemble']))}

"""

        summary += """
## Notes

- Forecasts are generated using ensemble methods combining multiple approaches:
  - Exponential smoothing for baseline trend
  - Trend-seasonality decomposition for patterns
  - Growth projection for long-term trajectory
- Confidence intervals widen over time due to increasing uncertainty
- Scenario analysis provides optimistic, baseline, and pessimistic projections
- All monetary values in millions of dollars

---
*Generated by RetailPRED Long-Term Forecaster*
"""

        return summary
