#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modeling Script for Retail Market Dynamics Project
Performs statistical analysis and trains ML models to predict retail dynamics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Statistical analysis imports
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    print("⚠ statsmodels not fully installed. Granger causality tests will be skipped.")
    grangercausalitytests = None

# Prophet import
try:
    from prophet import Prophet
except ImportError:
    print("⚠ Prophet not installed. Prophet model will be skipped.")
    Prophet = None


def load_features(file_path='data/processed/features.csv'):
    """
    Load the features dataset.
    
    Args:
        file_path (str): Path to features.csv
        
    Returns:
        pd.DataFrame: Features dataset
    """
    print("="*60)
    print("Loading Features Dataset...")
    print("="*60)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {df.shape[0]} records from {file_path}")
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"    - {col}")
        
        return df
        
    except FileNotFoundError:
        print(f"⚠ File not found: {file_path}")
        print("Creating sample data for demonstration...")
        
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        df = pd.DataFrame({
            'date': dates,
            'retail_sales': np.random.randint(500000, 1000000, len(dates)),
            'retail_growth': np.random.randn(len(dates)) * 2,
            'sp500_close': 300 + np.cumsum(np.random.randn(len(dates)) * 5),
            'sp500_return': np.random.randn(len(dates)) * 2,
            'cpi': 230 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'consumer_confidence': 90 + np.random.randn(len(dates)) * 5
        })
        
        print(f"✓ Created sample data with {len(df)} records")
        return df
        
    except Exception as e:
        print(f"⚠ Error loading data: {e}")
        return pd.DataFrame()


def run_correlation_tests(df):
    """
    Run correlation analysis between retail and stock variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Correlation results
    """
    print("\n" + "="*60)
    print("Running Correlation Analysis...")
    print("="*60)
    
    # Define variables of interest
    retail_vars = ['retail_sales', 'retail_growth']
    stock_vars = ['sp500_close', 'sp500_return']
    economic_vars = ['cpi', 'consumer_confidence']
    
    # Filter to available columns
    all_vars = retail_vars + stock_vars + economic_vars
    available_vars = [v for v in all_vars if v in df.columns]
    
    if len(available_vars) < 2:
        print("⚠ Not enough variables for correlation analysis")
        return {}
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr()
    
    print("\nCorrelations between Retail and Stock Variables:")
    for retail_var in retail_vars:
        if retail_var in available_vars:
            for stock_var in stock_vars:
                if stock_var in available_vars:
                    corr = corr_matrix.loc[retail_var, stock_var]
                    print(f"  {retail_var} ↔ {stock_var}: {corr:.3f}")
    
    print("\nCorrelations between Retail and Economic Variables:")
    for retail_var in retail_vars:
        if retail_var in available_vars:
            for econ_var in economic_vars:
                if econ_var in available_vars:
                    corr = corr_matrix.loc[retail_var, econ_var]
                    print(f"  {retail_var} ↔ {econ_var}: {corr:.3f}")
    
    return corr_matrix.to_dict()


def run_granger_causality_tests(df, max_lag=4):
    """
    Run Granger causality tests to assess predictive relationships.
    
    Args:
        df (pd.DataFrame): Input dataframe
        max_lag (int): Maximum lag to test
        
    Returns:
        dict: Granger test results
    """
    print("\n" + "="*60)
    print("Running Granger Causality Tests...")
    print("="*60)
    
    if grangercausalitytests is None:
        print("⚠ statsmodels not available. Skipping Granger tests.")
        return {}
    
    results = {}
    
    # Test relationships of interest
    # Does sp500_return Granger-cause retail_growth?
    if 'sp500_return' in df.columns and 'retail_growth' in df.columns:
        try:
            test_data = df[['sp500_return', 'retail_growth']].dropna()
            if len(test_data) > max_lag * 2:
                print(f"\nTesting: Does sp500_return Granger-cause retail_growth?")
                result = grangercausalitytests(test_data, max_lag, verbose=False)
                # Extract p-values for each lag
                p_values = {}
                for lag in range(1, max_lag + 1):
                    if lag in result:
                        p_val = result[lag][0]['ssr_ftest'][1]
                        p_values[f'lag_{lag}'] = p_val
                        print(f"  Lag {lag}: p-value = {p_val:.4f}")
                results['sp500_to_retail'] = p_values
        except Exception as e:
            print(f"  ⚠ Error testing sp500→retail: {e}")
    
    # Does retail_growth Granger-cause sp500_return?
    if 'retail_growth' in df.columns and 'sp500_return' in df.columns:
        try:
            test_data = df[['retail_growth', 'sp500_return']].dropna()
            if len(test_data) > max_lag * 2:
                print(f"\nTesting: Does retail_growth Granger-cause sp500_return?")
                result = grangercausalitytests(test_data, max_lag, verbose=False)
                p_values = {}
                for lag in range(1, max_lag + 1):
                    if lag in result:
                        p_val = result[lag][0]['ssr_ftest'][1]
                        p_values[f'lag_{lag}'] = p_val
                        print(f"  Lag {lag}: p-value = {p_val:.4f}")
                results['retail_to_sp500'] = p_values
        except Exception as e:
            print(f"  ⚠ Error testing retail→sp500: {e}")
    
    return results


def prepare_training_data(df, target='retail_growth'):
    """
    Prepare features and target for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target variable name
        
    Returns:
        tuple: X (features), y (target), feature_names
    """
    print("\n" + "="*60)
    print(f"Preparing Training Data (target: {target})...")
    print("="*60)
    
    # Select features (exclude date and target)
    exclude_cols = ['date', target]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"  Selected features: {len(numeric_cols)}")
    print(f"  Features: {', '.join(numeric_cols)}")
    
    # Extract X and y
    X = df[numeric_cols].copy()
    y = df[target].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y, numeric_cols


def train_linear_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Linear Regression model.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        dict: Model results
    """
    print("\n" + "="*60)
    print("Training Linear Regression Model...")
    print("="*60)
    
    try:
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✓ Model trained successfully")
        print(f"\nTraining Metrics:")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nTest Metrics:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'y_train': y_train,
            'model_type': 'linear_regression'
        }
        
    except Exception as e:
        print(f"⚠ Error training model: {e}")
        return {}


def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100):
    """
    Train and evaluate a Random Forest model.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        n_estimators (int): Number of trees
        
    Returns:
        dict: Model results
    """
    print("\n" + "="*60)
    print("Training Random Forest Model...")
    print("="*60)
    
    try:
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✓ Model trained successfully")
        print(f"\nTraining Metrics:")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nTest Metrics:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # Feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Feature Importances:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'y_train': y_train,
            'feature_importance': feature_importance,
            'model_type': 'random_forest'
        }
        
    except Exception as e:
        print(f"⚠ Error training model: {e}")
        return {}


def train_prophet(df, target='retail_growth', periods=12):
    """
    Train and evaluate Prophet model for time series forecasting.
    
    Args:
        df (pd.DataFrame): Input dataframe with date column
        target (str): Target variable name
        periods (int): Number of periods to forecast
        
    Returns:
        dict: Prophet model results
    """
    print("\n" + "="*60)
    print("Training Prophet Model...")
    print("="*60)
    
    if Prophet is None:
        print("⚠ Prophet not installed. Skipping.")
        return {}
    
    try:
        # Prepare data for Prophet
        prophet_df = df[['date', target]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        prophet_df = prophet_df.dropna()
        
        # Split train/test (use last periods as test)
        train_size = int(len(prophet_df) * 0.8)
        prophet_train = prophet_df.iloc[:train_size].copy()
        prophet_test = prophet_df.iloc[train_size:].copy()
        
        print(f"  Training samples: {len(prophet_train)}")
        print(f"  Test samples: {len(prophet_test)}")
        
        # Train Prophet
        model = Prophet()
        model.fit(prophet_train)
        
        # Predict on train
        train_forecast = model.predict(prophet_train[['ds']])
        y_pred_train = train_forecast['yhat'].values
        y_train = prophet_train['y'].values
        
        # Predict on test
        test_forecast = model.predict(prophet_test[['ds']])
        y_pred_test = test_forecast['yhat'].values
        y_test = prophet_test['y'].values
        
        # Forecast future periods
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✓ Model trained successfully")
        print(f"\nTraining Metrics:")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nTest Metrics:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'y_train': y_train,
            'forecast': forecast,
            'model_type': 'prophet'
        }
        
    except Exception as e:
        print(f"⚠ Error training Prophet model: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_results(all_results, output_path='models/results_summary.csv'):
    """
    Save model evaluation results to CSV.
    
    Args:
        all_results (dict): Dictionary of all model results
        output_path (str): Path to save results
        
    Returns:
        str: Path to saved file
    """
    print("\n" + "="*60)
    print("Saving Results...")
    print("="*60)
    
    try:
        # Prepare results dataframe
        results_list = []
        for model_name, results in all_results.items():
            if results and 'model_type' in results:
                results_list.append({
                    'model': model_name,
                    'model_type': results['model_type'],
                    'train_r2': results.get('train_r2', np.nan),
                    'test_r2': results.get('test_r2', np.nan),
                    'train_mae': results.get('train_mae', np.nan),
                    'test_mae': results.get('test_mae', np.nan),
                    'train_rmse': results.get('train_rmse', np.nan),
                    'test_rmse': results.get('test_rmse', np.nan)
                })
        
        results_df = pd.DataFrame(results_list)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        results_df.to_csv(output_path, index=False)
        print(f"✓ Saved results to {output_path}")
        print("\nResults Summary:")
        print(results_df.to_string(index=False))
        
        return output_path
        
    except Exception as e:
        print(f"⚠ Error saving results: {e}")
        return None


def save_models(all_results, models_dir='models'):
    """
    Save trained models using joblib.
    
    Args:
        all_results (dict): Dictionary of all model results
        models_dir (str): Directory to save models
    """
    print("\n" + "="*60)
    print("Saving Trained Models...")
    print("="*60)
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, results in all_results.items():
            if results and 'model' in results:
                model_path = os.path.join(models_dir, f'{model_name}.joblib')
                joblib.dump(results['model'], model_path)
                print(f"✓ Saved {model_name} to {model_path}")
        
    except Exception as e:
        print(f"⚠ Error saving models: {e}")


def create_visualizations(all_results, output_dir='visuals'):
    """
    Create visualization plots for model results.
    
    Args:
        all_results (dict): Dictionary of all model results
        output_dir (str): Directory to save visualizations
    """
    print("\n" + "="*60)
    print("Creating Visualizations...")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Actual vs. Predicted plots (matplotlib)
    print("\n1. Creating Actual vs. Predicted plots...")
    for model_name, results in all_results.items():
        if results and 'y_test' in results and 'y_pred_test' in results:
            plt.figure(figsize=(10, 6))
            plt.scatter(results['y_test'], results['y_pred_test'], alpha=0.6)
            plt.plot([results['y_test'].min(), results['y_test'].max()],
                    [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{model_name} - Actual vs. Predicted')
            plt.grid(True, alpha=0.3)
            
            # Add R² to plot
            r2 = results.get('test_r2', 0)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            output_path = os.path.join(output_dir, f'{model_name}_actual_vs_predicted.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {output_path}")
    
    # 2. Feature importances (Random Forest only)
    print("\n2. Creating Feature Importance plot...")
    for model_name, results in all_results.items():
        if results and 'feature_importance' in results:
            importance_df = results['feature_importance']
            
            plt.figure(figsize=(10, 6))
            top_features = importance_df.head(10)
            plt.barh(range(len(top_features)), top_features['importance'].values)
            plt.yticks(range(len(top_features)), top_features['feature'].values)
            plt.xlabel('Importance')
            plt.title(f'{model_name} - Feature Importances (Top 10)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'{model_name}_feature_importances.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {output_path}")
    
    # 3. Prophet forecast (plotly)
    print("\n3. Creating Prophet forecast plot...")
    for model_name, results in all_results.items():
        if results and 'forecast' in results and 'model' in results:
            forecast = results['forecast']
            
            fig = go.Figure()
            
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Predicted',
                line=dict(color='blue', width=2)
            ))
            
            # Add uncertainty bands
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    name='Upper bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    name='Uncertainty',
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(width=0)
                ))
            
            fig.update_layout(
                title=f'{model_name} - Forecast',
                xaxis_title='Date',
                yaxis_title='Target Value',
                hovermode='x unified',
                width=1200,
                height=600
            )
            
            output_path = os.path.join(output_dir, f'{model_name}_forecast.html')
            fig.write_html(output_path)
            print(f"  ✓ Saved {output_path}")


def main():
    """Main function to run all modeling tasks."""
    print("\n" + "="*60)
    print("RETAIL MARKET DYNAMICS - MODELING ANALYSIS")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Set up paths
    base_path = os.getcwd()
    if '/content' in base_path:
        data_processed = '/content/retail_market_dynamics/data/processed'
        models_dir = '/content/retail_market_dynamics/models'
        visuals_dir = '/content/retail_market_dynamics/visuals'
    else:
        data_processed = 'data/processed'
        models_dir = 'models'
        visuals_dir = 'visuals'
    
    print(f"\nWorking directory: {base_path}")
    print(f"Processed data directory: {data_processed}")
    print(f"Models directory: {models_dir}")
    print(f"Visuals directory: {visuals_dir}")
    
    # 1. Load features
    features_path = os.path.join(data_processed, 'features.csv')
    df = load_features(file_path=features_path)
    
    if df.empty:
        print("⚠ No data to process. Exiting.")
        return
    
    # 2. Run correlation tests
    correlation_results = run_correlation_tests(df)
    
    # 3. Run Granger causality tests
    granger_results = run_granger_causality_tests(df)
    
    # 4. Prepare training data
    X, y, feature_names = prepare_training_data(df, target='retail_growth')
    
    # 5. Train models
    all_results = {}
    
    if len(X) > 0 and len(y) > 0:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Train Linear Regression
        linear_results = train_linear_regression(X_train, X_test, y_train, y_test)
        if linear_results:
            all_results['linear_regression'] = linear_results
        
        # Train Random Forest
        rf_results = train_random_forest(X_train, X_test, y_train, y_test)
        if rf_results:
            all_results['random_forest'] = rf_results
        
        # Train Prophet
        prophet_results = train_prophet(df, target='retail_growth')
        if prophet_results:
            all_results['prophet'] = prophet_results
        
        # 6. Save results
        results_path = os.path.join(models_dir, 'results_summary.csv')
        save_results(all_results, output_path=results_path)
        
        # 7. Save models
        save_models(all_results, models_dir=models_dir)
        
        # 8. Create visualizations
        create_visualizations(all_results, output_dir=visuals_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("MODELING SUMMARY")
    print("="*60)
    print(f"✓ Completed at: {datetime.now()}")
    print(f"✓ Models trained: {len(all_results)}")
    print(f"✓ Results saved to: {os.path.join(models_dir, 'results_summary.csv')}")
    print(f"✓ Models saved to: {models_dir}")
    print(f"✓ Visualizations saved to: {visuals_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

