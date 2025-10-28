#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Engineering Script for Retail Market Dynamics Project
Creates derived features from combined dataset including lags, rolling averages, and normalized features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_combined_data(file_path='data/processed/combined.csv'):
    """
    Load the combined dataset.
    
    Args:
        file_path (str): Path to combined.csv
        
    Returns:
        pd.DataFrame: Combined dataset with date column as datetime
    """
    print("="*60)
    print("Loading Combined Data...")
    print("="*60)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {df.shape[0]} records from {file_path}")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"⚠ File not found: {file_path}")
        print("Creating sample data for demonstration...")
        
        # Create sample data
        dates = pd.date_range(start='2020-01-31', end=datetime.now().replace(day=1) - pd.offsets.Day(1), freq='M')
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


def create_lagged_features(df):
    """
    Create lagged features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with added lagged features
    """
    print("\n" + "="*60)
    print("Creating Lagged Features...")
    print("="*60)
    
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Define columns to lag
    columns_to_lag = ['retail_growth', 'cpi']
    
    for col in columns_to_lag:
        if col in df_features.columns:
            # 1-month lag
            df_features[f'{col}_lag1'] = df_features[col].shift(1)
            print(f"✓ Created {col}_lag1")
            
            # 3-month lag
            df_features[f'{col}_lag3'] = df_features[col].shift(3)
            print(f"✓ Created {col}_lag3")
        else:
            print(f"⚠ Column '{col}' not found, skipping...")
    
    return df_features


def create_rolling_averages(df):
    """
    Create rolling average features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with added rolling average features
    """
    print("\n" + "="*60)
    print("Creating Rolling Averages...")
    print("="*60)
    
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Define columns for rolling averages
    rolling_columns = ['retail_growth', 'sp500_return']
    
    for col in rolling_columns:
        if col in df_features.columns:
            # 3-month rolling average
            df_features[f'{col}_rolling3m'] = df_features[col].rolling(window=3, min_periods=1).mean()
            print(f"✓ Created {col}_rolling3m")
            
            # 6-month rolling average
            df_features[f'{col}_rolling6m'] = df_features[col].rolling(window=6, min_periods=1).mean()
            print(f"✓ Created {col}_rolling6m")
        else:
            print(f"⚠ Column '{col}' not found, skipping...")
    
    return df_features


def create_normalized_features(df, method='minmax'):
    """
    Normalize numerical features using Z-score or MinMax scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): 'zscore' or 'minmax'
        
    Returns:
        pd.DataFrame: Dataframe with added normalized features
        dict: Dictionary of scalers for each column
    """
    print("\n" + "="*60)
    print(f"Creating Normalized Features (method: {method})...")
    print("="*60)
    
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Define columns to normalize
    columns_to_normalize = ['retail_sales', 'retail_growth', 'sp500_close', 
                           'sp500_return', 'cpi', 'consumer_confidence']
    
    # Filter to columns that exist in dataframe
    columns_to_normalize = [col for col in columns_to_normalize if col in df_features.columns]
    
    scalers = {}
    
    for col in columns_to_normalize:
        # Skip if column has no variance
        if df_features[col].std() == 0:
            print(f"⚠ {col}: Zero variance, skipping normalization")
            continue
        
        try:
            if method == 'zscore':
                # Z-score normalization
                mean = df_features[col].mean()
                std = df_features[col].std()
                df_features[f'{col}_norm'] = (df_features[col] - mean) / std
                scalers[col] = {'method': 'zscore', 'mean': mean, 'std': std}
                print(f"✓ Created {col}_norm (Z-score)")
                
            elif method == 'minmax':
                # Min-Max normalization
                min_val = df_features[col].min()
                max_val = df_features[col].max()
                df_features[f'{col}_norm'] = (df_features[col] - min_val) / (max_val - min_val)
                scalers[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                print(f"✓ Created {col}_norm (MinMax)")
                
            else:
                print(f"⚠ Unknown method '{method}', skipping normalization")
                
        except Exception as e:
            print(f"⚠ Error normalizing {col}: {e}")
    
    return df_features, scalers


def create_correlation_matrix(df, output_path='visuals/correlation_heatmap.png'):
    """
    Create and save correlation heatmap of features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path to save the heatmap
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    print("\n" + "="*60)
    print("Creating Correlation Matrix...")
    print("="*60)
    
    try:
        # Select numerical columns (exclude date)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            print("⚠ No numerical columns found")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        print(f"✓ Correlation matrix computed for {len(numerical_cols)} features")
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True,
                   cbar_kws={"shrink": .5}, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save heatmap
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation heatmap to {output_path}")
        plt.close()
        
        # Print top correlations
        print("\nTop Correlations:")
        # Flatten correlation matrix and exclude diagonal (self-correlation)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    corr_pairs.append((col1, col2, corr_val))
        
        # Sort by absolute correlation
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        print(f"\nHighest Correlations:")
        for col1, col2, corr in corr_pairs_sorted[:10]:
            print(f"  {col1} <-> {col2}: {corr:.3f}")
        
        return corr_matrix
        
    except Exception as e:
        print(f"⚠ Error creating correlation matrix: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def save_features(df, output_path='data/processed/features.csv'):
    """
    Save engineered features to CSV.
    
    Args:
        df (pd.DataFrame): Dataframe with all features
        output_path (str): Path to save features
        
    Returns:
        str: Path to saved file
    """
    print("\n" + "="*60)
    print("Saving Engineered Features...")
    print("="*60)
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved features to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"    - {col}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(df.describe())
        
        return output_path
        
    except Exception as e:
        print(f"⚠ Error saving features: {e}")
        return None


def main():
    """Main function to run all feature engineering tasks."""
    print("\n" + "="*60)
    print("RETAIL MARKET DYNAMICS - FEATURE ENGINEERING")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Set up paths
    base_path = os.getcwd()
    if '/content' in base_path:
        data_processed = '/content/retail_market_dynamics/data/processed'
        visuals_dir = '/content/retail_market_dynamics/visuals'
    else:
        data_processed = 'data/processed'
        visuals_dir = 'visuals'
    
    print(f"\nWorking directory: {base_path}")
    print(f"Processed data directory: {data_processed}")
    print(f"Visuals directory: {visuals_dir}")
    
    # 1. Load combined data
    combined_path = os.path.join(data_processed, 'combined.csv')
    df = load_combined_data(file_path=combined_path)
    
    if df.empty:
        print("⚠ No data to process. Exiting.")
        return
    
    print(f"\nOriginal data shape: {df.shape}")
    
    # 2. Create lagged features
    df_features = create_lagged_features(df)
    print(f"After lags: {df_features.shape}")
    
    # 3. Create rolling averages
    df_features = create_rolling_averages(df_features)
    print(f"After rolling: {df_features.shape}")
    
    # 4. Create normalized features (using minmax by default, can change to 'zscore')
    df_features, scalers = create_normalized_features(df_features, method='minmax')
    print(f"After normalization: {df_features.shape}")
    
    # 5. Create correlation matrix
    corr_matrix = create_correlation_matrix(
        df_features, 
        output_path=os.path.join(visuals_dir, 'correlation_heatmap.png')
    )
    
    # 6. Save final features
    features_path = os.path.join(data_processed, 'features.csv')
    save_features(df_features, output_path=features_path)
    
    # Final summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Engineered columns: {len(df_features.columns)}")
    print(f"New features added: {len(df_features.columns) - len(df.columns)}")
    print(f"\n✓ Features saved to: {features_path}")
    
    # Display sample of features
    print("\nSample of engineered features:")
    print(df_features.head())
    
    print(f"\n✓ Completed at: {datetime.now()}")
    print("="*60)


if __name__ == "__main__":
    main()

