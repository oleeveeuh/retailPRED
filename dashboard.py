#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for Retail Market Dynamics Project
Interactive visualization of retail and market data with model forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os

# Page config
st.set_page_config(
    page_title="Retail Market Dynamics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_features_data(file_path='data/processed/features.csv'):
    """
    Load the features dataset.
    
    Args:
        file_path (str): Path to features.csv
        
    Returns:
        pd.DataFrame: Features dataset
    """
    try:
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.warning(f"File not found: {file_path}. Creating sample data.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()


def create_sample_data():
    """Create sample data for demonstration."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    return pd.DataFrame({
        'date': dates,
        'retail_sales': np.random.randint(500000, 1000000, len(dates)),
        'retail_growth': np.random.randn(len(dates)) * 2,
        'sp500_close': 300 + np.cumsum(np.random.randn(len(dates)) * 5),
        'sp500_return': np.random.randn(len(dates)) * 2,
        'cpi': 230 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'consumer_confidence': 90 + np.random.randn(len(dates)) * 5
    })


@st.cache_data(ttl=3600)
def load_model(model_name='prophet', model_dir='models'):
    """
    Load a trained model.
    
    Args:
        model_name (str): Name of the model
        model_dir (str): Directory containing models
        
    Returns:
        Model object or None
    """
    try:
        model_path = os.path.join(model_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def plot_retail_vs_sp500(df):
    """Create line chart comparing retail sales and SP500 over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Retail sales line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['retail_sales'],
            name='Retail Sales',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>%{x}</b><br>Retail Sales: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # SP500 line
    if 'sp500_close' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['sp500_close'],
                name='S&P 500',
                line=dict(color='#F24236', width=2),
                hovertemplate='<b>%{x}</b><br>S&P 500: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Retail Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="S&P 500 Close", secondary_y=True)
    
    fig.update_layout(
        title={
            'text': 'Retail Sales vs S&P 500 Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_retail_vs_sp500_scatter(df):
    """Create scatter plot comparing retail growth vs SP500 return."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['sp500_return'],
            y=df['retail_growth'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['date'].dt.year,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Year")
            ),
            text=df['date'].dt.strftime('%Y-%m'),
            hovertemplate='<b>%{text}</b><br>SP500 Return: %{x:.2f}%<br>Retail Growth: %{y:.2f}%<extra></extra>',
            name='Data Points'
        )
    )
    
    # Add trendline
    z = np.polyfit(df['sp500_return'].dropna(), df['retail_growth'].dropna(), 1)
    p = np.poly1d(z)
    
    fig.add_trace(
        go.Scatter(
            x=df['sp500_return'].dropna(),
            y=p(df['sp500_return'].dropna()),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Retail Growth vs S&P 500 Return',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='S&P 500 Return (%)',
        yaxis_title='Retail Growth (%)',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_correlation_heatmap(df):
    """Create correlation heatmap."""
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        st.warning("Not enough numerical columns for correlation analysis.")
        return None
    
    corr_matrix = df[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Feature Correlation Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=800,
        height=800,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig


def plot_prophet_forecast(df, model=None):
    """Plot Prophet forecast with confidence intervals."""
    if model is None:
        st.warning("No Prophet model available.")
        return None
    
    try:
        # Prepare data for Prophet
        prophet_df = df[['date', 'retail_growth']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Create future dataframe (12 months ahead)
        future = model.make_future_dataframe(periods=12, freq='M')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            name='Historical Data',
            mode='markers',
            marker=dict(color='blue', size=5)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title={
                'text': 'Prophet Forecast with Confidence Intervals',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Date',
            yaxis_title='Retail Growth (%)',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Prophet forecast: {e}")
        return None


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<p class="main-header">📊 Retail Market Dynamics Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.selectbox(
            "Select data file",
            ["features.csv", "combined.csv"],
            index=0
        )
        
        # Date range selector
        st.subheader("📅 Date Range")
        min_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        max_date = st.date_input("End Date", value=datetime.now())
        
        # Ticker selector
        st.subheader("📈 Stock Ticker")
        ticker = st.selectbox(
            "Select ticker",
            ["SPY", "AMZN", "WMT", "XRT"],
            index=0
        )
        
        # Model selection
        st.subheader("🤖 Model Forecast")
        show_forecast = st.checkbox("Show Prophet forecast", value=False)
    
    # Load data
    data_path = f'data/processed/{data_source}'
    
    with st.spinner("Loading data..."):
        df = load_features_data(data_path)
    
    if df.empty:
        st.error("No data available. Please check the data files.")
        return
    
    # Filter by date
    if 'date' in df.columns:
        df_filtered = df[
            (df['date'] >= pd.to_datetime(min_date)) &
            (df['date'] <= pd.to_datetime(max_date))
        ].copy()
    else:
        df_filtered = df.copy()
    
    # Key metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'retail_sales' in df_filtered.columns:
            avg_sales = df_filtered['retail_sales'].mean()
            st.metric("Avg Retail Sales", f"${avg_sales:,.0f}")
    
    with col2:
        if 'retail_growth' in df_filtered.columns:
            avg_growth = df_filtered['retail_growth'].mean()
            st.metric("Avg Retail Growth", f"{avg_growth:.2f}%")
    
    with col3:
        if 'sp500_close' in df_filtered.columns:
            avg_sp500 = df_filtered['sp500_close'].mean()
            st.metric("Avg S&P 500 Close", f"${avg_sp500:.2f}")
    
    with col4:
        if 'sp500_return' in df_filtered.columns:
            avg_return = df_filtered['sp500_return'].mean()
            st.metric("Avg S&P 500 Return", f"{avg_return:.2f}%")
    
    st.divider()
    
    # Visualization section
    st.header("📈 Visualizations")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Scatter Plot", "Correlations", "Forecasts"])
    
    with tab1:
        st.subheader("Retail Sales vs S&P 500 Over Time")
        fig = plot_retail_vs_sp500(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        with st.expander("📊 Summary Statistics"):
            st.dataframe(df_filtered[['retail_sales', 'sp500_close']].describe())
    
    with tab2:
        st.subheader("Retail Growth vs S&P 500 Return Scatter")
        fig = plot_retail_vs_sp500_scatter(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        if 'retail_growth' in df_filtered.columns and 'sp500_return' in df_filtered.columns:
            correlation = df_filtered['retail_growth'].corr(df_filtered['sp500_return'])
            st.metric("Correlation", f"{correlation:.3f}")
    
    with tab3:
        st.subheader("Feature Correlation Matrix")
        fig = plot_correlation_heatmap(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Model Forecasts")
        
        if show_forecast:
            # Load Prophet model
            prophet_model = load_model('prophet', 'models')
            
            if prophet_model:
                fig = plot_prophet_forecast(df_filtered, prophet_model)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("To see forecasts, please run the modeling script first to train the Prophet model.")
        
        # Display model performance if available
        try:
            results_df = pd.read_csv('models/results_summary.csv')
            st.subheader("📊 Model Performance Summary")
            st.dataframe(results_df, use_container_width=True)
        except FileNotFoundError:
            st.info("Model results not available. Run the modeling script to generate results.")
    
    # Data table
    st.divider()
    st.header("📋 Data Table")
    
    with st.expander("View processed data"):
        st.dataframe(df_filtered, use_container_width=True)
        
        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"retail_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📊 Retail Market Dynamics Dashboard | Built with Streamlit</p>
        <p>Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

