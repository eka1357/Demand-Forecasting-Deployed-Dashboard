import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import os
import sys

# Ensure we can import our models
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.baseline import prepare_data, naive_forecast, moving_average_forecast
from models.sarima_model import sarima_forecast
from models.lstm_model import lstm_forecast

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.title("📈 Retail Demand Forecasting (Store 85)")

# 1. Metrics Comparison Table
st.markdown("### 🏆 Benchmark Metrics")
results_path = 'evaluation/results.csv'
if os.path.exists(results_path):
    df_res = pd.read_csv(results_path)
    avg_res = df_res.groupby('Model').mean().reset_index().sort_values('MAPE')
    
    # Format for display
    avg_res['MAE'] = avg_res['MAE'].round(2)
    avg_res['RMSE'] = avg_res['RMSE'].round(2)
    avg_res['MAPE'] = avg_res['MAPE'].round(2).astype(str) + '%'
    
    st.dataframe(avg_res, use_container_width=True)
else:
    st.warning("No evaluation results found. Run the backtesting scripts first.")

st.divider()

# 2. Actual vs Predicted Plot
st.markdown("### 🔮 Forecast Viewer")

@st.cache_data
def get_data():
    df = prepare_data(85)
    # We will use the last 30 days of our dataset to visualize "Actual vs Predicted"
    train = df.iloc[:-30]
    val = df.iloc[-30:]
    return train, val

train, val = get_data()
val_start_date = val.index.min().strftime('%Y-%m-%d')

selected_model = st.selectbox("Select Model to Visualize:", 
                              ["Prophet (via API)", "SARIMA", "LSTM", "7-Day MA", "Naive"])

with st.spinner(f"Generating forecast using {selected_model}..."):
    predictions = None
    
    if selected_model == "Prophet (via API)":
        # Hit the FastAPI endpoint for the winning model
        try:
            res = requests.post("http://127.0.0.1:8000/forecast", json={"start_date": val_start_date})
            if res.status_code == 200:
                data = res.json()
                predictions = [item['predicted_sales'] for item in data['forecast']]
            else:
                st.error(f"API Error: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API. Make sure FastAPI is running on http://127.0.0.1:8000")
            
    elif selected_model == "SARIMA":
        predictions = sarima_forecast(train, horizon=30)
        
    elif selected_model == "LSTM":
        predictions = lstm_forecast(train, horizon=30, window_size=14)
        
    elif selected_model == "7-Day MA":
        predictions = moving_average_forecast(train, window=7, horizon=30)
        
    elif selected_model == "Naive":
        predictions = naive_forecast(train, horizon=30)

if predictions is not None:
    fig = go.Figure()
    
    # Show 60 days of history before the validation period for context
    history_to_show = 60
    plot_df = pd.concat([train.iloc[-history_to_show:], val])
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Sales'], 
                             mode='lines', name='Actual Sales', line=dict(color='#1f77b4')))
                             
    fig.add_trace(go.Scatter(x=val.index, y=predictions, 
                             mode='lines+markers', name=f'{selected_model} Forecast', 
                             line=dict(color='#ff7f0e', dash='dash')))
                             
    fig.update_layout(title=f"30-Day Forecast vs Actuals (Starting {val_start_date})",
                      xaxis_title="Date",
                      yaxis_title="Sales",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                      
    st.plotly_chart(fig, use_container_width=True)
