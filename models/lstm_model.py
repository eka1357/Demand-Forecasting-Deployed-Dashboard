"""
LSTM model for demand forecasting.

Purpose: Implement a small (1-2 layer) LSTM model.
Uses a sliding window of past N days (e.g., N=14) to predict the next day,
rolling forward for the 30-day horizon.
"""
