import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Set seed for reproducibility as per specs
tf.random.set_seed(42)
np.random.seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import get_backtest_splits, compute_metrics, log_results
from models.baseline import prepare_data

def create_dataset(data, window_size=14):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

def lstm_forecast(train, horizon=30, window_size=14):
    # Scale data for neural network
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train[['Sales']].values)
    
    X_train, y_train = create_dataset(scaled_train, window_size)
    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build simple 1-layer LSTM
    model = Sequential()
    # Using relu for simplicity, though tanh is default. 
    # With scaled [0,1] data, either is fine. We'll stick to defaults where possible to "keep it simple"
    model.add(LSTM(32, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit model (keeping epochs low for fast evaluation)
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    
    # Predict recursively for the horizon
    last_window = scaled_train[-window_size:]
    forecasts_scaled = []
    
    current_window = last_window.copy()
    for _ in range(horizon):
        # Reshape to [1, window_size, 1]
        x_input = np.reshape(current_window, (1, window_size, 1))
        pred = model.predict(x_input, verbose=0)[0][0]
        forecasts_scaled.append(pred)
        # Update window by sliding forward 1 day
        current_window = np.append(current_window[1:], [[pred]], axis=0)
        
    # Inverse transform
    forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1))
    return forecasts.flatten()

if __name__ == '__main__':
    df = prepare_data(85)
    splits = get_backtest_splits(df, horizon=30, n_splits=3)
    
    lstm_metrics = []
    
    print("\n--- Evaluating LSTM Model (Store 85) ---")
    for fold, (train_df, val_df) in enumerate(splits):
        print(f"Training Fold {fold+1} (Val Start: {val_df.index.min().date()})...")
        y_true = val_df['Sales'].values
        
        # LSTM Forecast
        pred_lstm = lstm_forecast(train_df, horizon=30, window_size=14)
        
        # Avoid negative predictions for sales
        pred_lstm = np.clip(pred_lstm, a_min=0, a_max=None)
        
        metrics = compute_metrics(y_true, pred_lstm)
        lstm_metrics.append(metrics)
        log_results('LSTM', fold+1, metrics['MAE'], metrics['RMSE'], metrics['MAPE'])
        
        print(f"  LSTM -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

    avg_lstm = pd.DataFrame(lstm_metrics).mean()
    
    print("\n=== Final Averaged Metrics (Over 3 Folds) ===")
    print("[LSTM Model]")
    print(f"MAE:  {avg_lstm['MAE']:.2f}")
    print(f"RMSE: {avg_lstm['RMSE']:.2f}")
    print(f"MAPE: {avg_lstm['MAPE']:.2f}%")
