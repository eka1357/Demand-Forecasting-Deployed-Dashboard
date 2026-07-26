import pandas as pd
import numpy as np
import sys
import os
import warnings

# Ignore statsmodels warnings for backtesting
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import get_backtest_splits, compute_metrics, log_results
from models.baseline import prepare_data
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(train, horizon=30):
    # We use weekly seasonality (m=7) per the project requirements.
    # Enforce constraints set to False helps prevent fitting crashes in automated backtesting
    model = SARIMAX(train['Sales'], 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return forecast.values

if __name__ == '__main__':
    df = prepare_data(85)
    splits = get_backtest_splits(df, horizon=30, n_splits=3)
    
    sarima_metrics = []
    
    print("\n--- Evaluating SARIMA Model (Store 85) ---")
    for fold, (train_df, val_df) in enumerate(splits):
        print(f"Training Fold {fold+1} (Val Start: {val_df.index.min().date()})...")
        y_true = val_df['Sales'].values
        
        # SARIMA Forecast
        pred_sarima = sarima_forecast(train_df, horizon=30)
        metrics = compute_metrics(y_true, pred_sarima)
        sarima_metrics.append(metrics)
        log_results('SARIMA', fold+1, metrics['MAE'], metrics['RMSE'], metrics['MAPE'])
        
        print(f"  SARIMA -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

    avg_sarima = pd.DataFrame(sarima_metrics).mean()
    
    print("\n=== Final Averaged Metrics (Over 3 Folds) ===")
    print("[SARIMA Model]")
    print(f"MAE:  {avg_sarima['MAE']:.2f}")
    print(f"RMSE: {avg_sarima['RMSE']:.2f}")
    print(f"MAPE: {avg_sarima['MAPE']:.2f}%")
