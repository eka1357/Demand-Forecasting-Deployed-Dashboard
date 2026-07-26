import pandas as pd
import numpy as np
import sys
import os
import logging

# Suppress Prophet logging spam
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import get_backtest_splits, compute_metrics, log_results
from models.baseline import prepare_data
from prophet import Prophet

def prophet_forecast(train, val, horizon=30):
    # Prophet requires 'ds' (datestamp) and 'y' (target) columns.
    # We rename the axis to 'Date' just in case the reindexing removed the name
    train = train.rename_axis('Date')
    df_train = train.reset_index()[['Date', 'Sales', 'Promo']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Seed 42 is used globally in AGENTS.md, but Prophet doesn't have a direct random_state for standard fitting,
    # it's deterministic unless using MCMC sampling.
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    
    # EDA showed Promo is highly impactful, so we add it as a regressor.
    m.add_regressor('Promo')
    
    m.fit(df_train)
    
    # We must provide the future dates AND the future regressor values (Promo)
    val = val.rename_axis('Date')
    future = val.reset_index()[['Date', 'Promo']].rename(columns={'Date': 'ds'})
    
    forecast = m.predict(future)
    return forecast['yhat'].values

if __name__ == '__main__':
    df = prepare_data(85)
    splits = get_backtest_splits(df, horizon=30, n_splits=3)
    
    prophet_metrics = []
    
    print("\n--- Evaluating Prophet Model (Store 85) ---")
    for fold, (train_df, val_df) in enumerate(splits):
        print(f"Training Fold {fold+1} (Val Start: {val_df.index.min().date()})...")
        y_true = val_df['Sales'].values
        
        # Prophet Forecast
        pred_prophet = prophet_forecast(train_df, val_df, horizon=30)
        
        # Clip to 0 since sales cannot be negative
        pred_prophet = np.clip(pred_prophet, a_min=0, a_max=None)
        
        metrics = compute_metrics(y_true, pred_prophet)
        prophet_metrics.append(metrics)
        log_results('Prophet', fold+1, metrics['MAE'], metrics['RMSE'], metrics['MAPE'])
        
        print(f"  Prophet -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

    avg_prophet = pd.DataFrame(prophet_metrics).mean()
    
    print("\n=== Final Averaged Metrics (Over 3 Folds) ===")
    print("[Prophet Model]")
    print(f"MAE:  {avg_prophet['MAE']:.2f}")
    print(f"RMSE: {avg_prophet['RMSE']:.2f}")
    print(f"MAPE: {avg_prophet['MAPE']:.2f}%")
