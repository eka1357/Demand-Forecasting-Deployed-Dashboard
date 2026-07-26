import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import get_backtest_splits, compute_metrics, log_results

def prepare_data(store_id=85):
    print("Loading data...")
    train = pd.read_csv('data/train.csv', low_memory=False)
    store = pd.read_csv('data/store.csv', low_memory=False)
    df = train.merge(store, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    
    store_df = df[df['Store'] == store_id].copy()
    store_df = store_df.sort_values('Date').reset_index(drop=True)
    store_df.set_index('Date', inplace=True)
    
    # Ensure continuous date range
    full_idx = pd.date_range(start=store_df.index.min(), end=store_df.index.max(), freq='D')
    store_df = store_df.reindex(full_idx)
    # Fill Sales with 0 for closed days
    store_df['Sales'] = store_df['Sales'].fillna(0)
    
    return store_df

def naive_forecast(train, horizon=30):
    last_value = train['Sales'].iloc[-1]
    return np.full(horizon, last_value)

def moving_average_forecast(train, window=7, horizon=30):
    ma_value = train['Sales'].rolling(window=window).mean().iloc[-1]
    if pd.isna(ma_value):
        ma_value = train['Sales'].mean()
    return np.full(horizon, ma_value)

if __name__ == '__main__':
    df = prepare_data(85)
    splits = get_backtest_splits(df, horizon=30, n_splits=3)
    
    naive_metrics = []
    ma7_metrics = []
    
    print("\n--- Evaluating Baseline Models (Store 85) ---")
    for fold, (train_df, val_df) in enumerate(splits):
        y_true = val_df['Sales'].values
        
        # Naive Forecast
        pred_naive = naive_forecast(train_df, horizon=30)
        naive_met = compute_metrics(y_true, pred_naive)
        naive_metrics.append(naive_met)
        log_results('Naive', fold+1, naive_met['MAE'], naive_met['RMSE'], naive_met['MAPE'])
        
        # 7-Day Moving Average
        pred_ma7 = moving_average_forecast(train_df, window=7, horizon=30)
        ma7_met = compute_metrics(y_true, pred_ma7)
        ma7_metrics.append(ma7_met)
        log_results('7-Day MA', fold+1, ma7_met['MAE'], ma7_met['RMSE'], ma7_met['MAPE'])
        
        print(f"Fold {fold+1} (Val Start: {val_df.index.min().date()})")
        print(f"  Naive -> MAE: {naive_metrics[-1]['MAE']:.2f}, RMSE: {naive_metrics[-1]['RMSE']:.2f}, MAPE: {naive_metrics[-1]['MAPE']:.2f}%")
        print(f"  MA7   -> MAE: {ma7_metrics[-1]['MAE']:.2f}, RMSE: {ma7_metrics[-1]['RMSE']:.2f}, MAPE: {ma7_metrics[-1]['MAPE']:.2f}%")

    avg_naive = pd.DataFrame(naive_metrics).mean()
    avg_ma7 = pd.DataFrame(ma7_metrics).mean()
    
    print("\n=== Final Averaged Metrics (Over 3 Folds) ===")
    print("[Naive Model]")
    print(f"MAE:  {avg_naive['MAE']:.2f}")
    print(f"RMSE: {avg_naive['RMSE']:.2f}")
    print(f"MAPE: {avg_naive['MAPE']:.2f}%")
    
    print("\n[7-Day Moving Average Model]")
    print(f"MAE:  {avg_ma7['MAE']:.2f}")
    print(f"RMSE: {avg_ma7['RMSE']:.2f}")
    print(f"MAPE: {avg_ma7['MAPE']:.2f}%")
