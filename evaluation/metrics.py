import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    # Avoid division by zero by masking
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def get_backtest_splits(df, horizon=30, min_train_size=365, n_splits=3):
    """
    Generate expanding window backtest splits.
    Returns a list of (train_df, val_df) tuples.
    """
    df = df.sort_index()
    splits = []
    n = len(df)
    
    for i in range(n_splits, 0, -1):
        val_end = n - (i - 1) * horizon
        val_start = val_end - horizon
        train_end = val_start
        
        if train_end < min_train_size:
            continue
            
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        splits.append((train_df, val_df))
        
    return splits

def log_results(model_name, fold, mae, rmse, mape, filepath='evaluation/results.csv'):
    import os
    res_df = pd.DataFrame([[model_name, fold, mae, rmse, mape]], 
                          columns=['Model', 'Fold', 'MAE', 'RMSE', 'MAPE'])
    if not os.path.exists(filepath):
        res_df.to_csv(filepath, index=False)
    else:
        res_df.to_csv(filepath, mode='a', header=False, index=False)
