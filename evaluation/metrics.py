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

def generate_evaluation_report():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    filepath = os.path.join(os.path.dirname(__file__), 'results.csv')
    if not os.path.exists(filepath):
        print("Results file not found.")
        return
        
    df = pd.read_csv(filepath)
    avg_df = df.groupby('Model').mean().reset_index()
    avg_df = avg_df.sort_values('MAPE')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_df, x='Model', y='MAPE', palette='viridis')
    plt.title('Model Comparison: Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Model')
    
    chart_path = os.path.join(os.path.dirname(__file__), 'comparison_chart.png')
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")

if __name__ == '__main__':
    generate_evaluation_report()
