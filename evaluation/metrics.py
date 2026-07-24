"""
Evaluation metrics and backtesting logic.

Purpose: Compute MAE, RMSE, and MAPE per fold, then average.
Creates a single results table: model x fold x metric.
Implements expanding-window backtest (min 3 folds, 30 days each).
"""
