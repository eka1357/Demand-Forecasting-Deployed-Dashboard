# Demand Forecasting Report

## 1. Exploratory Data Analysis (EDA) Findings
- **Store Selection**: Store 85 was selected due to having no missing sales data and being open consistently (100% open ratio over 942 days).
- **Trend & Seasonality**: The seasonal decomposition revealed a clear weekly seasonality pattern in the sales data.
- **Stationarity**: The ADF test indicated that the sales series is stationary (p-value < 0.05).
- **Exogenous Effects**: Promotions significantly increase average sales, while State Holidays drastically reduce sales (often to zero, as the store is closed).

## 2. Model Comparison

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 2242.68 | 2858.83 | 31.11% |
| 7-Day MA | 2067.67 | 2479.21 | 29.21% |
| SARIMA | 1038.70 | 1373.73 | 14.64% |
| Prophet | 953.26 | 1269.95 | 13.19% |
*(To be completed: LSTM models and a single chart comparing performance across folds)*

## 3. Conclusion
*(To be completed: 3-5 sentence summary of which model won and why)*
