---
trigger: always_on
---

# AGENTS.md — Demand Forecasting + Deployed Dashboard

## Project Goal
Build a demand forecasting system that benchmarks multiple models (naive, SARIMA,
Prophet, LSTM) on a real retail time series, backtests them properly, and serves
the best model through a FastAPI endpoint with a Streamlit dashboard.

This project exists to demonstrate: time-series EDA, classical + deep learning
forecasting, proper backtesting, model evaluation, and deployment. Do not add
scope beyond this list.

## Dataset
- Source: Kaggle "Rossmann Store Sales" (rossmann-store-sales)
- Files needed: train.csv, store.csv
- Filter to a single store ID (pick one with minimal missing data and no long
  closure gaps) to keep the problem tractable — do not model all stores at once.
- Granularity: daily
- Forecast horizon: 30 days ahead

## Fixed Decisions (do not re-derive or change these)
- Train/validation split: expanding-window backtest, minimum 3 folds, each fold
  validated on the next 30 days after its training cutoff
- Metrics: MAE, RMSE, MAPE — computed per fold, then averaged
- Random seed: 42 everywhere (numpy, tensorflow/torch, prophet)

## Build Order
1. **EDA** (`notebooks/eda.ipynb`)
   - Plot the raw series
   - Seasonal decomposition (trend/seasonal/residual)
   - ADF stationarity test, report the p-value in report.md
   - Note any holidays/promo effects visible in the Rossmann data (there's a
     `Promo` and `StateHoliday` column — flag these as candidate exogenous
     features but do not use them until step 2 baseline is done)

2. **Baseline** (`models/baseline.py`)
   - Naive forecast (last value) and 7-day moving average
   - This is the floor every other model must beat — log its metrics first

3. **Classical model** (`models/sarima_model.py`)
   - SARIMA via statsmodels, weekly seasonality (m=7)

4. **Prophet** (`models/prophet_model.py`)
   - Include `Promo` as an added regressor if step 1 flagged it as useful

5. **LSTM** (`models/lstm_model.py`)
   - Keep it small: 1–2 LSTM layers, don't over-engineer
   - Use a sliding window of past N days (start with N=14) to predict next day,
     roll forward for the 30-day horizon

6. **Evaluation** (`evaluation/metrics.py`)
   - Single results table: model x fold x (MAE, RMSE, MAPE)
   - Write the averaged comparison to report.md as both a table and one chart

7. **API** (`api/app.py`)
   - FastAPI, single endpoint: POST /forecast, takes a start date, returns 30-day
     forecast from the best model per the evaluation table
   - No auth needed

8. **Dashboard** (`dashboard.py`)
   - Streamlit: actual vs. predicted plot per model, metrics comparison table,
     dropdown to switch between models

## Guardrails
- Do not model all Rossmann stores at once — single store only, keep it explainable
- Do not skip the naive baseline
- Do not use test-period data (last 30 days) anywhere during training/tuning
- Every metric that goes in report.md needs a one-sentence plain-English
  explanation next to it — this will be explained live in interviews
- Stop at the scope above. No hyperparameter tuning frameworks, no ensembling,
  no additional models — this is a portfolio project, not a Kaggle leaderboard
  attempt

## Deliverables
- `report.md` — EDA findings, model comparison table + chart, 3-5 sentence
  conclusion on which model won and why
- Working `api/app.py` and `dashboard.py`, runnable with a single command each
- Clean README with setup instructions