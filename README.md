# Retail Demand Forecasting & Dashboard

This project demonstrates a complete end-to-end time-series forecasting pipeline applied to real retail data (Kaggle Rossmann Store Sales). It benchmarks multiple modeling approaches, properly evaluates them using an expanding-window backtest, and deploys the winning model via a FastAPI backend and an interactive Streamlit dashboard.

## Project Architecture
- `notebooks/eda.ipynb`: Exploratory Data Analysis, trend/seasonality decomposition, and stationarity testing.
- `evaluation/metrics.py`: Contains the 3-fold expanding-window backtest logic and evaluation metrics (MAE, RMSE, MAPE).
- `models/`: Implementations of various forecasting models:
  - `baseline.py`: Naive and 7-Day Moving Average models.
  - `sarima_model.py`: Classical statistical SARIMA model with weekly seasonality.
  - `prophet_model.py`: Meta's Prophet model integrating exogenous regressors (Promotions).
  - `lstm_model.py`: Deep learning LSTM using a sliding window.
- `api/app.py`: FastAPI application that serves the winning model (Prophet).
- `dashboard.py`: Streamlit frontend for viewing benchmark metrics and live interactive forecasts.
- `report.md`: Detailed breakdown of EDA findings, model comparisons, and the final conclusion.

## Setup Instructions

1. **Clone the repository and navigate to the directory.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install plotly
   ```
3. **Data Requirements**: Ensure the Kaggle Rossmann `train.csv`, `store.csv`, and `test.csv` files are located in the `data/` folder.

## Running the Application

### 1. Start the FastAPI Backend
The API serves the Prophet model. It trains automatically on startup and exposes a `/forecast` endpoint.
```bash
uvicorn api.app:app --reload
```
*The API will run locally on `http://127.0.0.1:8000`.*

### 2. Launch the Streamlit Dashboard
Open a **new, separate terminal window** and launch the dashboard frontend:
```bash
streamlit run dashboard.py
```
*This will open the interactive dashboard in your browser where you can view the benchmark metrics and interact with the Prophet forecast.*

## Reproducing Model Backtesting (Optional)
If you wish to reproduce the model backtesting and generate the comparison metrics yourself, run the models in order:
```bash
python models/baseline.py
python models/sarima_model.py
python models/prophet_model.py
python models/lstm_model.py
python evaluation/metrics.py
```
*This will generate the `evaluation/results.csv` table and the `evaluation/comparison_chart.png` chart.*
