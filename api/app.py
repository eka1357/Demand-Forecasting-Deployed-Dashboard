from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import logging
from prophet import Prophet
import os

# Suppress Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Global variables to hold data and model in memory
MODEL = None
PROMO_SCHEDULE = None

def load_and_train():
    global MODEL, PROMO_SCHEDULE
    print("Loading data and training Prophet model on Store 85 (this takes a few seconds)...")
    
    # Load train and store
    train = pd.read_csv('data/train.csv', low_memory=False)
    store = pd.read_csv('data/store.csv', low_memory=False)
    df = train.merge(store, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    
    store_df = df[df['Store'] == 85].copy()
    store_df = store_df.sort_values('Date').reset_index(drop=True)
    
    # Train Prophet model on all available historical data
    df_train = store_df[['Date', 'Sales', 'Promo']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    MODEL = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    MODEL.add_regressor('Promo')
    MODEL.fit(df_train)
    
    # Load test for future promo schedules
    if os.path.exists('data/test.csv'):
        test = pd.read_csv('data/test.csv', low_memory=False)
        test['Date'] = pd.to_datetime(test['Date'])
        test_store = test[test['Store'] == 85].copy()
        promo_test = test_store[['Date', 'Promo']]
    else:
        promo_test = pd.DataFrame(columns=['Date', 'Promo'])
    
    # Combine train and test promo schedules so we can look them up during forecasting
    promo_train = store_df[['Date', 'Promo']]
    PROMO_SCHEDULE = pd.concat([promo_train, promo_test]).drop_duplicates(subset=['Date']).set_index('Date')['Promo'].fillna(0)
    print("Model trained and API is ready to accept requests!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_and_train()
    yield
    # Shutdown
    pass

app = FastAPI(title="Demand Forecasting API", lifespan=lifespan)

class ForecastRequest(BaseModel):
    start_date: str  # YYYY-MM-DD

@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        start_date = pd.to_datetime(request.start_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
    if MODEL is None or PROMO_SCHEDULE is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
        
    # Generate next 30 days
    future_dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Lookup promo for these dates. Default to 0 if not found in our schedule.
    promos = []
    for d in future_dates:
        if d in PROMO_SCHEDULE.index:
            promos.append(PROMO_SCHEDULE.loc[d])
        else:
            promos.append(0)
            
    future_df = pd.DataFrame({
        'ds': future_dates,
        'Promo': promos
    })
    
    # Predict
    forecast_df = MODEL.predict(future_df)
    
    # Format output
    results = []
    for _, row in forecast_df.iterrows():
        pred_sales = max(0, row['yhat'])  # Clip negative predictions
        results.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_sales": round(pred_sales, 2)
        })
        
    return {"start_date": request.start_date, "horizon": 30, "forecast": results}
