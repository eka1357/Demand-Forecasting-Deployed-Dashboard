import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def main():
    with open('eda_results.txt', 'w') as f:
        # Load data
        train = pd.read_csv('data/train.csv', low_memory=False)
        store = pd.read_csv('data/store.csv', low_memory=False)
        df = train.merge(store, on='Store', how='left')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Find store
        store_stats = df.groupby('Store').agg(
            total_days=('Date', 'count'),
            open_days=('Open', 'sum'),
            sales_nulls=('Sales', lambda x: x.isnull().sum())
        )
        store_stats['open_ratio'] = store_stats['open_days'] / store_stats['total_days']
        best_stores = store_stats[(store_stats['sales_nulls'] == 0) & (store_stats['total_days'] >= 942)].sort_values(by='open_ratio', ascending=False)
        best_store_id = best_stores.index[0]
        
        f.write(f"Selected Store ID: {best_store_id}\n")
        
        # Filter
        store_df = df[df['Store'] == best_store_id].copy()
        store_df = store_df.sort_values('Date').reset_index(drop=True)
        store_df.set_index('Date', inplace=True)
        
        # ADF Test
        adf_result = adfuller(store_df['Sales'])
        f.write(f"ADF Statistic: {adf_result[0]:.4f}\n")
        f.write(f"p-value: {adf_result[1]:.4e}\n")
        
        # Promo and Holiday
        f.write(f"\nAverage Sales by Promo:\n{store_df.groupby('Promo')['Sales'].mean()}\n")
        store_df['StateHoliday'] = store_df['StateHoliday'].astype(str)
        f.write(f"\nAverage Sales by StateHoliday:\n{store_df.groupby('StateHoliday')['Sales'].mean()}\n")

if __name__ == '__main__':
    main()
