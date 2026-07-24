import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# 1. Load Data
print("Loading data...")
train = pd.read_csv("data/train.csv")
store = pd.read_csv("data/store.csv")

# 2. Merge Data
print("Merging data...")
df = train.merge(store, on="Store", how="left")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# 3. Find a store with minimal missing data and no long closure gaps
# Let's count open days per store
store_stats = df.groupby("Store").agg(
    total_days=("Date", "count"),
    open_days=("Open", "sum"),
    sales_nulls=("Sales", lambda x: x.isnull().sum())
)
# We want a store that is open most days (so no long closures)
# Rossmann train data goes from 2013-01-01 to 2015-07-31 (~942 days)
# Some stores are closed on Sundays, so ~784 days open is normal.
store_stats["open_ratio"] = store_stats["open_days"] / store_stats["total_days"]

# Let's pick a store that is consistently open (maybe even Sundays?) 
# or just has 0 nulls and max total_days
best_stores = store_stats[(store_stats["sales_nulls"] == 0) & (store_stats["total_days"] > 940)].sort_values(by="open_ratio", ascending=False)
best_store_id = best_stores.index[0]
print(f"Best Store ID: {best_store_id}")
print(f"Stats for Store {best_store_id}: \n{best_stores.loc[best_store_id]}")

# 4. Filter to best store
store_df = df[df["Store"] == best_store_id].copy()
store_df = store_df.sort_values("Date").reset_index(drop=True)
store_df.set_index("Date", inplace=True)

# 5. ADF test
# We test stationarity on Sales. Wait, some days are closed (Sales = 0). We should probably include them as 0 or maybe interpolate.
# But let's just run ADF on the raw sales series first.
sales_series = store_df["Sales"]
print("Running ADF Test on raw sales series...")
adf_result = adfuller(sales_series)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# Check Promo and StateHoliday impact on average sales
print("\nPromo Impact:")
print(store_df.groupby("Promo")["Sales"].mean())

print("\nStateHoliday Impact:")
# Convert StateHoliday to string because it might be mixed types ('0', 0, 'a', 'b', 'c')
store_df["StateHoliday"] = store_df["StateHoliday"].astype(str)
print(store_df.groupby("StateHoliday")["Sales"].mean())
