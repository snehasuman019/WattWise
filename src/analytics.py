
import os
import pandas as pd

# Load the existing hourly consumption data
input_file = "data/processed/hourly_consumption.csv"
output_dir = "data/processed"

df = pd.read_csv(input_file)

# Convert hour_start to datetime
df["hour_start"] = pd.to_datetime(df["hour_start"])

# Daily consumption
df["date"] = df["hour_start"].dt.date

daily = (
    df.groupby("date", as_index=False)
    ["hourly_consumption_kwh"]
    .sum()
    .rename(columns={
        "hourly_consumption_kwh": "daily_consumption_kwh"
    })
)

daily["daily_consumption_kwh"] = (
    daily["daily_consumption_kwh"].round(3)
)

# Monthly consumption
df["month"] = df["hour_start"].dt.to_period("M").astype(str)

monthly = (
    df.groupby("month", as_index=False)
    ["hourly_consumption_kwh"]
    .sum()
    .rename(columns={
        "hourly_consumption_kwh": "monthly_consumption_kwh"
    })
)

monthly["monthly_consumption_kwh"] = (
    monthly["monthly_consumption_kwh"].round(3)
)

# Save results
os.makedirs(output_dir, exist_ok=True)

daily.to_csv(
    os.path.join(output_dir, "daily_consumption.csv"),
    index=False
)

monthly.to_csv(
    os.path.join(output_dir, "monthly_consumption.csv"),
    index=False
)

print("Daily and monthly analytics completed!")
print(f"Daily records: {len(daily):,}")
print(f"Monthly records: {len(monthly):,}")

print("\nDaily consumption sample:")
print(daily.head())

print("\nMonthly consumption sample:")
print(monthly.head())
