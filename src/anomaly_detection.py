
import pandas as pd

# Load daily consumption data
file_path = "data/processed/daily_consumption.csv"
df = pd.read_csv(file_path)

# Calculate the IQR thresholds
q1 = df["daily_consumption_kwh"].quantile(0.25)
q3 = df["daily_consumption_kwh"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Flag unusual consumption
df["anomaly"] = (
    (df["daily_consumption_kwh"] < lower_bound)
    | (df["daily_consumption_kwh"] > upper_bound)
)

# Save the results
output_file = "data/processed/anomaly_detection.csv"
df.to_csv(output_file, index=False)

print("Anomaly detection completed!")
print(f"Lower threshold: {lower_bound:.2f} kWh")
print(f"Upper threshold: {upper_bound:.2f} kWh")
print(f"Unusual days detected: {df['anomaly'].sum()}")

print("\nSample unusual days:")
print(
    df[df["anomaly"]]
    .sort_values("daily_consumption_kwh", ascending=False)
    .head(10)
)
