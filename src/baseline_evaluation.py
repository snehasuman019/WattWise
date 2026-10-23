
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the saved test predictions
path = "data/processed/forecast_predictions.csv"
df = pd.read_csv(path)

df["hour_start"] = pd.to_datetime(df["hour_start"])
df = df.sort_values("hour_start").reset_index(drop=True)

# Actual and Random Forest predictions
actual = df["label"]
rf_predictions = df["prediction"]

# Baseline: predict the previous hour's actual consumption
baseline_predictions = actual.shift(1)

# Remove the first row because it has no previous test hour
valid = baseline_predictions.notna()

actual = actual[valid]
rf_predictions = rf_predictions[valid]
baseline_predictions = baseline_predictions[valid]

# Calculate metrics
rf_mae = mean_absolute_error(actual, rf_predictions)
rf_rmse = mean_squared_error(
    actual, rf_predictions
) ** 0.5

baseline_mae = mean_absolute_error(
    actual, baseline_predictions
)
baseline_rmse = mean_squared_error(
    actual, baseline_predictions
) ** 0.5

print("\nForecasting Model Comparison")
print("-" * 40)
print(f"Random Forest MAE:  {rf_mae:.4f} kWh")
print(f"Baseline MAE:       {baseline_mae:.4f} kWh")
print(f"Random Forest RMSE: {rf_rmse:.4f} kWh")
print(f"Baseline RMSE:      {baseline_rmse:.4f} kWh")

if rf_mae < baseline_mae:
    print("\nRandom Forest has lower MAE than the baseline.")
else:
    print("\nRandom Forest does not beat the baseline on MAE.")

if rf_rmse < baseline_rmse:
    print("Random Forest has lower RMSE than the baseline.")
else:
    print("Random Forest does not beat the baseline on RMSE.")
