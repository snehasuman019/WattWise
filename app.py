
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="WattWise | Electricity Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ WattWise")
st.subheader("Smart Electricity Consumption Analyzer")
st.caption(
    "Explore household electricity usage from "
    "December 2006 to November 2010."
)

# --------------------------------------------------
# Load processed datasets
# --------------------------------------------------

DATA_DIR = Path("data/processed")


@st.cache_data
def load_data():
    hourly = pd.read_csv(DATA_DIR / "hourly_consumption.csv")
    daily = pd.read_csv(DATA_DIR / "daily_consumption.csv")
    monthly = pd.read_csv(DATA_DIR / "monthly_consumption.csv")
    anomalies = pd.read_csv(DATA_DIR / "anomaly_detection.csv")

    hourly["hour_start"] = pd.to_datetime(hourly["hour_start"])
    daily["date"] = pd.to_datetime(daily["date"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    anomalies["date"] = pd.to_datetime(anomalies["date"])

    return hourly, daily, monthly, anomalies


try:
    hourly, daily, monthly, anomalies = load_data()
except FileNotFoundError as error:
    st.error(f"Required dataset not found: {error}")
    st.info("Run the data-processing and analytics scripts first.")
    st.stop()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

min_date = daily["date"].min().date()
max_date = daily["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("The start date must be before the end date.")
    st.stop()

# --------------------------------------------------
# Filter datasets
# --------------------------------------------------

start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

hourly_filtered = hourly[
    (hourly["hour_start"] >= start)
    & (hourly["hour_start"] < end)
].copy()

daily_filtered = daily[
    (daily["date"] >= start)
    & (daily["date"] < end)
].copy()

monthly_filtered = monthly[
    (monthly["month"] >= start.to_period("M").to_timestamp())
    & (monthly["month"] <= end.to_period("M").to_timestamp())
].copy()

if daily_filtered.empty:
    st.warning("No consumption records exist for this date range.")
    st.stop()

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

total_energy = daily_filtered["daily_consumption_kwh"].sum()
average_daily = daily_filtered["daily_consumption_kwh"].mean()

peak_index = daily_filtered["daily_consumption_kwh"].idxmax()
peak_day = daily_filtered.loc[peak_index, "date"]
peak_consumption = daily_filtered.loc[
    peak_index, "daily_consumption_kwh"
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Consumption",
    f"{total_energy:,.2f} kWh"
)

col2.metric(
    "Average Daily Consumption",
    f"{average_daily:,.2f} kWh"
)

col3.metric(
    "Peak Daily Consumption",
    f"{peak_consumption:,.2f} kWh"
)

st.caption(f"Peak day: {peak_day:%d %B %Y}")

# --------------------------------------------------
# Daily consumption chart
# --------------------------------------------------

st.subheader("Daily Electricity Consumption")

daily_chart = px.line(
    daily_filtered,
    x="date",
    y="daily_consumption_kwh",
    title="Daily Consumption Over Time",
    labels={
        "date": "Date",
        "daily_consumption_kwh": "Consumption (kWh)"
    }
)

daily_chart.update_layout(hovermode="x unified")
st.plotly_chart(daily_chart, use_container_width=True)


# --------------------------------------------------
# Unusual consumption detection
# --------------------------------------------------

st.subheader("🚨 Unusual Consumption Days")

anomalies_filtered = anomalies[
    (anomalies["date"] >= start)
    & (anomalies["date"] < end)
    & (anomalies["anomaly"] == True)
].copy()

st.metric(
    "Unusual days in selected period",
    len(anomalies_filtered)
)

if anomalies_filtered.empty:
    st.info("No unusual consumption days found in this period.")
else:
    st.warning(
        "These dates have statistically unusual daily "
        "consumption according to the IQR method."
    )

    st.dataframe(
        anomalies_filtered.sort_values(
            "daily_consumption_kwh",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    anomaly_chart = px.scatter(
        anomalies_filtered,
        x="date",
        y="daily_consumption_kwh",
        title="Detected Unusual Consumption Days",
        labels={
            "date": "Date",
            "daily_consumption_kwh": "Consumption (kWh)"
        }
    )

    st.plotly_chart(
        anomaly_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Hourly consumption chart
# --------------------------------------------------

st.subheader("Hourly Electricity Consumption")

hourly_chart = px.line(
    hourly_filtered,
    x="hour_start",
    y="hourly_consumption_kwh",
    title="Hourly Consumption Over Time",
    labels={
        "hour_start": "Time",
        "hourly_consumption_kwh": "Consumption (kWh)"
    }
)

hourly_chart.update_layout(hovermode="x unified")
st.plotly_chart(hourly_chart, use_container_width=True)

# --------------------------------------------------
# Monthly consumption chart
# --------------------------------------------------

st.subheader("Monthly Electricity Consumption")

if not monthly_filtered.empty:
    monthly_chart = px.bar(
        monthly_filtered,
        x="month",
        y="monthly_consumption_kwh",
        title="Monthly Consumption Comparison",
        labels={
            "month": "Month",
            "monthly_consumption_kwh": "Consumption (kWh)"
        }
    )

    st.plotly_chart(monthly_chart, use_container_width=True)

# --------------------------------------------------
# Data table
# --------------------------------------------------

with st.expander("View daily consumption data"):
    st.dataframe(
        daily_filtered,
        use_container_width=True,
        hide_index=True
    )

st.success("WattWise dashboard loaded successfully!")


# --------------------------------------------------
# Forecasting results
# --------------------------------------------------

st.subheader("🔮 Electricity Consumption Forecast Evaluation")

forecast_path = DATA_DIR / "forecast_predictions.csv"

if forecast_path.exists():
    forecast = pd.read_csv(forecast_path)
    forecast["hour_start"] = pd.to_datetime(forecast["hour_start"])

    forecast = forecast.rename(
        columns={
            "label": "actual_kwh",
            "prediction": "predicted_kwh"
        }
    )

    # Calculate evaluation metrics from the saved predictions
    mae = (
        forecast["actual_kwh"] - forecast["predicted_kwh"]
    ).abs().mean()

    rmse = (
        (forecast["actual_kwh"] - forecast["predicted_kwh"]) ** 2
    ).mean() ** 0.5

    metric1, metric2 = st.columns(2)

    metric1.metric("Test RMSE", f"{rmse:.3f} kWh")
    metric2.metric("Test MAE", f"{mae:.3f} kWh")

    # Display a manageable sample of test predictions
    chart_data = forecast.tail(168)

    fig = px.line(
        chart_data,
        x="hour_start",
        y=["actual_kwh", "predicted_kwh"],
        title="Actual vs Predicted Consumption — Last 168 Test Hours",
        labels={
            "hour_start": "Time",
            "value": "Consumption (kWh)",
            "variable": "Series"
        }
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View forecast prediction data"):
        st.dataframe(
            forecast.tail(100),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "These are predictions for held-out historical test records, "
        "not a forecast of future hours beyond the dataset."
    )
else:
    st.info(
        "Forecast predictions are not available yet. "
        "Run python src/forecasting.py first."
    )

    # Compare Random Forest with the baseline
    st.subheader("Model vs Baseline")

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE"],
        "Random Forest": [0.3511, 0.4942],
        "Baseline": [0.3974, 0.6033],
    })

    comparison_chart = px.bar(
        comparison,
        x="Metric",
        y=["Random Forest", "Baseline"],
        barmode="group",
        title="Forecast Error Comparison (kWh)",
        labels={
            "value": "Error (kWh)",
            "variable": "Model"
        },
        barmode_group_gap=0.15
    )

    st.plotly_chart(
        comparison_chart,
        use_container_width=True
    )

    st.caption(
        "Lower MAE and RMSE indicate better predictions. "
        "Random Forest outperformed this baseline on both metrics."
    )
    st.subheader("🤖 Model vs Baseline")

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE"],
        "Random Forest": [0.3511, 0.4942],
        "Baseline": [0.3974, 0.6033],
    })

    comparison_chart = px.bar(
        comparison,
        x="Metric",
        y=["Random Forest", "Baseline"],
        barmode="group",
        title="Forecast Error Comparison",
        labels={
            "value": "Error (kWh)",
            "variable": "Model"
        }
    )

    st.plotly_chart(comparison_chart, use_container_width=True)

    st.caption(
        "Lower error is better. Values are from the current model evaluation."
    )

    st.subheader("Model vs Baseline")

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Random Forest": [0.3511, 0.4942],
    "Baseline": [0.3974, 0.6033]
})

comparison_chart = px.bar(
    comparison,
    x="Metric",
    y=["Random Forest", "Baseline"],
    barmode="group",
    title="Forecast Error Comparison",
    labels={
        "value": "Error (kWh)",
        "variable": "Model"
    }
)

st.plotly_chart(comparison_chart, use_container_width=True)c