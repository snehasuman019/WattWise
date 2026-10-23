# WattWise — Smart Electricity Consumption Analyzer

WattWise is a data analytics project that uses **Apache PySpark** to process household electricity consumption data and identify consumption patterns, unusual usage days, and hourly consumption trends. A Streamlit dashboard presents the results through interactive visualizations.

## 🚀 Live Demo

**Explore the deployed dashboard here:** [WattWise — Smart Electricity Consumption Analyzer](https://wattwise-electricity-consumption.streamlit.app/)

## Features

- **Large-scale data processing:** Clean and transform household electricity records using PySpark.
- **Consumption analytics:** Explore hourly, daily, and monthly electricity consumption.
- **Anomaly detection:** Identify statistically unusual daily consumption using the Interquartile Range (IQR) method.
- **Consumption prediction:** Train a PySpark MLlib Random Forest regression model to estimate hourly consumption on held-out historical data.
- **Model evaluation:** Compare predictions with actual consumption using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Interactive dashboard:** Explore results with Streamlit and Plotly.

## Technology Stack

- Python
- Apache Spark / PySpark
- PySpark MLlib
- Pandas
- Streamlit
- Plotly
- scikit-learn (baseline evaluation)

## Project Structure

```text
WattWise/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   └── processed/
│       ├── anomaly_detection.csv
│       ├── daily_consumption.csv
│       ├── forecast_predictions.csv
│       ├── hourly_consumption.csv
│       └── monthly_consumption.csv
└── src/
    ├── data_processing.py
    ├── analytics.py
    ├── anomaly_detection.py
    ├── forecasting.py
    └── baseline_evaluation.py
```

## Dataset

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

The original dataset contains household electricity measurements recorded at approximately one-minute intervals. The project transforms the readings into hourly consumption summaries.

Dataset page: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Download the dataset and place the original text file at:

```text
data/raw/household_power_consumption.txt
```

The raw data and the large cleaned CSV are excluded from Git to keep the repository manageable.

## Installation

### Prerequisites

- Python 3.11
- Java compatible with your installed Spark version
- Git
- A supported Apache Spark/PySpark environment

### 1. Clone the repository

```bash
git clone https://github.com/snehasuman019/WattWise.git
cd WattWise
```

### 2. Create and activate a virtual environment

Windows Command Prompt:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

After placing the source dataset in `data/raw/`, run the processing pipeline in this order:

```bash
python src/data_processing.py
python src/analytics.py
python src/anomaly_detection.py
python src/forecasting.py
python src/baseline_evaluation.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

**Note:** Spark and Java must be configured correctly for your operating system. Windows users may encounter Hadoop-related warnings or file-writing limitations depending on their environment.

## Model Evaluation

The current evaluation produced these results:

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 0.3511 kWh | 0.3974 kWh |
| RMSE | 0.4942 kWh | 0.6033 kWh |

Lower MAE and RMSE indicate smaller prediction errors on the evaluated data. In the current run, the Random Forest model outperformed the baseline on both metrics.

These figures describe the current evaluation run and may change if the data split, features, or model configuration changes.

**Evaluation scope:** The model is evaluated on held-out historical records. This is not yet a forecast extending beyond the final timestamp in the dataset.

## Anomaly Detection

The project uses the IQR method to flag unusually low or high daily consumption values. An anomaly is a statistical outlier and does not automatically indicate an electrical fault or energy waste.

## Future Improvements

- Extend the model to forecast consumption beyond the available dataset.
- Improve model validation and compare additional forecasting approaches.
- Add configurable date ranges and more dashboard insights.
- Explore additional deployment and monitoring options.

## Author

**Sneha Suman**

B.Tech — Computer Science and Engineering  
Area of Interest: Data Science and Big Data Analytics

**GitHub:** [snehasuman019](https://github.com/snehasuman019)  
**Live Dashboard:** [WattWise](https://wattwise-electricity-consumption.streamlit.app/)
