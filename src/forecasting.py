
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    dayofweek,
    dayofmonth,
    month,
    lag,
    avg,
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("WattWise-Consumption-Forecasting")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # --------------------------------------------------
    # 2. Load hourly consumption data
    # --------------------------------------------------

    data_path = "data/processed/hourly_consumption.csv"

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(data_path)
    )

    df = df.withColumn(
        "hour_start",
        col("hour_start").cast("timestamp")
    )

    df = df.orderBy("hour_start")

    # --------------------------------------------------
    # 3. Create time-based features
    # --------------------------------------------------

    df = (
        df
        .withColumn("hour", hour("hour_start"))
        .withColumn("day_of_week", dayofweek("hour_start"))
        .withColumn("day_of_month", dayofmonth("hour_start"))
        .withColumn("month", month("hour_start"))
    )

    # Previous-hour consumption is a useful predictor.
    window_spec = Window.orderBy("hour_start")

    df = df.withColumn(
        "previous_hour_kwh",
        lag("hourly_consumption_kwh", 1).over(window_spec)
    )

    # Rolling average using the previous 24 hours,
    # excluding the current hour.
    rolling_window = (
        Window.orderBy("hour_start")
        .rowsBetween(-24, -1)
    )

    df = df.withColumn(
        "previous_24h_average_kwh",
        avg("hourly_consumption_kwh").over(rolling_window)
    )

    df = df.dropna()

    # --------------------------------------------------
    # 4. Assemble model features
    # --------------------------------------------------

    feature_columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "previous_hour_kwh",
        "previous_24h_average_kwh",
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    model_data = assembler.transform(df).select(
        "hour_start",
        "features",
        col("hourly_consumption_kwh").alias("label")
    )

    # --------------------------------------------------
    # 5. Chronological train/test split
    # --------------------------------------------------

    total_rows = model_data.count()
    split_point = int(total_rows * 0.8)

    indexed = model_data.withColumn(
        "row_number",
        __import__("pyspark.sql.functions", fromlist=["row_number"])
        .row_number().over(window_spec)
    )

    train_data = indexed.filter(
        col("row_number") <= split_point
    )

    test_data = indexed.filter(
        col("row_number") > split_point
    )

    print(f"Training records: {train_data.count():,}")
    print(f"Testing records: {test_data.count():,}")

    # --------------------------------------------------
    # 6. Train Random Forest regression model
    # --------------------------------------------------

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        numTrees=30,
        maxDepth=8,
        seed=42
    )

    model = rf.fit(train_data)

    # --------------------------------------------------
    # 7. Evaluate the model
    # --------------------------------------------------

    predictions = model.transform(test_data)

    rmse_evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )

    mae_evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="mae"
    )

    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    print("\nForecasting model evaluation")
    print(f"RMSE: {rmse:.3f} kWh")
    print(f"MAE:  {mae:.3f} kWh")

    print("\nSample predictions:")
    predictions.select(
        "hour_start",
        "label",
        "prediction"
    ).show(10, truncate=False)

    
    
    # --------------------------------------------------
    # 8. Save test predictions
    # --------------------------------------------------

    import os

    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/forecast_predictions.csv"

    predictions.select(
        "hour_start",
        "label",
        "prediction"
    ).toPandas().to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print("Forecasting completed successfully!")
    print("Model trained and evaluated; model file not saved.")


    # Model saving is temporarily skipped because the
    # Windows Hadoop utilities are not configured.
    print("Model trained and evaluated; model file not saved.")


    # Model saving is temporarily skipped because the
    # Windows Hadoop utilities are not configured.
    print("Model trained and evaluated; model file not saved.")

    # Use Pandas to avoid Spark's Windows CSV writer issue.
    output_path = "data/processed/forecast_predictions.csv"

    predictions.select(
        "hour_start",
        "label",
        "prediction"
    ).toPandas().to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print("Forecasting completed successfully!")
    print("Model trained and evaluated; model file not saved.")

finally:
    spark.stop()
