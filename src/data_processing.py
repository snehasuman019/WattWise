
import os
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat,
    lit,
    isnan,
    when,
    count,
    try_to_timestamp,
    hour,
    sum as spark_sum,
    round as spark_round,
    date_trunc,
)


# ============================================================
# 1. Start Spark
# ============================================================

spark = (
    SparkSession.builder
    .appName("WattWise-Electricity-Analysis")
    .master("local[*]")
    .config("spark.hadoop.fs.permissions.umask-mode", "000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:
    # ========================================================
    # 2. Load the electricity dataset
    # ========================================================

    file_path = "data/raw/household_power_consumption.txt"

    df = (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .option("nullValue", "?")
        .csv(file_path)
    )

    # ========================================================
    # 3. Convert measurement columns to numeric types
    # ========================================================

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for column_name in numeric_columns:
        df = df.withColumn(
            column_name,
            col(column_name).cast("double")
        )

    # ========================================================
    # 4. Create timestamp from Date and Time
    # ========================================================

    df = df.withColumn(
        "timestamp",
        try_to_timestamp(
            concat(col("Date"), lit(" "), col("Time")),
            lit("d/M/yyyy HH:mm:ss")
        )
    )

    # ========================================================
    # 5. Check missing measurement values
    # ========================================================

    print("\nMissing values in each measurement column:")

    df.select([
        count(
            when(
                col(c).isNull() | isnan(col(c)),
                c
            )
        ).alias(c)
        for c in numeric_columns
    ]).show(truncate=False)

    # ========================================================
    # 6. Clean the dataset
    # ========================================================

    df_clean = df.filter(
        col("timestamp").isNotNull()
        & col("Global_active_power").isNotNull()
    )

    cleaned_count = df_clean.count()

    print(f"\nCleaned record count: {cleaned_count:,}")

    # ========================================================
    # 7. Calculate hourly electricity consumption
    # ========================================================

    # Global_active_power is measured in kilowatts.
    # Each reading represents approximately one minute.
    # Therefore, kWh per reading = kW / 60.

    df_hourly = (
        df_clean
        .withColumn(
            "energy_kwh",
            col("Global_active_power") / 60.0
        )
        .groupBy(
            date_trunc(
                "hour",
                col("timestamp")
            ).alias("hour_start")
        )
        .agg(
            spark_round(
                spark_sum("energy_kwh"),
                3
            ).alias("hourly_consumption_kwh")
        )
        .orderBy("hour_start")
    )

    print("\nHourly electricity consumption:")
    df_hourly.show(10, truncate=False)

    hourly_count = df_hourly.count()
    print(f"Total hourly groups: {hourly_count:,}")

    # ========================================================
    # 8. Display schema and sample records
    # ========================================================

    print("\nCleaned dataset schema:")
    df_clean.printSchema()

    print("\nSample cleaned records:")
    df_clean.select(
        "timestamp",
        "Global_active_power",
        "Voltage"
    ).show(10, truncate=False)

    print(f"Total original records: {df.count():,}")

    # ========================================================
    # 9. Prepare output directory
    # ========================================================

    output_directory = "data/processed"
    os.makedirs(output_directory, exist_ok=True)

    # ========================================================
    # 10. Save hourly consumption CSV
    # ========================================================

    hourly_output = os.path.join(
        output_directory,
        "hourly_consumption.csv"
    )

    # This summary is small enough to convert to Pandas.
    df_hourly.toPandas().to_csv(
        hourly_output,
        index=False
    )

    print(f"\nHourly consumption saved to: {hourly_output}")

    # ========================================================
    # 11. Save cleaned records using Python CSV writer
    # ========================================================

    cleaned_output = os.path.join(
        output_directory,
        "cleaned_electricity.csv"
    )

    # Stream rows instead of loading the entire dataset
    # into Pandas memory.
    with open(
        cleaned_output,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(df_clean.columns)

        for row in df_clean.toLocalIterator():
            writer.writerow(row)

    print(f"Cleaned data saved to: {cleaned_output}")

    print("\nWattWise data processing completed successfully!")

finally:
    # Stop Spark only after all processing has finished,
    # including if an exception occurs.
    spark.stop()
