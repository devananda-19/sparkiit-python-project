"""
AQI Analysis & Forecasting
Steps: load -> clean -> EDA -> charts -> ARIMA forecast
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------
project_folder = Path(__file__).parent
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)  # charts are saved here

POLLUTANTS = ["pm25", "pm10", "no2"]
NUMERIC_COLS = ["aqi"] + POLLUTANTS

# ---------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------
data = pd.read_csv(project_folder / "aqi_data.csv")

print("First five rows:")
print(data.head())
print(f"\nShape: {data.shape[0]} rows, {data.shape[1]} columns")

# ---------------------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------------------
data["date"] = pd.to_datetime(data["date"], errors="coerce")
for col in NUMERIC_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Drop rows with an invalid date, remove duplicates, sort by date
data = data.dropna(subset=["date"])
data = data.drop_duplicates(subset="date")
data = data.sort_values("date").reset_index(drop=True)

# For time series, fill gaps by interpolation instead of deleting rows
data[NUMERIC_COLS] = data[NUMERIC_COLS].interpolate(method="linear")
data = data.dropna()  # only drops leftover gaps at the very start/end

# Flag inconsistent values (AQI/pollutants cannot be negative)
invalid = (data[NUMERIC_COLS] < 0).any(axis=1).sum()
print(f"\nRows with negative values: {invalid}")
data = data[(data[NUMERIC_COLS] >= 0).all(axis=1)]

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# ---------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
print("\nSummary statistics:")
print(data[NUMERIC_COLS].describe().round(2))

print("\nCorrelation between AQI and pollutants:")
print(data[NUMERIC_COLS].corr().round(2))

worst = data.loc[data["aqi"].idxmax()]
best = data.loc[data["aqi"].idxmin()]
print(f"\nHighest AQI: {worst['aqi']:.0f} on {worst['date'].date()}")
print(f"Lowest AQI:  {best['aqi']:.0f} on {best['date'].date()}")


def aqi_category(value):
    """Indian CPCB AQI categories."""
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


data["category"] = data["aqi"].apply(aqi_category)
print("\nDays per AQI category:")
print(data["category"].value_counts())

# ---------------------------------------------------------------
# 5. VISUALIZATION (charts are saved to the outputs folder)
# ---------------------------------------------------------------
# Chart 1: AQI with 7-day rolling average
data["aqi_7day_avg"] = data["aqi"].rolling(window=7).mean()

plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["aqi"], label="Daily AQI", alpha=0.6)
plt.plot(data["date"], data["aqi_7day_avg"], label="7-day average", linewidth=2)
plt.title("Air Quality Index (AQI) Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_folder / "aqi_over_time.png", dpi=150)
plt.close()

# Chart 2: Pollutant levels over time
plt.figure(figsize=(10, 5))
for col, label in zip(POLLUTANTS, ["PM2.5", "PM10", "NO2"]):
    plt.plot(data["date"], data[col], label=label)
plt.title("Air Pollutant Levels Over Time")
plt.xlabel("Date")
plt.ylabel("Pollutant Level")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_folder / "pollutants_over_time.png", dpi=150)
plt.close()

# Chart 3: Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data[NUMERIC_COLS].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between AQI and Pollutants")
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png", dpi=150)
plt.close()

print(f"\nCharts saved in: {output_folder}")

# ---------------------------------------------------------------
# 6. FORECASTING (ARIMA)
#    Needs:  pip install statsmodels
# ---------------------------------------------------------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np
except ImportError:
    print("\nSkipping forecasting: run 'pip install statsmodels' and try again.")
else:
    series = data.set_index("date")["aqi"].asfreq("D").interpolate()

    # Hold out the last 7 days to test how good the model is
    test_days = 7
    train, test = series[:-test_days], series[-test_days:]

    model = ARIMA(train, order=(1, 1, 1)).fit()
    test_pred = model.forecast(steps=test_days)

    rmse = np.sqrt(((test - test_pred) ** 2).mean())
    mae = (test - test_pred).abs().mean()
    print(f"\nARIMA(1,1,1) on last {test_days} days -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Refit on all data and forecast the next 14 days
    final_model = ARIMA(series, order=(1, 1, 1)).fit()
    forecast = final_model.get_forecast(steps=14)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series, label="Actual AQI")
    plt.plot(forecast_mean.index, forecast_mean, label="Forecast", linestyle="--")
    plt.fill_between(
        conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.2, label="95% interval"
    )
    plt.title("AQI Forecast (next 14 days)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_folder / "aqi_forecast.png", dpi=150)
    plt.close()
    print("Forecast chart saved.")

print("\nDone.")