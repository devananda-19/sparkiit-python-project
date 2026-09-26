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

POLLUTANTS = ["PM2.5", "PM10", "NO2"]
NUMERIC_COLS = ["AQI"] + POLLUTANTS

# ---------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------
data = pd.read_csv(project_folder / "data" / "city_day.csv")

print("First five rows:")
print(data.head())
print(f"\nShape: {data.shape[0]} rows, {data.shape[1]} columns")

# ---------------------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------------------
data["date"] = pd.to_datetime(data["Date"], errors="coerce")
for col in NUMERIC_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")
# Cities required for the internship project
TARGET_CITIES = ["Delhi", "Mumbai", "Kolkata", "Bengaluru"]
data = data[data["City"].isin(TARGET_CITIES)].copy()    

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Drop rows with an invalid date, remove duplicates, sort by date
data = data.dropna(subset=["date"])
data = data.drop_duplicates(subset=["City", "date"])
data = data.sort_values(["City", "date"]).reset_index(drop=True)
# Keep rows with an actual AQI value for AQI analysis/forecasting.
# Do not invent long missing AQI periods through interpolation.
data = data.dropna(subset=["AQI"]).copy()

# Interpolate only the selected pollutant columns within each city.
for col in POLLUTANTS:
    data[col] = data.groupby("City")[col].transform(
        lambda s: s.interpolate(limit=7)
    )

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

worst = data.loc[data["AQI"].idxmax()]
best = data.loc[data["AQI"].idxmin()]
print(
    f"\nHighest AQI: {worst['AQI']:.0f} "
    f"in {worst['City']} on {worst['date'].date()}"
)

print(
    f"Lowest AQI: {best['AQI']:.0f} "
    f"in {best['City']} on {best['date'].date()}"
)


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


data["category"] = data["AQI"].apply(aqi_category)
print("\nDays per AQI category:")
print(data["category"].value_counts())

# ---------------------------------------------------------------
# 5. VISUALIZATION (charts are saved to the outputs folder)
# ---------------------------------------------------------------
# Chart 1: AQI with 7-day rolling average
data["aqi_7day_avg"] = data["AQI"].rolling(window=7).mean()

plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["AQI"], label="Daily AQI", alpha=0.6)
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
# CITY-WISE AVERAGE AQI COMPARISON
# ---------------------------------------------------------------
city_avg_aqi = (
    data.groupby("City")["AQI"]
    .mean()
    .reindex(TARGET_CITIES)
)

plt.figure(figsize=(8, 5))
city_avg_aqi.plot(kind="bar")

plt.title("Average AQI by City")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    output_folder / "average_aqi_by_city.png",
    dpi=150
)
plt.close()

print("City comparison chart saved: average_aqi_by_city.png")
# ---------------------------------------------------------------
# CITY-WISE AQI TREND CHARTS
# ---------------------------------------------------------------
for city in TARGET_CITIES:
    city_data = data[data["City"] == city].sort_values("date").copy()

    city_data["aqi_7day_avg"] = city_data["AQI"].rolling(window=7).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(
        city_data["date"],
        city_data["AQI"],
        label="Daily AQI",
        alpha=0.5
    )
    plt.plot(
        city_data["date"],
        city_data["aqi_7day_avg"],
        label="7-day average",
        linewidth=2
    )

    plt.title(f"{city} - AQI Over Time")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"aqi_over_time_{city.lower()}.png"
    plt.savefig(output_folder / filename, dpi=150)
    plt.close()

    print(f"City trend chart saved: {filename}")
# ---------------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------------
cleaned_file = project_folder / "data" / "cleaned_aqi_data.csv"
data.to_csv(cleaned_file, index=False)

print(f"\nCleaned dataset saved to: {cleaned_file}")

# ---------------------------------------------------------------
# 6. FORECASTING (ARIMA) - CITY BY CITY
# ---------------------------------------------------------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
except ImportError:
    print("\nSkipping forecasting: run 'pip install statsmodels' and try again.")
else:
    forecast_days = 14
    comparison_results = []

    for city in TARGET_CITIES:
        city_data = data[data["City"] == city].copy()

        # Create one daily AQI time series for this city
        series = (
            city_data.set_index("date")["AQI"]
            .sort_index()
            .asfreq("D")
            .interpolate()
        )

        # Need enough observations for train/test forecasting
        if len(series) <= 14:
            print(f"\nSkipping {city}: not enough data for forecasting.")
            continue

        # Hold out the last 7 days for testing
        test_days = 7
        train = series.iloc[:-test_days]
        test = series.iloc[-test_days:]

        model = ARIMA(train, order=(1, 1, 1)).fit()
        test_pred = model.forecast(steps=test_days)

        rmse = np.sqrt(((test - test_pred) ** 2).mean())
        mae = (test - test_pred).abs().mean()
        comparison_results.append({
    "City": city,
    "Model": "ARIMA",
    "RMSE": rmse,
    "MAE": mae
})
        print(
            f"\n{city} - ARIMA(1,1,1) on last {test_days} days "
            f"-> RMSE: {rmse:.2f}, MAE: {mae:.2f}"
        )

        # Refit using all available data
        final_model = ARIMA(series, order=(1, 1, 1)).fit()

        forecast = final_model.get_forecast(steps=forecast_days)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Plot actual + forecast
        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series, label="Actual AQI")
        plt.plot(
            forecast_mean.index,
            forecast_mean,
            label="Forecast",
            linestyle="--"
        )

        plt.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            alpha=0.2,
            label="95% interval"
        )

        plt.title(f"{city} AQI Forecast (Next {forecast_days} Days)")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"aqi_forecast_{city.lower()}.png"
        plt.savefig(output_folder / filename, dpi=150)
        plt.close()

        print(f"Forecast chart saved: {filename}")



# ---------------------------------------------------------------
# 7. PROPHET FORECASTING - CITY BY CITY
# ---------------------------------------------------------------
forecast_days = 180

for city in TARGET_CITIES:
    city_data = data[data["City"] == city].copy()

    prophet_data = (
        city_data[["date", "AQI"]]
        .rename(columns={"date": "ds", "AQI": "y"})
        .sort_values("ds")
    )

    if len(prophet_data) <= 30:
        print(f"\nSkipping {city}: not enough data for Prophet.")
        continue

    # Use the same 7-day test period as ARIMA
    test_days = 7

    prophet_train = prophet_data.iloc[:-test_days].copy()
    prophet_test = prophet_data.iloc[-test_days:].copy()

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Train Prophet on the training data only
    model.fit(prophet_train)

    # Evaluate Prophet on the same 7-day test period as ARIMA
    prophet_test_forecast = model.predict(prophet_test[["ds"]])

    prophet_rmse = np.sqrt(
        mean_squared_error(
            prophet_test["y"],
            prophet_test_forecast["yhat"]
        )
    )

    prophet_mae = mean_absolute_error(
        prophet_test["y"],
        prophet_test_forecast["yhat"]
    )

    comparison_results.append({
        "City": city,
        "Model": "Prophet",
        "RMSE": prophet_rmse,
        "MAE": prophet_mae
    })

    print(
        f"{city} - Prophet on last {test_days} days "
        f"-> RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}"
    )

    # -----------------------------------------------------------
    # Final Prophet model for the 180-day future forecast
    # -----------------------------------------------------------
    final_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    final_model.fit(prophet_data)

    future = final_model.make_future_dataframe(
        periods=forecast_days,
        freq="D"
    )

    forecast = final_model.predict(future)

    forecast_future = forecast.tail(forecast_days)

    plt.figure(figsize=(10, 5))

    plt.plot(
        prophet_data["ds"],
        prophet_data["y"],
        label="Actual AQI",
        alpha=0.5
    )

    plt.plot(
        forecast_future["ds"],
        forecast_future["yhat"],
        label="Prophet Forecast",
        linestyle="--"
    )

    plt.fill_between(
        forecast_future["ds"],
        forecast_future["yhat_lower"],
        forecast_future["yhat_upper"],
        alpha=0.2,
        label="95% interval"
    )

    plt.title(f"{city} AQI Prophet Forecast (Next {forecast_days} Days)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"aqi_prophet_forecast_{city.lower()}.png"

    plt.savefig(
        output_folder / filename,
        dpi=150
    )

    plt.close()

    print(f"Prophet forecast chart saved: {filename}")
    # ---------------------------------------------------------------
# 8. SAVE MODEL COMPARISON RESULTS
# ---------------------------------------------------------------
comparison_df = pd.DataFrame(comparison_results)

comparison_file = output_folder / "model_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)

print("\nModel comparison results:")
print(comparison_df)

print(f"\nModel comparison saved to: {comparison_file}")
print("\nDone.")
output_folder.mkdir(exist_ok=True)