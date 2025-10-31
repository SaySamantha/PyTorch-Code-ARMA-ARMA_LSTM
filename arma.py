import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pipeline.armadata import ARMAVolumeDataset

train_dataset = ARMAVolumeDataset(dt.datetime(2022, 9, 30), dt.datetime(2023, 5, 8))
val_dataset   = ARMAVolumeDataset(dt.datetime(2023, 5, 9), dt.datetime(2023, 7, 19))
test_dataset  = ARMAVolumeDataset(dt.datetime(2023, 7, 20), dt.datetime(2023, 9, 29))

train_data = train_dataset.process_data()
val_data   = val_dataset.process_data()
test_data  = test_dataset.process_data()

# Combine train + val for fitting
train_val_data = pd.concat([train_data, val_data])

train_diff = train_val_data['log_diff_volume'].dropna().values
test_diff  = test_data['log_diff_volume'].dropna().values

train_series = pd.Series(train_diff, index=pd.RangeIndex(len(train_diff)))
test_series  = pd.Series(test_diff, index=pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(test_diff)))

# Starting log value from original full series before test period
log_start = train_val_data['log_volume'].iloc[-1]  # ✅ last log before test
print(f"[INFO] log_start: {log_start:.6f}")

model = ARIMA(train_series, order=(1, 0, 2))
model_fit = model.fit()

rolling_forecasts = []
updated_model = model_fit

# Append actual test data as if model gets new data every step
test_series_aligned = pd.Series(
    test_series.values,
    index=pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(test_series))
)

updated_model = model_fit.append(test_series_aligned)
rolling_forecasts_series = updated_model.predict(
    start=len(train_series),
    end=len(train_series) + len(test_series) - 1
)

forecast = rolling_forecasts_series.values
forecast = np.nan_to_num(forecast, nan=0.0)

df = pd.DataFrame({
    'Test_Diff': test_diff,
    'Forecast_Diff': forecast
}, index=pd.RangeIndex(len(test_diff)))

# Directly use the real test log values
df['Test_Log'] = test_data['log_volume'].reset_index(drop=True)

# Forecast logs: previous actual + forecasted diff
df['Forecast_Log'] = df['Test_Log'].shift(1) + df['Forecast_Diff']
df.loc[0, 'Forecast_Log'] = log_start + df.loc[0, 'Forecast_Diff']

mae = mean_absolute_error(df["Forecast_Log"], df["Test_Log"])
rmse = np.sqrt(mean_squared_error(df["Forecast_Log"], df["Test_Log"]))

print(f"\n[INFO] Log MAE: {mae:.6f}")
print(f"[INFO] Log RMSE: {rmse:.6f}")

plt.figure(figsize=(12, 6))
plt.plot(df['Test_Log'], label='Actual Test Log', linewidth=2)
plt.plot(df['Forecast_Log'], label='Forecast Log', linestyle='--', linewidth=2)
plt.xlabel("Index")
plt.ylabel("Log Volume")
plt.title("Test Data: Actual vs Forecasted Log Series")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

save_path = os.path.abspath("arma_forecast_results.csv")
df.to_csv(save_path, index=False)
print(f"[INFO] Saved full forecast results to: {save_path}")
