import os
import torch
import mlflow
import numpy as np
from torch import nn
import datetime as dt
from pprint import pprint
from torchinfo import summary
from pipeline.train import run, evaluate
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from torch.utils.data import Dataset
from pipeline.armadata import ARMAVolumeDataset

from pipeline.model import LSTMModel
from pipeline.utils import Hyperparameter, set_seed

class ResidualVolumeDataset(Dataset):
    def __init__(self, residuals, sequence_length):
        self.sequence_length = sequence_length
        self.residuals = np.array(residuals).flatten()
        self.features = []
        self.targets = []
        for i in range(len(self.residuals) - sequence_length):
            self.features.append(self.residuals[i:i+sequence_length])
            self.targets.append(self.residuals[i+sequence_length])
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(-1)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    def __len__(self):
        return len(self.features)

if __name__ == '__main__':
    # Specify hyperparameters
    hyperparams = Hyperparameter(
        sequence_length = 78,
        batch_size = 64,
        hidden_dim = 32,
        activation = nn.ReLU(),
        dropout = 0,
        norm = nn.BatchNorm1d,
        num_layers = 1,
        lr = 1e-3,
        wd = 0,
        min_lr = 1e-5,
        factor = 0.5,
        patience = 10,
        epochs = 10,
        
        # Set-up
        nworkers = 1,
        nruns = 1,
        log_every = 20,
        use_amp = False, #CHANGED THIS PART, NOT USING GPU
    )

    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify device
    device = torch.device('cpu') #CHANGED THIS PART, NOT USING GPU
    
    # [MLFlow] Experiment set-up
    ## Option 1: Run `mlflow server --host 127.0.0.1 --port 8080` (before running this script) and uncomment the code below
    # mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
    ## Option 2: Run `mlflow ui --port 8080` (after running this script) on the current directory 
    mlflow.set_experiment(f"Model Training @ {dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")

    # Perform training
    results = {metric: [] for metric in metrics.keys()}
    for i in range(hyperparams.nruns):
        with mlflow.start_run(run_name=f'run_{i}'):
            # [MLFlow] Log hyperparameters
            mlflow.log_params(hyperparams.__dict__)
            
            # [MLFlow] Set tag to current run
            mlflow.set_tag('Training Info', f'Model training with seed = {i}')
            
            # Set seed
            set_seed(i)

            # Load dataset    
            train_arma_dataset = ARMAVolumeDataset(dt.datetime(2022,9,30), dt.datetime(2023,5,8))
            val_arma_dataset   = ARMAVolumeDataset(dt.datetime(2023,5,9), dt.datetime(2023,7,19))
            test_arma_dataset  = ARMAVolumeDataset(dt.datetime(2023,7,20), dt.datetime(2023,9,29))

            # ------------------- ARMA FORMALITIES HERE -------------------
            train_data = train_arma_dataset.process_data()
            train_diff = train_data['log_diff_volume'].dropna().values
            train_series = pd.Series(train_diff, index=pd.RangeIndex(len(train_diff)))

            result = adfuller(train_series)
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            print('Critical Values:', result[4])

            lb_test = acorr_ljungbox(train_series, lags=[10, 20], return_df=True)
            print("=== Ljung-Box Test ===")
            print(lb_test)

            for lag in lb_test.index:
                pval = lb_test.loc[lag, 'lb_pvalue']
                if pval < 0.05:
                    print(f"Lag {lag}: Series shows significant autocorrelation (p-value={pval:.4f})")
                else:
                    print(f"Lag {lag}: Series does NOT show significant autocorrelation (p-value={pval:.4f})")

            if result[1] < 0.05:
                print("Series is stationary")
            else:
                print("Series is non-stationary; consider differencing")

            #plot_acf(train_series, lags=50)
            #plt.show()

            #plot_pacf(train_series, lags=50)
            #plt.show()

            # ------------------- ADDED ARMA TRAINING STEP -------------------
            arma_model = ARIMA(train_series, order=(1,0,2))
            arma_fit = arma_model.fit()

            # Keep full ARMA residuals first
            train_resid_full = arma_fit.resid.values

            # VAL residuals: actual - predicted using arma_fit
            val_data = val_arma_dataset.process_data()
            val_series = val_data['log_diff_volume'].dropna().values
            val_series = pd.Series(val_series)
            val_pred = arma_fit.predict(start=len(train_series), end=len(train_series)+len(val_series)-1)
            val_pred = pd.Series(val_pred.values, index=val_series.index)
            val_resid_full = val_series - val_pred

            # TEST residuals: actual - predicted using arma_fit
            test_data = test_arma_dataset.process_data()
            test_series = test_data['log_diff_volume'].dropna().values
            test_series = pd.Series(test_series)

            # --- TEST residuals: compute manually for unseen test data ---

            # Forecast differences for the test period
            forecast_diff = arma_fit.forecast(steps=len(test_series))  # one-step-ahead

            # Compute forecast log using actual previous log values
            log_start = 14.62308448145951 # HARD CODED IN THE MEANTIME, MANUALLY CHECKED THE DATASET
            forecast_log = []
            prev_log = 14.62308448145951  # hardcoded
            for diff in forecast_diff:
                next_log = prev_log + diff
                forecast_log.append(next_log)
                prev_log = next_log  # accumulate forecast sequentially

            # Residuals = actual log - forecast log
            test_resid_full = test_data['log_volume'].values - np.array(forecast_log)

            # Save forecast for CSV
            test_pred = np.array(forecast_log)

            # Slice residuals for LSTM 60-20-20 split according to dataset dates
            train_resid = train_resid_full                     # 60% train dates
            val_resid   = val_resid_full                       # 20% val dates
            test_resid  = test_resid_full                      # 20% test dates

            # Now create datasets
            train_dataset = ResidualVolumeDataset(train_resid, hyperparams.sequence_length)
            val_dataset   = ResidualVolumeDataset(val_resid, hyperparams.sequence_length)
            test_dataset  = ResidualVolumeDataset(test_resid, hyperparams.sequence_length)

            # SAVE THE DATA
            np.save('train_resid.npy', train_resid)
            np.save('val_resid.npy', val_resid)
            np.save('test_resid.npy', test_resid)

            np.save('train_arma_fitted.npy', arma_fit.fittedvalues.values)
            np.save('val_arma_pred.npy', val_pred)
            np.save('test_arma_pred.npy', test_pred)

            # ------------------- THE REST OF LSTM STEP HERE -------------------
            # CHANGED NUM_WORKERS=HYPERPARAMS.NWORKERS TO DROP_LAST; USING MAC
            train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, drop_last=False) 
            # last batch has to be of the same len for training stability
            val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, drop_last=False) 
            # ok even if last batch is smaller
            test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False, drop_last=False)

            # Extract input shapes
            input_dim = train_dataset[0][0].shape[-1]  # should be 1
            output_dim = train_dataset[0][1].shape[-1] # should be 1
            
            # Load model
            model = LSTMModel(input_dim, hyperparams.hidden_dim, output_dim, hyperparams.activation, 
                              hyperparams.dropout, hyperparams.norm, hyperparams.num_layers).to(device)
            model = torch.compile(model, mode='default')
            
            # [MLFlow] Log model summary
            with open('model_summary.txt', 'w') as f:
                f.write(str(summary(model)))
            mlflow.log_artifact('model_summary.txt')
            os.remove('model_summary.txt')

            # EDITED: Model training
            test_metrics, training_loss_list, val_loss_list = run(
                model, train_loader, val_loader, test_loader, device, loss_fn, metrics, hyperparams
            )

            # Get predictions and save them
            train_metrics, train_data = evaluate(model, train_loader, device, metrics, hyperparams)
            val_metrics, val_data     = evaluate(model, val_loader, device, metrics, hyperparams)
            test_metrics, test_data   = evaluate(model, test_loader, device, metrics, hyperparams)

            np.save('train_preds.npy', train_data['predictions'])
            np.save('train_targets.npy', train_data['targets'])
            np.save('val_preds.npy', val_data['predictions'])
            np.save('val_targets.npy', val_data['targets'])
            np.save('test_preds.npy', test_data['predictions'])
            np.save('test_targets.npy', test_data['targets'])

            np.save('training_loss.npy', np.array(training_loss_list))
            np.save('val_loss.npy', np.array(val_loss_list))

            # ADDED TO SAVE
            state_dict_path = f'lstm_model_run{i}_state_dict.pt'
            torch.save(model.state_dict(), state_dict_path)
            print(f"Saved model state_dict to {state_dict_path}")

    # EDITED Print results
    print('\nSummary of Results')

    results = {f'test_{metric}': [] for metric in metrics.keys()}

    # Append current run's test metrics
    for metric, value in test_metrics.items():
        results[f'test_{metric}'].append(value)

    # Print results
    pprint(results)
    summarize_results = lambda metrics: {metric: f'{np.mean(values):.6f} ± {np.std(values):.6f}' for metric, values in metrics.items()}    
    pprint(summarize_results(results))

    # ------------------- CREATE CSV OF TEST FORECASTS -------------------

# Convert to arrays
arma_forecast = np.array(test_pred)
arma_residuals = np.array(test_resid_full)
lstm_pred_residuals = np.array(test_data['predictions']).flatten()
timestamps = test_arma_dataset.process_data()['timestamp'].reset_index(drop=True)

# Align lengths in case of mismatch
min_len = min(len(timestamps), len(arma_forecast), len(arma_residuals), len(lstm_pred_residuals))
timestamps = timestamps[:min_len]
arma_forecast = arma_forecast[:min_len]
arma_residuals = arma_residuals[:min_len]
lstm_pred_residuals = lstm_pred_residuals[:min_len]

# ------------------- CREATE CSV OF TEST FORECASTS -------------------

# Convert to arrays
arma_forecast = np.array(forecast_log)                     # Forecasted log values
arma_residuals = np.array(test_resid_full)                 # ARMA residuals (log_actual - log_forecast)
test_processed = test_arma_dataset.process_data()
test_diff = test_processed['log_diff_volume'].dropna().values   # Actual differenced logs
test_log = test_processed['log_volume'].values[:len(forecast_log)]  # Actual log values
timestamps = test_processed['timestamp'].reset_index(drop=True)
forecast_diff = np.array(forecast_diff)                    # Forecasted differenced logs
test_log = test_processed['log_volume'].values[:len(arma_forecast)]  # Actual log values
forecast_log = np.array(forecast_log)                      # Already computed above
lstm_pred_residuals = np.load('test_preds.npy').flatten()  # LSTM residual predictions

timestamps = test_arma_dataset.process_data()['timestamp'].reset_index(drop=True)

# Align lengths in case of mismatch
min_len = min(
    len(timestamps),
    len(test_diff),
    len(forecast_diff),
    len(test_log),
    len(forecast_log),
    len(arma_forecast),
    len(arma_residuals),
    len(lstm_pred_residuals)
)

# Truncate to min length
timestamps = timestamps[:min_len]
test_diff = test_diff[:min_len]
forecast_diff = forecast_diff[:min_len]
test_log = test_log[:min_len]
forecast_log = forecast_log[:min_len]
arma_forecast = arma_forecast[:min_len]
arma_residuals = arma_residuals[:min_len]
lstm_pred_residuals = lstm_pred_residuals[:min_len]

# Combine into DataFrame
df_csv = pd.DataFrame({
    'timestamp': timestamps,
    'test_diff': test_diff,
    'forecast_diff': forecast_diff,
    'test_log': test_log,
    'forecast_log': forecast_log,
    'arma_forecast': arma_forecast,
    'arma_residual': arma_residuals,
    'lstm_pred_residual': lstm_pred_residuals
})

# Save CSV
csv_path = 'test_forecast_analysis.csv'
df_csv.to_csv(csv_path, index=False)
print(f"[INFO] CSV saved: {csv_path}")
