import os
import torch
import mlflow
import numpy as np
from torch import nn
import datetime as dt
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA
from torchinfo import summary
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from pipeline.armadata import ARMAVolumeDataset
from pipeline.train import run, evaluate
from pipeline.model import LSTMModel
from pipeline.utils import Hyperparameter, set_seed

class ResidualVolumeDataset(Dataset):
    def __init__(self, residuals, sequence_length):
        self.sequence_length = sequence_length
        self.residuals = np.array(residuals).flatten() # ensures 1D residuals
        self.features = []
        self.targets = []
        for i in range(len(self.residuals) - sequence_length): # sliding window to store seq, targets
            self.features.append(self.residuals[i:i+sequence_length])
            self.targets.append(self.residuals[i+sequence_length])
        # current features shape is (num of seq, seq len), unsqueeze(-1) to indicate input size is 1
        # current targets shape is (num of targets), unsqueeze(-1) to indicate input size is 1
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(-1)
    def __getitem__(self, idx): # idx for index
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
        epochs = 5,
        
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
    for i in range(hyperparams.nruns):
        with mlflow.start_run(run_name=f'run_{i}'):
            # [MLFlow] Log hyperparameters
            mlflow.log_params(hyperparams.__dict__)
            
            # [MLFlow] Set tag to current run
            mlflow.set_tag('Training Info', f'Model training with seed = {i}')
            
            # Set seed
            set_seed(i)
            # ------------------- CREATE DATASETS -------------------
            train_arma_dataset = ARMAVolumeDataset(dt.datetime(2022,9,30), dt.datetime(2023,5,8))
            val_arma_dataset   = ARMAVolumeDataset(dt.datetime(2023,5,9), dt.datetime(2023,7,19))
            test_arma_dataset  = ARMAVolumeDataset(dt.datetime(2023,7,20), dt.datetime(2023,9,30))

            train_data = train_arma_dataset.process_data()
            val_data   = val_arma_dataset.process_data()
            test_data  = test_arma_dataset.process_data()

            train_diff = train_data['log_diff_volume'].dropna().values
            val_diff = val_data['log_diff_volume'].dropna().values
            test_diff  = test_data['log_diff_volume'].dropna().values

            train_series = pd.Series(train_diff, index=pd.RangeIndex(len(train_diff)))
            val_series = pd.Series(val_diff, index=pd.RangeIndex(start=len(train_diff), stop=len(train_diff) + len(val_diff)))
            test_series  = pd.Series(test_diff, index=pd.RangeIndex(start=len(val_diff), stop=len(val_diff) + len(test_diff)))

            arma_model = ARIMA(train_series, order=(1,0,2))
            arma_fit = arma_model.fit()

            val_series_aligned = pd.Series(val_series.values,index=pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(val_series)))
            test_series_aligned = pd.Series(test_series.values,index=pd.RangeIndex(start=len(train_series) + len(val_series), stop=len(train_series) + len(val_series) + len(test_series)))

            train_resid_full = arma_fit.resid.values
            updated_model_val = arma_fit.append(val_series_aligned)

            val_pred = updated_model_val.predict(start=len(train_series),end=len(train_series) + len(val_series) - 1)
            val_resid_full = val_data['log_diff_volume'].values - val_pred.values

            updated_model_test = updated_model_val.append(test_series_aligned)

            test_pred = updated_model_test.predict(start=len(train_series) + len(val_series),end=len(train_series) + len(val_series) + len(test_series) - 1)
            test_resid_full = test_data['log_diff_volume'].values - test_pred.values

            # Prepare LSTM input (prepend last sequence_length residuals from val)
            test_lstm_input = np.concatenate([val_resid_full[-hyperparams.sequence_length:], test_resid_full])

            # Call ResidualVolumeDataset we see above
            train_dataset = ResidualVolumeDataset(train_resid_full, hyperparams.sequence_length)
            val_dataset   = ResidualVolumeDataset(val_resid_full, hyperparams.sequence_length)
            test_dataset  = ResidualVolumeDataset(test_lstm_input, hyperparams.sequence_length)
            
            # ------------------- THE REST OF LSTM STEP HERE -------------------
            # CHANGED NUM_WORKERS=HYPERPARAMS.NWORKERS TO DROP_LAST; USING MAC
            train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, drop_last=False) 
            val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, drop_last=False) 
            test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False, drop_last=False)

            # Extract input shapes
            input_dim = train_dataset[0][0].shape[-1] 
            output_dim = train_dataset[0][1].shape[-1]
            
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
            train_metrics, train_eval = evaluate(model, train_loader, device, metrics, hyperparams)
            val_metrics, val_eval     = evaluate(model, val_loader, device, metrics, hyperparams)
            test_metrics, test_eval   = evaluate(model, test_loader, device, metrics, hyperparams)

            # ADDED TO SAVE
            state_dict_path = f'lstm_model_run{i}_state_dict.pt'
            torch.save(model.state_dict(), state_dict_path)
            print(f"Saved model state_dict to {state_dict_path}")

    # EDITED Print results
    print("\n[INFO] Test Metrics Summary:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")

# ------------------- CREATE CSV OF TEST FORECASTS -------------------
timestamps = test_data['timestamp'].values
test_diff  = test_data['log_diff_volume'].values
test_log = test_data['log_volume'].values
arma_test_pred = test_pred         

lstm_pred_residuals = []
model.eval()
with torch.no_grad():
    for X, _ in test_loader:
        pred = model(X.to(device))
        lstm_pred_residuals.append(pred.cpu().numpy())
lstm_pred_residuals = np.concatenate(lstm_pred_residuals).flatten()

# Slice prepended part (first sequence_length predictions are extra)
lstm_pred_residuals = lstm_pred_residuals[hyperparams.sequence_length:]

# ------------------- ALIGN LENGTHS -------------------
min_len = min(len(test_resid_full), len(test_diff), len(arma_test_pred), len(lstm_pred_residuals), len(timestamps))
test_diff = test_diff[:min_len]
arma_test_pred = arma_test_pred[:min_len]
lstm_pred_residuals = lstm_pred_residuals[:min_len]
timestamps = timestamps[:min_len]
arma_test_resid = test_resid_full[:min_len]
hybrid_forecast_diff = arma_test_pred + lstm_pred_residuals

# ------------------- BUILD DATAFRAME -------------------
df_hybrid = pd.DataFrame({
    "timestamp": timestamps,
    "test_diff": test_diff,                     # actual log differences
    "forecast_diff": arma_test_pred,             # ARMA only
    "arma_residuals": arma_test_resid,
    "lstm_pred_residual": lstm_pred_residuals, # LSTM residuals
    "hybrid_forecast_diff": hybrid_forecast_diff # final hybrid prediction (log diff)
})

# Save CSV
df_hybrid.to_csv('test_forecast_analysis.csv', index=False)
print(f"[INFO] CSV saved successfully to test_forecast_analysis.csv")

# ------------------- PLOT RESULTS -------------------
plt.figure(figsize=(12, 6))
plt.plot(range(len(test_diff)), test_diff, label='Actual Test Diff', color='blue')
plt.plot(range(len(test_diff)), arma_test_pred, label='ARMA Forecast', color='red', linestyle='--')
plt.title('ARMA Forecast vs Actual Test')
plt.xlabel('Time')
plt.ylabel('Log Diff Volume')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(len(test_diff)), arma_test_resid, label='Actual Residuals', color='blue')
plt.plot(range(len(test_diff)), lstm_pred_residuals, label='LSTM Predicted Residuals', color='green', linestyle='--')
plt.title('LSTM Predicted Residuals vs Actual Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(len(test_diff)), test_diff, label='Actual Test Diff', color='blue')
plt.plot(range(len(test_diff)), hybrid_forecast_diff, label='Hybrid Forecast', color='purple', linestyle='--')
plt.title('Hybrid Forecast vs Actual Test')
plt.xlabel('Time')
plt.ylabel('Log Diff Volume')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Compute errors
arma_mae = mean_absolute_error(test_diff, arma_test_pred)
arma_mse = mean_squared_error(test_diff, arma_test_pred)

lstm_mae = mean_absolute_error(arma_test_resid, lstm_pred_residuals)
lstm_mse = mean_squared_error(arma_test_resid, lstm_pred_residuals)

hybrid_mae = mean_absolute_error(test_diff, hybrid_forecast_diff)
hybrid_mse = mean_squared_error(test_diff, hybrid_forecast_diff)

# Print results
print("\n[INFO] Test Set Errors:")
print(f"ARMA      - MAE: {arma_mae:.6f}, MSE: {arma_mse:.6f}")
print(f"LSTM      - MAE: {lstm_mae:.6f}, MSE: {lstm_mse:.6f}")
print(f"Hybrid    - MAE: {hybrid_mae:.6f}, MSE: {hybrid_mse:.6f}")
