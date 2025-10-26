import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pipeline.model import LSTMModel
from pipeline.data import VolumeDataset
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------- Device -------------------
device = torch.device('cpu')

# ------------------- Hyperparameters -------------------
sequence_length = 78
hidden_dim = 32
activation = torch.nn.ReLU()
dropout = 0
norm = torch.nn.BatchNorm1d
num_layers = 1
batch_size = 64

# ------------------- Load test dataset -------------------
test_dataset = VolumeDataset(pd.to_datetime('2024-07-01'), pd.to_datetime('2024-12-31'), sequence_length)

# ------------------- Load ARMA predictions and original targets -------------------
test_pred = np.load('test_arma_pred.npy')
test_series = np.load('test_series.npy')

# ------------------- Align lengths -------------------
min_len = min(len(test_series), len(test_pred))
test_series_aligned = test_series[:min_len]
test_pred_aligned = test_pred[:min_len]

# ------------------- Convert residuals to LSTM sequences -------------------
def create_lstm_targets_from_residuals(resid, seq_len):
    resid = np.array(resid).flatten()
    x_resid, y_resid = [], []
    for i in range(len(resid) - seq_len):
        seq = resid[i:i+seq_len]
        if len(seq) == seq_len:
            x_resid.append(seq)
            y_resid.append(resid[i+seq_len])
    x_resid = torch.tensor(np.array(x_resid), dtype=torch.float32).unsqueeze(-1)
    y_resid = torch.tensor(np.array(y_resid), dtype=torch.float32).unsqueeze(-1)
    return x_resid, y_resid

_, test_y_resid = create_lstm_targets_from_residuals(test_series_aligned - test_pred_aligned, sequence_length)

# ------------------- Align dataset features -------------------
test_dataset.features = test_dataset.features[:test_y_resid.shape[0]]
test_dataset.targets = test_y_resid

# ------------------- DataLoader -------------------
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------- Load LSTM model -------------------
state_dict = torch.load('lstm_model_run0_state_dict.pt', map_location=device)
state_dict_fixed = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

input_dim = test_dataset[0][0].shape[-1]
output_dim = test_dataset[0][1].shape[-1]

model = LSTMModel(input_dim, hidden_dim, output_dim, activation, dropout, norm, num_layers)
model.load_state_dict(state_dict_fixed)
model.to(device)
model.eval()

# ------------------- Prediction function -------------------
def compute_predictions(model, data_loader, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
    return np.array(preds)

# ------------------- LSTM residual predictions -------------------
test_preds = compute_predictions(model, test_loader, device)

# ------------------- Load original series -------------------
train_series = pd.Series(np.load('train_series.npy'))
val_series = pd.Series(np.load('val_series.npy'))
test_series = pd.Series(np.load('test_series.npy'))

# ------------------- Fit ARMA on training series -------------------
full_train = pd.concat([train_series, val_series]).reset_index(drop=True)
arma_model = ARIMA(full_train, order=(1,0,2))
arma_fit = arma_model.fit()

# ------------------- Rolling ARMA forecast using append() -------------------
test_series_aligned = pd.Series(test_series.values,
                                index=pd.RangeIndex(start=len(full_train),
                                                    stop=len(full_train)+len(test_series)))

updated_model = arma_fit.append(test_series_aligned)
rolling_arma_series = updated_model.predict(start=len(full_train), 
                                            end=len(full_train)+len(test_series)-1)
rolling_arma = rolling_arma_series.values


# ------------------- Align LSTM residuals -------------------
test_preds_full = np.zeros(len(test_series))
start_idx = sequence_length
end_idx = min(start_idx + len(test_preds), len(test_series))
test_preds_full[start_idx:end_idx] = test_preds[:end_idx-start_idx]

# ------------------- True residuals -------------------
true_residuals = test_series.values - rolling_arma

# ------------------- Hybrid predictions -------------------
test_hybrid = rolling_arma + test_preds_full

# ------------------- Convert arrays to Series for alignment -------------------
rolling_arma_series = pd.Series(rolling_arma, index=test_series.index)
test_preds_full_series = pd.Series(test_preds_full, index=test_series.index)
test_hybrid_series = pd.Series(test_hybrid, index=test_series.index)
true_residuals_series = pd.Series(true_residuals, index=test_series.index)

valid_range = slice(sequence_length, len(test_series))

# ------------------- ARMA Residuals Scatter Plot -------------------
plt.figure(figsize=(10,6))
plt.scatter(test_series.values, true_residuals, color='purple', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Actual Value')
plt.ylabel('Residual')
plt.title('ARMA Residuals Scatter Plot')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- Plot 1: ARMA vs Actual -------------------
plt.figure(figsize=(14,6))
plt.plot(test_series.values, label='Actual', color='blue', linewidth=2)
plt.plot(rolling_arma_series, label='ARMA Prediction', color='red', linestyle='--', linewidth=2)
plt.title('Test Series: Actual vs ARMA Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- Plot 2: LSTM Residuals vs True Residuals -------------------
plt.figure(figsize=(14,6))
plt.plot(true_residuals_series[valid_range], label='True Residuals', color='green', linewidth=2)
plt.plot(test_preds_full_series[valid_range], label='LSTM Predicted Residuals', color='lightgreen', linestyle='--', linewidth=2)
plt.title('Test Residuals: LSTM vs True (Aligned)')
plt.xlabel('Time Step')
plt.ylabel('Residual')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- Plot 3: Hybrid vs Actual -------------------
plt.figure(figsize=(14,6))
plt.plot(test_series[valid_range], label='Actual', color='blue', linewidth=2)
plt.plot(test_hybrid_series[valid_range], label='Hybrid (ARMA + LSTM)', color='green', linestyle='--', linewidth=2)
plt.title('Test Series: Actual vs Hybrid Prediction (Aligned)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- Metrics -------------------
mae_arma = mean_absolute_error(test_series, rolling_arma)
rmse_arma = np.sqrt(mean_squared_error(test_series, rolling_arma))

mae_lstm = mean_absolute_error(true_residuals, test_preds_full)
rmse_lstm = np.sqrt(mean_squared_error(true_residuals, test_preds_full))

mae_hybrid = mean_absolute_error(test_series, test_hybrid)
rmse_hybrid = np.sqrt(mean_squared_error(test_series, test_hybrid))

print("Test Metrics:")
print(f"ARMA vs Actual   -> MAE: {mae_arma:.6f}, RMSE: {rmse_arma:.6f}")
print(f"LSTM Residuals   -> MAE: {mae_lstm:.6f}, RMSE: {rmse_lstm:.6f}")
print(f"Hybrid vs Actual -> MAE: {mae_hybrid:.6f}, RMSE: {rmse_hybrid:.6f}")

