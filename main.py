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

from pipeline.model import LSTMModel
from pipeline.data import VolumeDataset
from pipeline.utils import Hyperparameter, set_seed

# ADDED: CONVERT ARMA RESIDUALS TO SEQ FOR LSTM TARGET; KEEP ONLY SEQ OF LN (SEQ_LEN)
def create_lstm_targets_from_residuals(resid, seq_len):
    resid = np.array(resid).flatten() # just ensure 1D list 
    x_resid, y_resid = [], []

    for i in range(len(resid) - seq_len): # sliding window, create seq
        seq = resid[i:i+seq_len]
        x_resid.append(seq) # the seq
        y_resid.append(resid[i+seq_len]) # the next value after the seq

    # need 3D so add #of features = 1
    x_resid = torch.tensor(np.array(x_resid), dtype=torch.float32).unsqueeze(-1)
    y_resid = torch.tensor(np.array(y_resid), dtype=torch.float32).unsqueeze(-1)
    return x_resid, y_resid

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
        epochs = 100,
        
        # Set-up
        nworkers = 1,
        nruns = 5,
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
            train_dataset = VolumeDataset(dt.datetime(2023,1,1), dt.datetime(2023,12,31), hyperparams.sequence_length)
            val_dataset = VolumeDataset(dt.datetime(2024,1,1), dt.datetime(2024,6,30), hyperparams.sequence_length)
            test_dataset = VolumeDataset(dt.datetime(2024,7,1), dt.datetime(2024,12,31), hyperparams.sequence_length)

            # ------------------- ARMA FORMALITIES HERE -------------------
            train_series = pd.Series(train_dataset.targets.numpy().flatten()) # NUMPY -> 1D -> PANDAS SERIES

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

            train_x_resid, train_y_resid = create_lstm_targets_from_residuals(arma_fit.resid.values, hyperparams.sequence_length) 
            # make sequences with this length; function returns x and Y

            # SET UP TRAINING SET FOR LSTM
            train_dataset.targets = train_y_resid # we want to predict the next residual

            # FIT THEN GET RESIDUALS TO SERVE AS VAL DATA FOR LSTM
            val_series = val_dataset.targets.numpy().flatten()
            val_pred = arma_fit.predict(start=len(train_dataset.targets), end=len(train_dataset.targets)+len(val_series)-1)
            _, val_y_resid = create_lstm_targets_from_residuals(val_series - val_pred, hyperparams.sequence_length)
            # ignore the x_resid returned by create_lstm_targets
            val_dataset.features = val_dataset.features[:val_y_resid.shape[0]]
            val_dataset.targets = val_y_resid

            # FIT THEN GET RESIDUALS TO SERVE AS TEST DATA FOR LSTM
            test_series = test_dataset.targets.numpy().flatten()
            test_pred = arma_fit.predict(start=len(train_dataset.targets) + len(val_series), end=len(train_dataset.targets)+len(val_series)+len(test_series)-1)
            _, test_y_resid = create_lstm_targets_from_residuals(test_series - test_pred, hyperparams.sequence_length)
            test_dataset.features = test_dataset.features[:test_y_resid.shape[0]]
            test_dataset.targets = test_y_resid

            # SAVE THE DATA
            np.save('train_series.npy', train_dataset.targets.numpy().flatten())
            np.save('val_series.npy', val_dataset.targets.numpy().flatten())
            np.save('test_series.npy', test_dataset.targets.numpy().flatten())

            np.save('train_arma_fitted.npy', arma_fit.fittedvalues.values)
            np.save('val_arma_pred.npy', val_pred)
            np.save('test_arma_pred.npy', test_pred)

            np.save('train_resid.npy', arma_fit.resid.values)        # full ARMA residuals
            np.save('train_resid_lstm.npy', train_y_resid.numpy().flatten())  # residuals as LSTM targets
            np.save('val_resid_lstm.npy', val_y_resid.numpy().flatten())
            np.save('test_resid_lstm.npy', test_y_resid.numpy().flatten())

            # ------------------- THE REST OF LSTM STEP HERE -------------------

            # CHANGED NUM_WORKERS=HYPERPARAMS.NWORKERS TO DROP_LAST; USING MAC
            train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, drop_last=False) 
            # last batch has to be of the same len for training stability
            val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, drop_last=False) 
            # ok even if last batch is smaller
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
