# This file contains training-related functions
import torch
import mlflow
import numpy as np
from torch import nn
import torch.nn.functional as F

# REMOVED SCALER
def train(model, train_loader, device, loss_fn, optimizer, hyperparams, **kwargs):
    model.train()
    
    total_loss, total = 0.0, 0
    for features, targets in train_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device
        
        optimizer.zero_grad()

        predictions = model(features)     # take last timestep
        loss = loss_fn(targets, predictions)  # Compute loss
        
        loss.backward()                     # Backprop
        optimizer.step()                    # Update weights

        total += targets.shape[0]
        total_loss += loss.item() * targets.shape[0]
        
    return total_loss / total

# REMOVED AMP, GRADSCALER
@torch.no_grad()
def evaluate(model, data_loader, device, metrics, hyperparams, **kwargs):
    model.eval()
   
    total = 0
    total_metrics = {metric: 0 for metric in metrics.keys()}
    targets_all, predictions_all = [], []
    for features, targets in data_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device

        with torch.amp.autocast(device_type=device.type, enabled=hyperparams.use_amp):
            predictions = model(features)    # Compute model predictions

        # Compute metrics
        total = total + targets.shape[0]
        for metric, metric_fn in metrics.items():
            total_metrics[metric] = total_metrics[metric] + metric_fn(targets, predictions).item() * targets.shape[0]
    
        # Save targets and predictions
        targets_all.append(targets)
        predictions_all.append(predictions)
    
    # Average metrics
    for metric in metrics.keys():
        total_metrics[metric] = total_metrics[metric] / total
    
    # Concatenate targets and predictions
    targets_all = torch.cat(targets_all).cpu().numpy()
    predictions_all = torch.cat(predictions_all).cpu().numpy()

    return total_metrics, {'targets': targets_all, 'predictions': predictions_all}


def run(model, train_loader, val_loader, test_loader, device, loss_fn, metrics, hyperparams, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=hyperparams.factor, patience=hyperparams.patience, min_lr=hyperparams.min_lr
    )

    # ADDED
    training_loss_list = []
    val_loss_list = []

    for epoch in range(1, hyperparams.epochs + 1):
        loss = train(model, train_loader, device, loss_fn, optimizer, hyperparams, **kwargs)  # no scaler
        val_metrics, _ = evaluate(model, val_loader, device, metrics, hyperparams, **kwargs)
        test_metrics, _ = evaluate(model, test_loader, device, metrics, hyperparams, **kwargs)
        scheduler.step(loss)

        # ADDED
        training_loss_list.append(loss)
        val_loss_list.append(val_metrics['mse'])  # or any metric you want for U-shape

        # MLFlow logging
        current_metrics = {f'val_{metric}': value for metric, value in val_metrics.items()} | \
                        {f'test_{metric}': value for metric, value in test_metrics.items()}
        mlflow.log_metrics(current_metrics, step=epoch)

        # Always print every epoch
        print(f'Epoch {epoch:04d} | loss: {loss:.6f} | ' +
            ' | '.join([f'val_{metric}: {value:.4f}' for metric, value in val_metrics.items()]) +
            ' | ' +
            ' | '.join([f'test_{metric}: {value:.4f}' for metric, value in test_metrics.items()]))

    return test_metrics, training_loss_list, val_loss_list

