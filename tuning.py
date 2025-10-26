import os
import ray
import torch
import mlflow
import tempfile
import numpy as np
from torch import nn
import datetime as dt
from torchinfo import summary
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.schedulers import ASHAScheduler

from pipeline.model import LSTMModel
from pipeline.data import VolumeDataset
from pipeline.train import train, evaluate
from pipeline.utils import Hyperparameter, set_seed


# Commonly used normalization layers
norm_layers = {
    'bn': nn.BatchNorm1d,
    'ln': nn.LayerNorm,
    'none': nn.Identity,
}


def run(config):
    # [MLFLow] Log hyperparameters
    tracking_uri = config.pop('tracking_uri', None)
    setup_mlflow(config, tracking_uri=tracking_uri)
    mlflow.log_params(config)
    
    # [MLFlow] Set tag to current run
    mlflow.set_tag('Training Info', 'Hyperparameter tuning')
    
    hyperparams = Hyperparameter(**config, nworkers=1, use_amp=True)
    hyperparams.norm = norm_layers[hyperparams.norm]
    hyperparams.activation = nn.ReLU() if hyperparams.activation == 'relu' else nn.Sigmoid()
    
    # Change directory
    os.chdir(base_path)
    
    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    # Set seed
    set_seed(0)

    # Load dataset    
    train_dataset = VolumeDataset(dt.datetime(2023,1,1), dt.datetime(2023,12,31), hyperparams.sequence_length)
    val_dataset = VolumeDataset(dt.datetime(2024,1,1), dt.datetime(2024,6,30), hyperparams.sequence_length)
    test_dataset = VolumeDataset(dt.datetime(2024,7,1), dt.datetime(2024,12,31), hyperparams.sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.nworkers)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)

    # Extract input shapes
    input_dim = train_dataset[0][0].shape[-1]
    output_dim = train_dataset[0][1].shape[-1]
    
    # Load model
    model = LSTMModel(input_dim, hyperparams.hidden_dim, output_dim, hyperparams.activation, 
                      hyperparams.dropout, hyperparams.norm, hyperparams.num_layers).to(device)
    model = torch.compile(model, mode='default')
    
    # [MLFlow] Log model summary
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_summary_path = os.path.join(tmp_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(model_summary_path)

    # Scaler + optimizer + scheduler
    scaler = torch.amp.GradScaler(device=device.type, enabled=hyperparams.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.wd)   # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams.factor, patience=hyperparams.patience, min_lr=hyperparams.min_lr)      # Scheduler for lr decay
    
    for epoch in range(1, hyperparams.epochs + 1):
        loss = train(model, train_loader, device, loss_fn, optimizer, scaler, hyperparams)
        val_metrics, _ = evaluate(model, val_loader, device, metrics, hyperparams)
        test_metrics, _ = evaluate(model, test_loader, device, metrics, hyperparams)
        scheduler.step(loss)
        
        # Checkpoint results
        checkpoint = {'loss': loss}
        checkpoint.update({f'val_{metric}': value for metric, value in val_metrics.items()})
        checkpoint.update({f'test_{metric}': value for metric, value in test_metrics.items()})
        mlflow.log_metrics(checkpoint, step=epoch)
        ray.train.report(checkpoint)

    # [MLFlow] Log model
    sample_features, sample_targets = train_dataset[0]
    sample_features, sample_targets = np.expand_dims(sample_features.numpy(), axis=0), np.expand_dims(sample_targets.numpy(), axis=0)
    signature = mlflow.models.infer_signature(sample_features, sample_targets)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name='model',
        signature=signature,
        code_paths=[os.path.join(base_path, 'pipeline')],
    )


if __name__ == '__main__':
    # Specify directories
    base_path = os.getcwd()
    directory = os.path.join(base_path, 'logs/')
    exp_name = f'Hyperparameter Tuning @ {dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}'
    
    # [MLFlow] Experiment set-up
    ## Run `mlflow server --host 127.0.0.1 --port 8080` (before running this script)
    mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
    mlflow.set_experiment(exp_name)
    
    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify hyperparameter space
    param_space = {
        'sequence_length': ray.tune.choice([78, 78 * 2]),
        'batch_size': ray.tune.choice([64, 128]),
        'hidden_dim': ray.tune.choice([16, 32]),
        'activation': ray.tune.choice(['relu', 'sigmoid']),
        'dropout': ray.tune.uniform(0, 1),
        'norm': ray.tune.choice(['bn', 'ln', 'none']),
        'num_layers': ray.tune.randint(1, 4),
        'lr': ray.tune.loguniform(1e-4, 1e-2),
        'wd': ray.tune.loguniform(1e-5, 1e-3),
        'min_lr': ray.tune.choice([1e-5]),
        'factor': ray.tune.quniform(0, 1, 0.1),
        'patience': ray.tune.choice([10]),
        'epochs': ray.tune.choice([100]),
        'tracking_uri': mlflow.get_tracking_uri(),
    }
    
    # Specify hyperparameters to evaluate
    hyperparams = [
    ]
    
    search_alg = BasicVariantGenerator(points_to_evaluate=hyperparams, max_concurrent=4)
    scheduler = None
    # search_alg = OptunaSearch(points_to_evaluate=hyperparams)
    # search_alg.restore_from_dir(os.path.join(directory, exp_name))
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=6)
    # scheduler = ASHAScheduler(max_t=50, grace_period=50)
    
    if ray.tune.Tuner.can_restore(os.path.join(directory, exp_name)):
        tuner = ray.tune.Tuner.restore(
            path=os.path.join(directory, exp_name), 
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            resume_unfinished=True,
        )
    else:
        tuner = ray.tune.Tuner(
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            tune_config=ray.tune.TuneConfig(mode='min', metric='val_mae', search_alg=search_alg, scheduler=scheduler, num_samples=100),
            run_config=ray.train.RunConfig(name=exp_name, storage_path=directory, failure_config=ray.train.FailureConfig(max_failures=2), 
                                           checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),
                                        #   [MLFlow] Alternative method for logging
                                        #    callbacks=[
                                        #        MLflowLoggerCallback(
                                        #            tracking_uri=None,   # [MLFlow] Include the tracking uri if available
                                        #            experiment_name=exp_name,
                                        #            save_artifact=True,
                                        #         )
                                        #        ],
                                           ),
            param_space=param_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result()

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial test mae: {best_result.metrics["test_mae"]}')
