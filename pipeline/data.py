# This file contains data-related classes and functions
import os
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    def __init__(self, start_date, end_date, sequence_length, raw_path='pipeline/dataset/raw/', cache_path='pipeline/dataset/cached/', force_reload=False):
        super(VolumeDataset, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.raw_path = raw_path
        self.cache_path = cache_path
        
        # Unique identifier for current dataset instance
        cache_file = f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_s{sequence_length}.pkl'
        
        # Process raw data if cached file not available or force_reload
        if force_reload or not os.path.exists(os.path.join(self.cache_path, cache_file)):
            self.index, self.features, self.targets = self.process_data()
            os.makedirs(self.cache_path, exist_ok=True)
            with open(os.path.join(self.cache_path, cache_file), 'wb') as file:
                pickle.dump((self.index, self.features, self.targets), file)
                
        # Load cached file if available and not force_reload
        else:
            with open(os.path.join(self.cache_path, cache_file), 'rb') as file:
                self.index, self.features, self.targets = pickle.load(file) 
        
    # We want to predict the next 5m log diff volume based on the features in the past sequence_length 5m information
    def process_data(self):
        # Load all csv files
        raw_data = pd.concat([
            pd.read_csv(os.path.join(self.raw_path, f'spy_{month.strftime("%Y_%m")}.csv'), parse_dates=['date'], index_col=['date'])
            for month in pd.date_range(start='2020-01-01', end='2025-06-30', freq='ME')
        ])
        
        # Pad missing entries and forward fill (for simplicity)
        market_open = pd.to_datetime('09:30:00').time()
        market_close = pd.to_datetime('16:00:00').time()
        full_index = pd.date_range(start='2020-01-01', end='2025-06-30', freq='5min')
        mask = pd.Series(full_index.date).isin(raw_data.index.date) & (pd.Series(full_index.time) >= market_open) & (pd.Series(full_index.time) < market_close)
        raw_data = raw_data.reindex(full_index[mask])
        print(f'[INFO] raw_data contains {raw_data.isna().any(axis=1).sum()} missing observations')
        raw_data = raw_data.ffill()                                 # Can be more complex imputation methods
        raw_data.index = raw_data.index + pd.Timedelta(minutes=5)   # Index on closing datetime

        # Perform feature engineering
        raw_data['log_diff_volume'] = np.log(raw_data['volume']) - np.log(raw_data['volume'].shift())
        raw_data['log_return'] = 100 * (np.log(raw_data['close']) - np.log(raw_data['close'].shift()))
        raw_data['mid_price'] = 0.5 * (raw_data['high'] + raw_data['low'])
        raw_data['relative_open'] = raw_data['open'] / raw_data['mid_price']
        raw_data['relative_high'] = raw_data['high'] / raw_data['mid_price']
        raw_data['relative_low'] = raw_data['low'] / raw_data['mid_price']
        raw_data['relative_close'] = raw_data['close'] / raw_data['mid_price']
        # TODO: Add technical indicators, rolling quantiles, etc.

        # Filter raw_data and select the datetime index, features, and targets
        processed_data = raw_data.dropna().loc[self.start_date : self.end_date]
        index = processed_data.index.values.astype(int)
        features = processed_data[['log_diff_volume', 'log_return', 'relative_open', 'relative_high', 'relative_low', 'relative_close']].values
        targets = processed_data[['log_diff_volume']].values
        
        # Convert to torch.Tensor
        index = torch.from_numpy(index).long()
        features = torch.from_numpy(features).float()
        targets = torch.from_numpy(targets).float()
        
        return index, features, targets
    
    def __getitem__(self, i):
        return self.features[i : (i + self.sequence_length)], self.targets[(i + self.sequence_length)]

    def __len__(self):
        return self.targets.shape[0] - (self.sequence_length + 1)
    
    def get_index(self, i):
        return self.index[i : (i + self.sequence_length)], self.index[(i + self.sequence_length)]

    def index_to_datetime(self, index):
        return pd.to_datetime(index.numpy(), unit='ns')
    