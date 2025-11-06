# This file contains data-related classes and functions
import os
import numpy as np
import pandas as pd

class ARMAVolumeDataset:
    def __init__(self, start_date, end_date, raw_path='pipeline/dataset/raw/'):
        self.start_date = start_date
        self.end_date = end_date
        self.raw_path = raw_path
        
    def process_data(self):
        # Load all csv files
        raw_data = pd.concat([
            pd.read_csv(os.path.join(self.raw_path, f'spy_{month.strftime("%Y_%m")}.csv'), parse_dates=['date'], index_col=['date'])
            for month in pd.date_range(start='2022-09-30', end='2023-09-30', freq='ME')
        ])
        
        # Pad missing entries and forward fill (for simplicity)
        market_open = pd.to_datetime('09:30:00').time()
        market_close = pd.to_datetime('16:00:00').time()
        full_index = pd.date_range(start='2022-09-30', end='2023-09-30', freq='5min')
        mask = (
            pd.Series(full_index.date).isin(raw_data.index.date)
            & (pd.Series(full_index.time) >= market_open)
            & (pd.Series(full_index.time) <= market_close)
        )
        raw_data = raw_data.reindex(full_index[mask])
        missing_rows = raw_data[raw_data.isna().any(axis=1)]
                          
        for col in ['open', 'close', 'high', 'low', 'volume']:
            raw_data[col] = raw_data[col].ffill()

        # Perform feature engineering
        raw_data['log_diff_volume'] = np.log(raw_data['volume']) - np.log(raw_data['volume'].shift())
        raw_data['log_return'] = 100 * (np.log(raw_data['close']) - np.log(raw_data['close'].shift()))
        raw_data['mid_price'] = 0.5 * (raw_data['high'] + raw_data['low'])
        raw_data['relative_open'] = raw_data['open'] / raw_data['mid_price']
        raw_data['relative_high'] = raw_data['high'] / raw_data['mid_price']
        raw_data['relative_low'] = raw_data['low'] / raw_data['mid_price']
        raw_data['relative_close'] = raw_data['close'] / raw_data['mid_price']
        raw_data['log_volume'] = np.log(raw_data['volume'])

        # EDITED SINCE SHOWING 7/18 AS FINAL VAL DATE, not 7/19
        # Filter raw_data and select the datetime index, features, and targets
        processed_data = raw_data.dropna().loc[
            self.start_date : pd.to_datetime(self.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        ]

        # index = processed_data.index.values.astype(int)
        index = processed_data.index
        features = processed_data[['log_diff_volume', 'log_return', 'relative_open', 'relative_high', 'relative_low', 'relative_close']].values

        save_path = os.path.abspath('processed_arma_volume_dataset.csv')
        processed_data.to_csv(save_path)
        print(f"[INFO] Saved processed dataset to: {save_path}")

        return pd.DataFrame({
            'timestamp': processed_data.index,
            'volume': processed_data['volume'],
            'log_volume': processed_data['log_volume'],
            'log_diff_volume': processed_data['log_diff_volume'],
            'log_return': processed_data['log_return'],
            'relative_open': processed_data['relative_open'],
            'relative_high': processed_data['relative_high'],
            'relative_low': processed_data['relative_low'],
            'relative_close': processed_data['relative_close']
        })
    
# This file will get overwritten later by the test dataset when the ARMAVolumeDataset is called
# Created check that log_volume and log_diff_volume are correct
if __name__ == "__main__":
    dataset = ARMAVolumeDataset(start_date='2022-09-30', end_date='2023-09-30')
    processed = dataset.process_data()
