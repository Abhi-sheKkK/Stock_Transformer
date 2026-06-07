import numpy as np
import torch
from torch.utils.data import Dataset

class GlobalStockDataset(Dataset):
    def __init__(self, data_list, seq_length=60, out_seq_len=5):
        """
        Dataset that slices multi-stock dataframes into 3D lookback sequences.
        Each sample contains:
          - X_temporal: [seq_length, num_features] continuous normalized features
          - X_stock_id: scalar integer stock index
          - X_sector_id: scalar integer sector index
          - Y_target: [out_seq_len] Close Log Returns to predict
        """
        self.seq_length = seq_length
        self.out_seq_len = out_seq_len
        self.samples = []
        
        for df in data_list:
            if len(df) < (seq_length + out_seq_len):
                continue
                
            # Extract identifiers (constant for this stock dataframe)
            stock_id = int(df['Stock_ID'].iloc[0])
            sector_id = int(df['Sector_ID'].iloc[0])
            
            # Continuous features (excluding Stock_ID, Sector_ID, and Target_Log_Return columns)
            temporal_features = df.drop(columns=['Stock_ID', 'Sector_ID', 'Target_Log_Return']).values.astype(np.float32)
            
            # Target is the Close_Log_Return scaled by 100
            targets = df['Target_Log_Return'].values.astype(np.float32)
            
            for i in range(len(df) - seq_length - out_seq_len + 1):
                x_temp = temporal_features[i : i + seq_length]
                y_targ = targets[i + seq_length : i + seq_length + out_seq_len]
                
                self.samples.append({
                    'X_temporal': torch.tensor(x_temp, dtype=torch.float32),
                    'X_stock_id': torch.tensor(stock_id, dtype=torch.long),
                    'X_sector_id': torch.tensor(sector_id, dtype=torch.long),
                    'Y_target': torch.tensor(y_targ, dtype=torch.float32)
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def create_sequences(input_features, close_prices, in_seq_length=100, out_seq_len=5):
    x, y = [], []
    for i in range(len(input_features) - in_seq_length - out_seq_len):
        x.append(input_features.iloc[i:i+in_seq_length].values)
        y_seq = close_prices[i+in_seq_length:i+in_seq_length+out_seq_len]
        if isinstance(y_seq, np.ndarray) and y_seq.ndim > 1:
            y_seq = y_seq.flatten()
        y.append(y_seq)
    return np.array(x), np.array(y)

def pad_batch(batch, batch_size):
    current_size = batch.shape[0]
    if current_size < batch_size:
        padding = torch.zeros((batch_size - current_size, *batch.shape[1:])).to(batch.device)
        return torch.cat([batch, padding], dim=0)
    return batch
