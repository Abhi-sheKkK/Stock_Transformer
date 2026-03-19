import numpy as np

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
    import torch
    current_size = batch.shape[0]
    if current_size < batch_size:
        padding = torch.zeros((batch_size - current_size, *batch.shape[1:])).to(batch.device)
        return torch.cat([batch, padding], dim=0)
    return batch
