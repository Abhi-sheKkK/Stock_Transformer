import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.features import create_input
from src.data import create_sequences
from src.model import StockTransformer
from src.train import train_and_evaluate
from src.visualization import plot_results

def main():
    parser = argparse.ArgumentParser(description="Train Stock Transformer Model")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=100, help='Input sequence length')
    
    args = parser.parse_args()
    
    print(f"Starting pipeline for {args.ticker}...")
    input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(args.ticker)
    
    x, y = create_sequences(input_features, scaled_close, in_seq_length=args.seq_length)
    x_tensor, y_tensor = torch.tensor(x), torch.tensor(y)
    
    X_train, X_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, shuffle=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    
    input_dim = input_features.shape[1]
    print(f"Input dimensions: {input_dim}")
    
    model = StockTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        seq_length=args.seq_length
    )
    
    predictions, targets, idx, train_losses, val_losses = train_and_evaluate(
        model, train_loader, test_loader, close_scaler, 
        num_epochs=args.epochs, learning_rate=args.lr
    )
    
    # Save enhanced plots
    plot_results(train_losses, val_losses, targets, predictions, idx, save_dir="results")

if __name__ == "__main__":
    main()
