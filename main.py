import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from src.features import create_input, TICKERS, save_scalers
from src.data import GlobalStockDataset
from src.model import StockTransformer
from src.train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Stock Global Transformer (Decoder-Only)")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=60, help='Lookback window (60 days)')
    parser.add_argument('--out_seq_len', type=int, default=5, help='Prediction horizon (5 days)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--stock_embed_dim', type=int, default=32, help='Stock embedding dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    args = parser.parse_args()

    # Dynamic year splits — automatically shift forward as time passes
    current_year = datetime.now().year
    val_year = current_year - 1
    train_cutoff = val_year - 1  # Train on everything up to and including this year

    print(f"Preparing global multi-stock dataset for {len(TICKERS)} tickers...")
    print(f"  Split: Train ≤ {train_cutoff} | Val = {val_year} | Test ≥ {current_year}")

    train_dfs, val_dfs, test_dfs = [], [], []

    # Baseline global scalers fit on AAPL
    print("Fitting baseline global scalers on AAPL...")
    base_input, feature_scaler, time_scaler, close_scaler, _ = create_input('AAPL')
    save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models')

    for ticker in TICKERS:
        try:
            df, _, _, _, _ = create_input(ticker, scalers_path='models')

            train_part = df[df.index.year <= train_cutoff]
            val_part = df[df.index.year == val_year]
            test_part = df[df.index.year >= current_year]

            if len(train_part) > 0:
                train_dfs.append(train_part)
            if len(val_part) > 0:
                val_dfs.append(val_part)
            if len(test_part) > 0:
                test_dfs.append(test_part)

            print(f"  {ticker}: train={len(train_part)}, val={len(val_part)}, test={len(test_part)}")
        except Exception as e:
            print(f"  Warning: {ticker} failed ({e})")

    if not train_dfs:
        print("Error: No training data loaded.")
        return

    print("\nCreating sliding-window datasets...")
    train_dataset = GlobalStockDataset(train_dfs, seq_length=args.seq_length, out_seq_len=args.out_seq_len)
    val_dataset = GlobalStockDataset(val_dfs, seq_length=args.seq_length, out_seq_len=args.out_seq_len)
    test_dataset = GlobalStockDataset(test_dfs, seq_length=args.seq_length, out_seq_len=args.out_seq_len)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation ({val_year}) samples: {len(val_dataset):,}")
    print(f"Test ({current_year}+) samples: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Derive num_continuous from the dataset (columns minus Stock_ID, Sector_ID, and Target_Log_Return)
    num_continuous = train_dfs[0].shape[1] - 3

    model = StockTransformer(
        num_continuous_features=num_continuous,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        prediction_horizon=args.out_seq_len,
        max_seq_len=args.seq_length + 10,
        stock_embed_dim=args.stock_embed_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Decoder-Only Causal Transformer")
    print(f"  d_model={args.d_model}, heads={args.nhead}, layers={args.num_layers}, dropout={args.dropout}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Params/samples ratio: {total_params/len(train_dataset):.4f}")

    print("\nStarting training...")
    preds, targets, idx, train_losses, val_losses = train_and_evaluate(
        model, train_loader, val_loader, close_scaler,
        num_epochs=args.epochs, learning_rate=args.lr,
        feature_scaler=feature_scaler, time_scaler=time_scaler
    )

    from src.visualization import plot_results
    plot_results(train_losses, val_losses, targets, preds, idx, save_dir="results_global")
    print("Training finished. Model saved as best_model.pth.")

if __name__ == "__main__":
    main()