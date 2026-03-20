import argparse
import torch
import pandas as pd
import numpy as np
import datetime
import warnings

from src.features import create_input
from src.model import StockTransformer

# Suppress yfinance and pandas warnings for cleaner CLI output
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Predict Next 5 Days Stock Prices")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--seq_length', type=int, default=100, help='Input sequence length')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fetching and preparing data for {args.ticker}...")
    
    # We use the full pipeline to get the exact scaled representation for the recent data
    try:
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(args.ticker)
    except Exception as e:
        print(f"Error fetching data for {args.ticker}: {e}")
        return
        
    # Take the last `seq_length` days for inference
    if len(input_features) < args.seq_length:
        print(f"Not enough data for {args.ticker}. Minimum {args.seq_length} days required.")
        return
        
    src_data = input_features.iloc[-args.seq_length:].values
    src = torch.tensor(src_data, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_length, input_dim]
    
    # The start token for our decoder is the scaled close price of the very last known day
    last_close_val = scaled_close[-1][0]
    
    input_dim = input_features.shape[1]
    
    model = StockTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        seq_length=args.seq_length
    )
    
    # Load the best model weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    except FileNotFoundError:
        print("Model file 'best_model.pth' not found. Please train the model first.")
        return
        
    model.to(device)
    model.eval()
    
    # Autoregressive decoding for the next 5 days
    predictions = []
    
    # Initialize the target sequence with the last known scaled close price
    tgt_input = torch.zeros((1, 1, input_dim), device=device)
    tgt_input[:, 0, 0] = last_close_val
    
    with torch.no_grad():
        for _ in range(5):
            output = model(src, tgt_input)
            
            # The model outputs predictions for the sequence [batch, seq_len]
            # output[:, -1] is the prediction for the newest step
            next_pred = output[:, -1].item()
            predictions.append(next_pred)
            
            # Append this prediction to tgt_input for the next iteration step
            new_step = torch.zeros((1, 1, input_dim), device=device)
            new_step[:, 0, 0] = next_pred
            tgt_input = torch.cat([tgt_input, new_step], dim=1)
            
    # Inverse transform predictions back to original $ scale
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    predicted_prices = close_scaler.inverse_transform(predictions_scaled).flatten()
    
    print("\n" + "="*40)
    print(f"  Expected Next 5 Days Prices: {args.ticker}")
    print("="*40)
    
    current_date = datetime.datetime.today().date()
    days_added = 0
    predictions_idx = 0
    
    while predictions_idx < 5:
        # Increment day
        current_date += datetime.timedelta(days=1)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if current_date.weekday() >= 5:
            continue
            
        print(f" Day {predictions_idx+1} ({current_date.strftime('%Y-%m-%d')}): ${predicted_prices[predictions_idx]:.2f}")
        predictions_idx += 1
        
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
