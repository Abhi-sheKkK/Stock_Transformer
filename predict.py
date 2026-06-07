import argparse
import torch
import numpy as np
import datetime
import warnings

from src.features import create_input
from src.model import StockTransformer
from src.cache import fetch_stock_data

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Predict Next 5 Days Stock Prices")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--seq_length', type=int, default=60, help='Lookback window')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--stock_embed_dim', type=int, default=32, help='Stock embedding dimension')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fetching and preparing data for {args.ticker}...")

    try:
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(
            args.ticker, scalers_path='models'
        )
        print("Using persisted scalers from training.")
    except Exception as e:
        print(f"Warning: Could not load persisted scalers ({e}), refitting from scratch.")
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(args.ticker)

    if len(input_features) < args.seq_length:
        print(f"Not enough data for {args.ticker}. Minimum {args.seq_length} days required.")
        return

    # Prepare continuous features (drop Stock_ID, Sector_ID, Target_Log_Return)
    stock_id = int(input_features['Stock_ID'].iloc[-1])
    sector_id = int(input_features['Sector_ID'].iloc[-1])
    continuous_cols = [c for c in input_features.columns if c not in ('Stock_ID', 'Sector_ID', 'Target_Log_Return')]
    x_cont = input_features[continuous_cols].iloc[-args.seq_length:].values

    x_temporal = torch.tensor(x_cont, dtype=torch.float32).unsqueeze(0).to(device)
    stock_id_t = torch.tensor([stock_id], dtype=torch.long).to(device)
    sector_id_t = torch.tensor([sector_id], dtype=torch.long).to(device)

    num_continuous = len(continuous_cols)

    model = StockTransformer(
        num_continuous_features=num_continuous,
        d_model=args.d_model, nhead=4, num_layers=4, dropout=0.25,
        prediction_horizon=args.horizon,
        max_seq_len=args.seq_length + 10,
        stock_embed_dim=args.stock_embed_dim,
    )

    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    except FileNotFoundError:
        print("Model file 'best_model.pth' not found. Please train the model first.")
        return

    model.to(device)
    model.eval()

    # Single forward pass → all 5 days predicted at once
    with torch.no_grad():
        predictions = model(x_temporal, stock_id_t, sector_id_t)  # [1, horizon]

    pred_log_returns = close_scaler.inverse_transform(
        predictions.cpu().numpy().reshape(-1, 1)
    ).flatten()

    # Fetch last actual close to compound into dollar prices
    raw_data = fetch_stock_data(args.ticker, period='6mo', ttl_seconds=14400)
    last_raw_close = float(raw_data['Close'].iloc[-1])

    predicted_prices = []
    price = last_raw_close
    for r in pred_log_returns:
        price = price * np.exp(r)
        predicted_prices.append(price)

    print("\n" + "=" * 40)
    print(f"  Expected Next {args.horizon} Days: {args.ticker}")
    print("=" * 40)

    current_date = datetime.datetime.today().date()
    idx = 0
    while idx < args.horizon:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
        print(f" Day {idx+1} ({current_date.strftime('%Y-%m-%d')}): ${predicted_prices[idx]:.2f}")
        idx += 1

    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
