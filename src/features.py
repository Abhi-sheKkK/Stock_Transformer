import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .cache import fetch_stock_data

# Global tickers list and sector mapping
TICKERS = [
    # 1. Technology & Semiconductors
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'AMD', 'CRM',
    # 2. Communication Services & Digital Media
    'GOOGL', 'META', 'NFLX', 'DIS', 'TMUS', 'CMCSA',
    # 3. Financials (Banking, Payments & Market Makers)
    'JPM', 'BAC', 'MS', 'GS', 'V', 'MA', 'AXP',
    # 4. Healthcare (Pharma, Devices & Insurance)
    'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ISRG',
    # 5. Consumer Discretionary & Staples
    'AMZN', 'TSLA', 'WMT', 'COST', 'HD', 'NKE', 'KO',
    # 6. Industrials & Energy
    'XOM', 'CVX', 'CAT', 'GE', 'UNP', 'HON', 'ETN'
]

TICKER_TO_SECTOR = {}
# Tech (Sector 0)
for t in ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'AMD', 'CRM']:
    TICKER_TO_SECTOR[t] = 0
# Comm (Sector 1)
for t in ['GOOGL', 'META', 'NFLX', 'DIS', 'TMUS', 'CMCSA']:
    TICKER_TO_SECTOR[t] = 1
# Fin (Sector 2)
for t in ['JPM', 'BAC', 'MS', 'GS', 'V', 'MA', 'AXP']:
    TICKER_TO_SECTOR[t] = 2
# Health (Sector 3)
for t in ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ISRG']:
    TICKER_TO_SECTOR[t] = 3
# Consumer (Sector 4)
for t in ['AMZN', 'TSLA', 'WMT', 'COST', 'HD', 'NKE', 'KO']:
    TICKER_TO_SECTOR[t] = 4
# Ind/Energy (Sector 5)
for t in ['XOM', 'CVX', 'CAT', 'GE', 'UNP', 'HON', 'ETN']:
    TICKER_TO_SECTOR[t] = 5


class IdentityScaler:
    """Scaler that returns the input data unchanged (used for raw cyclical temporal features)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X
    def inverse_transform(self, X):
        return X


class Scale100Scaler:
    """Scaler that multiplies inputs by 100.0 (used to scale log return targets)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X * 100.0
    def fit_transform(self, X, y=None):
        return X * 100.0
    def inverse_transform(self, X):
        return X / 100.0


def get_market_snapshot(ticker: str) -> dict:
    """
    Generate a market snapshot with current technical indicators.
    Returns dict with interpreted signals and a summary string for LLM consumption.
    """
    data = fetch_stock_data(ticker, period='6mo', ttl_seconds=900)

    if data.empty:
        return {"error": f"No data found for {ticker}", "ticker": ticker}

    latest = data.iloc[-1]
    prev = data.iloc[-2]
    price = float(latest["Close"])
    price_change = price - float(prev["Close"])
    price_change_pct = (price_change / float(prev["Close"])) * 100

    # Detect currency
    currency = "₹" if any(s in ticker.upper() for s in [".NS", ".BO"]) else "$"

    # --- Technical Indicators ---
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd = float(macd_line.iloc[-1])
    signal = float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])
    histogram = macd - signal

    bb_ma = float(data["Close"].rolling(20).mean().iloc[-1])
    bb_std = float(data["Close"].rolling(20).std().iloc[-1])
    bb_upper = bb_ma + (bb_std * 2)
    bb_lower = bb_ma - (bb_std * 2)
    bb_width = bb_upper - bb_lower
    bb_position = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

    hl = data["High"] - data["Low"]
    hc = np.abs(data["High"] - data["Close"].shift())
    lc = np.abs(data["Low"] - data["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = float((tp * data["Volume"]).rolling(14).sum().iloc[-1] / data["Volume"].rolling(14).sum().iloc[-1])

    avg_vol = float(data["Volume"].rolling(20).mean().iloc[-1])
    cur_vol = float(latest["Volume"])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

    # --- Signal Interpretation ---
    def _rsi_sig(v):
        if v > 70: return "Overbought"
        if v > 60: return "Mildly Overbought"
        if v < 30: return "Oversold"
        if v < 40: return "Mildly Oversold"
        return "Neutral"

    def _macd_sig(m, h):
        d = "Bullish" if m > 0 else "Bearish"
        mom = "Strengthening" if h > 0 else "Weakening"
        return f"{d}, {mom}"

    def _bb_sig(pos):
        if pos > 0.95: return "Near Upper Band (Overbought)"
        if pos > 0.8: return "Upper Region"
        if pos < 0.05: return "Near Lower Band (Oversold)"
        if pos < 0.2: return "Lower Region"
        return "Middle Band (Neutral)"

    def _vol_sig(r):
        if r > 2.0: return "Very High Volume"
        if r > 1.5: return "Above Average"
        if r < 0.5: return "Very Low Volume"
        if r < 0.75: return "Below Average"
        return "Normal Volume"

    signals = {
        "rsi": _rsi_sig(rsi),
        "macd": _macd_sig(macd, histogram),
        "bollinger": _bb_sig(bb_position),
        "volume": _vol_sig(vol_ratio),
        "trend": "Uptrend" if price > bb_ma else "Downtrend",
        "vwap_position": "Above VWAP (Bullish)" if price > vwap else "Below VWAP (Bearish)",
    }

    indicators = {
        "price": round(price, 2),
        "price_change": round(price_change, 2),
        "price_change_pct": round(price_change_pct, 2),
        "rsi": round(rsi, 1),
        "macd": round(macd, 4),
        "macd_signal": round(signal, 4),
        "macd_histogram": round(histogram, 4),
        "bb_upper": round(bb_upper, 2),
        "bb_middle": round(bb_ma, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_position": round(bb_position, 3),
        "atr": round(atr, 2),
        "vwap": round(vwap, 2),
        "volume": int(cur_vol),
        "avg_volume": int(avg_vol),
        "volume_ratio": round(vol_ratio, 2),
    }

    summary = "\n".join([
        f"=== Market Snapshot: {ticker.upper()} ===",
        f"Price: {currency}{indicators['price']} ({indicators['price_change_pct']:+.2f}%)",
        f"RSI (14): {indicators['rsi']} — {signals['rsi']}",
        f"MACD: {indicators['macd']:.4f} (Signal: {indicators['macd_signal']:.4f}, Hist: {indicators['macd_histogram']:.4f}) — {signals['macd']}",
        f"Bollinger: Lower {currency}{indicators['bb_lower']} | Mid {currency}{indicators['bb_middle']} | Upper {currency}{indicators['bb_upper']} — {signals['bollinger']}",
        f"ATR (14): {currency}{indicators['atr']}",
        f"VWAP: {currency}{indicators['vwap']} — {signals['vwap_position']}",
        f"Volume: {indicators['volume']:,} (avg {indicators['avg_volume']:,}) — {signals['volume']}",
        f"Trend: {signals['trend']}",
    ])

    return {
        "ticker": ticker.upper(),
        "currency": currency,
        "indicators": indicators,
        "signals": signals,
        "summary": summary,
    }


def create_input(stock_name, scalers_path=None):
    """
    Create scale-agnostic relative input features from raw stock data.
    """
    # 1. Fetch raw stock history
    stock_data = fetch_stock_data(stock_name, period='max', ttl_seconds=14400)
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {stock_name}")
        
    # 2. Fetch GSPC/SPY for market benchmark
    market_data = None
    try:
        market_data = fetch_stock_data('^GSPC', period='max', ttl_seconds=14400)
    except Exception as e:
        print(f"Warning: Could not fetch ^GSPC with period='max' ({e}). Retrying with period='10y'.")
        try:
            market_data = fetch_stock_data('^GSPC', period='10y', ttl_seconds=14400)
        except Exception as e2:
            print(f"Warning: Could not fetch ^GSPC with period='10y' ({e2}). Falling back to zero market returns.")
            
    if market_data is not None and not market_data.empty:
        market_data['spy_log_return'] = np.log(market_data['Close'] / market_data['Close'].shift(1))
    else:
        market_data = pd.DataFrame(index=stock_data.index)
        market_data['spy_log_return'] = 0.0
        
    # Join SPY/GSPC returns and align dates
    stock_data = stock_data.join(market_data[['spy_log_return']], how='left')
    stock_data['spy_log_return'] = stock_data['spy_log_return'].fillna(0.0)
    
    # 2b. Fetch VIX data for market volatility environment
    vix_data = None
    try:
        vix_data = fetch_stock_data('^VIX', period='max', ttl_seconds=14400)
    except Exception as e:
        print(f"Warning: Could not fetch ^VIX with period='max' ({e}). Retrying with period='10y'.")
        try:
            vix_data = fetch_stock_data('^VIX', period='10y', ttl_seconds=14400)
        except Exception as e2:
            print(f"Warning: Could not fetch ^VIX with period='10y' ({e2}). Falling back to default VIX.")
            
    if vix_data is not None and not vix_data.empty:
        vix_close = vix_data['Close']
        vix_rm20 = vix_close.rolling(20).mean().replace(0, 1e-8)
        vix_relative_height = vix_close / vix_rm20
        vix_df = pd.DataFrame(index=vix_data.index)
        vix_df['vix_relative_height'] = vix_relative_height
    else:
        vix_df = pd.DataFrame(index=stock_data.index)
        vix_df['vix_relative_height'] = 1.0
        
    stock_data = stock_data.join(vix_df[['vix_relative_height']], how='left')
    stock_data['vix_relative_height'] = stock_data['vix_relative_height'].fillna(1.0)
    
    # 3. Base calculations on raw prices
    close = stock_data['Close']
    open_p = stock_data['Open']
    high = stock_data['High']
    low = stock_data['Low']
    volume = stock_data['Volume']
    
    # 4. Construct stationary features DataFrame
    features_df = pd.DataFrame(index=stock_data.index)
    
    # --- Baseline Stationary Returns ---
    features_df['close_log_return'] = np.log(close / close.shift(1))
    features_df['open_log_return'] = np.log(open_p / close.shift(1))
    features_df['high_log_return'] = np.log(high / close.shift(1))
    features_df['low_log_return'] = np.log(low / close.shift(1))
    
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, 1e-8)
    features_df['volume_z_score'] = (volume - vol_mean_20) / vol_std_20
    
    # --- Volatility Breakout Bundle ---
    bb_mean_20 = close.rolling(window=20).mean()
    bb_std_20 = close.rolling(window=20).std()
    bb_upper_20 = bb_mean_20 + (bb_std_20 * 2)
    bb_lower_20 = bb_mean_20 - (bb_std_20 * 2)
    features_df['bb_band_squeeze'] = bb_mean_20 / (bb_upper_20 - bb_lower_20 + 1e-8)
    features_df['dist_channel_high'] = np.log(high.rolling(20).max() / close)
    features_df['volume_force_multiplier'] = features_df['close_log_return'] * features_df['volume_z_score']
    
    # --- Trend Pullback Bundle ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    features_df['trend_alignment_ratio'] = np.log(ema50 / ema20.replace(0, 1e-8))
    features_df['pullback_proximity'] = np.log(ema20 / close)
    features_df['macro_beta_shield'] = features_df['close_log_return'] - stock_data['spy_log_return']
    
    # --- Mean Reversion Bundle ---
    features_df['price_z_score_distance'] = (close - bb_mean_20) / bb_std_20.replace(0, 1e-8)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-8)
    rsi = 100 - (100 / (1 + rs))
    features_df['rsi_asymmetry'] = (rsi - 50) / 100.0
    
    # --- Macro Filters ---
    features_df['spy_log_return'] = stock_data['spy_log_return']
    features_df['vix_relative_height'] = stock_data['vix_relative_height']
    
    # --- Cyclical Time Encodings ---
    day_of_week = features_df.index.dayofweek.values
    day_of_week = np.clip(day_of_week, 0, 4)  # Clip to Mon-Fri trading days
    features_df['day_sin'] = np.sin(2. * np.pi * day_of_week / 5.0)
    features_df['day_cos'] = np.cos(2. * np.pi * day_of_week / 5.0)
    
    month_of_year = features_df.index.month.values
    features_df['month_sin'] = np.sin(2. * np.pi * month_of_year / 12.0)
    features_df['month_cos'] = np.cos(2. * np.pi * month_of_year / 12.0)
    
    # Drop rows with NaN resulting from rolling windows/shifts
    features_df.dropna(inplace=True)
    
    # 5. Extract category mappings
    stock_name_upper = stock_name.upper()
    if stock_name_upper in TICKERS:
        stock_id = TICKERS.index(stock_name_upper)
        sector_id = TICKER_TO_SECTOR[stock_name_upper]
    else:
        stock_id = 0
        sector_id = 0
        
    features_df['Stock_ID'] = stock_id
    features_df['Sector_ID'] = sector_id
    
    # Target: Close Log Return scaled by 100
    close_price = features_df['close_log_return'].values.reshape(-1, 1)
    
    # Split features into numerical and cyclical / categorical
    numerical_cols = [
        "close_log_return", "open_log_return", "high_log_return", "low_log_return",
        "volume_z_score", "bb_band_squeeze", "dist_channel_high", "volume_force_multiplier",
        "trend_alignment_ratio", "pullback_proximity", "macro_beta_shield",
        "price_z_score_distance", "rsi_asymmetry", "spy_log_return", "vix_relative_height"
    ]
    numerical_features = features_df[numerical_cols]
    
    # Fit / Transform numerical features
    loaded = False
    if scalers_path is not None:
        scalers_path = Path(scalers_path)
        if (scalers_path / 'feature_scaler.joblib').exists():
            feature_scaler = joblib.load(scalers_path / 'feature_scaler.joblib')
            time_scaler = joblib.load(scalers_path / 'time_scaler.joblib')
            close_scaler = joblib.load(scalers_path / 'close_scaler.joblib')
            loaded = True
            
    if not loaded:
        feature_scaler = StandardScaler()
        time_scaler = IdentityScaler()
        close_scaler = Scale100Scaler()
        
    if loaded:
        scaled_numerical = pd.DataFrame(feature_scaler.transform(numerical_features), index=numerical_features.index, columns=numerical_cols)
        scaled_close = close_scaler.transform(close_price)
    else:
        scaled_numerical = pd.DataFrame(feature_scaler.fit_transform(numerical_features), index=numerical_features.index, columns=numerical_cols)
        scaled_close = close_scaler.fit_transform(close_price)
        
    # Reconstruct input features DataFrame
    # Concatenate: [Scaled Numerical (15), Cyclical (4), Stock_ID (1), Sector_ID (1)]
    cyclical_df = features_df[['day_sin', 'day_cos', 'month_sin', 'month_cos']]
    cats_df = features_df[['Stock_ID', 'Sector_ID']]
    
    input_features = pd.concat([scaled_numerical, cyclical_df, cats_df], axis=1)
    input_features['Target_Log_Return'] = scaled_close
    
    return input_features, feature_scaler, time_scaler, close_scaler, scaled_close


def save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models'):
    """
    Persist fitted scalers to disk.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    joblib.dump(feature_scaler, save_dir / 'feature_scaler.joblib')
    joblib.dump(time_scaler, save_dir / 'time_scaler.joblib')
    joblib.dump(close_scaler, save_dir / 'close_scaler.joblib')
    print(f"Scalers saved to {save_dir}/")


def load_scalers(load_dir='models'):
    """
    Load previously saved scalers.
    """
    load_dir = Path(load_dir)
    required = ['feature_scaler.joblib', 'time_scaler.joblib', 'close_scaler.joblib']
    for f in required:
        if not (load_dir / f).exists():
            raise FileNotFoundError(
                f"Scaler file '{f}' not found in {load_dir}. "
                f"Train the model first to generate scalers."
            )
    return (
        joblib.load(load_dir / 'feature_scaler.joblib'),
        joblib.load(load_dir / 'time_scaler.joblib'),
        joblib.load(load_dir / 'close_scaler.joblib'),
    )
