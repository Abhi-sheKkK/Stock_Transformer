"""
Local file cache for yfinance stock data.
Avoids hitting the API on every call during training/testing/development.

Cache files are stored as Parquet in .cache/stocks/{TICKER}.parquet
with a configurable TTL (default: 4 hours for training, 15 min for snapshots).
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf

_CACHE_DIR = Path('.cache/stocks')


def fetch_stock_data(ticker: str, period: str = 'max', ttl_seconds: int = 14400) -> pd.DataFrame:
    """
    Fetch stock history with local file caching.
    
    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        period: yfinance period string ('max', '6mo', '1y', etc.)
        ttl_seconds: Cache validity in seconds.
                     Default 14400 (4 hours) — good for training.
                     Use 900 (15 min) for live dashboard snapshots.
    
    Returns:
        DataFrame with OHLCV data (same as yf.Ticker.history())
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize ticker for filename (e.g. RELIANCE.NS -> RELIANCE_NS)
    safe_name = ticker.upper().replace('.', '_').replace('/', '_')
    cache_file = _CACHE_DIR / f'{safe_name}_{period}.parquet'
    meta_file = _CACHE_DIR / f'{safe_name}_{period}.meta'
    
    # Check if cached data exists and is fresh
    if cache_file.exists() and meta_file.exists():
        cached_ts = float(meta_file.read_text().strip())
        age = time.time() - cached_ts
        if age < ttl_seconds:
            data = pd.read_parquet(cache_file)
            if not data.empty:
                data = data.dropna(subset=['Close', 'Open', 'High', 'Low'])
                if not data.empty:
                    print(f"  [cache] Using cached {ticker} data ({age/60:.0f}m old, TTL {ttl_seconds/60:.0f}m)")
                    return data
    
    # Fetch fresh data
    print(f"  [cache] Fetching fresh {ticker} data from yfinance (period={period})...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    if not data.empty:
        data = data.dropna(subset=['Close', 'Open', 'High', 'Low'])
        
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Save to cache
    data.to_parquet(cache_file)
    meta_file.write_text(str(time.time()))
    print(f"  [cache] Saved {len(data)} rows to cache")
    
    return data


def clear_cache(ticker: str = None):
    """Clear cache for a specific ticker or all cached data."""
    if not _CACHE_DIR.exists():
        return
    
    if ticker:
        safe_name = ticker.upper().replace('.', '_').replace('/', '_')
        for f in _CACHE_DIR.glob(f'{safe_name}_*'):
            f.unlink()
        print(f"Cache cleared for {ticker}")
    else:
        for f in _CACHE_DIR.iterdir():
            f.unlink()
        print("All cache cleared")
