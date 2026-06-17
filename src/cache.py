"""
Local file cache for stock data with multi-source fallback.

Source priority:
  1. Local Parquet cache (if fresh within TTL)
  2. yfinance (works locally / residential IPs)
  3. Alpha Vantage API (works from cloud — requires ALPHA_VANTAGE_KEY)

Cache files are stored as Parquet in .cache/stocks/{TICKER}.parquet
with a configurable TTL (default: 15 min for API snapshots).
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path('.cache/stocks')

# Map yfinance period strings → approximate day counts for Alpha Vantage
_PERIOD_DAYS = {
    '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365,
    '2y': 730, '5y': 1825, '10y': 3650, 'max': 7300,
}


# ---------------------------------------------------------------------------
# Alpha Vantage fallback
# ---------------------------------------------------------------------------

def _fetch_alpha_vantage(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Alpha Vantage (free, cloud-friendly).
    Requires env var ALPHA_VANTAGE_KEY.
    """
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        logger.warning("ALPHA_VANTAGE_KEY not set — cannot use Alpha Vantage fallback.")
        return pd.DataFrame()

    # Use compact (100 days) or full based on period
    days_needed = _PERIOD_DAYS.get(period, 180)
    output_size = "compact" if days_needed <= 100 else "full"

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&outputsize={output_size}"
        f"&apikey={api_key}"
    )

    try:
        logger.info(f"  [cache] Trying Alpha Vantage for {ticker}...")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        ts = payload.get("Time Series (Daily)")
        if not ts:
            # May hit rate limit or invalid ticker
            error_msg = payload.get("Note") or payload.get("Error Message") or payload.get("Information") or "unknown"
            logger.warning(f"  [cache] Alpha Vantage returned no data: {error_msg}")
            return pd.DataFrame()

        rows = []
        for date_str, vals in ts.items():
            rows.append({
                "Date": pd.Timestamp(date_str),
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["4. close"]),
                "Volume": int(vals["5. volume"]),
            })

        df = pd.DataFrame(rows).set_index("Date").sort_index()

        # Trim to requested period
        if days_needed < len(df):
            df = df.iloc[-days_needed:]

        logger.info(f"  [cache] Alpha Vantage returned {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.warning(f"  [cache] Alpha Vantage error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Primary fetch with fallback chain
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str, period: str = 'max', ttl_seconds: int = 900) -> pd.DataFrame:
    """
    Fetch stock history with local file caching and multi-source fallback.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        period: yfinance period string ('max', '6mo', '1y', etc.)
        ttl_seconds: Cache validity in seconds.
                     Default 900 (15 min) — serves fresh data with a
                     short cache window to avoid redundant API calls.

    Returns:
        DataFrame with OHLCV data
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

    # --- Source 1: yfinance direct (browser-like headers) ---
    data = pd.DataFrame()
    try:
        print(f"  [cache] Fetching fresh {ticker} data from yfinance (period={period})...")
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        stock = yf.Ticker(ticker, session=session)
        data = stock.history(period=period)
        if not data.empty:
            data = data.dropna(subset=['Close', 'Open', 'High', 'Low'])
    except Exception as e:
        logger.warning(f"  [cache] yfinance (direct) failed for {ticker}: {e}")

    # --- Source 2: yfinance via proxy (bypasses IP blocks) ---
    proxy_url = os.getenv("PROXY_URL")
    if data.empty and proxy_url:
        try:
            print(f"  [cache] Retrying {ticker} via proxy...")
            proxy_session = requests.Session()
            proxy_session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            })
            proxy_session.proxies = {
                'http': proxy_url,
                'https': proxy_url,
            }
            stock = yf.Ticker(ticker, session=proxy_session)
            data = stock.history(period=period)
            if not data.empty:
                data = data.dropna(subset=['Close', 'Open', 'High', 'Low'])
                if not data.empty:
                    print(f"  [cache] ✅ Proxy fetch succeeded for {ticker}")
        except Exception as e:
            logger.warning(f"  [cache] yfinance (proxy) failed for {ticker}: {e}")

    # --- Source 3: Alpha Vantage (cloud-friendly fallback) ---
    if data.empty:
        print(f"  [cache] yfinance returned no data, trying Alpha Vantage fallback...")
        data = _fetch_alpha_vantage(ticker, period)

    # --- Final check ---
    if data.empty:
        raise ValueError(
            f"No data found for ticker {ticker}. "
            f"Set PROXY_URL or ALPHA_VANTAGE_KEY env var for cloud deployment "
            f"(free AV key: https://www.alphavantage.co/support/#api-key)."
        )

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
