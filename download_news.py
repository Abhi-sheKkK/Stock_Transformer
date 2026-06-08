#!/usr/bin/env python3
"""
Parallel script to fetch and cache news articles for all 41 predefined stocks.
Uses ThreadPoolExecutor to run API calls in parallel.
"""

import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.features import TICKERS
from src.news import fetch_news
from src.config import config

def download_ticker_news(ticker):
    try:
        # bypass_cache=True forces hitting the APIs and regenerating cache files
        feed = fetch_news(ticker, bypass_cache=True)
        return ticker, True, len(feed.articles), feed.source_breakdown
    except Exception as e:
        return ticker, False, str(e), None

def main():
    print("==================================================")
    print("🚀 AI Financial Intelligence System — News Downloader (Parallel)")
    print(f"Target Tickers: {len(TICKERS)}")
    print(f"NewsAPI Key Configured: {'Yes' if config.news.news_api_key else 'No'}")
    print(f"Finnhub Key Configured: {'Yes' if config.news.finnhub_api_key else 'No'}")
    print("==================================================")

    success_count = 0
    failed_tickers = []
    
    # We use a balanced thread pool size of 5 to avoid overwhelming the APIs
    max_workers = 5
    print(f"Starting download using {max_workers} parallel workers...\n")

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(download_ticker_news, t): t for t in TICKERS}
        
        for idx, future in enumerate(as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            try:
                ticker, success, result, breakdown = future.result()
                if success:
                    print(f"[{idx}/{len(TICKERS)}] ✅ {ticker}: Cached {result} articles. (Breakdown: {breakdown})")
                    success_count += 1
                else:
                    print(f"[{idx}/{len(TICKERS)}] ❌ {ticker}: Failed with error: {result}")
                    failed_tickers.append(ticker)
            except Exception as exc:
                print(f"[{idx}/{len(TICKERS)}] ❌ {ticker}: Generated an exception: {exc}")
                failed_tickers.append(ticker)

    duration = time.time() - start_time
    print("\n" + "="*50)
    print("News Download Task Summary:")
    print(f"Total Tickers Attempted: {len(TICKERS)}")
    print(f"Successfully Cached: {success_count}")
    print(f"Time Taken: {duration:.2f} seconds")
    if failed_tickers:
        print(f"Failed Tickers: {', '.join(failed_tickers)}")
    print("==================================================")

if __name__ == "__main__":
    main()
