#!/usr/bin/env python3
"""
Download script to collect daily (1-day scale) stock price data from yfinance.
Saves the downloaded data into both Parquet and CSV formats in the 'data' directory.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

# List of 41 tickers specified by the user
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

def download_ticker_data(ticker: str, output_dir: Path, period: str = 'max'):
    """
    Download daily historical data for a ticker and save it to the output directory.
    """
    print(f"Downloading historical data for {ticker} (period={period})...")
    try:
        stock = yf.Ticker(ticker)
        # Fetch daily history (1-day scale)
        data = stock.history(period=period, interval='1d')
        
        if data.empty:
            print(f"Warning: No data returned for ticker {ticker}. Skipping.")
            return False
        
        # Ensure target directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save to Parquet format (preserves DatetimeIndex)
        parquet_path = output_dir / f"{ticker.upper()}.parquet"
        data.to_parquet(parquet_path)
        
        # 2. Save to CSV format (resets index to have 'Date' as a column)
        csv_path = output_dir / f"{ticker.upper()}.csv"
        csv_data = data.reset_index()
        csv_data.to_csv(csv_path, index=False)
        
        print(f"Successfully saved {ticker.upper()}:")
        print(f"  - Parquet: {parquet_path} ({len(data)} rows)")
        print(f"  - CSV: {csv_path} ({len(csv_data)} rows)")
        return True
    except Exception as e:
        print(f"Error downloading/saving data for {ticker}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Yahoo Finance Stock Data")
    parser.add_argument('--ticker', type=str, help="Specific ticker to download (default: download all 41 predefined tickers)")
    parser.add_argument('--period', type=str, default='max', help="Historical time period to download (default: max)")
    parser.add_argument('--output_dir', type=str, default='data', help="Output directory path (default: data)")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.ticker:
        tickers_to_download = [args.ticker.upper()]
    else:
        tickers_to_download = TICKERS
        
    print(f"Starting stock data download for {len(tickers_to_download)} tickers...")
    print(f"Output directory: {output_dir.resolve()}\n")
    
    success_count = 0
    failed_tickers = []
    
    for ticker in tickers_to_download:
        success = download_ticker_data(ticker, output_dir, period=args.period)
        if success:
            success_count += 1
        else:
            failed_tickers.append(ticker)
            
    print("\n" + "="*50)
    print("Download Task Summary:")
    print(f"Total Tickers Attempted: {len(tickers_to_download)}")
    print(f"Successfully Downloaded: {success_count}")
    if failed_tickers:
        print(f"Failed Tickers: {', '.join(failed_tickers)}")
    print("="*50)

if __name__ == "__main__":
    main()
