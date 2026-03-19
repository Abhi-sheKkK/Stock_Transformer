import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import yfinance as yf
from src.model import Time2Vec

def create_input(stock_name):
    stock_ticker = yf.Ticker(stock_name)
    stock_data = stock_ticker.history(period='max')
    
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {stock_name}")
        
    stock_data['12day_EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['24day_EMA'] = stock_data['Close'].ewm(span=24, adjust=False).mean()
    stock_data['MACD'] = stock_data['12day_EMA'] - stock_data['24day_EMA']
    
    # Feature 1: RSI (14-day)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Feature 2: Bollinger Bands (20-day)
    stock_data['BB_MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['BB_Std'] = stock_data['Close'].rolling(window=20).std()
    stock_data['BB_Upper'] = stock_data['BB_MA'] + (stock_data['BB_Std'] * 2)
    stock_data['BB_Lower'] = stock_data['BB_MA'] - (stock_data['BB_Std'] * 2)
    
    # Feature 3: ATR (14-day)
    high_low = stock_data['High'] - stock_data['Low']
    high_close = np.abs(stock_data['High'] - stock_data['Close'].shift())
    low_close = np.abs(stock_data['Low'] - stock_data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    stock_data['ATR'] = true_range.rolling(14).mean()
    
    # Feature 4: Rolling VWAP (14-day)
    typical_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    volume = (typical_price * stock_data['Volume']).rolling(14).sum()
    stock_data['VWAP'] = volume / stock_data['Volume'].rolling(14).sum()
    
    # Drop rows with NaN values resulting from rolling windows
    stock_data.dropna(inplace=True)
    
    features = stock_data.copy()
    
    close_price = features['Close'].values.reshape(-1, 1)
    
    feature_scaler = MinMaxScaler()
    time_scaler = MinMaxScaler()
    close_scaler = QuantileTransformer(output_distribution='normal')
    
    timestamps = pd.Series(stock_data.index).apply(lambda x: x.toordinal()).values.reshape(-1, 1)
    time_scaled = time_scaler.fit_transform(timestamps).astype(np.float32)
    time_df = pd.DataFrame(time_scaled, columns=['Timestamp'])
    
    scaled_features = pd.DataFrame(feature_scaler.fit_transform(features))
    scaled_close = close_scaler.fit_transform(close_price)
    
    input_features = pd.concat([time_df, scaled_features], axis=1)
    
    return input_features, feature_scaler, time_scaler, close_scaler, scaled_close
