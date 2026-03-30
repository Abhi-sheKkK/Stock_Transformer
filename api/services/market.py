"""
Market snapshot service.
Generates human-readable technical indicator summaries for LLM consumption.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def get_market_snapshot(ticker: str) -> dict:
    """
    Generate a market snapshot with current technical indicators.

    Returns dict with:
      - ticker: str
      - indicators: dict of raw numeric values
      - signals: dict of interpreted text signals
      - summary: formatted text for LLM consumption
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")

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

    # RSI (14)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd = float(macd_line.iloc[-1])
    signal = float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])
    histogram = macd - signal

    # Bollinger Bands (20)
    bb_ma = float(data["Close"].rolling(20).mean().iloc[-1])
    bb_std = float(data["Close"].rolling(20).std().iloc[-1])
    bb_upper = bb_ma + (bb_std * 2)
    bb_lower = bb_ma - (bb_std * 2)
    bb_width = bb_upper - bb_lower
    bb_position = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

    # ATR (14)
    hl = data["High"] - data["Low"]
    hc = np.abs(data["High"] - data["Close"].shift())
    lc = np.abs(data["Low"] - data["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # VWAP (14-day rolling)
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = float((tp * data["Volume"]).rolling(14).sum().iloc[-1] / data["Volume"].rolling(14).sum().iloc[-1])

    # Volume
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
