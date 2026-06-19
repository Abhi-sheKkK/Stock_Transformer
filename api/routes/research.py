"""
Research API route.
Generates comprehensive AI-powered research reports by combining
market data, news, sentiment, prediction, and LLM reasoning.
"""

import json
import os
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


def _gather_market_data(ticker: str) -> dict:
    """Fetch market snapshot with technical indicators."""
    from src.features import get_market_snapshot
    snapshot = get_market_snapshot(ticker)
    if "error" in snapshot:
        raise HTTPException(status_code=404, detail=snapshot["error"])
    return snapshot


def _gather_news_and_sentiment(ticker: str) -> dict:
    """Fetch news feed and sentiment report."""
    from src.news import fetch_news, get_news_summary_text
    from src.sentiment import get_sentiment_report

    feed = fetch_news(ticker)
    headlines = [a.title for a in feed.articles[:15]]
    sentiment = get_sentiment_report(ticker, headlines=headlines)

    return {
        "articles": feed.articles,
        "news_summary": get_news_summary_text(ticker),
        "sentiment": sentiment,
        "source_breakdown": feed.source_breakdown,
    }


def _gather_prediction(ticker: str) -> dict:
    """Run the Transformer model and return raw prediction data."""
    import torch
    import numpy as np

    try:
        from src.features import create_input
        from src.model import StockTransformer
        from src.cache import fetch_stock_data

        try:
            input_features, feature_scaler, time_scaler, close_scaler, _ = create_input(
                ticker, scalers_path='models'
            )
        except Exception:
            input_features, feature_scaler, time_scaler, close_scaler, _ = create_input(ticker)

        # Load model checkpoint with global caching to prevent OOM / RAM spikes
        global _cached_model, _cached_model_mtime, _cached_max_seq_len
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            return {"available": False, "reason": "No trained model found."}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtime = os.path.getmtime(model_path)

        if '_cached_model' not in globals() or _cached_model is None or _cached_model_mtime != mtime:
            print(f"🔄 Loading model weights from {model_path} into memory (research route)...")
            checkpoint = torch.load(model_path, map_location=device)

            # Auto-detect dimensions from checkpoint
            ckpt_max_seq_len = checkpoint['pos_embed'].shape[1]
            ckpt_d_model = checkpoint['input_proj.weight'].shape[0]
            ckpt_stock_embed_dim = checkpoint['stock_embed.weight'].shape[1]
            ckpt_num_stocks = checkpoint['stock_embed.weight'].shape[0]
            ckpt_sector_embed_dim = checkpoint['sector_embed.weight'].shape[1]
            ckpt_num_sectors = checkpoint['sector_embed.weight'].shape[0]
            ckpt_prediction_horizon = checkpoint['output_head.4.weight'].shape[0]

            model = StockTransformer(
                num_continuous_features=len([c for c in input_features.columns if c not in ('Stock_ID', 'Sector_ID', 'Target_Log_Return')]),
                d_model=ckpt_d_model, nhead=4, num_layers=4, dropout=0.25,
                prediction_horizon=ckpt_prediction_horizon,
                max_seq_len=ckpt_max_seq_len,
                stock_embed_dim=ckpt_stock_embed_dim, num_stocks=ckpt_num_stocks,
                sector_embed_dim=ckpt_sector_embed_dim, num_sectors=ckpt_num_sectors,
            )
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            _cached_model = model
            _cached_model_mtime = mtime
            _cached_max_seq_len = ckpt_max_seq_len
        else:
            model = _cached_model
            ckpt_max_seq_len = _cached_max_seq_len

        seq_length = ckpt_max_seq_len - 10
        if len(input_features) < seq_length:
            return {"available": False, "reason": f"Not enough data ({len(input_features)} < {seq_length})."}

        stock_id = int(input_features['Stock_ID'].iloc[-1])
        sector_id = int(input_features['Sector_ID'].iloc[-1])
        continuous_cols = [c for c in input_features.columns if c not in ('Stock_ID', 'Sector_ID', 'Target_Log_Return')]
        x_cont = input_features[continuous_cols].iloc[-seq_length:].values

        x_temporal = torch.tensor(x_cont, dtype=torch.float32).unsqueeze(0).to(device)
        stock_id_t = torch.tensor([stock_id], dtype=torch.long).to(device)
        sector_id_t = torch.tensor([sector_id], dtype=torch.long).to(device)

        with torch.no_grad():
            preds = model(x_temporal, stock_id_t, sector_id_t)
            predictions = preds[0].cpu().numpy().tolist()

        pred_log_returns = close_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        raw_data = fetch_stock_data(ticker, period='6mo', ttl_seconds=900)
        last_close = float(raw_data['Close'].iloc[-1])

        predicted_prices = []
        price = last_close
        for r in pred_log_returns:
            price *= np.exp(r)
            predicted_prices.append(round(price, 2))

        # Generate trading dates
        import datetime as dt
        current_date = dt.datetime.today().date()
        pred_dates = []
        while len(pred_dates) < len(predicted_prices):
            current_date += dt.timedelta(days=1)
            if current_date.weekday() < 5:
                pred_dates.append(current_date.isoformat())

        return {
            "available": True,
            "last_close": last_close,
            "predicted_prices": predicted_prices,
            "pred_log_returns": [round(float(r), 5) for r in pred_log_returns],
            "pred_dates": pred_dates,
        }
    except Exception as e:
        logger.warning(f"Prediction failed for research: {e}")
        return {"available": False, "reason": str(e)}


def _build_llm_prompt(ticker: str, market: dict, news_data: dict, prediction: dict) -> tuple:
    """Build system + user prompts for the research report LLM call."""
    currency = market.get("currency", "$")
    ind = market["indicators"]
    sig = market["signals"]
    sentiment = news_data["sentiment"]

    system_prompt = (
        "You are a senior equity research analyst at a top-tier investment bank. "
        "Your task is to produce a comprehensive, data-driven research report for a stock. "
        "You will be given real-time technical indicators, news headlines with truth scores, "
        "sentiment analysis, and a Transformer-based price forecast.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object only. Do NOT wrap in backticks or markdown codeblocks.\n"
        "The JSON must contain exactly these keys:\n"
        '- "executive_summary": A 2-3 sentence TL;DR of the investment thesis.\n'
        '- "market_position": An object with keys "current_price", "trend", "key_levels" (string descriptions).\n'
        '- "technical_analysis": An object with keys "signal" (bullish/bearish/neutral), '
        '"details" (2-3 sentence interpretation of RSI, MACD, Bollinger, volume).\n'
        '- "sentiment_analysis": An object with keys "overall" (bullish/bearish/neutral), '
        '"key_drivers" (2-3 sentence explanation of news sentiment).\n'
        '- "price_forecast": An object with keys "direction" (up/down/sideways), '
        '"target_5d" (string like "$xxx.xx"), "confidence" (high/medium/low), '
        '"reasoning" (2-3 sentence explanation).\n'
        '- "risk_factors": A list of 3-5 concise risk strings.\n'
        '- "catalysts": A list of 2-4 potential positive catalysts.\n'
        '- "investment_thesis": A 3-4 sentence final recommendation paragraph.\n'
        '- "rating": Exactly one of: "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL".\n'
    )

    # Build news headlines for prompt
    headlines = []
    for idx, art in enumerate(news_data["articles"][:10], 1):
        headlines.append(
            f'{idx}. "{art.title}" (Source: {art.source}, Truth Score: {art.truth_score:.2f})'
        )
    headlines_str = "\n".join(headlines) if headlines else "No recent news available."

    # Build prediction section
    if prediction.get("available"):
        pred_section = (
            f"Transformer 5-Day Forecast:\n"
            f"  Last Close: {currency}{prediction['last_close']:.2f}\n"
            f"  Predicted Prices: {prediction['predicted_prices']}\n"
            f"  Predicted Log Returns: {prediction['pred_log_returns']}\n"
            f"  Dates: {prediction['pred_dates']}"
        )
    else:
        pred_section = f"Transformer forecast unavailable: {prediction.get('reason', 'Unknown error')}"

    user_prompt = (
        f"RESEARCH REPORT REQUEST FOR: {ticker.upper()}\n\n"
        f"═══ TECHNICAL DATA ═══\n"
        f"Price: {currency}{ind['price']} ({ind['price_change_pct']:+.2f}%)\n"
        f"RSI (14): {ind['rsi']:.1f} — {sig['rsi']}\n"
        f"MACD: {ind['macd']:.4f} (Signal: {ind['macd_signal']:.4f}, Histogram: {ind['macd_histogram']:.4f}) — {sig['macd']}\n"
        f"Bollinger Position: {ind['bb_position']:.3f} — {sig['bollinger']}\n"
        f"  Upper: {currency}{ind['bb_upper']:.2f} | Middle: {currency}{ind['bb_middle']:.2f} | Lower: {currency}{ind['bb_lower']:.2f}\n"
        f"ATR (14): {currency}{ind['atr']:.2f}\n"
        f"VWAP: {currency}{ind['vwap']:.2f} — {sig['vwap_position']}\n"
        f"Volume: {ind['volume']:,} (Ratio: {ind['volume_ratio']:.2f}x avg) — {sig['volume']}\n"
        f"Trend: {sig['trend']}\n\n"
        f"═══ SENTIMENT ═══\n"
        f"Overall: {sentiment.overall_label.upper()} (Score: {sentiment.overall_score:+.3f})\n"
        f"Distribution: Bullish={sentiment.bullish_count}, Neutral={sentiment.neutral_count}, Bearish={sentiment.bearish_count}\n"
        f"Method: {sentiment.method}\n\n"
        f"═══ NEWS HEADLINES ═══\n"
        f"{headlines_str}\n\n"
        f"═══ TRANSFORMER FORECAST ═══\n"
        f"{pred_section}\n\n"
        f"Generate the full research report JSON now."
    )

    return system_prompt, user_prompt


def _generate_data_only_report(ticker: str, market: dict, news_data: dict, prediction: dict) -> dict:
    """Generate a report using only structured data (no LLM)."""
    ind = market["indicators"]
    sig = market["signals"]
    sentiment = news_data["sentiment"]
    currency = market.get("currency", "$")

    # Derive signal from technicals
    bull_signals = 0
    bear_signals = 0
    if ind["rsi"] > 50:
        bull_signals += 1
    else:
        bear_signals += 1
    if ind["macd"] > ind["macd_signal"]:
        bull_signals += 1
    else:
        bear_signals += 1
    if sig["trend"] == "Uptrend":
        bull_signals += 1
    else:
        bear_signals += 1
    if "Bullish" in sig["vwap_position"]:
        bull_signals += 1
    else:
        bear_signals += 1

    tech_signal = "bullish" if bull_signals > bear_signals else ("bearish" if bear_signals > bull_signals else "neutral")

    # Derive rating
    sent_label = sentiment.overall_label
    if tech_signal == "bullish" and sent_label == "bullish":
        rating = "BUY"
    elif tech_signal == "bearish" and sent_label == "bearish":
        rating = "SELL"
    else:
        rating = "HOLD"

    # Build prediction section
    if prediction.get("available"):
        last = prediction["last_close"]
        final_pred = prediction["predicted_prices"][-1]
        direction = "up" if final_pred > last else ("down" if final_pred < last else "sideways")
        forecast = {
            "direction": direction,
            "target_5d": f"{currency}{final_pred:.2f}",
            "confidence": "medium",
            "reasoning": f"Transformer model predicts a move from {currency}{last:.2f} to {currency}{final_pred:.2f} over 5 trading days based on technical pattern recognition.",
        }
    else:
        forecast = {
            "direction": "unknown",
            "target_5d": "N/A",
            "confidence": "low",
            "reasoning": prediction.get("reason", "Model unavailable."),
        }

    top_headlines = [a.title for a in news_data["articles"][:3]]

    forecast_text = f"targets {forecast['target_5d']}" if prediction.get('available') else "is currently unavailable"

    return {
        "executive_summary": f"{ticker.upper()} is currently trading at {currency}{ind['price']:.2f} with a {sig['trend'].lower()} bias. Technical indicators show {tech_signal} signals with RSI at {ind['rsi']:.1f} and sentiment is {sent_label}.",
        "market_position": {
            "current_price": f"{currency}{ind['price']:.2f}",
            "trend": sig["trend"],
            "key_levels": f"Support at {currency}{ind['bb_lower']:.2f} (BB Lower), resistance at {currency}{ind['bb_upper']:.2f} (BB Upper), VWAP at {currency}{ind['vwap']:.2f}.",
        },
        "technical_analysis": {
            "signal": tech_signal,
            "details": f"RSI at {ind['rsi']:.1f} ({sig['rsi']}). MACD {sig['macd']}. Bollinger Bands show price in {sig['bollinger'].lower()} zone. Volume is {sig['volume'].lower()} with {ind['volume_ratio']:.1f}x average ratio.",
        },
        "sentiment_analysis": {
            "overall": sent_label,
            "key_drivers": f"News sentiment is {sent_label} based on {sentiment.total_analyzed} analyzed headlines. {sentiment.bullish_count} bullish vs {sentiment.bearish_count} bearish signals detected via {sentiment.method} analysis.",
        },
        "price_forecast": forecast,
        "risk_factors": [
            f"ATR of {currency}{ind['atr']:.2f} indicates {'elevated' if ind['atr'] > ind['price'] * 0.02 else 'moderate'} daily volatility.",
            f"{'Overbought' if ind['rsi'] > 70 else 'Oversold' if ind['rsi'] < 30 else 'Neutral'} RSI may {'limit upside' if ind['rsi'] > 70 else 'present reversal risk' if ind['rsi'] < 30 else 'suggest continuation'}.",
            "Model predictions are based on historical patterns and may not account for unexpected events.",
        ],
        "catalysts": [
            h for h in top_headlines if h
        ][:3] or ["No specific catalysts identified from recent news."],
        "investment_thesis": f"Based on the combined analysis, {ticker.upper()} shows a {tech_signal} technical outlook with {sent_label} market sentiment. The Transformer model's 5-day forecast {forecast_text}. Investors should monitor key support/resistance levels and upcoming catalysts.",
        "rating": rating,
        "llm_generated": False,
    }


@router.post("/{ticker}")
async def generate_research_report(ticker: str):
    """
    Generate a comprehensive AI-powered research report for a stock.
    Combines market data, news, sentiment, prediction, and LLM reasoning.

    Example: POST /research/AAPL
    """
    try:
        # 1. Gather all data in parallel-ready fashion
        market = _gather_market_data(ticker)
        news_data = _gather_news_and_sentiment(ticker)
        prediction = _gather_prediction(ticker)

        currency = market.get("currency", "$")

        # 2. Try LLM-powered report via Groq
        from src.config import config
        groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY", "").strip().strip("'\"")

        report = None

        if groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)

                system_prompt, user_prompt = _build_llm_prompt(ticker, market, news_data, prediction)

                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=config.llm.groq_model,
                    temperature=0.4,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )

                response_text = chat_completion.choices[0].message.content
                report = json.loads(response_text)
                report["llm_generated"] = True
                logger.info(f"Research report generated via Groq for {ticker}")
            except Exception as e:
                logger.warning(f"Groq research report failed: {e}. Falling back to data-only report.")
                report = None

        # 3. Fallback to structured data report
        if report is None:
            report = _generate_data_only_report(ticker, market, news_data, prediction)

        # 4. Attach metadata
        response = {
            "ticker": ticker.upper(),
            "currency": currency,
            "generated_at": datetime.now().isoformat(),
            "report": report,
            "data": {
                "price": market["indicators"]["price"],
                "price_change_pct": market["indicators"]["price_change_pct"],
                "rsi": market["indicators"]["rsi"],
                "trend": market["signals"]["trend"],
                "sentiment": news_data["sentiment"].overall_label,
                "sentiment_score": round(news_data["sentiment"].overall_score, 3),
                "prediction_available": prediction.get("available", False),
                "predicted_prices": prediction.get("predicted_prices", []),
                "pred_dates": prediction.get("pred_dates", []),
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research report failed: {str(e)}")
