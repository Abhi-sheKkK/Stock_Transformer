"""
Prediction API routes.
Runs the Stock Transformer model for price forecasting.
"""

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/{ticker}")
async def predict_prices(
    ticker: str,
    days: int = Query(default=5, ge=1, le=30, description="Number of days to predict"),
    include_reasoning: bool = Query(default=True, description="Include AI reasoning"),
):
    """
    Predict future stock prices using the Transformer model.
    Optionally includes AI-generated reasoning for the prediction.

    Example: POST /predict/RELIANCE.NS?days=5&include_reasoning=true
    """
    import torch
    import numpy as np
    import os

    try:
        from src.features import create_input
        from src.model import StockTransformer
        from src.config import config

        # Use persisted scalers from training to ensure same distribution
        try:
            input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(
                ticker, scalers_path='models'
            )
        except Exception:
            # Fallback: refit scalers (may produce different distribution than training)
            input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(ticker)

        seq_length = config.model.seq_length
        if len(input_features) < seq_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for {ticker}. Need {seq_length} days, got {len(input_features)}.",
            )

        # Load model checkpoint
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="No trained model found. Train the model first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        # Auto-detect dimensions from checkpoint to avoid size mismatches
        ckpt_max_seq_len = checkpoint['pos_embed'].shape[1]
        ckpt_d_model = checkpoint['input_proj.weight'].shape[0]
        ckpt_stock_embed_dim = checkpoint['stock_embed.weight'].shape[1]
        ckpt_num_stocks = checkpoint['stock_embed.weight'].shape[0]
        ckpt_sector_embed_dim = checkpoint['sector_embed.weight'].shape[1]
        ckpt_num_sectors = checkpoint['sector_embed.weight'].shape[0]
        ckpt_prediction_horizon = checkpoint['output_head.4.weight'].shape[0]

        # Lookback sequence length is max_seq_len - 10 (or capped by features length)
        seq_length = ckpt_max_seq_len - 10
        if len(input_features) < seq_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for {ticker}. Need {seq_length} days, got {len(input_features)}.",
            )

        # Prepare continuous features (drop Stock_ID, Sector_ID, Target_Log_Return)
        stock_id = int(input_features['Stock_ID'].iloc[-1])
        sector_id = int(input_features['Sector_ID'].iloc[-1])
        continuous_cols = [c for c in input_features.columns if c not in ('Stock_ID', 'Sector_ID', 'Target_Log_Return')]
        x_cont = input_features[continuous_cols].iloc[-seq_length:].values

        x_temporal = torch.tensor(x_cont, dtype=torch.float32).unsqueeze(0).to(device)
        stock_id_t = torch.tensor([stock_id], dtype=torch.long).to(device)
        sector_id_t = torch.tensor([sector_id], dtype=torch.long).to(device)

        num_continuous = len(continuous_cols)

        model = StockTransformer(
            num_continuous_features=num_continuous,
            d_model=ckpt_d_model,
            nhead=4,
            num_layers=4,
            dropout=0.25,
            prediction_horizon=ckpt_prediction_horizon,
            max_seq_len=ckpt_max_seq_len,
            stock_embed_dim=ckpt_stock_embed_dim,
            num_stocks=ckpt_num_stocks,
            sector_embed_dim=ckpt_sector_embed_dim,
            num_sectors=ckpt_num_sectors,
        )
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # Single forward pass -> all 5 days predicted at once
        with torch.no_grad():
            preds = model(x_temporal, stock_id_t, sector_id_t)  # [1, 5]
            predictions = preds[0].cpu().numpy().tolist()

        # Inverse transform to get raw log returns
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        pred_log_returns = close_scaler.inverse_transform(predictions_scaled).flatten()

        # --- Groq LLM Sentiment Adjustment ---
        llm_applied = False
        llm_rationale = "Pure mathematical Transformer forecast (Groq API key not configured)."
        adjusted_log_returns = list(pred_log_returns)

        # Fetch truth-verified articles
        from src.news import fetch_news
        try:
            news_feed = fetch_news(ticker)
            verified_articles = news_feed.articles
        except Exception as e:
            print(f"Failed to fetch news for LLM adjustment: {e}")
            verified_articles = []

        groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY", "").strip().strip("'\"")

        if groq_api_key and verified_articles:
            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)

                orig_returns_list = [round(float(r), 5) for r in pred_log_returns]

                # Format articles cleanly for prompt
                headlines = []
                for idx, art in enumerate(verified_articles[:10], 1):
                    headlines.append(f"{idx}. \"{art.title}\" (Source: {art.source}, Truth Score: {art.truth_score:.2f})")
                headlines_str = "\n".join(headlines)

                system_prompt = (
    "You are the Chief Investment Officer Agent in a multi-agent quantitative framework. "
    "Your job is to synthesize a hard mathematical structural forecast with real-time news context. "
    "You have final executive authority over the 5-day horizon forecast.\n\n"
    "YOUR CONSTRAINT RULES:\n"
    "1. You are NOT bound by any percentage limits or directional constraints. The Transformer provides the underlying "
    "structural market geometry, but it is completely blind to external events. If you believe current news completely "
    "invalidates or heavily magnifies the chart geometry, you have full authority to override, scale up, suppress, "
    "or completely reverse the direction of the Transformer's numbers.\n"
    "2. Balance quantitative probability with structural macro/fundamental reality. Use extreme modifications only when "
    "news represents a genuine, high-impact catalyst (e.g., earnings blowouts, regulatory shocks, black swan events).\n"
    "3. Every single item in the 'adjusted_log_returns' JSON array MUST be a raw, single, fully evaluated float number "
    "(e.g., -0.01425). DO NOT write mathematical formulas, additions, subtractions, multiplications, or variables "
    "inside the JSON array. Evaluate all arithmetic yourself.\n\n"
    "OUTPUT FORMAT:\n"
    "Return a JSON object only. Do NOT wrap in backticks or markdown codeblocks. The JSON must contain exactly these keys:\n"
    '- "adjusted_log_returns": list of 5 floats representing your final adjusted log returns.\n'
    '- "transformer_weight_used": a string or float (0-100%) indicating how much you trusted the base math versus the news context.\n'
    '- "rationale": a 2-sentence crisp, data-backed explanation of your adjustments for the UI dashboard.'
)

                user_prompt = (
                    f"INPUT DATA:\n"
                    f"- Target Stock: {ticker.upper()}\n"
                    f"- Transformer's 5-Day Predicted Log Returns: {orig_returns_list}\n"
                    f"- Live Scraped Headlines From Last 24 Hours:\n{headlines_str}"
                )

                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=config.llm.groq_model,
                    response_format={"type": "json_object"}
                )

                response_content = chat_completion.choices[0].message.content
                import json
                llm_response = json.loads(response_content)

                raw_adjusted = llm_response.get("adjusted_log_returns", [])
                if len(raw_adjusted) == len(orig_returns_list):
                    bounded_adjusted = []
                    for orig, adj in zip(orig_returns_list, raw_adjusted):
                        delta = adj - orig
                        # Core Rule: Hard clipping of adjustment delta to +/- 0.003
                        clamped_delta = max(min(delta, 0.003), -0.003)
                        bounded_adjusted.append(orig + clamped_delta)
                    adjusted_log_returns = bounded_adjusted
                    llm_rationale = llm_response.get("rationale", "Log returns adjusted based on news sentiment.")
                    llm_applied = True
                else:
                    llm_rationale = f"Log returns not adjusted: LLM returned invalid array length. {llm_response.get('rationale', '')}"
            except Exception as e:
                llm_rationale = f"Transformer forecast (Groq adjustment failed: {str(e)})."
        elif verified_articles:
            llm_rationale = "Pure Transformer forecast. Configure GROQ_API_KEY for sentiment adjustment."
        else:
            llm_rationale = "Pure Transformer forecast. No recent news headlines available for sentiment adjustment."

        # Fetch last raw close to compound into dollar prices
        from src.cache import fetch_stock_data
        raw_data = fetch_stock_data(ticker, period='6mo', ttl_seconds=900)
        last_raw_close = float(raw_data['Close'].iloc[-1])

        # Compute adjusted prices
        predicted_prices = []
        price = last_raw_close
        for r in adjusted_log_returns:
            price = price * np.exp(r)
            predicted_prices.append(price)

        # Compute original (unadjusted) prices
        original_prices = []
        orig_price = last_raw_close
        for r in pred_log_returns:
            orig_price = orig_price * np.exp(r)
            original_prices.append(orig_price)

        # Cap prediction horizon to available predicted prices (5 days)
        horizon = min(days, len(predicted_prices))

        # Generate trading dates
        import datetime
        current_date = datetime.datetime.today().date()
        pred_dates = []
        while len(pred_dates) < horizon:
            current_date += datetime.timedelta(days=1)
            if current_date.weekday() < 5:
                pred_dates.append(current_date.isoformat())

        # Currency detection
        currency = "₹" if any(s in ticker.upper() for s in [".NS", ".BO"]) else "$"

        response = {
            "ticker": ticker.upper(),
            "currency": currency,
            "predictions": [
                {"date": d, "price": round(p, 2)}
                for d, p in zip(pred_dates, predicted_prices[:horizon])
            ],
            "original_predictions": [
                {"date": d, "price": round(p, 2)}
                for d, p in zip(pred_dates, original_prices[:horizon])
            ],
            "model_info": {
                "architecture": "Decoder-Only Causal Transformer",
                "input_window": seq_length,
                "prediction_horizon": horizon,
            },
            "llm_adjustment": {
                "applied": llm_applied,
                "original_log_returns": [round(float(r), 5) for r in pred_log_returns],
                "adjusted_log_returns": [round(float(r), 5) for r in adjusted_log_returns],
                "rationale": llm_rationale,
            }
        }

        # AI reasoning (Phase 2 — will be implemented with Ollama)
        if include_reasoning:
            response["reasoning"] = {
                "status": "active" if llm_applied else "inactive",
                "message": llm_rationale,
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
