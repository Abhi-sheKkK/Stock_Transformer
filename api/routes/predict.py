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

        # Fetch and prepare data
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(ticker)

        seq_length = config.model.seq_length
        if len(input_features) < seq_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for {ticker}. Need {seq_length} days, got {len(input_features)}.",
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="No trained model found. Train the model first.")

        input_dim = input_features.shape[1]
        model = StockTransformer(
            input_dim=input_dim,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            seq_length=seq_length,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Prepare source sequence
        src_data = input_features.iloc[-seq_length:].values
        src = torch.tensor(src_data, dtype=torch.float32).unsqueeze(0).to(device)
        last_close_val = scaled_close[-1][0]

        # Autoregressive prediction
        predictions = []
        tgt_input = torch.zeros((1, 1, input_dim), device=device)
        tgt_input[:, 0, 0] = last_close_val

        with torch.no_grad():
            for _ in range(days):
                output = model(src, tgt_input)
                next_pred = output[:, -1].item()
                predictions.append(next_pred)
                new_step = torch.zeros((1, 1, input_dim), device=device)
                new_step[:, 0, 0] = next_pred
                tgt_input = torch.cat([tgt_input, new_step], dim=1)

        # Inverse transform
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        predicted_prices = close_scaler.inverse_transform(predictions_scaled).flatten().tolist()

        # Generate trading dates
        import datetime
        current_date = datetime.datetime.today().date()
        pred_dates = []
        while len(pred_dates) < days:
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
                for d, p in zip(pred_dates, predicted_prices)
            ],
            "model_info": {
                "architecture": "StockTransformer + Time2Vec",
                "input_window": seq_length,
                "prediction_horizon": days,
            },
        }

        # AI reasoning (Phase 2 — will be implemented with Ollama)
        if include_reasoning:
            response["reasoning"] = {
                "status": "available_after_phase2",
                "message": "AI reasoning will be powered by Llama 3 via Ollama.",
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
