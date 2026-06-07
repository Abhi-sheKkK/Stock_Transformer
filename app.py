import streamlit as st
import torch
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import os

from src.features import create_input
from src.model import StockTransformer
from src.train import train_and_evaluate
from torch.utils.data import DataLoader
from src.cache import fetch_stock_data

st.set_page_config(page_title="Stock Transformer AI", layout="wide", page_icon="📈")

st.title("📈 Stock Transformer AI Dashboard")
st.markdown("Decoder-Only Causal Transformer · Multi-Stock Global Forecasting · Relative Stationarized Metrics")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")

st.sidebar.subheader("Model Parameters")
seq_length = st.sidebar.number_input("Lookback Window (Days)", min_value=10, max_value=365, value=60)
out_seq_len = st.sidebar.number_input("Prediction Horizon (Days)", min_value=1, max_value=5, value=5)

st.sidebar.subheader("Training Hyperparameters")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=500, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=0.01, value=0.0001, format="%.5f")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tab1, tab2 = st.tabs(["🔮 Inference", "🏋️ Global Training"])


@st.cache_data(show_spinner=False)
def get_stock_data(tkr, use_saved_scalers=False):
    try:
        sp = 'models' if use_saved_scalers else None
        return create_input(tkr, scalers_path=sp)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None


def plot_interactive(actual_dates, actual_prices, pred_dates, pred_prices, tkr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_prices, mode='lines',
                             name='Actual Price', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines+markers',
                             name='Predicted', line=dict(color='#ef4444', dash='dash')))
    fig.update_layout(title=f"Forecast — {tkr}", xaxis_title="Date", yaxis_title="Price (USD)",
                      template="plotly_dark", hovermode="x unified")
    return fig


with tab1:
    st.header("Forecast")

    if st.button("Run Inference", use_container_width=True):
        if not os.path.exists("best_model.pth"):
            st.warning("⚠️ No trained model found. Train the model first.")
        else:
            with st.spinner(f"Preparing features for {ticker}..."):
                input_features, feature_scaler, time_scaler, close_scaler, scaled_close = get_stock_data(ticker, True)

            if input_features is not None and len(input_features) >= seq_length:
                st.success("Data ready!")

                with st.spinner("Running Transformer inference..."):
                    stock_id = int(input_features['Stock_ID'].iloc[-1])
                    sector_id = int(input_features['Sector_ID'].iloc[-1])
                    cont_cols = [c for c in input_features.columns if c not in ('Stock_ID', 'Sector_ID', 'Target_Log_Return')]
                    x_cont = input_features[cont_cols].iloc[-seq_length:].values

                    x_temporal = torch.tensor(x_cont, dtype=torch.float32).unsqueeze(0).to(device)
                    stock_id_t = torch.tensor([stock_id], dtype=torch.long).to(device)
                    sector_id_t = torch.tensor([sector_id], dtype=torch.long).to(device)

                    model = StockTransformer(
                        num_continuous_features=len(cont_cols),
                        d_model=128, nhead=4, num_layers=4, dropout=0.25,
                        prediction_horizon=int(out_seq_len),
                        max_seq_len=int(seq_length) + 10,
                        stock_embed_dim=32,
                    )
                    model.load_state_dict(torch.load('best_model.pth', map_location=device))
                    model.to(device)
                    model.eval()

                    with torch.no_grad():
                        preds = model(x_temporal, stock_id_t, sector_id_t)

                    pred_log_returns = close_scaler.inverse_transform(
                        preds.cpu().numpy().reshape(-1, 1)
                    ).flatten()

                    raw_data = fetch_stock_data(ticker, period='6mo', ttl_seconds=14400)
                    last_raw_close = float(raw_data['Close'].iloc[-1])
                    last_30 = raw_data['Close'].iloc[-30:].values
                    actual_dates = raw_data.index[-30:].tolist()

                    predicted_prices = []
                    price = last_raw_close
                    for r in pred_log_returns:
                        price *= np.exp(r)
                        predicted_prices.append(price)

                    pred_dates = []
                    cd = pd.to_datetime(actual_dates[-1]).date()
                    while len(pred_dates) < out_seq_len:
                        cd += datetime.timedelta(days=1)
                        if cd.weekday() < 5:
                            pred_dates.append(cd)

                    plot_dates = [actual_dates[-1]] + pred_dates
                    plot_preds = [last_30[-1]] + list(predicted_prices)

                    fig = plot_interactive(actual_dates, last_30, plot_dates, plot_preds, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    df_preds = pd.DataFrame({"Date": pred_dates, "Expected Price": predicted_prices})
                    st.dataframe(df_preds.style.format({"Expected Price": "${:.2f}"}), use_container_width=True)


with tab2:
    st.header("Train Multi-Stock Global Transformer")
    st.markdown("Trains the decoder-only causal transformer across all 41 tickers with chronological train/val split.")

    if st.button("🚀 Start Global Training", use_container_width=True):
        my_bar = st.progress(0, text="Preparing...")

        from src.features import TICKERS, save_scalers
        from src.data import GlobalStockDataset

        train_dfs, val_dfs = [], []

        try:
            my_bar.progress(5, text="Fitting scalers on AAPL...")
            base_input, feature_scaler, time_scaler, close_scaler, _ = create_input('AAPL')
            save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models')

            for i, t in enumerate(TICKERS):
                my_bar.progress(10 + int(i / len(TICKERS) * 40), text=f"Loading {t}...")
                try:
                    df, _, _, _, _ = create_input(t, scalers_path='models')
                    tr = df[df.index.year <= 2023]
                    vl = df[df.index.year == 2024]
                    if len(tr) > 0: train_dfs.append(tr)
                    if len(vl) > 0: val_dfs.append(vl)
                except Exception as ex:
                    st.warning(f"Failed: {t} ({ex})")

            if not train_dfs:
                st.error("No training data loaded.")
            else:
                my_bar.progress(50, text="Building sliding-window datasets...")
                train_ds = GlobalStockDataset(train_dfs, seq_length=int(seq_length), out_seq_len=int(out_seq_len))
                val_ds = GlobalStockDataset(val_dfs, seq_length=int(seq_length), out_seq_len=int(out_seq_len))

                train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
                val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

                num_continuous = train_dfs[0].shape[1] - 3
                model = StockTransformer(
                    num_continuous_features=num_continuous,
                    d_model=128, nhead=4, num_layers=4, dropout=0.25,
                    prediction_horizon=int(out_seq_len),
                    max_seq_len=int(seq_length) + 10,
                    stock_embed_dim=32,
                )

                my_bar.progress(60, text=f"Training for {int(epochs)} epochs...")

                with st.spinner("Training in progress..."):
                    preds, targets, idx, train_losses, val_losses = train_and_evaluate(
                        model, train_loader, val_loader, close_scaler,
                        num_epochs=int(epochs), learning_rate=lr,
                        feature_scaler=feature_scaler, time_scaler=time_scaler
                    )

                my_bar.progress(100, text="Done!")
                st.success("Model trained! Saved as 'best_model.pth'.")
                st.balloons()

                st.markdown("### Training Summary")
                st.write(f"Final Val Loss: {val_losses[-1]:.6f}")

                loss_fig = go.Figure()
                loss_fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train'))
                loss_fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Val'))
                loss_fig.update_layout(title="Learning Curve", xaxis_title="Epoch",
                                       yaxis_title="Huber Loss", template="plotly_dark")
                st.plotly_chart(loss_fig, use_container_width=True)
        except Exception as ex:
            st.error(f"Training failed: {ex}")
