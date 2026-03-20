import streamlit as st
import torch
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import os

from src.features import create_input
from src.data import create_sequences
from src.model import StockTransformer
from src.train import train_and_evaluate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stock Transformer AI", layout="wide", page_icon="📈")

st.title("📈 Stock Transformer AI Dashboard")
st.markdown("Advanced Time-Series Forecasting using Transformers & Time2Vec")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")

st.sidebar.subheader("Model Parameters")
seq_length = st.sidebar.number_input("Input Window Size (Days)", min_value=10, max_value=365, value=100)
out_seq_len = st.sidebar.number_input("Output Prediction Size (Days)", min_value=1, max_value=30, value=5)

st.sidebar.subheader("Training Hyperparameters")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=500, value=50)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=0.01, value=0.0001, format="%.5f")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🔮 Inference (Predict Next Days)", "🏋️ Training Dashboard"])

@st.cache_data(show_spinner=False)
def get_stock_data(ticker):
    try:
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = create_input(ticker)
        return input_features, feature_scaler, time_scaler, close_scaler, scaled_close
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None

def plot_interactive(actual_dates, actual_prices, pred_dates, pred_prices, ticker):
    fig = go.Figure()
    # Actual Prices
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_prices, mode='lines', 
                             name='Recent Actual Price', line=dict(color='blue')))
    # Predicted Prices
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines+markers', 
                             name='Predicted Price', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title=f"Stock Price Forecast for {ticker}",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template="plotly_dark",
                      hovermode="x unified")
    return fig

with tab1:
    st.header("Autoregressive Forecast")
    
    if st.button("Run Inference", use_container_width=True):
        if not os.path.exists("best_model.pth"):
            st.warning("⚠️ No trained model found ('best_model.pth'). Please train the model first in the Training tab!")
        else:
            with st.spinner(f"Fetching data and preparing features for {ticker}..."):
                input_features, feature_scaler, time_scaler, close_scaler, scaled_close = get_stock_data(ticker)
            
            if input_features is not None:
                if len(input_features) < seq_length:
                    st.error(f"Not enough historical data for {ticker}. Required: {seq_length} days.")
                else:
                    st.success("Data successfully fetched!")
                    
                    with st.spinner("Decoding future prices from the Transformer..."):
                        src_data = input_features.iloc[-seq_length:].values
                        src = torch.tensor(src_data, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        last_close_val = scaled_close[-1][0]
                        input_dim = input_features.shape[1]
                        
                        model = StockTransformer(
                            input_dim=input_dim, d_model=64, nhead=4, 
                            num_encoder_layers=2, num_decoder_layers=2, seq_length=seq_length
                        )
                        model.load_state_dict(torch.load('best_model.pth', map_location=device))
                        model.to(device)
                        model.eval()
                        
                        predictions = []
                        tgt_input = torch.zeros((1, 1, input_dim), device=device)
                        tgt_input[:, 0, 0] = last_close_val
                        
                        with torch.no_grad():
                            for _ in range(out_seq_len):
                                output = model(src, tgt_input)
                                next_pred = output[:, -1].item()
                                predictions.append(next_pred)
                                new_step = torch.zeros((1, 1, input_dim), device=device)
                                new_step[:, 0, 0] = next_pred
                                tgt_input = torch.cat([tgt_input, new_step], dim=1)
                                
                        predictions_scaled = np.array(predictions).reshape(-1, 1)
                        predicted_prices = close_scaler.inverse_transform(predictions_scaled).flatten()
                        
                        # Get real historical dates/prices for plotting
                        last_30_days_close = close_scaler.inverse_transform(scaled_close[-30:]).flatten()
                        today = datetime.datetime.today().date()
                        
                        actual_dates = [(today - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
                        pred_dates = []
                        current_date = today
                        
                        while len(pred_dates) < out_seq_len:
                            current_date += datetime.timedelta(days=1)
                            if current_date.weekday() < 5:  # Skip weekends
                                pred_dates.append(current_date)
                                
                        # Combine last actual point to predicted to connect the graph
                        plot_dates = [actual_dates[-1]] + pred_dates
                        plot_preds = [last_30_days_close[-1]] + list(predicted_prices)
                        
                        fig = plot_interactive(actual_dates, last_30_days_close, plot_dates, plot_preds, ticker)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show raw prediction table
                        df_preds = pd.DataFrame({"Date": pred_dates, "Expected Price": predicted_prices})
                        st.dataframe(df_preds.style.format({"Expected Price": "${:.2f}"}), use_container_width=True)


with tab2:
    st.header("Train the Transformer")
    st.markdown("Launch an end-to-end training process on the selected stock. This will overwrite `best_model.pth`.")
    
    if st.button(" Start Training", use_container_width=True):
        progress_text = "Fetching Full History..."
        my_bar = st.progress(0, text=progress_text)
        
        input_features, feature_scaler, time_scaler, close_scaler, scaled_close = get_stock_data(ticker)
        
        if input_features is not None:
            my_bar.progress(10, text="Generating Sequences...")
            
            x, y = create_sequences(input_features, scaled_close, in_seq_length=seq_length, out_seq_len=5)
            x_tensor, y_tensor = torch.tensor(x), torch.tensor(y)
            X_train, X_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, shuffle=False)
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
            
            input_dim = input_features.shape[1]
            model = StockTransformer(input_dim=input_dim, d_model=64, nhead=4, 
                                     num_encoder_layers=2, num_decoder_layers=2, seq_length=seq_length)
            
            my_bar.progress(30, text=f"Training for {epochs} epochs...")
            
            with st.spinner("Training in progress... Check terminal for detailed logs."):
                predictions, targets, idx, train_losses, val_losses = train_and_evaluate(
                    model, train_loader, test_loader, close_scaler, 
                    num_epochs=epochs, learning_rate=lr
                )
            
            my_bar.progress(100, text="Training Complete!")
            st.success("Model trained successfully! Saved as 'best_model.pth'.")
            st.balloons()
            
            st.markdown("### Training Summary")
            st.write(f"Final Validation Loss: {val_losses[-1]:.6f}")
            
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
            loss_fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
            loss_fig.update_layout(title="Learning Curve", xaxis_title="Epoch", yaxis_title="Huber Loss", template="plotly_dark")
            st.plotly_chart(loss_fig, use_container_width=True)
