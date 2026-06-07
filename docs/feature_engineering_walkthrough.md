# Feature Engineering Walkthrough

To transition from a single-stock model to a **Multi-Stock Global Transformer**, we overhauled the data preprocessing pipeline to produce scale-agnostic, stationary inputs, and updated the model architecture to support joint categorical and numerical embeddings.

## 1. Feature Engineering Overhaul
Every feature based on raw prices, moving averages, or shares volume has been transformed to ensure stationarity and comparability across stocks with vastly different valuations.

### Stationary Transformations
* **Log Returns (OHLCV)**: Raw prices (`Open`, `High`, `Low`, `Close`) are converted to log returns relative to the previous day's close.
  $$\text{Close\_Log\_Return}_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$
  $$\text{Open\_Log\_Return}_t = \ln\left(\frac{\text{Open}_t}{\text{Close}_{t-1}}\right)$$
  $$\text{Volume\_Log\_Return}_t = \ln\left(\frac{\text{Volume}_t}{\text{Volume}_{t-1}}\right)$$
* **Distance Percentages (Moving Averages)**: Moving averages (`12-day EMA`, `24-day EMA`, `BB_MA`, `VWAP`) are converted to logarithmic distance percentages from the daily close:
  $$\text{EMA12\_dist}_t = \ln\left(\frac{\text{Close}_t}{\text{EMA12}_t}\right)$$
* **Bollinger Bands %B**: Expresses the price position relative to the Bollinger Bands (1.0 = upper band, 0.0 = lower band):
  $$\%B = \frac{\text{Close} - \text{BB\_Lower}}{\text{BB\_Upper} - \text{BB\_Lower}}$$
* **Percentage MACD**: The raw MACD spread is scaled by the 24-day EMA:
  $$\text{Percentage\_MACD} = \frac{\text{EMA12} - \text{EMA24}}{\text{EMA24}}$$
* **Relative Volatility**: Rolling standard deviation and Average True Range (ATR) are scaled by the daily close:
  $$\text{Relative\_ATR} = \frac{\text{ATR}}{\text{Close}}$$

### Cyclical Calendar Features (Method B)
* Linear ordinal timestamps have been replaced by the raw **Day of Year (1–366)**. This allows the `Time2Vec` layer to learn recurring cyclical calendar patterns (e.g., tax-loss harvesting, window dressing) that affect all stocks simultaneously.

### Cross-Asset Market Context
* Added daily **S&P 500 (^GSPC) Log Returns** aligned with the stock's timeline. This helps the Transformer distinguish between systemic market moves and stock-specific volatility. A two-tier download fallback (trying `period='max'` first, then `period='10y'`) guarantees robust downloads.

### Categorical Identifiers
* Added integer-encoded identifiers:
  * `Stock_ID` (0 to 40)
  * `Sector_ID` (0 to 5: Technology, Communication, Financials, Healthcare, Consumer, Industrials/Energy)

---

## 2. Model Architecture & Concatenation Flow
The `StockTransformer` embeds both numeric inputs and discrete stock/sector metadata, feeding them jointly into the encoder.

```mermaid
graph TD
    subgraph Input Sequences
        T[Day of Year [batch, seq_len, 1]]
        N[15 Scaled Numerical Features [batch, seq_len, 15]]
        S[Stock ID [batch, seq_len, 1]]
        SEC[Sector ID [batch, seq_len, 1]]
    end

    T --> T2V[Time2Vec Layer]
    T2V --> |33 dimensions| C[Concat Layer]
    N --> |15 dimensions| C
    S --> SE[Stock Embedding Layer] --> |16 dimensions| C
    SEC --> SCE[Sector Embedding Layer] --> |8 dimensions| C

    C --> |72 dimensions| FC[fc_in Linear Projection]
    FC --> |d_model=64| LN[LayerNorm]
    LN --> TE[Transformer Encoder]
```

---

## 3. Autoregressive Prediction and Inverse Transformation
Because the model predicts log returns ($r_t$), predicting future dollar prices requires reconstructing the compounding return path from the last known actual close price ($P_0$):

$$P_{k} = P_{k-1} \times e^{r_k}$$

The updated `predict.py` and Streamlit dashboard (`app.py`) handle this automatically:
1. Perform autoregressive inference to predict next 5 days of scaled log returns.
2. Invert the scaling using the fitted `close_scaler`.
3. Fetch the last valid closing price of the stock.
4. Calculate the compounding price trajectory and display actual vs. predicted values in original USD format.
