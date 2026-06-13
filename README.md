# AIFIS — AI Financial Intelligence System

**A production-grade, Transformer-powered stock forecasting platform with live news verification, LLM sentiment adjustment, and an interactive web dashboard.**

**Author:** Abhishek Kotwani
**Status:** Production Ready

---

## Overview

AIFIS is a full-stack financial intelligence platform that combines a custom **Decoder-Only Causal Transformer** with real-time market data, multi-source news aggregation, and LLM-powered sentiment analysis to generate 5-day price forecasts for **41 US equities** across 6 sectors.

The system goes beyond raw model predictions — it cross-references live news through a proprietary **Truth Engine**, then passes verified headlines to a **Groq-hosted Llama 3.1** model that applies bounded sentiment adjustments to the Transformer's output, producing both a baseline and an AI-enhanced forecast.

Everything is served through a single FastAPI backend powering a responsive web dashboard with real-time charts, technical indicators, and AI-generated research reports.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)                     │
│    Charts · Dual Predictions · Research Reports · News      │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend (api/)                     │
│                                                             │
│  /market/{ticker}    → Technical snapshot (RSI, MACD, BB)   │
│  /predict/{ticker}   → Transformer + LLM-adjusted forecast  │
│  /news/{ticker}      → Truth-verified news feed             │
│  /research/{ticker}  → Full AI research report              │
│  /analyze/{ticker}   → Combined analysis                    │
└────┬──────────┬───────────┬──────────┬──────────────────────┘
     │          │           │          │
     ▼          ▼           ▼          ▼
  yfinance   News APIs   Transformer  Groq LLM
  (market)   (GNews,     (PyTorch)    (Llama 3.1)
             Finnhub,
             NewsAPI)
```

---

## Supported Stock Universe (41 Tickers)

| Sector | Tickers |
|---|---|
| **Technology & Semiconductors** | AAPL, MSFT, NVDA, AVGO, ORCL, AMD, CRM |
| **Communication & Digital Media** | GOOGL, META, NFLX, DIS, TMUS, CMCSA |
| **Financials** | JPM, BAC, MS, GS, V, MA, AXP |
| **Healthcare** | LLY, UNH, JNJ, MRK, ABBV, TMO, ISRG |
| **Consumer** | AMZN, TSLA, WMT, COST, HD, NKE, KO |
| **Industrials & Energy** | XOM, CVX, CAT, GE, UNP, HON, ETN |

---

## Key Features

### Transformer Model
- **Decoder-Only Causal Architecture** (GPT-style) with causal self-attention masking
- Learnable positional embeddings with stock-specific and sector-specific conditioning via `nn.Embedding` layers
- Pre-LayerNorm, GELU activations, and high dropout (0.25) — tuned for noisy financial data
- Predicts **5-day log returns** from a 60-day input window
- Trained globally across all 41 stocks — shared attention weights learn cross-asset patterns

### Scale-Agnostic Feature Engineering
All input features are stationary and scale-independent, enabling a single global model to handle stocks at any price level:

| Category | Features |
|---|---|
| **Returns** | Close/Open/High/Low log returns |
| **Volatility** | Bollinger Band Squeeze, Distance to Channel High, Volume Force Multiplier |
| **Trend** | Trend Alignment Ratio (EMA50/EMA20), Pullback Proximity |
| **Mean Reversion** | Price Z-Score Distance, RSI Asymmetry |
| **Macro** | S&P 500 log return (beta shield), VIX relative height |
| **Temporal** | Cyclical day-of-week and month-of-year encodings (sin/cos) |

### News Truth Engine
A 3-pillar verification system that scores every scraped headline before it reaches the model:
1. **Cross-Source Consistency** — Matches headlines across yfinance, NewsAPI, and Finnhub to verify corroboration
2. **Source Credibility** — Weighted scoring based on outlet reliability (yfinance: 1.0, Finnhub: 0.9, NewsAPI: 0.7)
3. **Temporal Freshness** — Exponential decay penalizes stale articles

Only articles passing the truth threshold are forwarded to the LLM for sentiment analysis.

### LLM Sentiment Adjustment (Groq + Llama 3.1)
- The Transformer's raw 5-day log returns are sent to Llama 3.1 8B along with truth-verified headlines
- The LLM suggests per-day adjustments within a **strict ±0.3% delta boundary**
- Programmatic clamping in Python enforces the boundary — the LLM prompt alone is never trusted
- The UI renders both the **Original Prediction** and the **AI-Enhanced Prediction** side by side

### AI Research Reports
- Full-length equity research reports generated via Groq LLM
- Combines technical indicators, sentiment analysis, and Transformer forecasts into structured JSON
- Includes: Executive Summary, Technical Analysis, Sentiment Drivers, Price Forecast, Risk Factors, Catalysts, and a final Rating (Strong Buy → Strong Sell)
- Falls back to a deterministic data-only report if the LLM is unavailable

### Interactive Web Dashboard
- Real-time market snapshot with RSI, MACD, Bollinger Bands, ATR, VWAP, and volume analysis
- Dual-line prediction chart (dashed baseline + solid AI-adjusted)
- Prediction schedule table with Original Price and AI Enhanced Price columns
- Live news feed with truth scores and source badges
- LLM Sentiment Context display showing the model's rationale
- Fully responsive design with dark mode, glassmorphism, and micro-animations

---

## Project Structure

```
Stock_Transformer/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry point, CORS, static file serving
│   └── routes/
│       ├── market.py           # GET  /market/{ticker}
│       ├── predict.py          # POST /predict/{ticker}
│       ├── news.py             # GET  /news/{ticker}
│       ├── research.py         # POST /research/{ticker}
│       └── analyze.py          # POST /analyze/{ticker}
│
├── src/                        # Core Python modules
│   ├── model.py                # StockTransformer (decoder-only causal)
│   ├── features.py             # Scale-agnostic feature engineering
│   ├── train.py                # Training loop with TensorBoard logging
│   ├── news.py                 # Multi-source scraper + Truth Engine
│   ├── sentiment.py            # FinBERT / keyword-fallback sentiment
│   ├── cache.py                # yfinance data caching (15-min TTL)
│   ├── config.py               # Centralized .env-driven configuration
│   ├── data.py                 # Dataset and DataLoader utilities
│   └── visualization.py        # Matplotlib plotting helpers
│
├── frontend/                   # Single-page web dashboard
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── models/                     # Pre-fitted scalers (.joblib)
├── best_model.pth              # Trained model weights
├── main.py                     # CLI training entry point
├── predict.py                  # CLI prediction script
├── requirements.txt            # Python dependencies
├── build_deployable.sh         # Automated deployment packager
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/Abhi-sheKkK/Stock_Transformer.git
cd Stock_Transformer
```

### 2. Create Virtual Environment
```bash
python3 -m venv stock_env
source stock_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and set your API keys:
```env
# Required for LLM sentiment adjustment
GROQ_API_KEY=your_groq_api_key_here

# Optional — enriches news coverage
NEWS_API_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key
```

### 4. Launch the Application
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000** — the dashboard loads automatically.

---

## Usage

### Web Dashboard
1. Select a stock ticker from the dropdown (e.g., AAPL, NFLX, TSLA)
2. View the live market snapshot with technical indicators and signal interpretations
3. Click **Run Forecast** to generate the 5-day prediction with LLM sentiment adjustment
4. Click **Generate Research Report** for a full AI equity analysis

### CLI Prediction
```bash
python predict.py --ticker AAPL
```
Outputs the 5-day price forecast directly to terminal.

### CLI Training
```bash
python main.py
```
Trains the global Transformer model across all 41 stocks with TensorBoard logging.

---

## Caching Strategy

All external API calls follow a **15-minute cache window**:

| Data Source | Cache Location | TTL |
|---|---|---|
| Stock prices (yfinance) | `.cache/stocks/` | 15 minutes |
| News articles (GNews, Finnhub, NewsAPI) | `.cache/news/` | 15 minutes |

- **First request**: Fetches fresh data from the API and saves to local cache
- **Within 15 minutes**: Serves instantly from cache (no API call)
- **After 15 minutes**: Automatically fetches fresh data on next request

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/market/{ticker}` | Live technical snapshot (RSI, MACD, BB, ATR, VWAP) |
| `GET` | `/news/{ticker}` | Truth-verified news feed with sentiment scores |
| `POST` | `/predict/{ticker}` | 5-day forecast (original + AI-enhanced prices) |
| `POST` | `/research/{ticker}` | Full AI-generated research report |
| `POST` | `/analyze/{ticker}` | Combined analysis with reasoning |

---

## Deployment

### Quick Deploy
```bash
chmod +x build_deployable.sh
./build_deployable.sh
```

This creates `Stock_Transformer_deploy.zip` containing only the files needed for production (excludes virtual environments, caches, git history, and dev files).

### On Your Server
```bash
unzip Stock_Transformer_deploy.zip -d app && cd app
cp .env.example .env        # Configure API keys
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Production Recommendations
- Place uvicorn behind **NGINX** for SSL termination and static asset caching
- Restrict CORS origins in `api/main.py` from `"*"` to your specific domain
- Use a process manager (systemd / PM2) to keep the server running
- The `.cache/` directory is auto-created at runtime — no manual setup needed

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model** | PyTorch (Decoder-Only Causal Transformer) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JavaScript (Canvas charts) |
| **LLM** | Groq Cloud (Llama 3.1 8B Instant) |
| **Data** | yfinance, NewsAPI, Finnhub |
| **Sentiment** | FinBERT (optional) / Keyword fallback |
| **Config** | python-dotenv (.env driven) |
| **Scaling** | scikit-learn StandardScaler + custom Scale100Scaler |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq Cloud API key for LLM sentiment adjustment |
| `GROQ_MODEL` | No | Model name (default: `llama-3.1-8b-instant`) |
| `NEWS_API_KEY` | YES | NewsAPI key for additional news coverage |
| `FINNHUB_API_KEY` | YES | Finnhub key for additional news coverage |
| `NEWS_CACHE_TTL` | No | News cache TTL in minutes (default: `15`) |
| `OLLAMA_BASE_URL` | No | Ollama endpoint for local LLM (default: `http://localhost:11434`) |
| `API_HOST` | No | Server bind host (default: `0.0.0.0`) |
| `API_PORT` | No | Server bind port (default: `8000`) |

---

## License

This project is licensed under the MIT License.

---

## Contact

- **Email:** abhishek.9.kotwani@gmail.com
- **GitHub:** [Abhi-sheKkK](https://github.com/Abhi-sheKkK)
