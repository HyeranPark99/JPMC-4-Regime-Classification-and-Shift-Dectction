# Market Regime Classification & Shift Detection Dashboard

A near-realtime financial dashboard that classifies market regimes for S&P 500 and Bitcoin using Markov Switching models, LSTM price forecasting, and AI-generated commentary via a local LLM — all at zero ongoing cost.

Originally built as a static Jupyter notebook (`SP_500_with_Taylor_rule_(inflation_and_GDP)_RNN_Final_.ipynb`). Modernized into a fully modular Streamlit application with live data feeds and persistent models.

---

## What It Does

- **Regime Classification** — Identifies whether the market is currently in a Bull, Bear, or Transitional regime using a 3-state Markov Switching model
- **Taylor Rule Model** — Enhances S&P 500 regime detection with macro signals (GDP growth + 10-year breakeven inflation) from FRED
- **LSTM Price Forecast** — 12-month forward forecast for S&P 500 and 30-day forecast for Bitcoin
- **Monthly Returns Heatmap** — Year × Month grid showing historical % returns, color-coded red/green
- **AI Commentary** — 2–3 sentence market analysis generated locally via Ollama (phi3:mini), no API costs
- **Live Data** — S&P 500 and Bitcoin prices from yfinance, macro data from FRED, refreshed every 4 hours

---

## Screenshots

> Dashboard renders in a browser at `http://localhost:8501` with:
> - Regime overlay chart (price + colored background bands)
> - Smoothed regime probability chart with actual price overlay
> - Expected regime duration bar chart
> - Monthly returns heatmap (S&P 500 + Bitcoin side by side)
> - LSTM price prediction charts
> - AI commentary panel

---

## Architecture

```
yfinance (^GSPC, BTC-USD)  ──►  data/fetcher.py  ──►  data/cache/*.csv (TTL: 4h)
fredapi (T10YIE, GDP)       ──►       │
                                      ▼
                              data/preprocessor.py
                              (resample, align, Taylor exog)
                                      │
                                      ▼
                              models/trainer.py
                              (staleness check → load or retrain every 7 days)
                              ├── markov_regime.py   (3 variants)
                              └── lstm_predictor.py  (LSTM 64→32)
                                      │
                              llm/commentary.py  (Ollama HTTP → phi3:mini)
                                      │
                              ui/  (Plotly charts + Streamlit layout)
                                      │
                              app.py  (Streamlit entry point)
```

---

## Project Structure

```
├── app.py                      # Streamlit entry point
├── config.py                   # All constants (paths, model params, colors, limits)
├── requirements.txt
├── .env.example                # Copy to .env and add your FRED API key
│
├── data/
│   ├── fetcher.py              # yfinance + FRED API, TTL-based CSV cache
│   └── preprocessor.py        # Resample, interpolate GDP, build Taylor exog matrix
│
├── models/
│   ├── markov_regime.py        # MarkovRegimeModel wrapper (statsmodels)
│   ├── lstm_predictor.py       # LSTMPredictor (Keras LSTM 64→32)
│   └── trainer.py              # Orchestrates training, 7-day staleness check
│
├── llm/
│   └── commentary.py           # Ollama HTTP client (no ollama package required)
│
└── ui/
    ├── charts.py               # Plotly figure factories
    ├── sidebar.py              # Sidebar controls
    ├── layout.py               # Page section renderers
    └── guards.py               # Server safeguards (rate limits, concurrency)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your free FRED API key (get one at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)):

```
FRED_API_KEY=your_key_here
```

> The app works without a FRED key — the Taylor Rule model will be disabled and only the Basic and Volatility-Switching Markov models will be available.

### 5. Install Ollama (for AI commentary)

Download from [ollama.com](https://ollama.com) and pull the model:

```bash
ollama pull phi3:mini
```

> The dashboard works without Ollama — the AI commentary panel will show setup instructions instead.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First run:** Models train from scratch — expect ~2 minutes. Subsequent runs load from disk instantly.

---

## Configuration

All tuneable parameters are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `DATA_START` | `"2018-01-01"` | Historical data start date |
| `DATA_CACHE_TTL_HOURS` | `4` | How often live data refreshes |
| `MODEL_RETRAIN_INTERVAL_DAYS` | `7` | Days before models auto-retrain |
| `MARKOV_K_REGIMES` | `3` | Number of market regimes |
| `LSTM_N_STEPS` | `50` | Lookback window for BTC LSTM (days) |
| `LSTM_EPOCHS` | `100` | Training epochs |
| `MAX_CONCURRENT_USERS` | `5` | Server concurrency limit |
| `RETRAIN_COOLDOWN_HOURS` | `6` | Minimum time between retrains |
| `MIN_REFRESH_INTERVAL_MINUTES` | `5` | Fastest allowed auto-refresh |

---

## Models

### Markov Regime Switching (statsmodels)

Three variants selectable from the sidebar:

| Variant | Asset | Exogenous Variables |
|---|---|---|
| Basic | S&P 500 or Bitcoin | None |
| Volatility-Switching | S&P 500 | None (switching variance) |
| Taylor Rule | S&P 500 | Lagged S&P 500, GDP, Inflation |

Regime labels are assigned dynamically by sorting estimated regime means — lowest mean = Bear, highest = Bull, middle = Transitional.

### LSTM Price Predictor (TensorFlow/Keras)

Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`

| Asset | Input frequency | Lookback | Forecast horizon |
|---|---|---|---|
| S&P 500 | Monthly | 12 months | 12 months |
| Bitcoin | Daily | 50 days | 30 days |

Scaler is fitted on training data only (fixes data leakage present in the original notebook).

---

## Bugs Fixed from Original Notebook

| Bug | Original | Fixed |
|---|---|---|
| Price resampling | `.sum()` on close prices | `.last()` |
| Scaler data leakage | Scaler re-fitted on each split | Fitted on train only, transform applied to val/test |
| No model persistence | Retrains from scratch every run | mtime-based 7-day cache |
| GDP interpolation | Broke when FRED uses quarter-start dates | Union-index interpolation fix |

---

## Deployment (Oracle Cloud Free Tier)

The recommended zero-cost deployment option is Oracle Cloud Always Free (4 ARM cores, 24 GB RAM — enough to run Streamlit + Ollama simultaneously).

See the [full deployment guide](#) for step-by-step instructions covering:
- VM creation and SSH setup
- Firewall configuration
- Ollama installation
- systemd service for auto-restart on reboot

---

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- See `requirements.txt` for full list

---

## License

MIT
