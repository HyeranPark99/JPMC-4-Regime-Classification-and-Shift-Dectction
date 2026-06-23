from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
MODELS_DIR = ROOT_DIR / "models" / "saved"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
DATA_START = "2018-01-01"
DATA_CACHE_TTL_HOURS = 4
MODEL_RETRAIN_INTERVAL_DAYS = 7

# ── Markov ─────────────────────────────────────────────────────────────────
MARKOV_K_REGIMES = 3

# ── LSTM ───────────────────────────────────────────────────────────────────
LSTM_N_STEPS = 50
LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# ── Regime display ─────────────────────────────────────────────────────────
REGIME_COLORS = {
    0: "rgba(214, 39, 40, 0.15)",   # red   – bear / high vol
    1: "rgba(44, 160, 44, 0.15)",   # green – bull / low vol
    2: "rgba(31, 119, 180, 0.15)",  # blue  – transitional
}
REGIME_LINE_COLORS = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}
REGIME_LABELS = {0: "Bear / High Volatility", 1: "Bull / Low Volatility", 2: "Transitional"}

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"

# ── Server safeguards ───────────────────────────────────────────────────────
MAX_CONCURRENT_USERS = 5          # sessions active within the last 5 minutes
RETRAIN_COOLDOWN_HOURS = 6        # minimum hours between model retrains
MIN_REFRESH_INTERVAL_MINUTES = 5  # fastest auto-refresh allowed
