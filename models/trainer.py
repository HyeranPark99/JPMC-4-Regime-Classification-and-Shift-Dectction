import time
from pathlib import Path

from config import MODEL_RETRAIN_INTERVAL_DAYS, MODELS_DIR


def _stale(path: Path, days: int = MODEL_RETRAIN_INTERVAL_DAYS) -> bool:
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > days


def train_all_models(
    sp_monthly,
    btc_daily,
    gdp_monthly,
    inflation_monthly,
    force: bool = False,
    progress_cb=None,
) -> dict:
    """
    Train (or load) all models. Returns a dict with fitted model objects.
    progress_cb(message: str) is called at each step for UI feedback.
    """
    from models.lstm_predictor import LSTMPredictor
    from models.markov_regime import MarkovRegimeModel
    from data.preprocessor import build_taylor_exog, resample_to_monthly

    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    results = {}

    # ── Paths ────────────────────────────────────────────────────────────────
    p_markov_sp = MODELS_DIR / "markov_sp500.pkl"
    p_markov_btc = MODELS_DIR / "markov_btc.pkl"
    p_markov_taylor = MODELS_DIR / "markov_taylor.pkl"
    p_lstm_model = MODELS_DIR / "lstm_btc.keras"
    p_lstm_meta = MODELS_DIR / "lstm_btc_meta.pkl"

    # ── Markov S&P 500 ───────────────────────────────────────────────────────
    if force or _stale(p_markov_sp):
        _log("Training Markov model (S&P 500)…")
        m_sp = MarkovRegimeModel()
        m_sp.fit(sp_monthly)
        m_sp.save(p_markov_sp)
    else:
        _log("Loading Markov model (S&P 500)…")
        m_sp = MarkovRegimeModel.load(p_markov_sp)
    results["markov_sp"] = m_sp

    # ── Markov Bitcoin ───────────────────────────────────────────────────────
    btc_monthly = resample_to_monthly(btc_daily, "Close")
    if force or _stale(p_markov_btc):
        _log("Training Markov model (Bitcoin)…")
        m_btc = MarkovRegimeModel()
        m_btc.fit(btc_monthly)
        m_btc.save(p_markov_btc)
    else:
        _log("Loading Markov model (Bitcoin)…")
        m_btc = MarkovRegimeModel.load(p_markov_btc)
    results["markov_btc"] = m_btc

    # ── Markov Taylor Rule ───────────────────────────────────────────────────
    taylor_available = (
        gdp_monthly is not None
        and len(gdp_monthly) > 10
        and inflation_monthly is not None
        and len(inflation_monthly) > 10
    )
    if taylor_available:
        if force or _stale(p_markov_taylor):
            _log("Training Markov model (Taylor Rule)…")
            endog, exog = build_taylor_exog(sp_monthly, gdp_monthly, inflation_monthly)
            if len(endog) < 10:
                _log("Taylor Rule model skipped: insufficient aligned data after joining series.")
                results["markov_taylor"] = None
            else:
                m_taylor = MarkovRegimeModel()
                m_taylor.fit(endog, exog=exog)
                m_taylor.save(p_markov_taylor)
                results["markov_taylor"] = m_taylor
        else:
            _log("Loading Markov model (Taylor Rule)…")
            results["markov_taylor"] = MarkovRegimeModel.load(p_markov_taylor)
    else:
        results["markov_taylor"] = None

    # ── LSTM Bitcoin ─────────────────────────────────────────────────────────
    if force or _stale(p_lstm_model):
        _log("Training LSTM (Bitcoin)…")
        btc_close = btc_daily["Close"].dropna()
        lstm = LSTMPredictor()
        lstm.fit(btc_close)
        lstm.save(p_lstm_model, p_lstm_meta)
    else:
        _log("Loading LSTM (Bitcoin)…")
        lstm = LSTMPredictor.load(p_lstm_model, p_lstm_meta)
    results["lstm_btc"] = lstm

    # ── LSTM S&P 500 (monthly) ────────────────────────────────────────────────
    p_lstm_sp_model = MODELS_DIR / "lstm_sp500.keras"
    p_lstm_sp_meta = MODELS_DIR / "lstm_sp500_meta.pkl"
    if force or _stale(p_lstm_sp_model):
        _log("Training LSTM (S&P 500)…")
        # Monthly series — use 12-month lookback window instead of 50-day default
        lstm_sp = LSTMPredictor(n_steps=12)
        lstm_sp.fit(sp_monthly)
        lstm_sp.save(p_lstm_sp_model, p_lstm_sp_meta)
    else:
        _log("Loading LSTM (S&P 500)…")
        lstm_sp = LSTMPredictor.load(p_lstm_sp_model, p_lstm_sp_meta)
    results["lstm_sp500"] = lstm_sp

    _log("Done.")
    return results


def model_status() -> dict:
    """Returns age info for each saved model file."""
    files = {
        "Markov S&P 500": MODELS_DIR / "markov_sp500.pkl",
        "Markov Bitcoin": MODELS_DIR / "markov_btc.pkl",
        "Markov Taylor": MODELS_DIR / "markov_taylor.pkl",
        "LSTM Bitcoin": MODELS_DIR / "lstm_btc.keras",
    }
    status = {}
    for label, path in files.items():
        if not path.exists():
            status[label] = "not trained"
        else:
            age_days = (time.time() - path.stat().st_mtime) / 86400
            status[label] = f"trained {age_days:.1f}d ago"
    return status
