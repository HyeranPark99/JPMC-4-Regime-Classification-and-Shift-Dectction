import streamlit as st

from data.fetcher import cache_status
from llm.commentary import is_ollama_running, list_available_models
from models.trainer import model_status


def render_sidebar() -> dict:
    """Render sidebar controls and return user selections as a dict."""
    st.sidebar.title("Controls")

    # ── Asset & model ─────────────────────────────────────────────────────
    asset = st.sidebar.radio("Asset", ["S&P 500", "Bitcoin"], index=0)

    markov_variant = st.sidebar.radio(
        "Markov Model",
        ["Basic", "Volatility-Switching", "Taylor Rule (S&P 500)"],
        index=2,
    )

    # ── Date range ────────────────────────────────────────────────────────
    import pandas as pd

    min_year, max_year = 2018, pd.Timestamp.now().year
    date_start, date_end = st.sidebar.select_slider(
        "Year range",
        options=list(range(min_year, max_year + 1)),
        value=(min_year, max_year),
    )

    # ── Auto-refresh ──────────────────────────────────────────────────────
    refresh_options = {"Off": 0, "1 min": 1, "5 min": 5, "15 min": 15}
    refresh_label = st.sidebar.selectbox("Auto-refresh", list(refresh_options.keys()), index=0)
    refresh_interval = refresh_options[refresh_label]

    # ── Retrain ───────────────────────────────────────────────────────────
    force_retrain = st.sidebar.button("Retrain Models")

    st.sidebar.divider()

    # ── Ollama status ─────────────────────────────────────────────────────
    st.sidebar.subheader("AI Commentary (Ollama)")
    ollama_ok = is_ollama_running()
    if ollama_ok:
        models = list_available_models()
        st.sidebar.success(f"Online — {len(models)} model(s)")
        selected_model = st.sidebar.selectbox(
            "Model", models if models else ["phi3:mini"], index=0
        )
    else:
        st.sidebar.error("Offline")
        st.sidebar.caption(
            "To enable AI commentary:\n"
            "1. Install [Ollama](https://ollama.com)\n"
            "2. Run: `ollama serve`\n"
            "3. Pull a model: `ollama pull phi3:mini`"
        )
        selected_model = "phi3:mini"

    # ── Data status ───────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Data Cache")
    for label, info in cache_status().items():
        icon = "✓" if "fresh" in info or "m ago" in info else "○"
        st.sidebar.caption(f"{icon} {label}: {info}")

    # ── Model status ──────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Model Status")
    for label, info in model_status().items():
        st.sidebar.caption(f"• {label}: {info}")

    return {
        "asset": asset,
        "markov_variant": markov_variant,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "refresh_interval": refresh_interval,
        "force_retrain": force_retrain,
        "ollama_ok": ollama_ok,
        "ollama_model": selected_model,
    }
