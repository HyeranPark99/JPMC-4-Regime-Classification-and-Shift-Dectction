from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.charts import (
    lstm_prediction_chart,
    monthly_returns_heatmap,
    regime_duration_bar,
    regime_overlay_chart,
    regime_probability_chart,
    training_loss_chart,
)


def render_header(last_updated: pd.Timestamp | None = None):
    st.title("Market Regime Dashboard")
    if last_updated:
        st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")


def render_regime_metrics(model, asset_label: str):
    """Top-row metric cards for current regime."""
    if model is None:
        st.warning("No model available for this selection.")
        return
    summary = model.summary_dict()
    labels = model.get_regime_labels()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Regime", summary.get("label", "—"))
    c2.metric("Regime Probability", f"{summary.get('probability', 0):.0%}")
    c3.metric(
        "Est. Duration",
        f"{summary.get('duration_months', 0):.1f} mo"
        if summary.get("duration_months") else "—",
    )
    c4.metric("Model AIC", f"{summary.get('aic', 0):.1f}")


def render_main_chart(price_series, model, regime_labels, asset_label):
    if model is None:
        st.info("Markov model not available. Select a different variant or retrain.")
        return
    try:
        dominant = model.get_dominant_regime()
        fig = regime_overlay_chart(price_series, dominant, regime_labels,
                                   title=f"{asset_label} — Regime Overlay")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render regime chart: {e}")


def render_probability_chart(model, regime_labels, price_series: pd.Series | None = None):
    if model is None:
        return
    try:
        probs = model.get_smoothed_probs()
        fig = regime_probability_chart(probs, regime_labels, price_series=price_series)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Probability chart error: {e}")


def render_duration_bar(model, regime_labels):
    if model is None:
        return
    try:
        durations = model.get_expected_durations()
        fig = regime_duration_bar(durations, regime_labels)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Duration chart error: {e}")


def render_commentary(model, price: float, asset: str, ollama_model: str,
                      inflation=None, gdp=None):
    st.subheader("AI Commentary")
    from llm.commentary import generate_regime_commentary, is_ollama_running

    if model is None:
        st.info("No model data for commentary.")
        return

    if not is_ollama_running():
        st.info("Ollama is not running. Start Ollama and pull `phi3:mini` for AI commentary.")
        return

    summary = model.summary_dict()
    metrics = {
        "asset": asset,
        "price": price,
        "inflation": float(inflation) if inflation is not None else None,
        "gdp_growth": float(gdp) if gdp is not None else None,
    }

    with st.spinner("Generating commentary…"):
        text = generate_regime_commentary(summary, metrics, model=ollama_model)
    st.markdown(f"> {text}")


def render_monthly_heatmap(price_series: pd.Series, asset_label: str):
    try:
        fig = monthly_returns_heatmap(price_series, title=f"{asset_label} — Monthly Returns (%)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Monthly heatmap error: {e}")


def render_lstm_section(
    lstm_model,
    series: pd.Series,
    title: str = "Bitcoin — LSTM Price Prediction",
    n_forecast: int = 30,
    freq: str = "D",
):
    st.subheader(title)
    if lstm_model is None:
        st.info("LSTM model not available.")
        return
    try:
        insample = lstm_model.predict_insample(series)
        forecast = lstm_model.predict_next_n(series, n=n_forecast, freq=freq)
        fig = lstm_prediction_chart(series, insample, forecast, title=title)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("LSTM Training Details"):
            if lstm_model.history:
                st.plotly_chart(training_loss_chart(lstm_model.history), use_container_width=True)
            final_loss = (lstm_model.history.get("loss") or [None])[-1]
            final_val = (lstm_model.history.get("val_loss") or [None])[-1]
            if final_loss:
                c1, c2 = st.columns(2)
                c1.metric("Final Train MSE", f"{final_loss:.6f}")
                if final_val:
                    c2.metric("Final Val MSE", f"{final_val:.6f}")
    except Exception as e:
        st.error(f"LSTM section error: {e}")
