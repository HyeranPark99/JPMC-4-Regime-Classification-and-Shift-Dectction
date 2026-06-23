import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data.fetcher import fetch_btc, fetch_inflation, fetch_gdp, fetch_sp500
from data.preprocessor import (
    interpolate_gdp_to_monthly,
    resample_inflation_monthly,
    resample_to_monthly,
)
from models.trainer import train_all_models
from ui.layout import (
    render_commentary,
    render_duration_bar,
    render_header,
    render_lstm_section,
    render_main_chart,
    render_monthly_heatmap,
    render_probability_chart,
    render_regime_metrics,
)
from ui.guards import (
    check_concurrent_users,
    check_retrain_allowed,
    enforce_min_refresh,
    record_retrain,
    update_session_heartbeat,
)
from ui.sidebar import render_sidebar


def _filter_by_year(series: pd.Series, start_year: str, end_year: str) -> pd.Series:
    mask = (series.index.year >= int(start_year)) & (series.index.year <= int(end_year))
    return series[mask]


def _filter_df_by_year(df: pd.DataFrame, start_year: str, end_year: str) -> pd.DataFrame:
    mask = (df.index.year >= int(start_year)) & (df.index.year <= int(end_year))
    return df[mask]


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    sp500 = fetch_sp500()
    btc = fetch_btc()
    inflation_raw = fetch_inflation()
    gdp_raw = fetch_gdp()
    return sp500, btc, inflation_raw, gdp_raw


def main():
    # ── Server safeguards ─────────────────────────────────────────────────────
    update_session_heartbeat()
    check_concurrent_users()

    controls = render_sidebar()

    # Auto-refresh — enforce minimum interval
    refresh_interval = enforce_min_refresh(controls["refresh_interval"])
    if refresh_interval > 0:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=refresh_interval * 60_000, key="autorefresh")
        except ImportError:
            st.sidebar.warning("`streamlit-autorefresh` not installed. Run: pip install streamlit-autorefresh")

    render_header(last_updated=pd.Timestamp.now())

    # ── Load raw data ─────────────────────────────────────────────────────
    with st.spinner("Loading market data…"):
        sp500_df, btc_df, inflation_raw, gdp_raw = load_data()

    # ── Preprocess ────────────────────────────────────────────────────────
    sp_monthly = resample_to_monthly(sp500_df, "Close")
    btc_monthly = resample_to_monthly(btc_df, "Close")
    btc_daily = btc_df["Close"].dropna()

    inflation_monthly = (
        resample_inflation_monthly(inflation_raw) if len(inflation_raw) > 0 else None
    )
    gdp_monthly = (
        interpolate_gdp_to_monthly(gdp_raw) if len(gdp_raw) > 0 else None
    )

    # Apply year filter
    ds, de = controls["date_start"], controls["date_end"]
    sp_filtered = _filter_by_year(sp_monthly, ds, de)
    btc_daily_filtered = _filter_by_year(btc_daily, ds, de)
    btc_monthly_filtered = _filter_by_year(btc_monthly, ds, de)

    # ── Retrain guard ─────────────────────────────────────────────────────
    force_retrain = controls["force_retrain"]
    if force_retrain:
        retrain_ok, retrain_msg = check_retrain_allowed()
        if not retrain_ok:
            st.warning(f"Retrain blocked: {retrain_msg}")
            force_retrain = False

    # ── Train / load models ───────────────────────────────────────────────
    with st.spinner("Loading models (first run trains from scratch — may take a minute)…"):
        logs = []
        models = train_all_models(
            sp_monthly=sp_monthly,
            btc_daily=btc_df,
            gdp_monthly=gdp_monthly,
            inflation_monthly=inflation_monthly,
            force=force_retrain,
            progress_cb=logs.append,
        )
        if force_retrain:
            record_retrain()

    # ── Select active Markov model ────────────────────────────────────────
    variant = controls["markov_variant"]
    if variant == "Basic":
        if controls["asset"] == "S&P 500":
            active_markov = models["markov_sp"]
            active_monthly = sp_filtered
            asset_label = "S&P 500"
        else:
            active_markov = models["markov_btc"]
            active_monthly = btc_monthly_filtered
            asset_label = "Bitcoin"
    elif variant == "Volatility-Switching":
        active_markov = models["markov_sp"]
        active_monthly = sp_filtered
        asset_label = "S&P 500"
    else:  # Taylor Rule
        active_markov = models.get("markov_taylor")
        active_monthly = sp_filtered
        asset_label = "S&P 500 (Taylor Rule)"

    regime_labels = active_markov.get_regime_labels() if active_markov else {}

    # ── Latest macro values for commentary ───────────────────────────────
    latest_inflation = (
        float(inflation_monthly.iloc[-1]) if inflation_monthly is not None and len(inflation_monthly) else None
    )
    latest_gdp = (
        float(gdp_monthly.iloc[-1]) if gdp_monthly is not None and len(gdp_monthly) else None
    )
    latest_price = float(active_monthly.iloc[-1]) if len(active_monthly) else 0.0

    # ── Page layout ───────────────────────────────────────────────────────
    render_regime_metrics(active_markov, asset_label)
    st.divider()

    # Hero chart
    render_main_chart(active_monthly, active_markov, regime_labels, asset_label)

    # Probability + duration side by side
    col_left, col_right = st.columns([3, 1])
    with col_left:
        render_probability_chart(active_markov, regime_labels, price_series=active_monthly)
    with col_right:
        render_duration_bar(active_markov, regime_labels)

    # Monthly returns heatmaps
    st.divider()
    st.subheader("Monthly Returns")
    col_sp, col_btc = st.columns(2)
    with col_sp:
        render_monthly_heatmap(sp_filtered, "S&P 500")
    with col_btc:
        render_monthly_heatmap(btc_monthly_filtered, "Bitcoin")

    # AI commentary (only re-generate on button press or first load)
    st.divider()
    if st.button("Generate AI Commentary") or "commentary_cache" not in st.session_state:
        render_commentary(
            active_markov,
            price=latest_price,
            asset=asset_label,
            ollama_model=controls["ollama_model"],
            inflation=latest_inflation,
            gdp=latest_gdp,
        )
        # Cache the commentary generation trigger
        st.session_state["commentary_cache"] = True
    else:
        st.info("Click 'Generate AI Commentary' to refresh the AI analysis.")

    # LSTM S&P 500 section
    st.divider()
    render_lstm_section(
        models.get("lstm_sp500"),
        sp_filtered,
        title="S&P 500 — LSTM Price Prediction",
        n_forecast=12,
        freq="ME",
    )

    # LSTM Bitcoin section
    st.divider()
    render_lstm_section(
        models.get("lstm_btc"),
        btc_daily_filtered,
        title="Bitcoin — LSTM Price Prediction",
        n_forecast=30,
        freq="D",
    )

    # Training log expander
    if logs:
        with st.expander("Training Log"):
            for msg in logs:
                st.text(msg)


if __name__ == "__main__":
    main()
