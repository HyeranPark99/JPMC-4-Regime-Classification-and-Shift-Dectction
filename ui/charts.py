from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import REGIME_COLORS, REGIME_LINE_COLORS


def regime_overlay_chart(
    price_series: pd.Series,
    dominant_regime: pd.Series,
    regime_labels: dict,
    title: str = "Price with Regime Overlay",
) -> go.Figure:
    """Line chart with shaded background bands per market regime."""
    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name="Price",
            line=dict(color="#444", width=1.5),
        )
    )

    # Shade each regime period
    if dominant_regime is not None and len(dominant_regime) > 0:
        combined = dominant_regime.reindex(price_series.index, method="ffill").dropna()
        if len(combined) > 0:
            prev_regime = int(combined.iloc[0])
            start = combined.index[0]
            for date, regime in combined.items():
                regime = int(regime)
                if regime != prev_regime:
                    fig.add_vrect(
                        x0=start,
                        x1=date,
                        fillcolor=REGIME_COLORS.get(prev_regime, "rgba(128,128,128,0.1)"),
                        layer="below",
                        line_width=0,
                        annotation_text=regime_labels.get(prev_regime, f"R{prev_regime}"),
                        annotation_position="top left",
                        annotation=dict(font_size=9, font_color="grey"),
                    )
                    start = date
                    prev_regime = regime
            # Final segment
            fig.add_vrect(
                x0=start,
                x1=combined.index[-1],
                fillcolor=REGIME_COLORS.get(prev_regime, "rgba(128,128,128,0.1)"),
                layer="below",
                line_width=0,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def regime_probability_chart(
    smoothed_probs: pd.DataFrame,
    regime_labels: dict,
    price_series: pd.Series | None = None,
    title: str = "Smoothed Regime Probabilities",
) -> go.Figure:
    """Stacked area chart of regime probabilities over time.

    If price_series is provided it is overlaid on a secondary Y-axis so the
    reader can see how probability shifts track actual price movements.
    """
    has_price = price_series is not None and len(price_series) > 0
    fig = make_subplots(specs=[[{"secondary_y": has_price}]])

    for col in smoothed_probs.columns:
        regime_idx = int(col.split()[-1])
        label = regime_labels.get(regime_idx, col)
        color = REGIME_LINE_COLORS.get(regime_idx, "#888")
        fig.add_trace(
            go.Scatter(
                x=smoothed_probs.index,
                y=smoothed_probs[col],
                mode="lines",
                name=label,
                stackgroup="one",
                line=dict(color=color, width=1),
                fillcolor=REGIME_COLORS.get(regime_idx, "rgba(128,128,128,0.2)"),
            ),
            secondary_y=False,
        )

    if has_price:
        asset_name = price_series.name or "Price"
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                name=f"{asset_name} (actual)",
                line=dict(color="#222", width=1.5, dash="dot"),
                opacity=0.7,
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(title="Probability", range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=60, t=60, b=40),
    )
    return fig


def regime_duration_bar(
    expected_durations: dict,
    regime_labels: dict,
    title: str = "Expected Regime Duration (months)",
) -> go.Figure:
    labels = [regime_labels.get(k, f"Regime {k}") for k in sorted(expected_durations)]
    values = [expected_durations[k] for k in sorted(expected_durations)]
    colors = [REGIME_LINE_COLORS.get(k, "#888") for k in sorted(expected_durations)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f} mo" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Months",
        template="plotly_white",
        margin=dict(l=20, r=60, t=50, b=30),
        height=200,
    )
    return fig


def lstm_prediction_chart(
    actual: pd.Series,
    insample_pred: pd.Series | None,
    forecast: pd.Series | None,
    title: str = "Bitcoin Price — LSTM Prediction",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    if insample_pred is not None and len(insample_pred) > 0:
        fig.add_trace(
            go.Scatter(
                x=insample_pred.index,
                y=insample_pred.values,
                mode="lines",
                name="In-sample fit",
                line=dict(color="#ff7f0e", width=1, dash="dot"),
            )
        )

    if forecast is not None and len(forecast) > 0:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode="lines",
                name="30-day forecast",
                line=dict(color="#2ca02c", width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def monthly_returns_heatmap(
    price_series: pd.Series,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Year x Month heatmap of percentage returns, green=positive, red=negative."""
    monthly = price_series.resample("ME").last().dropna()
    returns = monthly.pct_change().dropna() * 100

    pivot = returns.groupby([returns.index.year, returns.index.month]).first().unstack()
    # Reindex so all 12 months are always present (missing months become NaN)
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    z = pivot.values
    text = [[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=[str(yr) for yr in pivot.index.tolist()],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "rgba(214,39,40,0.85)"],
                [0.5, "rgba(255,255,255,0.9)"],
                [1.0, "rgba(44,160,44,0.85)"],
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(title="Return %", thickness=12),
            hoverongaps=False,
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=60, r=60, t=80, b=20),
        height=max(250, 40 * len(pivot) + 100),
    )
    return fig


def training_loss_chart(history: dict) -> go.Figure:
    fig = go.Figure()
    if "loss" in history:
        fig.add_trace(
            go.Scatter(y=history["loss"], mode="lines", name="Train Loss",
                       line=dict(color="#1f77b4"))
        )
    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(y=history["val_loss"], mode="lines", name="Val Loss",
                       line=dict(color="#ff7f0e", dash="dash"))
        )
    fig.update_layout(
        title="LSTM Training Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
        height=250,
    )
    return fig
