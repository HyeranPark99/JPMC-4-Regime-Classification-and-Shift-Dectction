from __future__ import annotations

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

_PROMPT_TEMPLATE = """\
You are a concise financial analyst writing for an investor dashboard. \
The {asset} market is currently in a '{regime_label}' regime (Regime {regime_id} of 3, \
probability {probability:.0%}). \
Current price: {price:.2f}. \
{macro_line}\
The model estimates this regime has lasted approximately {duration_months:.1f} months. \
In 2-3 sentences, give a factual, plain-English market commentary. No advice, no bullet points.\
"""


def is_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def generate_regime_commentary(
    regime_summary: dict,
    latest_metrics: dict,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    regime_summary keys: current_regime, label, probability, duration_months
    latest_metrics keys: price, asset (str), inflation (optional), gdp_growth (optional)
    """
    macro_parts = []
    if "inflation" in latest_metrics and latest_metrics["inflation"] is not None:
        macro_parts.append(f"Inflation: {latest_metrics['inflation']:.2f}%.")
    if "gdp_growth" in latest_metrics and latest_metrics["gdp_growth"] is not None:
        macro_parts.append(f"GDP: {latest_metrics['gdp_growth']:.2f}.")
    macro_line = " ".join(macro_parts) + " " if macro_parts else ""

    prompt = _PROMPT_TEMPLATE.format(
        asset=latest_metrics.get("asset", "S&P 500"),
        regime_label=regime_summary.get("label", "Unknown"),
        regime_id=regime_summary.get("current_regime", 0),
        probability=regime_summary.get("probability", 0.0),
        price=latest_metrics.get("price", 0.0),
        macro_line=macro_line,
        duration_months=regime_summary.get("duration_months", 0.0),
    )

    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as exc:
        return f"[Commentary unavailable: {exc}]"


def list_available_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
