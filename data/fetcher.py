import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from config import CACHE_DIR, DATA_CACHE_TTL_HOURS, DATA_START

load_dotenv()


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _is_fresh(path: Path, ttl_hours: float = DATA_CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < ttl_hours


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def fetch_sp500(start: str = DATA_START, force: bool = False) -> pd.DataFrame:
    path = _cache_path("sp500_daily")
    if not force and _is_fresh(path):
        return _load_csv(path)
    df = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    _save_csv(df, path)
    return df


def fetch_btc(start: str = DATA_START, force: bool = False) -> pd.DataFrame:
    path = _cache_path("btc_daily")
    if not force and _is_fresh(path):
        return _load_csv(path)
    df = yf.download("BTC-USD", start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    _save_csv(df, path)
    return df


def fetch_inflation(start: str = DATA_START, force: bool = False) -> pd.Series:
    """10-Year Breakeven Inflation Rate (T10YIE) from FRED."""
    path = _cache_path("inflation_daily")
    if not force and _is_fresh(path):
        return _load_csv(path).squeeze()
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        if path.exists():
            return _load_csv(path).squeeze()
        return pd.Series(dtype=float, name="T10YIE")
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        series = fred.get_series("T10YIE", observation_start=start)
        series.name = "T10YIE"
        series.index = pd.to_datetime(series.index).tz_localize(None)
        df = series.to_frame()
        _save_csv(df, path)
        return series
    except Exception:
        if path.exists():
            return _load_csv(path).squeeze()
        return pd.Series(dtype=float, name="T10YIE")


def fetch_gdp(start: str = DATA_START, force: bool = False) -> pd.Series:
    """Real GDP (quarterly) from FRED."""
    path = _cache_path("gdp_quarterly")
    if not force and _is_fresh(path):
        return _load_csv(path).squeeze()
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        if path.exists():
            return _load_csv(path).squeeze()
        return pd.Series(dtype=float, name="GDP")
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        series = fred.get_series("GDP", observation_start=start)
        series.name = "GDP"
        series.index = pd.to_datetime(series.index).tz_localize(None)
        df = series.to_frame()
        _save_csv(df, path)
        return series
    except Exception:
        if path.exists():
            return _load_csv(path).squeeze()
        return pd.Series(dtype=float, name="GDP")


def cache_status() -> dict:
    """Returns freshness info for the sidebar data status panel."""
    import time as _time

    result = {}
    names = {
        "sp500_daily": "S&P 500",
        "btc_daily": "Bitcoin",
        "inflation_daily": "Inflation",
        "gdp_quarterly": "GDP",
    }
    for key, label in names.items():
        path = _cache_path(key)
        if not path.exists():
            result[label] = "not cached"
        else:
            age_h = (_time.time() - path.stat().st_mtime) / 3600
            if age_h < 1:
                result[label] = f"fresh ({int(age_h * 60)}m ago)"
            else:
                result[label] = f"cached ({age_h:.1f}h ago)"
    return result
