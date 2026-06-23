from __future__ import annotations

import pandas as pd
import numpy as np


def resample_to_monthly(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Resample a daily price series to month-end using last observation."""
    series = df[col].resample("ME").last()
    series.name = col
    return series.dropna()


def resample_to_monthly_pct(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Month-end price resampled then converted to percentage change."""
    monthly = resample_to_monthly(df, col)
    return monthly.pct_change().dropna() * 100


def interpolate_gdp_to_monthly(gdp_quarterly: pd.Series) -> pd.Series:
    """Linearly interpolate quarterly GDP to monthly frequency."""
    monthly_idx = pd.date_range(
        start=gdp_quarterly.index.min(),
        end=gdp_quarterly.index.max(),
        freq="ME",
    )
    # FRED GDP uses quarter-start dates (e.g. 2018-01-01) while the monthly
    # target index uses month-end dates (2018-01-31). A plain reindex finds no
    # exact matches, leaving all NaN with nothing to interpolate from.
    # Fix: union the two indexes so the quarterly anchor points are preserved,
    # interpolate over the combined index, then select only month-end dates.
    combined_idx = monthly_idx.union(gdp_quarterly.index).sort_values()
    gdp_all = gdp_quarterly.reindex(combined_idx).interpolate(method="time")
    gdp_monthly = gdp_all.reindex(monthly_idx)
    gdp_monthly.name = "GDP"
    return gdp_monthly


def resample_inflation_monthly(inflation_daily: pd.Series) -> pd.Series:
    """Average daily T10YIE values to a monthly series."""
    monthly = inflation_daily.resample("ME").mean()
    monthly.name = "T10YIE"
    return monthly.dropna()


def align_series(*series: pd.Series) -> pd.DataFrame:
    """Inner-join on date index, drop any remaining NaN rows."""
    df = pd.concat(series, axis=1).dropna()
    return df


def build_taylor_exog(
    sp_monthly: pd.Series,
    gdp_monthly: pd.Series,
    inflation_monthly: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Replicates notebook cell 55: endog = SP500, exog = [lagged SP500, GDP, Inflation].
    Returns (endog, exog) with the first row dropped (due to lag).
    """
    combined = align_series(sp_monthly, gdp_monthly, inflation_monthly)
    sp = combined.iloc[:, 0]
    gdp = combined.iloc[:, 1]
    infl = combined.iloc[:, 2]

    exog = pd.DataFrame(
        {
            "sp_lag1": sp.shift(1),
            "GDP": gdp,
            "T10YIE": infl,
        }
    ).iloc[1:]
    endog = sp.iloc[1:]
    return endog, exog


def make_sequences(series: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sliding-window sequences for LSTM training."""
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i : i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)
