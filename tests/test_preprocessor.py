"""Unit tests for data/preprocessor.py — pure functions, no network access."""

import numpy as np
import pandas as pd
import pytest

from data.preprocessor import (
    align_series,
    build_taylor_exog,
    interpolate_gdp_to_monthly,
    make_sequences,
    resample_inflation_monthly,
    resample_to_monthly,
    resample_to_monthly_pct,
)


def daily_prices(start="2020-01-01", periods=90, value=100.0):
    idx = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame({"Close": np.linspace(value, value + periods - 1, periods)}, index=idx)


class TestResampleToMonthly:
    def test_takes_last_observation_of_month(self):
        df = daily_prices("2020-01-01", 60)
        monthly = resample_to_monthly(df)
        assert monthly.index[0] == pd.Timestamp("2020-01-31")
        assert monthly.iloc[0] == df["Close"].loc["2020-01-31"]

    def test_pct_change_drops_first_month(self):
        df = daily_prices("2020-01-01", 90)
        pct = resample_to_monthly_pct(df)
        monthly = resample_to_monthly(df)
        assert len(pct) == len(monthly) - 1
        expected = (monthly.iloc[1] / monthly.iloc[0] - 1) * 100
        assert pct.iloc[0] == pytest.approx(expected)


class TestInterpolateGdpToMonthly:
    def test_quarter_start_anchors_produce_no_all_nan_output(self):
        # Regression test for the documented reindex bug: FRED GDP uses
        # quarter-START dates, the monthly target uses month-END dates.
        gdp = pd.Series(
            [100.0, 104.0, 108.0, 112.0],
            index=pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]),
        )
        monthly = interpolate_gdp_to_monthly(gdp)
        assert monthly.name == "GDP"
        # Interior months must be interpolated, not NaN
        interior = monthly.loc["2020-02-29":"2020-09-30"]
        assert not interior.isna().any()
        # Values must be monotonically increasing for increasing anchors
        assert (interior.diff().dropna() > 0).all()

    def test_values_stay_within_anchor_bounds(self):
        gdp = pd.Series(
            [100.0, 110.0],
            index=pd.to_datetime(["2020-01-01", "2020-04-01"]),
        )
        monthly = interpolate_gdp_to_monthly(gdp).dropna()
        assert (monthly >= 100.0).all()
        assert (monthly <= 110.0).all()


class TestResampleInflationMonthly:
    def test_averages_daily_values(self):
        idx = pd.date_range("2020-01-01", "2020-01-31", freq="D")
        daily = pd.Series(2.0, index=idx)
        monthly = resample_inflation_monthly(daily)
        assert monthly.name == "T10YIE"
        assert monthly.loc["2020-01-31"] == pytest.approx(2.0)


class TestAlignAndTaylorExog:
    def _monthly(self, values, name, start="2020-01-31"):
        idx = pd.date_range(start=start, periods=len(values), freq="ME")
        return pd.Series(values, index=idx, name=name)

    def test_align_inner_joins_and_drops_nan(self):
        a = self._monthly([1, 2, 3, 4], "a")
        b = self._monthly([10, 20, 30], "b")  # one month shorter
        joined = align_series(a, b)
        assert len(joined) == 3
        assert list(joined.columns) == ["a", "b"]

    def test_taylor_exog_shapes_and_lag(self):
        sp = self._monthly([100.0, 102.0, 101.0, 105.0], "Close")
        gdp = self._monthly([1.0, 1.1, 1.2, 1.3], "GDP")
        infl = self._monthly([2.0, 2.1, 2.2, 2.3], "T10YIE")
        endog, exog = build_taylor_exog(sp, gdp, infl)
        # First row dropped because of the 1-month lag
        assert len(endog) == 3
        assert list(exog.columns) == ["sp_lag1", "GDP", "T10YIE"]
        # Lag column equals previous month's endog value
        assert exog["sp_lag1"].iloc[0] == pytest.approx(100.0)


class TestMakeSequences:
    def test_sliding_window_shapes(self):
        series = np.arange(10, dtype=float)
        X, y = make_sequences(series, n_steps=3)
        assert X.shape == (7, 3)
        assert y.shape == (7,)
        np.testing.assert_array_equal(X[0], [0.0, 1.0, 2.0])
        assert y[0] == 3.0

    def test_series_shorter_than_window_returns_empty(self):
        X, y = make_sequences(np.arange(2, dtype=float), n_steps=5)
        assert len(X) == 0
        assert len(y) == 0
