from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import MARKOV_K_REGIMES, REGIME_LABELS


class MarkovRegimeModel:
    def __init__(self, k_regimes: int = MARKOV_K_REGIMES, switching_variance: bool = False):
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self._result = None
        self._endog = None

    def fit(self, endog: pd.Series, exog: pd.DataFrame | None = None) -> None:
        self._endog = endog
        kwargs = dict(
            k_regimes=self.k_regimes,
            trend="c",
            switching_variance=self.switching_variance,
        )
        if exog is not None:
            kwargs["exog"] = exog
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = sm.tsa.MarkovRegression(endog, **kwargs)
            self._result = mod.fit(disp=False)

    def get_smoothed_probs(self) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Model not fitted yet.")
        probs = pd.DataFrame(
            self._result.smoothed_marginal_probabilities,
            index=self._endog.index,
            columns=[f"Regime {i}" for i in range(self.k_regimes)],
        )
        return probs.dropna()

    def get_dominant_regime(self) -> pd.Series:
        probs = self.get_smoothed_probs()
        return probs.idxmax(axis=1).map(lambda x: int(x.split()[-1]))

    def get_regime_labels(self) -> dict[int, str]:
        """Assign labels by sorting regimes on estimated mean (lowest=Bear)."""
        if self._result is None:
            return REGIME_LABELS
        try:
            means = [self._result.params[f"regime{i}.const"] for i in range(self.k_regimes)]
            sorted_idx = np.argsort(means)
            label_map = {
                int(sorted_idx[0]): "Bear / High Volatility",
                int(sorted_idx[-1]): "Bull / Low Volatility",
            }
            for i in sorted_idx[1:-1]:
                label_map[int(i)] = "Transitional"
            return label_map
        except Exception:
            return REGIME_LABELS

    def get_expected_durations(self) -> dict[int, float]:
        """Expected duration of each regime in months (1 / (1 - p_ii))."""
        if self._result is None:
            return {}
        durations = {}
        try:
            trans = self._result.regime_transition
            for i in range(self.k_regimes):
                p_stay = trans[i, i]
                # regime_transition may be (k, k, T) — take mean over time
                p_stay_scalar = float(np.mean(np.asarray(p_stay).flatten()))
                durations[i] = 1.0 / max(1.0 - p_stay_scalar, 1e-6)
        except Exception:
            pass
        return durations

    def get_current_regime(self) -> tuple[int, float]:
        """Returns (regime_index, probability) for the most recent observation."""
        probs = self.get_smoothed_probs()
        if probs.empty:
            return 0, 0.0
        latest = probs.iloc[-1]
        idx = int(np.argmax(latest.values))
        return idx, float(latest.iloc[idx])

    def summary_dict(self) -> dict:
        if self._result is None:
            return {}
        regime_idx, prob = self.get_current_regime()
        labels = self.get_regime_labels()
        durations = self.get_expected_durations()
        return {
            "current_regime": regime_idx,
            "label": labels.get(regime_idx, f"Regime {regime_idx}"),
            "probability": prob,
            "duration_months": durations.get(regime_idx, float("nan")),
            "aic": self._result.aic,
            "bic": self._result.bic,
        }

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "MarkovRegimeModel":
        return joblib.load(path)
