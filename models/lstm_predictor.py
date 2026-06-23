from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LSTM_BATCH_SIZE, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_N_STEPS, LSTM_UNITS
from data.preprocessor import make_sequences

warnings.filterwarnings("ignore", category=UserWarning)


class LSTMPredictor:
    def __init__(
        self,
        n_steps: int = LSTM_N_STEPS,
        units: list = None,
        dropout: float = LSTM_DROPOUT,
    ):
        self.n_steps = n_steps
        self.units = units or LSTM_UNITS
        self.dropout = dropout
        self._model = None
        self._scaler = None
        self.history = {}

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                LSTM(self.units[0], return_sequences=True, input_shape=(self.n_steps, 1)),
                Dropout(self.dropout),
                LSTM(self.units[1]),
                Dropout(self.dropout),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(
        self,
        series: pd.Series,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        validation_split: float = 0.1,
    ) -> dict:
        from sklearn.preprocessing import MinMaxScaler

        values = series.values.reshape(-1, 1).astype("float32")

        # Train/val/test split — scaler fitted on train only (bug fix vs. notebook)
        n = len(values)
        n_train = int(n * 0.8)

        self._scaler = MinMaxScaler()
        train_scaled = self._scaler.fit_transform(values[:n_train])
        rest_scaled = self._scaler.transform(values[n_train:])
        all_scaled = np.concatenate([train_scaled, rest_scaled])

        X, y = make_sequences(all_scaled.flatten(), self.n_steps)
        X = X.reshape(-1, self.n_steps, 1)

        self._model = self._build_model()
        h = self._model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )
        self.history = h.history
        return self.history

    def predict_insample(self, series: pd.Series) -> pd.Series:
        """In-sample predictions aligned with series index."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted.")
        values = series.values.reshape(-1, 1).astype("float32")
        scaled = self._scaler.transform(values)
        X, _ = make_sequences(scaled.flatten(), self.n_steps)
        X = X.reshape(-1, self.n_steps, 1)
        preds_scaled = self._model.predict(X, verbose=0)
        preds = self._scaler.inverse_transform(preds_scaled).flatten()
        idx = series.index[self.n_steps :]
        return pd.Series(preds, index=idx, name="Predicted")

    def predict_next_n(self, series: pd.Series, n: int = 30, freq: str = "D") -> pd.Series:
        """Roll-forward forecast n steps beyond the last observation.

        freq controls the output date index: "D" for daily (BTC), "ME" for monthly (SP500).
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted.")
        values = series.values.astype("float32").reshape(-1, 1)
        scaled = self._scaler.transform(values).flatten()
        window = list(scaled[-self.n_steps :])
        preds_scaled = []
        for _ in range(n):
            x = np.array(window[-self.n_steps :]).reshape(1, self.n_steps, 1)
            p = float(self._model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(p)
            window.append(p)
        preds = self._scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        last_date = series.index[-1]
        # Generate n future dates starting just after last_date at the given frequency
        future_idx = pd.date_range(start=last_date, periods=n + 1, freq=freq)[1:]
        return pd.Series(preds, index=future_idx, name="Forecast")

    def save(self, model_path: Path, scaler_path: Path) -> None:
        self._model.save(model_path)
        joblib.dump({"scaler": self._scaler, "n_steps": self.n_steps,
                     "units": self.units, "dropout": self.dropout,
                     "history": self.history}, scaler_path)

    @classmethod
    def load(cls, model_path: Path, scaler_path: Path) -> "LSTMPredictor":
        import tensorflow as tf
        meta = joblib.load(scaler_path)
        obj = cls(n_steps=meta["n_steps"], units=meta["units"], dropout=meta["dropout"])
        obj._scaler = meta["scaler"]
        obj.history = meta.get("history", {})
        obj._model = tf.keras.models.load_model(model_path)
        return obj
