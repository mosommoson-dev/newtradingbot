"""ML model wrappers: XGBoost, optional LSTM, logistic regression baseline,
and a simple weighted Ensemble.

The wrappers expose a ``predict_proba`` method returning a 2-column numpy
array ``[p_down, p_up]`` so the strategy layer can be model-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logistic regression baseline (always available)
# ---------------------------------------------------------------------------

class LogRegModel:
    def __init__(self, **params: Any) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            [("scaler", StandardScaler()),
             ("clf", LogisticRegression(max_iter=200, **params))]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogRegModel:
        self.pipeline.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X.values)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel:
    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=2,
            tree_method="hist",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        self.model.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.values)


# ---------------------------------------------------------------------------
# LSTM (optional -- requires tensorflow or torch)
# ---------------------------------------------------------------------------

class LSTMModel:
    def __init__(self, sequence_length: int = 60, units_l1: int = 64,
                 units_l2: int = 32, dense: int = 16, dropout: float = 0.2,
                 epochs: int = 30, batch_size: int = 32) -> None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "tensorflow is required for the LSTM model; "
                "install via `pip install tensorflow`"
            ) from exc
        self.sequence_length = sequence_length
        self.units_l1 = units_l1
        self.units_l2 = units_l2
        self.dense = dense
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None

    def _build(self, n_features: int) -> Any:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                LSTM(self.units_l1, return_sequences=True,
                     input_shape=(self.sequence_length, n_features)),
                Dropout(self.dropout),
                LSTM(self.units_l2),
                Dropout(self.dropout),
                Dense(self.dense, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray | None]:
        seqs: list[np.ndarray] = []
        labels: list[float] = []
        for i in range(self.sequence_length, len(X)):
            seqs.append(X[i - self.sequence_length: i])
            if y is not None:
                labels.append(y[i])
        seq_arr = np.stack(seqs) if seqs else np.empty((0, self.sequence_length, X.shape[1]))
        lab_arr = np.array(labels) if labels else None
        return seq_arr, lab_arr

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMModel:
        Xv = X.values.astype(np.float32)
        yv = y.values.astype(np.float32)
        seqs, labels = self._make_sequences(Xv, yv)
        if len(seqs) == 0:
            raise ValueError("Not enough data to form LSTM sequences")
        self._model = self._build(Xv.shape[1])
        self._model.fit(seqs, labels, epochs=self.epochs,
                        batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LSTMModel not fitted")
        Xv = X.values.astype(np.float32)
        seqs, _ = self._make_sequences(Xv, None)
        if len(seqs) == 0:
            # Pad with priors when the input is shorter than the sequence length.
            return np.full((len(X), 2), 0.5)
        probs_up = self._model.predict(seqs, verbose=0).ravel()
        # Prepend 0.5 priors for the warmup rows.
        out_up = np.concatenate([np.full(self.sequence_length, 0.5), probs_up])
        return np.column_stack([1 - out_up, out_up])


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class EnsembleModel:
    weights: dict[str, float] = field(default_factory=lambda:
                                      {"xgb": 0.5, "logreg": 0.2, "lstm": 0.3})
    members: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, members: Sequence[str] | None = None
            ) -> EnsembleModel:
        members = members or list(self.weights)
        for m in members:
            if m == "xgb":
                self.members[m] = XGBoostModel().fit(X, y)
            elif m == "logreg":
                self.members[m] = LogRegModel().fit(X, y)
            elif m == "lstm":
                try:
                    self.members[m] = LSTMModel().fit(X, y)
                except RuntimeError:
                    # LSTM is optional -- skip if tensorflow isn't available.
                    self.weights.pop(m, None)
        # Re-normalise weights over members that actually fitted.
        present = {k: w for k, w in self.weights.items() if k in self.members}
        total = sum(present.values()) or 1.0
        self.weights = {k: w / total for k, w in present.items()}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.members:
            return np.full((len(X), 2), 0.5)
        out = np.zeros((len(X), 2))
        for name, model in self.members.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            try:
                p = model.predict_proba(X)
            except Exception:
                continue
            if p.shape[0] != out.shape[0]:
                # LSTM padding mismatch; resize via reindex on tail.
                pad = out.shape[0] - p.shape[0]
                if pad > 0:
                    p = np.vstack([np.full((pad, 2), 0.5), p])
                else:
                    p = p[-out.shape[0]:]
            out += w * p
        return out
