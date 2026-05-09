"""ML ensemble strategy: XGBoost + (optional) LSTM + logistic regression.

The actual training code lives in :mod:`eurusd_quant_bot.ml.training` to keep
this module focused on signal generation.  At inference time the strategy
either:

1. Loads cached predictions from ``ml/artifacts/predictions.parquet`` (the
   typical production path; predictions are produced by the training
   pipeline which runs on a schedule), or
2. Calls a fitted model wrapper passed via the ``model`` constructor argument
   (used in tests and one-shot backtests).

The strategy emits ``+1/-1/0`` based on the configured confidence threshold
and predicted move size.  Walk-forward optimization is handled by the ML
training module, not by Optuna at the strategy level.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from ..config import get_settings
from ..config.settings import ARTIFACT_DIR
from .base_strategy import BaseStrategy


class _PredictiveModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class MLEnsembleStrategy(BaseStrategy):
    name = "ml_ensemble"

    def __init__(
        self,
        *,
        model: _PredictiveModel | None = None,
        feature_columns: list[str] | None = None,
        cached_predictions_path: str | None = None,
        **overrides: object,
    ) -> None:
        d = get_settings().strategy.ml
        merged: dict[str, object] = {
            "horizon_bars": d.horizon_bars,
            "confidence_threshold": d.confidence_threshold,
            "move_threshold_pips": d.move_threshold_pips,
        }
        merged.update(overrides)
        super().__init__(**merged)
        self._model = model
        self._feature_columns = feature_columns
        self._cached_predictions_path = cached_predictions_path

    def required_columns(self) -> set[str]:
        return {"close"}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        proba = self._predict_proba(data)
        if proba is None:
            # Without a model and without cached predictions, emit no trades.
            return pd.Series(0, index=data.index, dtype=int, name="signal")

        threshold = float(self.params["confidence_threshold"])
        move_pips = float(self.params["move_threshold_pips"])
        pip = get_settings().instrument.pip
        atr = data["close"].diff().abs().rolling(14).mean()
        # Up probability minus down probability gives a signed confidence.
        up_p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        signed_conf = pd.Series(up_p - 0.5, index=data.index) * 2  # in [-1, 1]
        magnitude = atr * float(self.params["horizon_bars"]) / pip

        out = pd.Series(0, index=data.index, dtype=int, name="signal")
        long_mask = (up_p > threshold) & (magnitude > move_pips)
        short_mask = (up_p < (1 - threshold)) & (magnitude > move_pips)
        out.loc[long_mask] = 1
        out.loc[short_mask] = -1
        return out * (signed_conf.abs() > 0).astype(int)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_proba(self, data: pd.DataFrame) -> np.ndarray | None:
        if self._model is not None:
            X = data[self._feature_columns] if self._feature_columns else data
            return self._model.predict_proba(X)

        path = self._cached_predictions_path or str(ARTIFACT_DIR / "predictions.parquet")
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            return None
        df = df.reindex(data.index)
        if "p_up" not in df.columns:
            return None
        return np.column_stack([1 - df["p_up"].values, df["p_up"].values])
