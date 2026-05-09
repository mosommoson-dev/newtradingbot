"""Walk-forward training pipeline for the ML strategy.

Steps:
    1. Build features from raw OHLCV (uses :mod:`ml.features`).
    2. Form a binary classification target: next-N-bar direction.
    3. Drop highly-correlated features (>0.9 abs).
    4. Iterate over expanding-window folds.
    5. For each fold: fit Ensemble (XGBoost + LogReg + optional LSTM),
       compute SHAP-based feature importance for XGBoost, retain top-K.
    6. Persist the final model and predictions to ``ml/artifacts/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from ..config import get_settings
from ..config.settings import ARTIFACT_DIR
from .features import build_features, feature_names
from .models import EnsembleModel


@dataclass
class TrainingResult:
    fold_metrics: pd.DataFrame
    final_model: EnsembleModel
    feature_columns: list[str]
    predictions_path: str


def make_target(close: pd.Series, horizon: int) -> pd.Series:
    fwd_return = close.shift(-horizon) / close - 1.0
    return (fwd_return > 0).astype(int).rename("target")


def drop_correlated(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    if drop:
        logger.info("Dropping {} highly-correlated features", len(drop))
    return X.drop(columns=drop)


def _fold_indices(idx: pd.DatetimeIndex, train_months: int,
                  test_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp,
                                                  pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = idx[0].normalize()
    end = idx[-1].normalize()
    train_end = start + pd.DateOffset(months=train_months)
    while train_end + pd.DateOffset(months=test_months) <= end:
        test_end = train_end + pd.DateOffset(months=test_months)
        folds.append((start, train_end, train_end, test_end))
        train_end = test_end
    return folds


def train_walk_forward(
    ohlcv: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    cot: pd.DataFrame | None = None,
    horizon: int | None = None,
    train_months: int | None = None,
    test_months: int | None = None,
) -> TrainingResult:
    settings = get_settings().strategy.ml
    horizon = horizon or settings.horizon_bars
    train_months = train_months or settings.train_window_months
    test_months = test_months or settings.test_window_months

    features = build_features(ohlcv, macro=macro, cot=cot)
    target = make_target(features["close"], horizon)
    df = features.join(target).dropna(subset=["target"])

    # Important: derive feature names from the *features* frame, not the
    # joined frame.  Otherwise the target column leaks into the predictor
    # set and the model trivially memorises the label.
    cols = feature_names(features)
    X = df[cols]
    X = drop_correlated(X, settings.correlation_drop_threshold)
    y = df["target"].astype(int)
    assert "target" not in X.columns, "Target leaked into features"

    folds = _fold_indices(X.index, train_months, test_months)
    fold_rows: list[dict[str, float]] = []
    final_model: EnsembleModel | None = None

    for ts0, ts1, ts2, ts3 in folds:
        train_mask = (X.index >= ts0) & (X.index < ts1)
        test_mask = (X.index >= ts2) & (X.index < ts3)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if len(X_train) < 500 or len(X_test) < 100:
            continue

        model = EnsembleModel().fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        preds = (proba[:, 1] > 0.5).astype(int)
        acc = float(np.mean(preds == y_test.values))
        fold_rows.append({"test_start": ts2, "test_end": ts3, "accuracy": acc,
                          "n_train": len(X_train), "n_test": len(X_test)})
        final_model = model

    if final_model is None:
        raise RuntimeError("Walk-forward training produced no folds")

    full_proba = final_model.predict_proba(X)
    pred_df = pd.DataFrame({"p_up": full_proba[:, 1]}, index=X.index)
    out_path = ARTIFACT_DIR / "predictions.parquet"
    pred_df.to_parquet(out_path)

    return TrainingResult(
        fold_metrics=pd.DataFrame(fold_rows),
        final_model=final_model,
        feature_columns=list(X.columns),
        predictions_path=str(out_path),
    )
