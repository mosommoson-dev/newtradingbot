"""Real-time prediction service.

In production, training writes predictions to a Parquet file and updates a
model artifact.  At decision time the trader reads the latest prediction for
the current bar's timestamp and forwards it to the ML strategy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.settings import ARTIFACT_DIR


class PredictionStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else ARTIFACT_DIR / "predictions.parquet"
        self._cache: pd.DataFrame | None = None
        self._mtime: float | None = None

    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=["p_up"])
        mtime = self._path.stat().st_mtime
        if self._cache is None or mtime != self._mtime:
            self._cache = pd.read_parquet(self._path)
            self._mtime = mtime
        return self._cache

    def latest_for(self, ts: pd.Timestamp) -> float | None:
        df = self.load()
        if df.empty:
            return None
        idx = df.index
        if ts in idx:
            return float(df.at[ts, "p_up"])
        try:
            i = idx.searchsorted(ts) - 1
        except Exception:
            return None
        if i < 0:
            return None
        return float(df.iloc[i]["p_up"])
