"""Mean-reversion strategy: Z-score + RSI + Bollinger %B with ADX filter.

Logic (all conditions must hold, evaluated *at bar close*):

LONG entry:
    * Z-score of close over ``z_score_period`` < ``-z_score_entry``
    * RSI(``rsi_period``) < ``rsi_low``
    * Bollinger %B(``bb_period``, ``bb_std``) < ``bb_low``
    * ADX(``adx_period``) < ``adx_max`` (ranging market)
    * Outside the configured low-volatility windows

SHORT entry: mirror.

Exit:
    * Z-score crosses zero, OR
    * RSI crosses 50, OR
    * The execution layer's ATR trailing stop is hit (handled outside).

Signals are generated using ONLY past data (`shift` not required because
indicator values at time ``t`` already use only data up to ``t``).
"""

from __future__ import annotations

import pandas as pd

from ..config import get_settings
from ..ml import _indicators as ind
from .base_strategy import BaseStrategy


def _in_low_vol_window(idx: pd.DatetimeIndex,
                       windows: tuple[tuple[int, int], ...]) -> pd.Series:
    hour = pd.Series(idx.hour, index=idx)
    mask = pd.Series(False, index=idx)
    for start, end in windows:
        mask = mask | ((hour >= start) & (hour < end))
    return mask


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy implementing the spec in the project brief."""

    name = "mean_reversion"

    def __init__(self, **overrides: object) -> None:
        defaults = get_settings().strategy.mean_reversion
        merged: dict[str, object] = {
            "z_score_period": defaults.z_score_period,
            "z_score_entry": defaults.z_score_entry,
            "rsi_period": defaults.rsi_period,
            "rsi_low": defaults.rsi_low,
            "rsi_high": defaults.rsi_high,
            "bb_period": defaults.bb_period,
            "bb_std": defaults.bb_std,
            "bb_low": defaults.bb_low,
            "bb_high": defaults.bb_high,
            "adx_period": defaults.adx_period,
            "adx_max": defaults.adx_max,
            "atr_period": defaults.atr_period,
            "atr_stop_mult": defaults.atr_stop_mult,
        }
        merged.update(overrides)
        super().__init__(**merged)

    def required_columns(self) -> set[str]:
        return {"open", "high", "low", "close"}

    @staticmethod
    def search_space() -> dict[str, tuple[float | int, float | int, str]]:
        d = get_settings().strategy.mean_reversion
        return {
            "z_score_period": (*d.z_score_period_range, "int"),
            "z_score_entry": (*d.z_score_entry_range, "float"),
            "rsi_period": (*d.rsi_period_range, "int"),
            "bb_period": (*d.bb_period_range, "int"),
            "bb_std": (*d.bb_std_range, "float"),
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._ensure_columns(data)
        cfg = self.params
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        zscore = (
            close - close.rolling(int(cfg["z_score_period"])).mean()
        ) / close.rolling(int(cfg["z_score_period"])).std(ddof=0)
        rsi = ind.rsi(close, int(cfg["rsi_period"]))
        upper, _mid, lower = ind.bollinger(close, int(cfg["bb_period"]), float(cfg["bb_std"]))
        pctb = (close - lower) / (upper - lower).replace(0, pd.NA)
        adx_, _, _ = ind.adx(high, low, close, int(cfg["adx_period"]))

        bad_window = _in_low_vol_window(data.index, get_settings().schedule.avoid_low_vol_windows)
        ranging = adx_ < float(cfg["adx_max"])

        long_entry = (
            (zscore < -float(cfg["z_score_entry"]))
            & (rsi < float(cfg["rsi_low"]))
            & (pctb < float(cfg["bb_low"]))
            & ranging
            & ~bad_window
        )
        short_entry = (
            (zscore > float(cfg["z_score_entry"]))
            & (rsi > float(cfg["rsi_high"]))
            & (pctb > float(cfg["bb_high"]))
            & ranging
            & ~bad_window
        )

        # Build a position series using a simple state machine: enter on signal,
        # hold until z-score crosses 0 or RSI crosses 50, then flat.
        position = pd.Series(0, index=data.index, dtype=int)
        cur = 0
        z = zscore.fillna(0).to_numpy()
        r = rsi.fillna(50).to_numpy()
        long_arr = long_entry.fillna(False).to_numpy()
        short_arr = short_entry.fillna(False).to_numpy()
        out = position.to_numpy()

        for i in range(len(out)):
            if cur == 0:
                if long_arr[i]:
                    cur = 1
                elif short_arr[i]:
                    cur = -1
            elif cur == 1:
                if z[i] >= 0 or r[i] >= 50:
                    cur = 0
            elif cur == -1:
                if z[i] <= 0 or r[i] <= 50:
                    cur = 0
            out[i] = cur

        return pd.Series(out, index=data.index, dtype=int, name="signal")
