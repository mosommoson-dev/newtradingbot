"""Trend-following strategy: triple-EMA alignment with MACD + ADX confirm.

LONG entry:
    * EMA(fast) > EMA(slow) > EMA(trend) (alignment)
    * MACD histogram > 0 (momentum positive)
    * ADX(14) > ``adx_threshold`` (sufficient trend strength)
    * Cross-up: EMA(fast) crosses EMA(slow) on this bar

SHORT entry: mirror.

Exit:
    * EMA(fast) crosses back through EMA(slow) (anti-alignment), OR
    * ADX < 20 (trend died), OR
    * ATR trailing stop hit (handled by execution layer).

The strategy emits a position series; the execution layer sizes positions
and applies the trailing stop.
"""

from __future__ import annotations

import pandas as pd

from ..config import get_settings
from ..ml import _indicators as ind
from .base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def __init__(self, **overrides: object) -> None:
        d = get_settings().strategy.trend_following
        merged: dict[str, object] = {
            "ema_fast": d.ema_fast,
            "ema_slow": d.ema_slow,
            "ema_trend": d.ema_trend,
            "adx_period": d.adx_period,
            "adx_threshold": d.adx_threshold,
            "atr_period": d.atr_period,
            "atr_multiplier": d.atr_multiplier,
            "macd_fast": d.macd_fast,
            "macd_slow": d.macd_slow,
            "macd_signal": d.macd_signal,
        }
        merged.update(overrides)
        super().__init__(**merged)

    @staticmethod
    def search_space() -> dict[str, tuple[float | int, float | int, str]]:
        d = get_settings().strategy.trend_following
        return {
            "ema_fast": (*d.ema_fast_range, "int"),
            "ema_slow": (*d.ema_slow_range, "int"),
            "ema_trend": (*d.ema_trend_range, "int"),
            "adx_threshold": (*d.adx_threshold_range, "float"),
            "atr_multiplier": (*d.atr_multiplier_range, "float"),
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._ensure_columns(data)
        cfg = self.params
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ema_fast = ind.ema(close, int(cfg["ema_fast"]))
        ema_slow = ind.ema(close, int(cfg["ema_slow"]))
        ema_trend = ind.ema(close, int(cfg["ema_trend"]))
        _, _, hist = ind.macd(close, int(cfg["macd_fast"]),
                              int(cfg["macd_slow"]), int(cfg["macd_signal"]))
        adx_, _, _ = ind.adx(high, low, close, int(cfg["adx_period"]))

        long_align = (ema_fast > ema_slow) & (ema_slow > ema_trend)
        short_align = (ema_fast < ema_slow) & (ema_slow < ema_trend)
        prev_long = long_align.shift(1, fill_value=False)
        prev_short = short_align.shift(1, fill_value=False)
        long_cross = long_align & ~prev_long
        short_cross = short_align & ~prev_short

        long_entry = long_cross & (hist > 0) & (adx_ > float(cfg["adx_threshold"]))
        short_entry = short_cross & (hist < 0) & (adx_ > float(cfg["adx_threshold"]))

        # Position state machine: hold while alignment holds AND ADX > 20.
        position = pd.Series(0, index=data.index, dtype=int)
        cur = 0
        la = long_align.fillna(False).to_numpy()
        sa = short_align.fillna(False).to_numpy()
        le = long_entry.fillna(False).to_numpy()
        se = short_entry.fillna(False).to_numpy()
        adx_arr = adx_.fillna(0).to_numpy()
        out = position.to_numpy()

        for i in range(len(out)):
            if cur == 0:
                if le[i]:
                    cur = 1
                elif se[i]:
                    cur = -1
            elif cur == 1:
                if not la[i] or adx_arr[i] < 20:
                    cur = 0
            elif cur == -1:
                if not sa[i] or adx_arr[i] < 20:
                    cur = 0
            out[i] = cur

        return pd.Series(out, index=data.index, dtype=int, name="signal")
