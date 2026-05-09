"""Multi-strategy portfolio allocator (risk parity + adaptive weighting).

Given per-strategy signal series and per-strategy historical returns, this
module computes a combined position and per-strategy weight that:

1. Starts from the configured static weights.
2. Re-weights using inverse-volatility (risk parity) on a rolling window.
3. Zeroes out any strategy whose 30-day Sharpe ratio is below the configured
   floor (default 0).
4. Caps the ensemble weight if pairwise correlations exceed the configured
   correlation cap (>0.8 by default).

The combined position is the weighted sum of strategy positions, clipped to
``[-1, +1]`` so we never request more total exposure than one full position
worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_settings


@dataclass
class AllocatorOutput:
    position: pd.Series              # combined -1..+1 series
    weights: pd.DataFrame            # per-strategy weight per bar


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=0).replace(0, np.nan)
    return mean / std


def combine(
    strategy_positions: dict[str, pd.Series],
    strategy_returns: dict[str, pd.Series] | None = None,
    *,
    bars_per_day: int = 24,
) -> AllocatorOutput:
    """Combine multiple strategies into a single position series."""
    cfg = get_settings().strategy.portfolio
    base_weights = pd.Series(cfg.weights)
    base_weights = base_weights / base_weights.sum()

    df_pos = pd.DataFrame(strategy_positions).fillna(0).astype(float)

    # Risk-parity / Sharpe-floor weights ------------------------------
    if strategy_returns is not None and len(strategy_returns) > 0:
        df_ret = pd.DataFrame(strategy_returns).fillna(0)
        window = cfg.rolling_sharpe_window_days * bars_per_day

        sharpe = pd.DataFrame(
            {k: _rolling_sharpe(df_ret[k], window) for k in df_ret.columns}
        )
        inv_vol = 1.0 / df_ret.rolling(window).std(ddof=0).replace(0, np.nan)
        rp_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(method="ffill")
        floored = rp_weights.where(sharpe > cfg.drop_strategy_below_sharpe, 0.0)
        floored = floored.div(floored.sum(axis=1), axis=0).fillna(method="ffill").fillna(base_weights)
        weights = floored.combine_first(pd.DataFrame(
            np.tile(base_weights.values, (len(df_pos), 1)),
            index=df_pos.index, columns=df_pos.columns,
        ))

        # Correlation cap -------------------------------------------
        corr = df_ret.rolling(window).corr().unstack().mean(axis=1).rolling(window).mean()
        if (corr > cfg.correlation_cap).any():
            weights = weights * 0.5  # halve allocation when correlations are too high
    else:
        weights = pd.DataFrame(
            np.tile(base_weights.reindex(df_pos.columns).values, (len(df_pos), 1)),
            index=df_pos.index,
            columns=df_pos.columns,
        )

    aligned_weights = weights.reindex(df_pos.index).ffill().fillna(base_weights)
    combined = (df_pos * aligned_weights).sum(axis=1).clip(-1.0, 1.0)
    return AllocatorOutput(position=combined.astype(float), weights=aligned_weights)
