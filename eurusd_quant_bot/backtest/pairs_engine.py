"""Dual-leg backtest engine for pairs / spread strategies.

Given a long-short position on the spread ``a - beta*b``, applies realistic
transaction costs whenever the position changes and computes equity, trades,
and the standard performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, compute_metrics


@dataclass
class PairsBacktestResult:
    equity: pd.Series           # equity curve, $10,000 start
    pnl: pd.Series              # per-bar PnL fraction
    trades: pd.DataFrame        # trade ledger (entry/exit_time, pnl, ...)
    stats: PerformanceMetrics


def _build_pairs_trades(
    price_a: pd.Series,
    price_b: pd.Series,
    position: pd.Series,
    pnl_per_bar: pd.Series,
) -> pd.DataFrame:
    """Convert a long/short spread position series into an entry/exit ledger.

    Each contiguous non-zero block of ``position`` is one trade.  Entry is at
    the first non-zero bar, exit at the bar where it returns to zero (or at
    the last bar if still open).
    """
    rows = []
    current_dir = 0
    entry_idx = None
    cum_pnl = 0.0
    for i, (ts, p) in enumerate(position.items()):
        if p != current_dir:
            if current_dir != 0 and entry_idx is not None:
                exit_ts = ts
                rows.append({
                    "entry_time": entry_idx,
                    "exit_time": exit_ts,
                    "direction": current_dir,
                    "entry_a": float(price_a.loc[entry_idx]),
                    "exit_a": float(price_a.loc[exit_ts]),
                    "entry_b": float(price_b.loc[entry_idx]),
                    "exit_b": float(price_b.loc[exit_ts]),
                    "pnl": cum_pnl,
                })
            cum_pnl = 0.0
            current_dir = int(p)
            entry_idx = ts if p != 0 else None
        if current_dir != 0:
            cum_pnl += float(pnl_per_bar.iloc[i])
    if current_dir != 0 and entry_idx is not None:
        exit_ts = position.index[-1]
        rows.append({
            "entry_time": entry_idx,
            "exit_time": exit_ts,
            "direction": current_dir,
            "entry_a": float(price_a.loc[entry_idx]),
            "exit_a": float(price_a.loc[exit_ts]),
            "entry_b": float(price_b.loc[entry_idx]),
            "exit_b": float(price_b.loc[exit_ts]),
            "pnl": cum_pnl,
        })
    return pd.DataFrame(rows)


def run_pairs_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    position: pd.Series,
    beta: pd.Series,
    cost_bps: float = 4.0,
    starting_equity: float = 10_000.0,
) -> PairsBacktestResult:
    """Backtest a spread position with realistic costs.

    Parameters
    ----------
    price_a, price_b :
        Aligned closing-price series for the two instruments.
    position :
        Integer ``-1/0/+1`` Series on the spread.  ``+1`` = long A / short
        ``beta`` units of B; ``-1`` = inverse.
    beta :
        Time-varying hedge ratio (e.g. from the Kalman filter).
    cost_bps :
        Round-trip transaction cost in bps applied each time the position
        magnitude changes.  ``4`` bps ≈ 0.8 pip spread + commission per leg.
    """
    if not price_a.index.equals(price_b.index) or not price_a.index.equals(position.index):
        idx = price_a.index.intersection(price_b.index).intersection(position.index)
        price_a, price_b, position, beta = (
            price_a.reindex(idx), price_b.reindex(idx),
            position.reindex(idx), beta.reindex(idx),
        )

    ret_a = price_a.pct_change().fillna(0.0)
    ret_b = price_b.pct_change().fillna(0.0)
    pos_lag = position.shift(1).fillna(0)
    beta_lag = beta.shift(1).fillna(0.0)
    raw_pnl = pos_lag * (ret_a - beta_lag * ret_b)
    raw_pnl = raw_pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos_change = position.diff().abs().fillna(0.0)
    cost = pos_change * (cost_bps / 1e4)
    pnl = (raw_pnl - cost).fillna(0.0)

    equity = (1.0 + pnl).cumprod() * starting_equity

    trades = _build_pairs_trades(price_a, price_b, position, pnl)

    # Convert PnL fractions to per-trade pip estimates for the metrics layer.
    if not trades.empty:
        trades = trades.assign(
            pnl_pips=trades["pnl"] * 1e4,  # 1 pip ~ 1e-4 fractional move on EUR/USD
        )

    stats = compute_metrics(equity, trades)

    return PairsBacktestResult(
        equity=equity,
        pnl=pnl,
        trades=trades,
        stats=stats,
    )


__all__ = ["PairsBacktestResult", "run_pairs_backtest"]
