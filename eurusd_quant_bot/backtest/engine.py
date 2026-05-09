"""Vectorized backtester with realistic costs and no look-ahead.

Design
======
The engine consumes:
    - A clean OHLCV frame indexed by UTC timestamps.
    - A position series in {-1, 0, +1} aligned to the same index, where the
      position at row ``t`` is the desired position to hold *during* bar
      ``t+1`` (i.e. after the close of ``t``).

Fills are simulated at the **next bar's open price** to avoid look-ahead.
Spread, commission and slippage are subtracted at every position change.

Outputs:
    - ``equity``     : the equity curve as a Series
    - ``trades``     : DataFrame with one row per closed trade
    - ``stats``      : :class:`PerformanceMetrics`

The backtest is vectorized for speed; for tighter event-driven simulation
(needed only when modelling intra-bar SL/TP), this engine can be paired with
the dedicated event loop in ``live/trader.py`` running over the same data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import get_settings
from ..execution.slippage_model import calculate_slippage
from .metrics import PerformanceMetrics, compute_metrics


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: PerformanceMetrics
    positions: pd.Series
    bar_returns: pd.Series


@dataclass
class BacktestConfig:
    initial_balance: float | None = None
    risk_per_trade: float | None = None
    spread_pips: float | None = None
    commission_per_lot_per_side_usd: float | None = None
    base_slippage_pips: float | None = None
    use_dynamic_slippage: bool = True


def run_backtest(
    data: pd.DataFrame,
    positions: pd.Series,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    data : DataFrame
        OHLCV bars. Must contain at least ``open``, ``high``, ``low``, ``close``.
    positions : Series
        Desired position in {-1, 0, +1} aligned to ``data.index``.  The
        engine assumes the position decided at the close of bar ``t`` is held
        from the open of bar ``t+1`` to the open of bar ``t+2``.
    """
    if not data.index.equals(positions.index):
        positions = positions.reindex(data.index, fill_value=0)

    settings = get_settings()
    cfg = config or BacktestConfig()
    initial = cfg.initial_balance if cfg.initial_balance is not None else settings.backtest.initial_balance
    risk_per_trade = cfg.risk_per_trade if cfg.risk_per_trade is not None else settings.risk.risk_per_trade
    spread_pips = cfg.spread_pips if cfg.spread_pips is not None else settings.instrument.avg_spread_pips
    commission = (
        cfg.commission_per_lot_per_side_usd
        if cfg.commission_per_lot_per_side_usd is not None
        else settings.instrument.commission_per_lot_per_side_usd
    )
    base_slip = cfg.base_slippage_pips if cfg.base_slippage_pips is not None else settings.instrument.base_slippage_pips
    pip = settings.instrument.pip
    pip_value = settings.instrument.pip_value_per_lot

    # Fills happen at next-bar open to avoid look-ahead.
    next_open = data["open"].shift(-1)
    bar_return = next_open.pct_change(fill_method=None).shift(-1).fillna(0.0)
    # Realized return for holding the position during the *next* bar:
    pos = positions.astype(float).fillna(0.0)
    held_return = pos * bar_return

    # Detect position changes to apply spread + commission + slippage on entry/exit.
    changes = pos.diff().fillna(pos)
    # ATR for slippage model
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().bfill()

    # Per-trade lot size: convert risk amount to lots assuming 50-pip stop default.
    # The strategy / risk manager actually sets stops at run time; for the
    # vectorized engine we assume a constant 50-pip stop, which is conservative.
    default_stop_pips = 50.0
    risk_amount = initial * risk_per_trade
    lots = max(0.01, round(risk_amount / (default_stop_pips * pip_value), 2))

    # Compute per-bar costs in pct of price (so we can apply directly to returns)
    cost_pips = pd.Series(0.0, index=data.index)
    abs_changes = changes.abs()
    if cfg.use_dynamic_slippage:
        slippage = atr.combine(close, lambda a, c: a) * 0  # placeholder, replaced below
        slippage = pd.Series(
            [calculate_slippage(a, lots * settings.instrument.standard_lot_units * c, False)
             for a, c in zip(atr.values, close.values, strict=True)],
            index=data.index,
        )
    else:
        slippage = pd.Series(base_slip, index=data.index)

    cost_pips = cost_pips.add(spread_pips * abs_changes, fill_value=0.0)
    cost_pips = cost_pips.add(slippage * abs_changes, fill_value=0.0)
    cost_in_price = cost_pips * pip
    cost_pct = (cost_in_price / close).fillna(0.0)
    held_return = held_return - cost_pct

    # Commission cost (USD) -> convert to pct of equity at each change.
    commission_usd = commission * lots * abs_changes
    equity = pd.Series(initial, index=data.index, dtype=float)
    eq = initial
    eq_arr = equity.to_numpy()
    ret_arr = held_return.to_numpy()
    com_arr = commission_usd.to_numpy()
    for i in range(len(eq_arr)):
        eq = eq * (1 + ret_arr[i]) - com_arr[i]
        eq_arr[i] = eq
    equity[:] = eq_arr

    trades = _extract_trades(data, pos, slippage_pips=slippage,
                             spread_pips=spread_pips, lots=lots,
                             commission_per_side=commission)
    stats = compute_metrics(equity, trades)
    return BacktestResult(equity=equity, trades=trades, stats=stats,
                          positions=pos, bar_returns=held_return)


def _extract_trades(
    data: pd.DataFrame,
    positions: pd.Series,
    *,
    slippage_pips: pd.Series,
    spread_pips: float,
    lots: float,
    commission_per_side: float,
) -> pd.DataFrame:
    """Walk the position series and emit one row per closed round-trip."""
    settings = get_settings()
    pip = settings.instrument.pip
    pip_value = settings.instrument.pip_value_per_lot

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    direction = 0
    entry_idx: int | None = None
    entry_ts: pd.Timestamp | None = None
    entry_price: float = 0.0
    open_arr = data["open"].to_numpy()
    pos_arr = positions.to_numpy()
    idx = data.index

    def _close_trade(i: int, exit_price: float) -> None:
        assert entry_idx is not None and entry_ts is not None
        slip_in = float(slippage_pips.iloc[entry_idx])
        slip_out = float(slippage_pips.iloc[i])
        pnl_pips = (exit_price - entry_price) / pip * direction
        pnl_pips -= spread_pips + slip_in + slip_out
        pnl = pnl_pips * pip_value * lots - 2 * commission_per_side * lots
        rows.append(
            {
                "strategy": "vectorized",
                "pair": settings.instrument.pair,
                "direction": int(direction),
                "entry_time": entry_ts,
                "exit_time": idx[i],
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "lot_size": float(lots),
                "pnl": float(pnl),
                "pnl_pips": float(pnl_pips),
                "commission": float(2 * commission_per_side * lots),
                "slippage_pips": float(slip_in + slip_out),
            }
        )

    for i in range(len(pos_arr) - 1):
        next_open = float(open_arr[i + 1]) if i + 1 < len(open_arr) else float(open_arr[i])
        cur = int(pos_arr[i])
        if direction == 0 and cur != 0:
            direction = cur
            entry_idx = i + 1 if i + 1 < len(open_arr) else i
            entry_ts = idx[entry_idx]
            entry_price = next_open
        elif direction != 0 and cur != direction:
            _close_trade(i + 1 if i + 1 < len(open_arr) else i, next_open)
            direction = 0
            entry_idx = None
            entry_ts = None
            entry_price = 0.0
            if cur != 0:
                direction = cur
                entry_idx = i + 1 if i + 1 < len(open_arr) else i
                entry_ts = idx[entry_idx]
                entry_price = next_open

    if direction != 0 and entry_idx is not None:
        _close_trade(len(pos_arr) - 1, float(open_arr[-1]))

    if not rows:
        return pd.DataFrame(
            columns=[
                "strategy", "pair", "direction", "entry_time", "exit_time",
                "entry_price", "exit_price", "lot_size", "pnl", "pnl_pips",
                "commission", "slippage_pips",
            ]
        )
    return pd.DataFrame(rows)
