"""Performance metrics for an equity / trade history."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: float
    win_rate: float
    profit_factor: float
    avg_win_pips: float
    avg_loss_pips: float
    avg_trade_duration_hours: float
    total_trades: int
    long_trades: int
    short_trades: int
    consecutive_losses: int
    recovery_factor: float
    payoff_ratio: float
    expectancy_pips: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min())
    in_dd = drawdown < 0
    if in_dd.any():
        groups = (in_dd != in_dd.shift()).cumsum()
        durations = in_dd.groupby(groups).sum()
        longest = float(durations.max())
        # convert "rows" to days using the index frequency
        if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
            avg_seconds = float(np.diff(equity.index.values.astype("datetime64[s]").astype(int)).mean())
            longest_days = longest * avg_seconds / 86400.0
        else:
            longest_days = longest
    else:
        longest_days = 0.0
    return max_dd, longest_days


def _annualize_factor(equity: pd.Series) -> float:
    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 2:
        return 252.0
    seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    bars = len(equity) - 1
    if seconds <= 0:
        return 252.0
    bars_per_year = bars * (365.25 * 86400.0) / seconds
    return float(bars_per_year)


def _consecutive_losses(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    streak = 0
    best = 0
    for pnl in trades["pnl"].to_numpy():
        if pnl < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> PerformanceMetrics:
    """Compute all metrics from an equity curve and a trades DataFrame.

    ``trades`` is expected to contain the columns:
        ``entry_time``, ``exit_time``, ``direction``, ``pnl``, ``pnl_pips``.
    """
    if equity.empty:
        zeros = {f.name: 0.0 for f in PerformanceMetrics.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        zeros["total_trades"] = 0
        zeros["long_trades"] = 0
        zeros["short_trades"] = 0
        zeros["consecutive_losses"] = 0
        return PerformanceMetrics(**zeros)

    returns = equity.pct_change().dropna()
    factor = _annualize_factor(equity)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    if len(returns) > 0 and returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(factor))
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=0) * np.sqrt(factor))
        if len(downside) > 0 and downside.std(ddof=0) > 0
        else 0.0
    )
    max_dd, dd_dur_days = _max_drawdown(equity)
    # Use safe sign-preserving annualisation to avoid complex numbers when
    # total_return < -1 (a fully blown-up account).
    if len(returns):
        base = 1.0 + total_return
        exponent = factor / max(len(returns), 1)
        if base <= 0:
            annual_return = -1.0  # account fully wiped
        else:
            annual_return = base ** exponent - 1.0
    else:
        annual_return = 0.0
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    if not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_rate = float(len(wins) / len(trades))
        gross_win = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_win = float(wins["pnl_pips"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["pnl_pips"].mean()) if not losses.empty else 0.0
        durations_h = (
            (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600.0
        )
        avg_dur = float(durations_h.mean()) if len(durations_h) else 0.0
        long_n = int((trades["direction"] == 1).sum())
        short_n = int((trades["direction"] == -1).sum())
        cons_loss = _consecutive_losses(trades)
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    else:
        win_rate = profit_factor = avg_win = avg_loss = avg_dur = payoff = expectancy = 0.0
        long_n = short_n = cons_loss = 0

    recovery = total_return / abs(max_dd) if max_dd < 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=float(annual_return),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=float(calmar),
        max_drawdown=float(max_dd),
        max_drawdown_duration_days=float(dd_dur_days),
        win_rate=win_rate,
        profit_factor=float(profit_factor) if np.isfinite(profit_factor) else 0.0,
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
        avg_trade_duration_hours=avg_dur,
        total_trades=len(trades),
        long_trades=long_n,
        short_trades=short_n,
        consecutive_losses=cons_loss,
        recovery_factor=float(recovery),
        payoff_ratio=float(payoff),
        expectancy_pips=float(expectancy),
    )
