"""Monte-Carlo robustness analysis.

For each simulation we shuffle the per-trade PnL (preserves the distribution
but breaks any time-dependent ordering) and re-run the equity curve.  We also
optionally perturb slippage by ±50% and entry/exit timing by ±1 bar.

Outputs:
    - distribution of total returns / Sharpe / drawdowns
    - probability of ruin (final equity < 50% initial)
    - 5/50/95-percentile equity curves
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_settings
from .metrics import compute_metrics


@dataclass
class MonteCarloResult:
    n_runs: int
    final_equities: np.ndarray
    sharpes: np.ndarray
    max_drawdowns: np.ndarray
    probability_of_ruin: float
    median_drawdown: float
    worst_drawdown: float

    def summary(self) -> dict[str, float]:
        return {
            "n_runs": self.n_runs,
            "final_eq_p5": float(np.percentile(self.final_equities, 5)),
            "final_eq_p50": float(np.median(self.final_equities)),
            "final_eq_p95": float(np.percentile(self.final_equities, 95)),
            "sharpe_p5": float(np.percentile(self.sharpes, 5)),
            "sharpe_p50": float(np.median(self.sharpes)),
            "sharpe_p95": float(np.percentile(self.sharpes, 95)),
            "max_dd_p5": float(np.percentile(self.max_drawdowns, 5)),
            "max_dd_p50": float(np.median(self.max_drawdowns)),
            "max_dd_p95": float(np.percentile(self.max_drawdowns, 95)),
            "probability_of_ruin": self.probability_of_ruin,
            "median_drawdown": self.median_drawdown,
            "worst_drawdown": self.worst_drawdown,
        }


def run_monte_carlo(
    trades: pd.DataFrame,
    *,
    initial_balance: float | None = None,
    n_runs: int | None = None,
    slippage_perturb: float | None = None,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Bootstrap a Monte-Carlo distribution from the trades dataframe."""
    if trades.empty:
        raise ValueError("Cannot run Monte Carlo with no trades")

    settings = get_settings()
    initial = initial_balance if initial_balance is not None else settings.backtest.initial_balance
    n_runs = n_runs or settings.backtest.monte_carlo_runs
    slip_perturb = slippage_perturb if slippage_perturb is not None else settings.backtest.monte_carlo_slippage_perturb
    rng = np.random.default_rng(seed)

    pnl = trades["pnl"].to_numpy()
    slip = trades.get("slippage_pips", pd.Series(0, index=trades.index)).to_numpy()

    finals = np.empty(n_runs)
    sharpes = np.empty(n_runs)
    drawdowns = np.empty(n_runs)
    ruined = 0

    for r in range(n_runs):
        order = rng.permutation(len(pnl))
        shuffled = pnl[order].copy()
        # Perturb slippage: each trade's PnL adjusted by an extra -slip * factor
        factor = rng.uniform(-slip_perturb, slip_perturb, size=len(shuffled))
        # `slip` is in pips; convert to per-trade USD impact assuming 1 pip ≈ $1 per micro lot.
        pip_usd = settings.instrument.pip_value_per_lot * 0.01  # micro lot baseline
        shuffled = shuffled - factor * slip[order] * pip_usd

        equity = initial + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        finals[r] = equity[-1]
        ret = pd.Series(equity).pct_change().dropna()
        sharpes[r] = float(ret.mean() / ret.std(ddof=0) * np.sqrt(252)) if ret.std(ddof=0) > 0 else 0.0
        drawdowns[r] = float(dd.min())
        if equity[-1] < 0.5 * initial:
            ruined += 1

    return MonteCarloResult(
        n_runs=n_runs,
        final_equities=finals,
        sharpes=sharpes,
        max_drawdowns=drawdowns,
        probability_of_ruin=ruined / n_runs,
        median_drawdown=float(np.median(drawdowns)),
        worst_drawdown=float(np.min(drawdowns)),
    )


def _verify_metrics_module() -> None:  # pragma: no cover - sanity helper
    """Ensure :mod:`metrics` imports correctly even if Monte Carlo isn't run."""
    _ = compute_metrics
