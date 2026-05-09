"""Backtesting suite (vectorized engine, metrics, walk-forward, Monte Carlo)."""

from __future__ import annotations

from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import PerformanceMetrics, compute_metrics
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .pairs_engine import PairsBacktestResult, run_pairs_backtest
from .walk_forward import WalkForwardFold, WalkForwardResult, walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MonteCarloResult",
    "PairsBacktestResult",
    "PerformanceMetrics",
    "WalkForwardFold",
    "WalkForwardResult",
    "compute_metrics",
    "run_backtest",
    "run_monte_carlo",
    "run_pairs_backtest",
    "walk_forward",
]
