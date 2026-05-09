"""Backtesting suite (vectorized engine, metrics, walk-forward, Monte Carlo)."""

from __future__ import annotations

from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import PerformanceMetrics, compute_metrics
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .walk_forward import WalkForwardFold, WalkForwardResult, walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MonteCarloResult",
    "PerformanceMetrics",
    "WalkForwardFold",
    "WalkForwardResult",
    "compute_metrics",
    "run_backtest",
    "run_monte_carlo",
    "walk_forward",
]
