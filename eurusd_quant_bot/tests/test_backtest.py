"""End-to-end tests for the backtester + metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eurusd_quant_bot.backtest import (
    compute_metrics,
    run_backtest,
    run_monte_carlo,
)
from eurusd_quant_bot.strategy import MeanReversionStrategy


def test_run_backtest_returns_equity_curve(synthetic_ohlcv):
    s = MeanReversionStrategy(z_score_period=30, z_score_entry=1.0,
                              rsi_low=40, rsi_high=60, bb_low=0.4, bb_high=0.6,
                              adx_max=80)
    sigs = s.generate_signals(synthetic_ohlcv)
    res = run_backtest(synthetic_ohlcv, sigs)
    assert len(res.equity) == len(synthetic_ohlcv)
    assert res.equity.iloc[0] > 0
    assert isinstance(res.stats.sharpe_ratio, float)


def test_metrics_zero_for_empty_inputs():
    eq = pd.Series([], dtype=float)
    trades = pd.DataFrame()
    m = compute_metrics(eq, trades)
    assert m.total_trades == 0
    assert m.sharpe_ratio == 0.0


def test_metrics_basic_equity_growth():
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    eq = pd.Series(np.linspace(10_000, 11_000, len(idx)), index=idx)
    trades = pd.DataFrame({
        "entry_time": idx[::20][:-1],
        "exit_time": idx[10::20][: len(idx[::20]) - 1],
        "direction": [1, -1, 1, -1, 1, -1, 1, -1, 1],
        "pnl": [10, -5, 15, -7, 20, -10, 25, -8, 30],
        "pnl_pips": [5, -3, 8, -4, 10, -5, 12, -4, 15],
    })
    trades = trades.iloc[: 9]
    m = compute_metrics(eq, trades)
    assert m.total_return > 0.0
    assert m.win_rate > 0.0
    assert m.profit_factor > 1.0


def test_monte_carlo_summary_keys():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    trades = pd.DataFrame({
        "entry_time": idx[:30],
        "exit_time": idx[1:31],
        "direction": [1] * 30,
        "pnl": np.random.default_rng(0).normal(loc=2.0, scale=10.0, size=30),
        "pnl_pips": np.random.default_rng(1).normal(loc=2.0, scale=5.0, size=30),
        "slippage_pips": np.full(30, 0.5),
    })
    res = run_monte_carlo(trades, n_runs=50, seed=0)
    summary = res.summary()
    for k in ("final_eq_p50", "sharpe_p50", "max_dd_p50", "probability_of_ruin"):
        assert k in summary
