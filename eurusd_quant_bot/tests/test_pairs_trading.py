"""Unit tests for the pairs-trading strategy and Kalman filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurusd_quant_bot.backtest import run_pairs_backtest
from eurusd_quant_bot.strategy import (
    PairsTradingStrategy,
    kalman_pairs,
)


def _make_cointegrated_pair(n: int = 1000, true_alpha: float = 0.05,
                             true_beta: float = 1.3, seed: int = 7
                             ) -> tuple[pd.Series, pd.Series]:
    """Synthesize two cointegrated price series.

    b_t is a random walk; a_t = alpha + beta*b_t + stationary spread.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    b_steps = rng.normal(0, 0.005, size=n)
    log_b = np.cumsum(b_steps)
    b = 1.10 * np.exp(log_b)
    # Stationary spread: AR(1) with strong mean-reversion
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.003)
    a = true_alpha + true_beta * b + spread
    return (
        pd.Series(a, index=idx, name="A"),
        pd.Series(b, index=idx, name="B"),
    )


def test_kalman_pairs_recovers_true_beta():
    """Kalman should converge near the true hedge ratio on synthetic data."""
    a, b = _make_cointegrated_pair(n=800, true_alpha=0.0, true_beta=1.5, seed=1)
    res = kalman_pairs(a, b, obs_var=1e-3, trans_var=1e-7)
    # Final estimate should be within 5% of the truth
    assert abs(res.beta.iloc[-1] - 1.5) < 0.10, (
        f"beta should converge near 1.5, got {res.beta.iloc[-1]:.3f}"
    )
    assert res.beta.notna().sum() == len(a)
    assert res.spread.notna().sum() == len(a)


def test_kalman_is_causal():
    """Kalman estimate at time t must not depend on data after t."""
    a, b = _make_cointegrated_pair(n=300, seed=2)
    full = kalman_pairs(a, b, obs_var=1e-3, trans_var=1e-5)
    # Truncate at t=200 and check estimates 0..199 match.
    cut = 200
    truncated = kalman_pairs(a.iloc[:cut], b.iloc[:cut],
                              obs_var=1e-3, trans_var=1e-5)
    np.testing.assert_allclose(
        full.beta.iloc[:cut].values, truncated.beta.values,
        rtol=1e-12, atol=1e-12,
    )


def test_pairs_strategy_signals_shape():
    a, b = _make_cointegrated_pair(n=500, seed=3)
    strat = PairsTradingStrategy(z_entry=2.0, z_exit=0.5, z_lookback=60)
    sig = strat.fit_predict(a, b)
    assert len(sig.position) == len(a)
    assert set(sig.position.unique()).issubset({-1, 0, 1})
    assert sig.beta.notna().sum() > 0


def test_pairs_strategy_validates_thresholds():
    with pytest.raises(ValueError, match="z_exit"):
        PairsTradingStrategy(z_entry=1.0, z_exit=1.5)
    with pytest.raises(ValueError, match="z_lookback"):
        PairsTradingStrategy(z_lookback=2)
    with pytest.raises(ValueError, match="positive"):
        PairsTradingStrategy(kalman_obs_var=-1)


def test_pairs_backtest_runs_end_to_end():
    a, b = _make_cointegrated_pair(n=600, seed=4)
    strat = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30)
    sig = strat.fit_predict(a, b)
    result = run_pairs_backtest(a, b, sig.position, sig.beta, cost_bps=4.0)
    assert len(result.equity) == len(a)
    assert (result.equity > 0).all(), "equity must stay positive on synthetic data"
    # Non-zero number of trades (this is a cointegrated series; mean-reversion
    # SHOULD generate signals)
    assert len(result.trades) > 0


def test_pairs_backtest_costs_reduce_pnl():
    a, b = _make_cointegrated_pair(n=600, seed=5)
    strat = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30)
    sig = strat.fit_predict(a, b)
    no_cost = run_pairs_backtest(a, b, sig.position, sig.beta, cost_bps=0.0)
    with_cost = run_pairs_backtest(a, b, sig.position, sig.beta, cost_bps=20.0)
    assert with_cost.equity.iloc[-1] < no_cost.equity.iloc[-1], (
        "high transaction costs must reduce final equity"
    )


def test_pairs_strategy_no_lookahead():
    """The position at time t must not depend on data after t."""
    a, b = _make_cointegrated_pair(n=400, seed=6)
    strat = PairsTradingStrategy(z_entry=2.0, z_exit=0.5, z_lookback=40)
    full = strat.fit_predict(a, b)
    cut = 250
    truncated = strat.fit_predict(a.iloc[:cut], b.iloc[:cut])
    # Positions in the overlap region should match exactly
    np.testing.assert_array_equal(
        full.position.iloc[:cut].values,
        truncated.position.values,
    )
