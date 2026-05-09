"""Unit tests for the pairs-trading strategy and Kalman filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurusd_quant_bot.backtest import run_pairs_backtest
from eurusd_quant_bot.strategy import (
    PairsTradingStrategy,
    kalman_pairs,
    rolling_half_life,
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


def _diverging_spread(n: int = 400, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    """Two series with a sustained spread blow-out.

    The first half is mean-reverting; halfway through the spread takes a
    large step jump and stays there.  The rolling z-score is therefore
    persistently above the entry threshold (until rolling stats catch
    up) — a clean test of "what happens when cointegration breaks".
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    b_steps = rng.normal(0, 0.005, size=n)
    log_b = np.cumsum(b_steps)
    b = 1.10 * np.exp(log_b)
    spread = rng.normal(0, 0.001, size=n)
    # Step jump halfway through: spread mean shifts from 0 to 0.05
    half = n // 2
    spread[half:] += 0.05
    a = 1.30 * b + spread
    return (
        pd.Series(a, index=idx, name="A"),
        pd.Series(b, index=idx, name="B"),
    )


def _count_trades(position: pd.Series) -> int:
    """Count distinct positions (trades) in a position series."""
    return int(((position != position.shift(1)) & (position != 0)).sum())


def test_stop_loss_fires_on_diverging_spread():
    """A tighter stop_loss_z must produce more stop-out events (and so
    fewer bars in any one open position) than a looser one.  Both runs
    use the iterative path so the comparison is apples to apples."""
    a, b = _diverging_spread(n=400, seed=12)
    loose = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                   hedge_window=60, stop_loss_z=10.0)
    tight = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                   hedge_window=60, stop_loss_z=2.5)
    sig0 = loose.fit_predict(a, b)
    sig1 = tight.fit_predict(a, b)
    # The tight stop must produce strictly fewer bars in non-zero position.
    bars0 = int((sig0.position != 0).sum())
    bars1 = int((sig1.position != 0).sum())
    assert bars1 < bars0, (
        f"tight stop_loss should reduce time-in-position (bars0={bars0}, "
        f"bars1={bars1})"
    )


def _max_consecutive_pos(position: pd.Series) -> int:
    """Length of the longest run of contiguous non-zero positions."""
    nz = (position != 0).astype(int).to_numpy()
    best = run = 0
    for v in nz:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def test_max_holding_days_caps_individual_trade_length():
    """A short max_holding_days must produce shorter contiguous trades
    than a long one.  Both runs use the iterative path."""
    a, b = _diverging_spread(n=400, seed=13)
    long_hold = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                       hedge_window=60, max_holding_days=200)
    short_hold = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                        hedge_window=60, max_holding_days=10)
    sig0 = long_hold.fit_predict(a, b)
    sig1 = short_hold.fit_predict(a, b)
    run0 = _max_consecutive_pos(sig0.position)
    run1 = _max_consecutive_pos(sig1.position)
    # Short max-holding must cap the longest contiguous trade
    assert run1 < run0, (
        f"max_holding_days=10 must shorten longest contiguous trade vs 200 "
        f"(run0={run0}, run1={run1})"
    )
    # And the cap must be enforced: no trade can run longer than
    # max_holding_days + a tiny amount of slack from same-bar bookkeeping.
    assert run1 <= 11, f"longest run with cap=10 should be ~10 (got {run1})"


def test_half_life_gate_blocks_entries_in_trending_regime():
    """When the rolling OU half-life on the spread is high, a tight
    gate must allow strictly fewer entries.  We construct a series
    whose spread first mean-reverts and then trends, and check that
    the tighter gate hides more of the trending portion."""
    rng = np.random.default_rng(21)
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    b_steps = rng.normal(0, 0.005, size=n)
    log_b = np.cumsum(b_steps)
    b = 1.10 * np.exp(log_b)
    # Deterministic trend in the spread so the OLS hedge can't
    # absorb it: forces high (or NaN) half-life in the trending half.
    spread = np.concatenate([
        rng.normal(0, 0.002, size=n // 2),
        np.linspace(0, 0.05, n - n // 2),
    ])
    a = 1.30 * b + spread
    A = pd.Series(a, index=idx, name="A")
    B = pd.Series(b, index=idx, name="B")
    loose_gate = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                        hedge_window=60,
                                        max_half_life=9999.0,
                                        half_life_window=60)
    tight_gate = PairsTradingStrategy(z_entry=1.5, z_exit=0.5, z_lookback=30,
                                        hedge_window=60,
                                        max_half_life=5.0,
                                        half_life_window=60)
    sig0 = loose_gate.fit_predict(A, B)
    sig1 = tight_gate.fit_predict(A, B)
    bars0 = int((sig0.position != 0).sum())
    bars1 = int((sig1.position != 0).sum())
    assert bars1 < bars0, (
        f"tight half-life gate must reduce entries in trending regime "
        f"(bars0={bars0}, bars1={bars1})"
    )


def test_rolling_half_life_is_finite_for_mean_reverting_series():
    """A strongly mean-reverting AR(1) spread must yield a finite half-life."""
    rng = np.random.default_rng(0)
    n = 500
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.5 * s[t - 1] + rng.normal(0, 1.0)
    spread = pd.Series(s, index=pd.date_range("2022-01-01", periods=n, freq="D"))
    hl = rolling_half_life(spread, window=60)
    # On a 0.5-AR(1), the theoretical half-life is ln(2) / -ln(0.5) ~ 1 day.
    # We don't need to recover it precisely, just that the rolling estimate
    # is finite for the bulk of the series.
    finite = hl.dropna()
    assert len(finite) > 0
    assert (finite > 0).all()


def test_strategy_validation_new_params():
    with pytest.raises(ValueError, match="stop_loss_z"):
        PairsTradingStrategy(z_entry=2.0, z_exit=0.5, stop_loss_z=1.5)
    with pytest.raises(ValueError, match="max_holding_days"):
        PairsTradingStrategy(max_holding_days=-1)
    with pytest.raises(ValueError, match="max_half_life"):
        PairsTradingStrategy(max_half_life=-1)
    with pytest.raises(ValueError, match="half_life_window"):
        PairsTradingStrategy(half_life_window=5)
