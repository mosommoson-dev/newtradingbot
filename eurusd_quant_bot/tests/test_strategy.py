"""Tests for the strategy layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurusd_quant_bot.strategy import (
    MeanReversionStrategy,
    MLEnsembleStrategy,
    TrendFollowingStrategy,
    combine,
    get_strategy,
)


def test_registry_returns_correct_class(synthetic_ohlcv):
    s = get_strategy("mean_reversion")
    assert isinstance(s, MeanReversionStrategy)
    s = get_strategy("trend_following")
    assert isinstance(s, TrendFollowingStrategy)
    s = get_strategy("ml_ensemble")
    assert isinstance(s, MLEnsembleStrategy)


def test_mean_reversion_signal_shape_and_values(synthetic_ohlcv):
    s = MeanReversionStrategy()
    sigs = s.generate_signals(synthetic_ohlcv)
    assert sigs.index.equals(synthetic_ohlcv.index)
    assert sigs.dtype == int
    assert set(sigs.unique()).issubset({-1, 0, 1})


def test_mean_reversion_generates_some_signals(synthetic_ohlcv):
    s = MeanReversionStrategy(z_score_period=30, z_score_entry=1.5,
                              rsi_low=40, rsi_high=60, bb_low=0.4, bb_high=0.6,
                              adx_max=80)
    sigs = s.generate_signals(synthetic_ohlcv)
    nonzero = (sigs != 0).sum()
    assert nonzero > 0, "loose-parameter mean-reversion should produce signals on synthetic data"


def test_trend_following_signals_values(synthetic_ohlcv):
    s = TrendFollowingStrategy(adx_threshold=10)
    sigs = s.generate_signals(synthetic_ohlcv)
    assert sigs.index.equals(synthetic_ohlcv.index)
    assert set(sigs.unique()).issubset({-1, 0, 1})


def test_ml_strategy_no_model_returns_zeros(synthetic_ohlcv):
    s = MLEnsembleStrategy()
    sigs = s.generate_signals(synthetic_ohlcv)
    assert sigs.eq(0).all()


def test_combine_clamps_to_unit_range(synthetic_ohlcv):
    pos_a = pd.Series(1, index=synthetic_ohlcv.index)
    pos_b = pd.Series(1, index=synthetic_ohlcv.index)
    pos_c = pd.Series(1, index=synthetic_ohlcv.index)
    out = combine({"mean_reversion": pos_a, "trend_following": pos_b, "ml_ensemble": pos_c})
    assert out.position.between(-1, 1).all()


def test_signal_dataclass_validates_direction():
    from eurusd_quant_bot.strategy.base_strategy import Signal

    Signal(ts=pd.Timestamp("2024-01-01", tz="UTC"), direction=1)
    Signal(ts=pd.Timestamp("2024-01-01", tz="UTC"), direction=-1)
    with pytest.raises(ValueError):
        Signal(ts=pd.Timestamp("2024-01-01", tz="UTC"), direction=2)


def test_position_holds_until_exit_condition(synthetic_ohlcv):
    """The mean-reversion strategy must not flip on every bar."""
    s = MeanReversionStrategy(z_score_period=30, z_score_entry=1.5,
                              rsi_low=45, rsi_high=55, bb_low=0.45, bb_high=0.55,
                              adx_max=80)
    sigs = s.generate_signals(synthetic_ohlcv)
    # No more than 30% of consecutive bars should change position.
    flips = (sigs.diff().fillna(0) != 0).sum()
    assert flips < 0.3 * len(sigs)


def test_signals_are_strictly_causal(synthetic_ohlcv):
    """Cutting off the last bar must not change earlier signals."""
    s = MeanReversionStrategy()
    full = s.generate_signals(synthetic_ohlcv)
    truncated = s.generate_signals(synthetic_ohlcv.iloc[:-1])
    np.testing.assert_array_equal(full.iloc[:-1].to_numpy(), truncated.to_numpy())
