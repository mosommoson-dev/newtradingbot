"""Critical tests: verify no look-ahead bias in features or backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eurusd_quant_bot.backtest import run_backtest
from eurusd_quant_bot.ml import _indicators as ind
from eurusd_quant_bot.ml.features import build_features
from eurusd_quant_bot.strategy import MeanReversionStrategy


def test_rsi_does_not_see_future(synthetic_ohlcv):
    close = synthetic_ohlcv["close"]
    full = ind.rsi(close, 14)
    truncated = ind.rsi(close.iloc[:-1], 14)
    # Last value of truncated must equal the value at the same index in full.
    np.testing.assert_allclose(
        truncated.iloc[-1], full.iloc[-2], rtol=1e-9, equal_nan=True
    )


def test_features_are_strictly_causal(synthetic_ohlcv):
    """Truncating data must not change earlier feature rows."""
    full = build_features(synthetic_ohlcv)
    truncated = build_features(synthetic_ohlcv.iloc[:-50])
    common = full.index.intersection(truncated.index)
    pd.testing.assert_frame_equal(
        full.loc[common, :].reset_index(drop=True),
        truncated.loc[common, :].reset_index(drop=True),
        check_exact=False, rtol=1e-9, atol=1e-9,
    )


def test_features_no_negative_lag(synthetic_ohlcv):
    """No feature column should depend on tomorrow's close."""
    df = synthetic_ohlcv.copy()
    feat_real = build_features(df)
    # Mutate the *last* close drastically, and confirm earlier features are unchanged.
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 0.5
    feat_perturbed = build_features(df2)
    common = feat_real.index.intersection(feat_perturbed.index)[:-1]
    pd.testing.assert_frame_equal(
        feat_real.loc[common, :].reset_index(drop=True),
        feat_perturbed.loc[common, :].reset_index(drop=True),
        check_exact=False, rtol=1e-9, atol=1e-9,
    )


def test_backtest_fills_at_next_bar_open(synthetic_ohlcv):
    """Engineered fills happen at next bar's open, never current close."""
    s = MeanReversionStrategy(z_score_period=20, z_score_entry=1.0,
                              rsi_low=45, rsi_high=55, bb_low=0.45, bb_high=0.55,
                              adx_max=80)
    sigs = s.generate_signals(synthetic_ohlcv)
    res = run_backtest(synthetic_ohlcv, sigs)
    # If we have any trades, every entry_price must equal the open of the bar
    # AFTER the bar that produced the signal.
    if not res.trades.empty:
        for _, t in res.trades.iterrows():
            ts = t["entry_time"]
            actual_open = synthetic_ohlcv.loc[ts, "open"]
            assert abs(actual_open - t["entry_price"]) < 1e-9
