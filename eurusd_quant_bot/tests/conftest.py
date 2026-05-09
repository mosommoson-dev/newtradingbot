"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate a deterministic synthetic EUR/USD-like OHLCV series.

    The series alternates between a mean-reverting regime and a trending
    regime so that both strategies can produce signals.
    """
    rng = np.random.default_rng(42)
    n = 2_000
    idx = pd.date_range("2022-01-03", periods=n, freq="1h", tz="UTC")
    drift = np.concatenate([
        np.zeros(n // 2),                      # mean-reverting
        np.linspace(0, 0.01, n - n // 2),      # gentle uptrend
    ])
    noise = rng.normal(0, 0.0008, size=n)
    log_price = np.cumsum(drift / n + noise)
    close = 1.10 * np.exp(log_price)
    high = close * (1 + rng.uniform(0, 0.0008, size=n))
    low = close * (1 - rng.uniform(0, 0.0008, size=n))
    open_ = close * (1 + rng.normal(0, 0.0003, size=n))
    volume = rng.integers(1000, 5000, size=n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "ts"
    return df
