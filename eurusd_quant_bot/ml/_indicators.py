"""Pure-numpy / pandas implementations of the technical indicators we use.

We deliberately avoid making ``ta-lib`` (the C-bound wheel) a hard runtime
dependency: the underlying C library is awkward to install in many CI
environments.  When the optional ``talib`` import succeeds we delegate to it
for speed; otherwise we fall back to the pandas implementations below.

All functions accept and return ``pandas.Series`` indexed by timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dep
    import talib as _talib  # type: ignore[import-not-found]

    _HAS_TALIB = True
except Exception:
    _talib = None  # type: ignore[assignment]
    _HAS_TALIB = False


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TALIB:
        out = _talib.RSI(close.astype(float).values, timeperiod=period)
        return pd.Series(out, index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               fastk: int = 14, slowk: int = 3, slowd: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(fastk).min()
    highest = high.rolling(fastk).max()
    k_fast = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = k_fast.rolling(slowk).mean()
    d = k.rolling(slowd).mean()
    return k, d


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------

def bollinger(close: pd.Series, period: int = 20, std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return upper, mid, lower


# ---------------------------------------------------------------------------
# ATR / true range
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# ADX / directional movement
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_, plus_di, minus_di


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_ = tp.rolling(period).mean()
    md = (tp - sma_).abs().rolling(period).mean()
    return (tp - sma_) / (0.015 * md.replace(0, np.nan))


# ---------------------------------------------------------------------------
# OBV / MFI / Williams %R
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    raw_flow = tp * volume
    direction = np.sign(tp.diff().fillna(0.0))
    pos_flow = raw_flow.where(direction > 0, 0.0)
    neg_flow = raw_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        short_p: int = 7, mid_p: int = 14, long_p: int = 28) -> pd.Series:
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = true_range(high, low, close)

    def _avg(p: int) -> pd.Series:
        return bp.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)

    return 100 * (4 * _avg(short_p) + 2 * _avg(mid_p) + _avg(long_p)) / 7


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def realized_vol(returns: pd.Series, period: int = 20) -> pd.Series:
    return returns.rolling(period).std(ddof=0) * np.sqrt(period)


def parkinson_vol(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    factor = 1.0 / (4.0 * np.log(2.0))
    log_hl = np.log(high / low) ** 2
    return np.sqrt(factor * log_hl.rolling(period).mean())


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(rs.rolling(period).mean())


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP, reset daily on the timestamp index."""
    tp = (high + low + close) / 3
    pv = tp * volume
    grouped = pd.DataFrame({"pv": pv, "v": volume}).groupby(close.index.normalize())
    cum_pv = grouped["pv"].cumsum()
    cum_v = grouped["v"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, period: int = 100) -> pd.Series:
    """Rolling Hurst exponent estimate via R/S analysis.

    H ~ 0.5 -> random walk, H > 0.5 -> trending, H < 0.5 -> mean-reverting.
    """

    def _rs(window: np.ndarray) -> float:
        n = len(window)
        if n < 16:
            return np.nan
        try:
            lags = range(2, max(8, n // 4))
            tau = [np.std(np.subtract(window[lag:], window[:-lag])) for lag in lags]
            tau = [t for t in tau if t > 0]
            if len(tau) < 2:
                return np.nan
            poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
            return float(poly[0])
        except Exception:
            return np.nan

    return series.rolling(period).apply(_rs, raw=True)


# ---------------------------------------------------------------------------
# Candlestick patterns (no ta-lib): a few common ones
# ---------------------------------------------------------------------------

def candlestick_doji(o: pd.Series, h: pd.Series, lo: pd.Series, c: pd.Series,
                     thresh: float = 0.1) -> pd.Series:
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    return (body / rng < thresh).astype(int)


def candlestick_hammer(o: pd.Series, h: pd.Series, lo: pd.Series, c: pd.Series) -> pd.Series:
    body = (c - o).abs()
    upper = h - c.combine(o, max)
    lower_wick = c.combine(o, min) - lo
    rng = (h - lo).replace(0, np.nan)
    cond = (lower_wick > 2 * body) & (upper < body) & (body / rng < 0.4)
    return cond.astype(int)


def candlestick_engulfing(o: pd.Series, c: pd.Series) -> pd.Series:
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    bull = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)
    bear = (prev_c > prev_o) & (c < o) & (c < prev_o) & (o > prev_c)
    return bull.astype(int) - bear.astype(int)
