"""Feature engineering pipeline.

The function :func:`build_features` consumes a clean OHLCV frame (and
optionally a macro frame from FRED + a CFTC frame) and emits more than 50
strictly-causal features.  Strict causality is enforced by:

1. Every rolling/exponential window only uses past observations.
2. We **never** call ``shift(-n)`` for features (only for the *target* used in
   ML training, never for inference).
3. The optional macro / sentiment series are forward-filled, never backfilled.

The output frame is left-joined onto the price index, so the row at time ``t``
contains only information available *at the close of bar ``t``*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _indicators as ind


def _returns_block(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    log_ret = np.log(close).diff()
    simple_ret = close.pct_change()
    for n in (1, 5, 10, 20):
        out[f"log_return_{n}"] = log_ret.rolling(n).sum()
        out[f"simple_return_{n}"] = simple_ret.rolling(n).sum()
    out["log_return"] = log_ret
    out["simple_return"] = simple_ret
    return out


def _volatility_block(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
                      ) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    log_ret = np.log(close).diff()
    out["atr_14"] = ind.atr(high, low, close, 14)
    out["atr_ratio"] = out["atr_14"] / close
    out["realized_vol_20"] = ind.realized_vol(log_ret, 20)
    out["parkinson_vol_20"] = ind.parkinson_vol(high, low, 20)
    out["garman_klass_vol_20"] = ind.garman_klass_vol(open_, high, low, close, 20)
    return out


def _price_level_block(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
                       ) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    vwap_series = ind.vwap(high, low, close, volume.replace(0, np.nan).fillna(1.0))
    out["dist_to_vwap"] = (close - vwap_series) / close
    for p in (20, 50, 100, 200):
        out[f"sma_{p}"] = ind.sma(close, p)
        out[f"ema_{p}"] = ind.ema(close, p)
        out[f"dist_sma_{p}"] = (close - out[f"sma_{p}"]) / close
        out[f"dist_ema_{p}"] = (close - out[f"ema_{p}"]) / close
    return out


def _ta_block(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out["rsi_14"] = ind.rsi(close, 14)
    out["rsi_7"] = ind.rsi(close, 7)
    k, d = ind.stochastic(high, low, close, 14, 3, 3)
    out["stoch_k"] = k
    out["stoch_d"] = d
    out["cci_20"] = ind.cci(high, low, close, 20)

    line, sig, hist = ind.macd(close, 12, 26, 9)
    out["macd"] = line
    out["macd_signal"] = sig
    out["macd_hist"] = hist
    out["macd_signal_slope"] = sig.diff()

    upper, mid, lower = ind.bollinger(close, 20, 2.0)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_width"] = (upper - lower) / mid
    out["bb_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)
    out["dist_bb_upper"] = (upper - close) / close
    out["dist_bb_lower"] = (close - lower) / close

    adx_, plus_di, minus_di = ind.adx(high, low, close, 14)
    out["adx_14"] = adx_
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["dx_slope"] = (plus_di - minus_di).diff()

    out["obv"] = ind.obv(close, volume.replace(0, np.nan).fillna(0.0))
    out["mfi_14"] = ind.money_flow_index(high, low, close, volume.fillna(0.0), 14)
    out["williams_r_14"] = ind.williams_r(high, low, close, 14)
    out["ultimate_osc"] = ind.ultimate_oscillator(high, low, close)

    # Candlesticks
    out["cdl_doji"] = ind.candlestick_doji(open_, high, low, close)
    out["cdl_hammer"] = ind.candlestick_hammer(open_, high, low, close)
    out["cdl_engulfing"] = ind.candlestick_engulfing(open_, close)
    return out


def _statistical_block(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    log_ret = np.log(close).diff()
    out["zscore_50"] = (close - close.rolling(50).mean()) / close.rolling(50).std(ddof=0)
    out["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0)
    for lag in (1, 5, 10, 20):
        out[f"autocorr_{lag}"] = (
            log_ret.rolling(50)
            .apply(lambda x, lag=lag: pd.Series(x).autocorr(lag), raw=False)
        )
    out["hurst_100"] = ind.hurst_exponent(close, 100)
    return out


def _macro_block(macro: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if macro is None or macro.empty:
        return pd.DataFrame(index=index)
    aligned = macro.reindex(index, method="ffill")
    out = pd.DataFrame(index=index)

    if "DTWEXBGS" in aligned.columns:
        out["dxy_return_1d"] = np.log(aligned["DTWEXBGS"]).diff()
    if {"DFF", "ECBDFR"}.issubset(aligned.columns):
        out["rate_diff_us_eu"] = aligned["DFF"] - aligned["ECBDFR"]
        out["rate_diff_chg"] = out["rate_diff_us_eu"].diff()
    if "CPIAUCSL" in aligned.columns:
        out["cpi_yoy"] = aligned["CPIAUCSL"].pct_change(12)
    if "PAYEMS" in aligned.columns:
        out["nfp_chg"] = aligned["PAYEMS"].diff()
    if "VIXCLS" in aligned.columns:
        out["vix_level"] = aligned["VIXCLS"]
        out["vix_chg"] = aligned["VIXCLS"].pct_change()
    return out


def _cot_block(cot: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if cot is None or cot.empty:
        return pd.DataFrame(index=index)
    aligned = cot.reindex(index, method="ffill")
    out = pd.DataFrame(index=index)
    if "net_noncom" in aligned.columns:
        out["cot_net_noncom"] = aligned["net_noncom"]
    if "net_noncom_chg" in aligned.columns:
        out["cot_net_chg"] = aligned["net_noncom_chg"]
    if "net_noncom_pct_oi" in aligned.columns:
        out["cot_pct_oi"] = aligned["net_noncom_pct_oi"]
    return out


def build_features(
    ohlcv: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    cot: pd.DataFrame | None = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Build the full causal feature matrix.

    Parameters
    ----------
    ohlcv : DataFrame
        Cleaned OHLCV frame (columns ``open``, ``high``, ``low``, ``close``,
        ``volume``).
    macro : DataFrame, optional
        Macro series indexed by date, joined with forward-fill.
    cot : DataFrame, optional
        Weekly CFTC COT positioning, joined with forward-fill.
    drop_warmup : bool
        If True, drop leading rows that contain any NaN due to indicator
        warmup (longest window is 200 for the EMA / SMA pair).
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")

    open_ = ohlcv["open"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    volume = ohlcv.get("volume", pd.Series(0.0, index=ohlcv.index)).astype(float)

    blocks = [
        ohlcv[[c for c in ("open", "high", "low", "close", "volume") if c in ohlcv.columns]].copy(),
        _returns_block(close),
        _volatility_block(open_, high, low, close),
        _price_level_block(high, low, close, volume),
        _ta_block(open_, high, low, close, volume),
        _statistical_block(close),
        _macro_block(macro, close.index),
        _cot_block(cot, close.index),
    ]
    out = pd.concat(blocks, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    if drop_warmup:
        out = out.dropna(subset=["sma_200"])

    out.index.name = "ts"
    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    """Return the engineered feature columns (excluding raw OHLCV and any
    label/target columns that may have been joined in)."""
    excluded = {"open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns if c not in excluded]
