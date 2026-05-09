"""Cleaning + resampling utilities.

The fetcher modules return raw dataframes; the preprocessor turns them into the
canonical OHLCV structure the rest of the bot expects:
    - UTC tz-aware index named ``ts``
    - columns: ``open``, ``high``, ``low``, ``close``, ``volume`` (volume optional)
    - sorted, deduplicated, no rows where ``high < low`` or close <= 0
    - missing bars: forward-filled when the gap is short (default <= 3),
      dropped when longer.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

OHLCV = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(df: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Validate, sort, and gap-fill an OHLCV frame.

    Parameters
    ----------
    df : DataFrame
        Raw frame as returned by any of the fetchers.
    max_ffill : int
        Maximum number of consecutive missing bars that may be forward-filled.
        Anything longer is dropped to avoid manufacturing trades over real
        market closures.
    """
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[[c for c in OHLCV if c in df.columns]]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "ts"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    invalid = (df["high"] < df["low"]) | (df["close"] <= 0) | (df["open"] <= 0)
    if invalid.any():
        logger.warning("Dropping {} invalid OHLC rows", int(invalid.sum()))
        df = df[~invalid]

    df = df.ffill(limit=max_ffill).dropna(subset=["open", "high", "low", "close"])
    return df


def resample(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Resample a clean OHLCV frame to a coarser pandas offset alias.

    ``granularity`` accepts pandas aliases (``"4H"``, ``"1D"``) or OANDA
    granularities (``"H4"``, ``"D1"``).
    """
    alias_map = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
                 "H1": "1H", "H4": "4H", "D1": "1D"}
    rule = alias_map.get(granularity, granularity)
    out = (
        df.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(how="any")
    )
    return out


def align_macro(price: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Join a daily/weekly macro frame onto a price frame using forward-fill.

    Macro data is observed at lower frequency than prices, so we use as-of
    forward-fill -- never look-ahead.
    """
    if macro is None or macro.empty:
        return price.copy()
    aligned = macro.reindex(price.index, method="ffill")
    return price.join(aligned, how="left")
