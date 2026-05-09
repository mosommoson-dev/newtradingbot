"""Multi-source historical data fetchers.

Sources implemented:
    * **yfinance**           : free, no key, daily-to-1m EUR/USD bars from 2015+.
    * **OANDA v20**          : authoritative M1/M5/.../D1 candles; requires API token.
    * **FRED**               : macro time series (Fed Funds, ECB, CPI, NFP, DXY, VIX).
    * **CFTC Commitments**   : weekly net positioning via the public CFTC ZIP feed.
    * **Dukascopy (stub)**   : tick downloader is stubbed -- implementing the full
                               binary BI5 parser is left as a follow-up; the helper
                               below documents the format and download URL.

Each fetcher returns a tidy ``pandas.DataFrame`` indexed by UTC timestamp.  The
preprocessor consumes those frames, so behaviour is consistent across sources.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from ..config import get_settings
from ..config.brokers import load_oanda_credentials

# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

_YF_INTERVAL_MAP: dict[str, str] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "1h",   # yfinance has no native H4 -> we will resample
    "D1": "1d",
}


def fetch_yfinance(
    pair_yf: str | None = None,
    granularity: str = "H1",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch EUR/USD bars from yfinance.

    Returns a UTC-indexed dataframe with columns
    ``[open, high, low, close, volume]``.  H4 is produced by resampling H1.
    """
    import yfinance as yf  # imported lazily so tests don't pay the cost

    settings = get_settings()
    pair_yf = pair_yf or settings.instrument.pair_yf
    interval = _YF_INTERVAL_MAP.get(granularity, "1h")
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("yfinance: downloading {} {} {} -> {}", pair_yf, interval, start, end)
    df = yf.download(
        tickers=pair_yf,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        logger.warning("yfinance returned empty data for {}", pair_yf)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    df = df[["open", "high", "low", "close", "volume"]].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if granularity == "H4":
        df = (
            df.resample("4H", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(how="any")
        )

    df.index.name = "ts"
    return df


# ---------------------------------------------------------------------------
# OANDA
# ---------------------------------------------------------------------------

@dataclass
class OandaHistoricalFetcher:
    """Pulls historical candles from the OANDA REST API in pages of 5000 bars."""

    pair: str | None = None
    granularity: str = "H1"
    price_type: str = "M"  # M = mid, B = bid, A = ask

    def __post_init__(self) -> None:
        self.pair = self.pair or get_settings().instrument.pair

    def fetch(self, start: str, end: str | None = None) -> pd.DataFrame:
        creds = load_oanda_credentials()
        if not creds.is_configured:
            logger.warning(
                "OANDA credentials missing; skipping OANDA fetch for {} {}",
                self.pair,
                self.granularity,
            )
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            from oandapyV20 import API
            from oandapyV20.endpoints.instruments import InstrumentsCandles
        except Exception as exc:
            logger.warning("oandapyV20 not installed: {}", exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        api = API(access_token=creds.token, environment=creds.environment)
        params = {
            "from": _to_oanda_ts(start),
            "to": _to_oanda_ts(end or datetime.now(timezone.utc).isoformat()),
            "granularity": self.granularity,
            "price": self.price_type,
            "count": 5000,
        }

        rows: list[dict[str, float]] = []
        cursor = params["from"]
        while True:
            page_params = dict(params, **{"from": cursor})
            page_params.pop("to", None)  # use count-based paging forward
            req = InstrumentsCandles(instrument=self.pair, params=page_params)
            try:
                api.request(req)
            except Exception as exc:
                logger.error("OANDA fetch failed at cursor={}: {}", cursor, exc)
                break

            candles = req.response.get("candles", [])
            if not candles:
                break
            for c in candles:
                if not c.get("complete"):
                    continue
                price = c.get("mid") or c.get("bid") or c.get("ask") or {}
                rows.append(
                    {
                        "ts": c["time"],
                        "open": float(price["o"]),
                        "high": float(price["h"]),
                        "low": float(price["l"]),
                        "close": float(price["c"]),
                        "volume": float(c.get("volume", 0)),
                    }
                )
            last_ts = candles[-1]["time"]
            if last_ts == cursor or last_ts >= params["to"]:
                break
            cursor = last_ts

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index().drop_duplicates()
        return df


def _to_oanda_ts(value: str) -> str:
    """Convert an ISO date / datetime string into OANDA's RFC3339 format."""
    dt = pd.Timestamp(value)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


# ---------------------------------------------------------------------------
# FRED (macro)
# ---------------------------------------------------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch one FRED series.

    Returns a UTC-indexed DataFrame with a single column named ``series_id``.
    Values are NaN-aware (FRED returns ``"."`` for missing observations).
    """
    settings = get_settings()
    if not settings.fred.enabled:
        logger.warning("FRED_API_KEY not set; skipping fetch of {}", series_id)
        return pd.DataFrame(columns=[series_id])

    params = {
        "series_id": series_id,
        "api_key": settings.fred.api_key,
        "file_type": "json",
        "observation_start": start,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[series_id])

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").rename(columns={"value": series_id})
    return df


def fetch_fred_bundle(series: Iterable[str] | None = None) -> pd.DataFrame:
    """Fetch all configured FRED series and join them on their date index."""
    settings = get_settings()
    series = list(series) if series is not None else list(settings.fred.series)
    frames = []
    for s in series:
        try:
            frames.append(fetch_fred_series(s))
        except Exception as exc:
            logger.warning("FRED fetch for {} failed: {}", s, exc)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index().ffill()
    df.index.name = "ts"
    return df


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders
# ---------------------------------------------------------------------------

CFTC_FUT_URL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)


def fetch_cftc_eurusd(years: Iterable[int]) -> pd.DataFrame:
    """Fetch the CFTC Disaggregated COT report for EUR/USD futures.

    Parses the public ZIP archives provided by the CFTC and returns a tidy
    weekly frame with non-commercial net positioning, change vs prior week,
    and percent of open interest.
    """
    out: list[pd.DataFrame] = []
    for year in years:
        url = CFTC_FUT_URL_TEMPLATE.format(year=year)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("CFTC {} fetch failed: {}", year, exc)
            continue
        try:
            import zipfile

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    df = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            logger.warning("CFTC {} parse failed: {}", year, exc)
            continue

        eur = df[df["Market_and_Exchange_Names"].str.contains("EURO FX", case=False, na=False)]
        if eur.empty:
            continue
        eur = eur.assign(date=pd.to_datetime(eur["Report_Date_as_YYYY-MM-DD"], utc=True))
        eur = eur.set_index("date")
        eur["net_noncom"] = (
            eur["NonComm_Positions_Long_All"] - eur["NonComm_Positions_Short_All"]
        )
        out.append(
            eur[
                [
                    "Open_Interest_All",
                    "NonComm_Positions_Long_All",
                    "NonComm_Positions_Short_All",
                    "net_noncom",
                ]
            ]
        )

    if not out:
        return pd.DataFrame()
    df = pd.concat(out).sort_index()
    df["net_noncom_chg"] = df["net_noncom"].diff()
    df["net_noncom_pct_oi"] = df["net_noncom"] / df["Open_Interest_All"]
    df.index.name = "ts"
    return df


# ---------------------------------------------------------------------------
# Dukascopy
# ---------------------------------------------------------------------------

_DUKASCOPY_GRANULARITY: dict[str, str] = {
    "M1": "INTERVAL_MIN_1",
    "M5": "INTERVAL_MIN_5",
    "M15": "INTERVAL_MIN_15",
    "M30": "INTERVAL_MIN_30",
    "H1": "INTERVAL_HOUR_1",
    "H4": "INTERVAL_HOUR_4",
    "D1": "INTERVAL_DAY_1",
}


def dukascopy_url(pair: str, day: datetime, hour: int) -> str:
    """Public Dukascopy URL for one hour of raw BI5 tick data.

    Provided for reference / direct download workflows; the recommended path
    is :func:`fetch_dukascopy`, which uses the maintained ``dukascopy-python``
    library and returns clean OHLCV frames.
    """
    p = pair.replace("_", "").upper()
    return (
        f"https://datafeed.dukascopy.com/datafeed/{p}/"
        f"{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
    )


def fetch_dukascopy(
    pair: str = "EUR/USD",
    granularity: str = "H1",
    start: str | datetime = "2015-01-01",
    end: str | datetime | None = None,
    offer_side: str = "BID",
) -> pd.DataFrame:
    """Fetch aggregated OHLCV from Dukascopy via ``dukascopy-python``.

    Parameters
    ----------
    pair :
        Instrument (e.g. ``"EUR/USD"``).
    granularity :
        One of ``M1, M5, M15, M30, H1, H4, D1``.
    start, end :
        ISO-format strings or timezone-aware :class:`datetime` objects in UTC.
    offer_side :
        ``"BID"`` or ``"ASK"`` -- BID is the conservative default for backtests.

    Returns
    -------
    DataFrame with columns ``open, high, low, close, volume`` and a UTC
    ``DatetimeIndex``.
    """
    import dukascopy_python as dp

    if granularity not in _DUKASCOPY_GRANULARITY:
        raise ValueError(f"Unsupported granularity {granularity!r}")
    interval = getattr(dp, _DUKASCOPY_GRANULARITY[granularity])
    side = dp.OFFER_SIDE_BID if offer_side.upper() == "BID" else dp.OFFER_SIDE_ASK

    def _to_dt(value: str | datetime | None, default: datetime) -> datetime:
        if value is None:
            return default
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

    start_dt = _to_dt(start, datetime(2015, 1, 1, tzinfo=timezone.utc))
    end_dt = _to_dt(end, datetime.now(timezone.utc))

    logger.info(
        "Dukascopy: downloading {} {} {} -> {}",
        pair, granularity, start_dt.date(), end_dt.date(),
    )
    df = dp.fetch(
        instrument=pair,
        interval=interval,
        offer_side=side,
        start=start_dt,
        end=end_dt,
    )
    if df.empty:
        logger.warning("Dukascopy returned empty frame for {} {}", pair, granularity)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def cache_dataframe(df: pd.DataFrame, name: str) -> Path:
    """Persist a frame to ``data/raw/<name>.parquet`` and return the path."""
    from ..config.settings import RAW_DATA_DIR

    path = RAW_DATA_DIR / f"{name}.parquet"
    if df.empty:
        logger.warning("Refusing to cache empty frame to {}", path)
        return path
    df.to_parquet(path)
    logger.info("Cached {} rows to {}", len(df), path)
    return path
