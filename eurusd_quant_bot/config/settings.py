"""Centralised, fully-typed configuration for the EUR/USD quant trading bot.

All tunables in the bot live here.  Environment variables (loaded from `.env` via
`python-dotenv`) override defaults so containers / CI can change behaviour without
code changes.

Magic numbers are not allowed elsewhere in the codebase: every numeric literal
that has any business meaning must come from one of the dataclasses below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load `.env` exactly once at import time.  `override=False` means real env vars
# (e.g. those set by docker-compose) take precedence over `.env`.
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
FEATURES_DIR: Final[Path] = DATA_DIR / "features"
REPORTS_DIR: Final[Path] = REPO_ROOT / "reports"
LOG_DIR: Final[Path] = REPO_ROOT / "logs"
ARTIFACT_DIR: Final[Path] = PACKAGE_ROOT / "ml" / "artifacts"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, REPORTS_DIR, LOG_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None or value == "":
        return None
    return value


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


# ---------------------------------------------------------------------------
# Bot runtime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeSettings:
    mode: str = _env("BOT_MODE", "paper") or "paper"
    log_level: str = _env("LOG_LEVEL", "INFO") or "INFO"
    dashboard_host: str = _env("DASHBOARD_HOST", "0.0.0.0") or "0.0.0.0"
    dashboard_port: int = _env_int("DASHBOARD_PORT", 5000)


# ---------------------------------------------------------------------------
# Instrument / market structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSettings:
    """All EUR/USD-specific contract parameters."""

    pair: str = "EUR_USD"        # OANDA naming
    pair_yf: str = "EURUSD=X"    # yfinance ticker
    pip: float = 0.0001          # 1 pip in EUR/USD
    pip_value_per_lot: float = 10.0  # USD per pip per standard lot (100k)
    min_lot: float = 0.01
    lot_step: float = 0.01
    standard_lot_units: int = 100_000

    # Cost model -- OANDA averages used in backtests.
    avg_spread_pips: float = 0.8
    commission_per_lot_per_side_usd: float = 7.0
    base_slippage_pips: float = 0.3


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskSettings:
    """Risk envelope shared by backtest and live."""

    risk_per_trade: float = _env_float("RISK_PER_TRADE", 0.01)        # 1%
    max_daily_loss: float = _env_float("MAX_DAILY_LOSS", 0.02)        # 2%
    max_daily_loss_soft: float = 0.01                                  # halve risk above this
    max_total_drawdown: float = _env_float("MAX_TOTAL_DRAWDOWN", 0.05)  # 5%
    emergency_drawdown: float = _env_float("EMERGENCY_DRAWDOWN", 0.10)  # 10%
    max_open_positions: int = _env_int("MAX_OPEN_POSITIONS", 3)
    max_total_exposure: float = 0.05  # 5% of account at risk in open trades
    max_correlated_exposure: float = 0.03
    max_spread_pips: float = _env_float("MAX_SPREAD_PIPS", 1.5)
    kelly_fraction: float = 0.5  # half-Kelly
    kelly_lookback_trades: int = 50
    vol_target_annual: float = 0.10
    news_blackout_minutes: int = 30


# ---------------------------------------------------------------------------
# Strategy params (mid-range starting points; walk-forward will re-optimize)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeanReversionParams:
    z_score_period: int = 50
    z_score_entry: float = 2.0
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    bb_low: float = 0.20
    bb_high: float = 0.80
    adx_period: int = 14
    adx_max: float = 25.0   # only trade in ranging markets
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    # Walk-forward search ranges (used by Optuna).
    z_score_period_range: tuple[int, int] = (30, 80)
    z_score_entry_range: tuple[float, float] = (1.5, 2.5)
    rsi_period_range: tuple[int, int] = (7, 21)
    bb_period_range: tuple[int, int] = (15, 25)
    bb_std_range: tuple[float, float] = (1.5, 2.5)


@dataclass(frozen=True)
class TrendFollowingParams:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_period: int = 14
    atr_multiplier: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Walk-forward search ranges.
    ema_fast_range: tuple[int, int] = (10, 30)
    ema_slow_range: tuple[int, int] = (40, 80)
    ema_trend_range: tuple[int, int] = (100, 250)
    adx_threshold_range: tuple[float, float] = (20.0, 35.0)
    atr_multiplier_range: tuple[float, float] = (1.5, 3.0)


@dataclass(frozen=True)
class MLParams:
    horizon_bars: int = 4              # next 4-hour candle direction
    confidence_threshold: float = 0.60
    move_threshold_pips: float = 5.0
    train_window_months: int = 24
    test_window_months: int = 6
    retrain_frequency_days: int = 30
    sequence_length: int = 60          # LSTM lookback
    lstm_units_l1: int = 64
    lstm_units_l2: int = 32
    lstm_dense: int = 16
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    xgb_max_depth_range: tuple[int, int] = (3, 8)
    xgb_lr_range: tuple[float, float] = (0.01, 0.1)
    xgb_n_estimators_range: tuple[int, int] = (100, 500)
    top_k_features: int = 20
    correlation_drop_threshold: float = 0.9


@dataclass(frozen=True)
class PortfolioParams:
    weights: dict[str, float] = field(
        default_factory=lambda: {"mean_reversion": 0.40, "trend_following": 0.35, "ml_ensemble": 0.25}
    )
    rolling_sharpe_window_days: int = 30
    drop_strategy_below_sharpe: float = 0.0
    correlation_cap: float = 0.8


@dataclass(frozen=True)
class StrategySettings:
    mean_reversion: MeanReversionParams = field(default_factory=MeanReversionParams)
    trend_following: TrendFollowingParams = field(default_factory=TrendFollowingParams)
    ml: MLParams = field(default_factory=MLParams)
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestSettings:
    initial_balance: float = 10_000.0
    granularity: str = "H1"
    in_sample_start: str = "2015-01-01"
    in_sample_end: str = "2023-06-30"
    out_of_sample_start: str = "2023-07-01"
    out_of_sample_end: str = "2024-12-31"
    walk_forward_train_months: int = 24
    walk_forward_test_months: int = 6
    walk_forward_step_months: int = 6
    optuna_trials: int = 50
    monte_carlo_runs: int = 1000
    monte_carlo_slippage_perturb: float = 0.5  # ±50%
    monte_carlo_timing_perturb_bars: int = 1
    accept_min_sharpe: float = 1.0
    accept_max_drawdown: float = 0.15
    accept_min_profit_factor: float = 1.3


# ---------------------------------------------------------------------------
# Schedule / news filter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleSettings:
    """All times are UTC."""

    week_open_dow: int = 6        # Sunday (Mon=0 ... Sun=6)
    week_open_hour: int = 22
    week_close_dow: int = 4       # Friday
    week_close_hour: int = 21
    skip_first_minutes_of_session: int = 30
    avoid_low_vol_windows: tuple[tuple[int, int], ...] = (
        (7, 10),    # 07-10 UTC
        (14, 17),   # 14-17 UTC
    )
    high_value_windows: tuple[tuple[int, int], ...] = (
        (8, 12),    # London
        (13, 17),   # NY morning
    )


# ---------------------------------------------------------------------------
# Database / cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseSettings:
    host: str = _env("DB_HOST", "localhost") or "localhost"
    port: int = _env_int("DB_PORT", 5432)
    name: str = _env("DB_NAME", "forex_bot") or "forex_bot"
    user: str = _env("DB_USER", "trader") or "trader"
    password: str = _env("DB_PASSWORD", "change-me") or "change-me"

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class RedisSettings:
    host: str = _env("REDIS_HOST", "localhost") or "localhost"
    port: int = _env_int("REDIS_PORT", 6379)


# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str | None = _env("TELEGRAM_BOT_TOKEN")
    chat_id: str | None = _env("TELEGRAM_CHAT_ID")

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class FREDSettings:
    api_key: str | None = _env("FRED_API_KEY")
    series: tuple[str, ...] = (
        "DFF",          # Effective Federal Funds Rate
        "ECBDFR",       # ECB Deposit Facility Rate
        "CPIAUCSL",     # US CPI
        "PAYEMS",       # US Nonfarm Payrolls
        "DTWEXBGS",     # Trade Weighted USD Index
        "VIXCLS",       # VIX
    )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    instrument: InstrumentSettings = field(default_factory=InstrumentSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    fred: FREDSettings = field(default_factory=FREDSettings)


SETTINGS: Final[Settings] = Settings()


def get_settings() -> Settings:
    """Return the global, immutable :class:`Settings` instance.

    Tests that need to override values should construct their own ``Settings``
    instance and pass it explicitly rather than mutating this global.
    """
    return SETTINGS
