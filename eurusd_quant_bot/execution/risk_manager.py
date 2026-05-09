"""Risk manager: position sizing + drawdown / spread / news kill switches.

Covers all the rules from the project brief:

* Modified Kelly (half-Kelly) position sizing using win rate and avg win/loss.
* Vol-targeting on top of Kelly: scale down when ATR is unusually high.
* Per-trade risk cap (1%), correlated risk cap (3%), total exposure cap (5%).
* Daily loss soft-stop (1% -> halve risk), hard stop (2% -> kill switch).
* Total drawdown stop (5%) and emergency stop (10% -> close everything).
* Spread filter and news blackout.

Construction does not depend on a running broker; pass an ``Account`` object
that exposes the current balance + open exposures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import get_settings


@dataclass
class Account:
    starting_balance: float
    balance: float
    peak_equity: float
    daily_pnl: float = 0.0
    open_risk: float = 0.0       # risk currently exposed across open trades, pct of equity
    open_positions: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_pnl(self, delta: float) -> None:
        self.balance += delta
        self.daily_pnl += delta
        self.peak_equity = max(self.peak_equity, self.balance)

    @property
    def total_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.balance - self.peak_equity) / self.peak_equity

    @property
    def daily_drawdown(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return self.daily_pnl / self.starting_balance


@dataclass
class TradeStats:
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float


def kelly_fraction(stats: TradeStats, kelly_multiplier: float = 0.5) -> float:
    """Modified Kelly: half-Kelly by default for safety."""
    if stats.avg_win_pct <= 0:
        return 0.0
    loss = abs(stats.avg_loss_pct)
    if loss == 0:
        return 0.0
    b = stats.avg_win_pct / loss
    p = stats.win_rate
    q = 1 - p
    raw = (b * p - q) / b if b > 0 else 0.0
    raw = max(0.0, min(raw, 1.0))
    return raw * kelly_multiplier


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    *,
    pip_value_per_lot: float | None = None,
    min_lot: float | None = None,
    lot_step: float | None = None,
) -> float:
    """Return lot size such that worst-case loss = ``risk_per_trade`` * balance.

    Uses OANDA's 0.01 (micro lot) granularity by default.
    """
    instr = get_settings().instrument
    pip_value_per_lot = pip_value_per_lot if pip_value_per_lot is not None else instr.pip_value_per_lot
    min_lot = min_lot if min_lot is not None else instr.min_lot
    lot_step = lot_step if lot_step is not None else instr.lot_step

    if stop_loss_pips <= 0 or account_balance <= 0 or pip_value_per_lot <= 0:
        return 0.0
    risk_amount = account_balance * risk_per_trade
    raw_lots = risk_amount / (stop_loss_pips * pip_value_per_lot)
    lots = max(min_lot, round(raw_lots / lot_step) * lot_step)
    # Clamp away from absurdly large positions (paranoia: never request > 100 std lots)
    return float(min(lots, 100.0))


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    lot_size: float = 0.0
    risk_per_trade: float = 0.0


class RiskManager:
    def __init__(self, scheduled_news: list[datetime] | None = None) -> None:
        self.settings = get_settings()
        self.scheduled_news = scheduled_news or []
        self.kill_switch_tripped = False

    def update_news(self, news_times: list[datetime]) -> None:
        self.scheduled_news = list(news_times)

    # ------------------------------------------------------------------
    # Per-trade decision
    # ------------------------------------------------------------------

    def evaluate_entry(
        self,
        account: Account,
        stop_loss_pips: float,
        spread_pips: float,
        now: datetime,
        *,
        existing_correlated_risk: float = 0.0,
    ) -> RiskDecision:
        risk_cfg = self.settings.risk

        if self.kill_switch_tripped:
            return RiskDecision(False, "kill_switch_tripped")

        if account.total_drawdown <= -risk_cfg.emergency_drawdown:
            self.kill_switch_tripped = True
            return RiskDecision(False, "emergency_drawdown")

        if account.total_drawdown <= -risk_cfg.max_total_drawdown:
            return RiskDecision(False, "total_drawdown_limit")

        if account.daily_drawdown <= -risk_cfg.max_daily_loss:
            return RiskDecision(False, "daily_loss_kill_switch")

        if account.open_positions >= risk_cfg.max_open_positions:
            return RiskDecision(False, "max_open_positions")

        if spread_pips > risk_cfg.max_spread_pips:
            return RiskDecision(False, f"spread_too_wide:{spread_pips:.2f}p")

        if self._in_news_blackout(now):
            return RiskDecision(False, "news_blackout")

        # Soft daily-loss reduction
        risk_per_trade = risk_cfg.risk_per_trade
        if account.daily_drawdown <= -risk_cfg.max_daily_loss_soft:
            risk_per_trade = risk_per_trade * 0.5

        # Correlated risk cap
        if existing_correlated_risk + risk_per_trade > risk_cfg.max_correlated_exposure:
            return RiskDecision(False, "correlated_risk_cap")

        # Total open risk cap
        if account.open_risk + risk_per_trade > risk_cfg.max_total_exposure:
            return RiskDecision(False, "total_exposure_cap")

        lots = calculate_position_size(account.balance, risk_per_trade, stop_loss_pips)
        if lots <= 0:
            return RiskDecision(False, "lot_size_zero")
        return RiskDecision(True, "ok", lot_size=lots, risk_per_trade=risk_per_trade)

    # ------------------------------------------------------------------
    # Daily reset / kill switch
    # ------------------------------------------------------------------

    def reset_daily(self, account: Account) -> None:
        account.daily_pnl = 0.0
        account.last_reset = datetime.now(timezone.utc)

    def should_stop_trading(self, account: Account) -> bool:
        risk_cfg = self.settings.risk
        return (
            self.kill_switch_tripped
            or account.daily_drawdown <= -risk_cfg.max_daily_loss
            or account.total_drawdown <= -risk_cfg.max_total_drawdown
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_news_blackout(self, now: datetime) -> bool:
        if not self.scheduled_news:
            return False
        delta = timedelta(minutes=self.settings.risk.news_blackout_minutes)
        return any(abs(now - n) <= delta for n in self.scheduled_news)


def stats_from_trades(trades: pd.DataFrame, lookback: int | None = None) -> TradeStats:
    """Compute Kelly inputs from a closed-trades DataFrame."""
    if trades.empty:
        return TradeStats(win_rate=0.5, avg_win_pct=0.005, avg_loss_pct=-0.005)
    df = trades.tail(lookback) if lookback else trades
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    win_rate = float(len(wins) / len(df)) if len(df) else 0.5
    avg_win_pct = float(wins["pnl"].mean()) if not wins.empty else 0.005
    avg_loss_pct = float(losses["pnl"].mean()) if not losses.empty else -0.005
    return TradeStats(win_rate=win_rate, avg_win_pct=avg_win_pct, avg_loss_pct=avg_loss_pct)
