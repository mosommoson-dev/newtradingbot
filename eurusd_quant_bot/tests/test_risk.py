"""Tests for the risk manager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from eurusd_quant_bot.execution.risk_manager import Account, RiskManager


def _make_account(balance: float = 10_000.0, daily_pnl: float = 0.0,
                  peak: float | None = None) -> Account:
    return Account(
        starting_balance=balance,
        balance=balance,
        peak_equity=peak if peak is not None else balance,
        daily_pnl=daily_pnl,
    )


def test_kill_switch_at_2pct_daily_dd():
    rm = RiskManager()
    acc = _make_account(daily_pnl=-250)  # -2.5% of starting balance
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8,
                                 now=datetime.now(timezone.utc))
    assert not decision.allowed
    assert decision.reason == "daily_loss_kill_switch"


def test_total_dd_blocks_trading():
    rm = RiskManager()
    # Drop from 10_000 -> 9_400 (-6%) > 5%
    acc = Account(starting_balance=10_000, balance=9_400, peak_equity=10_000)
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8,
                                 now=datetime.now(timezone.utc))
    assert not decision.allowed
    assert decision.reason == "total_drawdown_limit"


def test_emergency_dd_trips_kill_switch():
    rm = RiskManager()
    acc = Account(starting_balance=10_000, balance=8_900, peak_equity=10_000)
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8,
                                 now=datetime.now(timezone.utc))
    assert not decision.allowed
    assert decision.reason == "emergency_drawdown"
    assert rm.kill_switch_tripped


def test_spread_filter_blocks_trade():
    rm = RiskManager()
    acc = _make_account()
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=2.0,
                                 now=datetime.now(timezone.utc))
    assert not decision.allowed
    assert decision.reason.startswith("spread_too_wide")


def test_news_blackout_blocks_trade():
    now = datetime.now(timezone.utc)
    rm = RiskManager(scheduled_news=[now])
    acc = _make_account()
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8, now=now)
    assert not decision.allowed
    assert decision.reason == "news_blackout"


def test_news_blackout_allows_trade_outside_window():
    now = datetime.now(timezone.utc)
    rm = RiskManager(scheduled_news=[now + timedelta(hours=2)])
    acc = _make_account()
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8, now=now)
    assert decision.allowed


def test_daily_loss_soft_halves_risk():
    rm = RiskManager()
    # daily PnL at -1.2% of starting balance: above soft (1%) but below hard (2%).
    acc = Account(starting_balance=10_000, balance=9_880, peak_equity=10_000,
                  daily_pnl=-120)
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8,
                                 now=datetime.now(timezone.utc))
    assert decision.allowed
    assert decision.risk_per_trade < 0.01


def test_max_open_positions_blocks():
    rm = RiskManager()
    acc = _make_account()
    acc.open_positions = 3
    decision = rm.evaluate_entry(acc, stop_loss_pips=20, spread_pips=0.8,
                                 now=datetime.now(timezone.utc))
    assert not decision.allowed
    assert decision.reason == "max_open_positions"
