"""Tests for the execution layer."""

from __future__ import annotations

import pytest

from eurusd_quant_bot.execution import (
    MockOandaClient,
    OrderManager,
    RiskDecision,
    calculate_position_size,
    calculate_slippage,
    kelly_fraction,
)
from eurusd_quant_bot.execution.risk_manager import TradeStats


def test_calculate_position_size_basic():
    lots = calculate_position_size(account_balance=10_000, risk_per_trade=0.01,
                                   stop_loss_pips=50)
    # risk = $100, $1 per pip per micro lot ⇒ $10 per pip per mini lot
    # $100 / (50 pips * $10/pip/lot) = 0.2 lots
    assert pytest.approx(lots, rel=0.05) == 0.20


def test_calculate_position_size_returns_zero_for_invalid_inputs():
    assert calculate_position_size(0, 0.01, 50) == 0
    assert calculate_position_size(10_000, 0.01, 0) == 0


def test_calculate_position_size_respects_min_lot():
    lots = calculate_position_size(100, 0.01, 50)  # tiny account
    assert lots >= 0.01


def test_kelly_fraction_zero_for_negative_edge():
    stats = TradeStats(win_rate=0.4, avg_win_pct=0.005, avg_loss_pct=-0.01)
    # Negative edge -> Kelly should be 0.
    assert kelly_fraction(stats) == 0.0


def test_kelly_fraction_positive_for_positive_edge():
    stats = TradeStats(win_rate=0.55, avg_win_pct=0.01, avg_loss_pct=-0.005)
    # Half-Kelly should be > 0 but <= 1
    f = kelly_fraction(stats)
    assert 0 < f <= 1.0


def test_slippage_realistic_range():
    s = calculate_slippage(atr_in_price=0.0010, notional_usd=15_000_000, is_news=False)
    assert 0.2 <= s <= 1.5


def test_slippage_news_penalty():
    s_quiet = calculate_slippage(atr_in_price=0.0010, notional_usd=15_000_000, is_news=False)
    s_news = calculate_slippage(atr_in_price=0.0010, notional_usd=15_000_000, is_news=True)
    assert s_news > s_quiet


def test_mock_oanda_round_trip():
    client = MockOandaClient(balance=10_000, price=1.10)
    om = OrderManager(client)
    decision = RiskDecision(allowed=True, reason="ok", lot_size=0.05, risk_per_trade=0.01)
    result = om.submit(direction=1, decision=decision, entry_price=1.10,
                       atr_in_price=0.0010, client_tag="test")
    assert result.submitted
    assert result.ticket is not None
    pos = client.open_positions()
    assert len(pos) == 1
    assert pos[0].units == int(0.05 * 100_000)
    om.emergency_close_all()
    assert client.closed


def test_order_manager_rejects_disallowed_decision():
    client = MockOandaClient()
    om = OrderManager(client)
    decision = RiskDecision(allowed=False, reason="kill_switch_tripped")
    result = om.submit(direction=1, decision=decision, entry_price=1.10,
                       atr_in_price=0.0010)
    assert not result.submitted
    assert result.reason == "kill_switch_tripped"
