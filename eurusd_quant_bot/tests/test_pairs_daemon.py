"""Tests for the pairs paper-trading daemon state model."""

from __future__ import annotations

import json

from eurusd_quant_bot.live.pairs_daemon import PaperState, PaperTrade


def test_state_round_trips_through_json(tmp_path):
    state = PaperState(equity=10_500.0, starting_equity=10_000.0)
    state.open_trades["EUR/USD vs EUR/JPY"] = PaperTrade(
        pair="EUR/USD vs EUR/JPY",
        direction=1,
        entry_a=1.10,
        entry_b=160.0,
        entry_beta=140.0,
        entry_time="2026-01-01",
    )
    state.daily_pnl_history["2026-01-01"] = 12.5
    raw = json.dumps(state.to_dict())
    rebuilt = PaperState.from_dict(json.loads(raw))
    assert rebuilt.equity == 10_500.0
    assert "EUR/USD vs EUR/JPY" in rebuilt.open_trades
    ot = rebuilt.open_trades["EUR/USD vs EUR/JPY"]
    assert ot.direction == 1
    assert ot.entry_a == 1.10
    assert rebuilt.daily_pnl_history["2026-01-01"] == 12.5


def test_state_back_compat_with_single_open_trade():
    """Old-format state (single open_trade field) still loads."""
    raw = {
        "equity": 10_000.0,
        "starting_equity": 10_000.0,
        "open_trade": {
            "pair": "EUR/USD vs EUR/JPY",
            "direction": -1,
            "entry_a": 1.20,
            "entry_b": 170.0,
            "entry_beta": 141.0,
            "entry_time": "2026-01-02",
            "pnl_pct": 0.0,
        },
        "closed_trades": [],
    }
    state = PaperState.from_dict(raw)
    assert "EUR/USD vs EUR/JPY" in state.open_trades
    assert state.open_trades["EUR/USD vs EUR/JPY"].direction == -1


def test_state_handles_missing_fields():
    state = PaperState.from_dict({})
    assert state.equity == 10_000.0
    assert state.open_trades == {}
    assert state.closed_trades == []
