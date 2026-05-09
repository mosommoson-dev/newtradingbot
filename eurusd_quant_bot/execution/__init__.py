"""Execution layer: broker, orders, risk, slippage."""

from __future__ import annotations

from .oanda_client import MockOandaClient, OandaClient, OrderTicket, PositionSnapshot, Quote
from .order_manager import OrderManager, OrderResult
from .risk_manager import (
    Account,
    RiskDecision,
    RiskManager,
    TradeStats,
    calculate_position_size,
    kelly_fraction,
    stats_from_trades,
)
from .slippage_model import calculate_slippage

__all__ = [
    "Account",
    "MockOandaClient",
    "OandaClient",
    "OrderManager",
    "OrderResult",
    "OrderTicket",
    "PositionSnapshot",
    "Quote",
    "RiskDecision",
    "RiskManager",
    "TradeStats",
    "calculate_position_size",
    "calculate_slippage",
    "kelly_fraction",
    "stats_from_trades",
]
