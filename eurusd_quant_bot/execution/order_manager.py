"""Order manager: turns a Signal + RiskDecision into broker calls.

The order manager owns the (small) state machine that decides:
    * What kind of order to place (market / stop / limit) -- we default to
      market with bracket SL/TP for simplicity and predictability.
    * Where to place SL: ``ATR(14) * atr_stop_mult`` away from entry, in pips.
    * Where to place TP: 2x stop distance (R:R = 2:1).
    * Whether to update an existing trailing stop.

It is broker-agnostic: it talks to anything that quacks like
``OandaClient`` / ``MockOandaClient``.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from ..config import get_settings
from .oanda_client import OrderTicket
from .risk_manager import RiskDecision


@dataclass
class OrderResult:
    submitted: bool
    reason: str
    ticket: OrderTicket | None = None


class OrderManager:
    """Stateless dispatcher. Holds a reference to the broker client only."""

    def __init__(self, broker, *, take_profit_r_multiple: float = 2.0) -> None:
        self.broker = broker
        self.tp_r_multiple = take_profit_r_multiple

    def submit(
        self,
        *,
        direction: int,
        decision: RiskDecision,
        entry_price: float,
        atr_in_price: float,
        client_tag: str | None = None,
    ) -> OrderResult:
        if not decision.allowed:
            return OrderResult(False, decision.reason)

        instr = get_settings().instrument
        atr_stop_mult = get_settings().strategy.mean_reversion.atr_stop_mult
        stop_distance_pips = max(5.0, (atr_in_price * atr_stop_mult) / instr.pip)
        stop_distance_price = stop_distance_pips * instr.pip

        sl = entry_price - stop_distance_price * direction
        tp = entry_price + self.tp_r_multiple * stop_distance_price * direction

        units = round(decision.lot_size * instr.standard_lot_units * direction)
        if units == 0:
            return OrderResult(False, "zero_units")

        try:
            ticket = self.broker.place_market_order(
                instrument=instr.pair,
                units=units,
                sl_price=round(sl, 5),
                tp_price=round(tp, 5),
                client_tag=client_tag,
            )
        except Exception as exc:
            logger.exception("Order submission failed: {}", exc)
            return OrderResult(False, f"broker_error:{exc}")

        logger.info(
            "Submitted order {}: {} {} @ {:.5f} | SL {:.5f} | TP {:.5f}",
            ticket.order_id, "BUY" if direction > 0 else "SELL", units, ticket.price, sl, tp,
        )
        return OrderResult(True, "ok", ticket=ticket)

    def update_trailing_stop(
        self,
        *,
        instrument: str,
        position_units: int,
        current_price: float,
        new_stop: float,
    ) -> None:
        """Hook for ATR-trailing stops; full implementation requires
        OANDA's TrailingStopLoss order or a manual modify."""
        logger.debug("Trailing stop update requested for {} -> {:.5f}", instrument, new_stop)
        # Implementing OANDA trailing-stop modify is left as a follow-up; the
        # bracket SL/TP attached on entry is sufficient for the paper stage.

    def emergency_close_all(self) -> None:
        instr = get_settings().instrument.pair
        try:
            self.broker.close_position(instr)
            logger.warning("Emergency close issued for {}", instr)
        except Exception as exc:
            logger.exception("Emergency close failed: {}", exc)
