"""Thin wrapper around ``oandapyV20``.

The wrapper centralises:
    * authentication / endpoint selection (practice vs live)
    * resilient HTTP error handling with exponential backoff
    * a small, typed surface area: ``account_summary``, ``current_price``,
      ``open_positions``, ``place_market_order``, ``close_position``,
      ``get_pricing_stream``.

Tests use the ``MockOandaClient`` defined at the bottom of the file so they
do not need real OANDA credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..config import get_settings
from ..config.brokers import OandaCredentials, load_oanda_credentials


@dataclass
class Quote:
    bid: float
    ask: float
    spread_pips: float
    timestamp: str


@dataclass
class PositionSnapshot:
    instrument: str
    units: int            # signed (positive = long)
    average_price: float
    unrealized_pnl: float


@dataclass
class OrderTicket:
    order_id: str
    instrument: str
    units: int
    price: float
    sl_price: float | None
    tp_price: float | None


class OandaClient:
    """Real OANDA REST client wrapper."""

    def __init__(self, credentials: OandaCredentials | None = None,
                 max_retries: int = 3, backoff: float = 1.5) -> None:
        self._creds = credentials or load_oanda_credentials()
        self._max_retries = max_retries
        self._backoff = backoff

        try:
            from oandapyV20 import API
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("oandapyV20 not installed") from e

        if not self._creds.is_configured:
            raise RuntimeError("OANDA credentials missing -- set OANDA_TOKEN and OANDA_ACCOUNT_ID")
        self._api = API(access_token=self._creds.token, environment=self._creds.environment)
        self._account_id = self._creds.account_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request(self, endpoint: Any) -> Any:
        for attempt in range(self._max_retries):
            try:
                self._api.request(endpoint)
                return endpoint.response
            except Exception as exc:
                wait = self._backoff ** attempt
                logger.warning("OANDA request failed ({}/{}): {} -- retrying in {:.1f}s",
                               attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError("OANDA request exhausted retries")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def account_summary(self) -> dict[str, Any]:
        from oandapyV20.endpoints.accounts import AccountSummary

        return self._request(AccountSummary(accountID=self._account_id))

    def current_price(self, instrument: str | None = None) -> Quote:
        from oandapyV20.endpoints.pricing import PricingInfo

        instrument = instrument or get_settings().instrument.pair
        resp = self._request(PricingInfo(accountID=self._account_id,
                                         params={"instruments": instrument}))
        prices = resp["prices"][0]
        bid = float(prices["bids"][0]["price"])
        ask = float(prices["asks"][0]["price"])
        pip = get_settings().instrument.pip
        return Quote(bid=bid, ask=ask, spread_pips=(ask - bid) / pip, timestamp=prices["time"])

    def open_positions(self) -> list[PositionSnapshot]:
        from oandapyV20.endpoints.positions import OpenPositions

        resp = self._request(OpenPositions(accountID=self._account_id))
        positions = []
        for p in resp.get("positions", []):
            long = p.get("long", {})
            short = p.get("short", {})
            units = int(long.get("units", 0)) + int(short.get("units", 0))
            avg_long = float(long.get("averagePrice") or 0.0) if int(long.get("units", 0)) > 0 else 0.0
            avg_short = float(short.get("averagePrice") or 0.0) if int(short.get("units", 0)) < 0 else 0.0
            avg = avg_long or avg_short
            positions.append(
                PositionSnapshot(
                    instrument=p["instrument"],
                    units=units,
                    average_price=avg,
                    unrealized_pnl=float(p.get("unrealizedPL", 0.0)),
                )
            )
        return positions

    def place_market_order(
        self,
        instrument: str,
        units: int,
        sl_price: float | None = None,
        tp_price: float | None = None,
        client_tag: str | None = None,
    ) -> OrderTicket:
        from oandapyV20.endpoints.orders import OrderCreate

        order: dict[str, Any] = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if sl_price is not None:
            order["order"]["stopLossOnFill"] = {"price": f"{sl_price:.5f}"}
        if tp_price is not None:
            order["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.5f}"}
        if client_tag:
            order["order"]["clientExtensions"] = {"tag": client_tag}

        resp = self._request(OrderCreate(accountID=self._account_id, data=order))
        fill = resp.get("orderFillTransaction") or {}
        return OrderTicket(
            order_id=str(fill.get("id") or fill.get("orderID") or ""),
            instrument=instrument,
            units=units,
            price=float(fill.get("price", 0.0)),
            sl_price=sl_price,
            tp_price=tp_price,
        )

    def close_position(self, instrument: str) -> dict[str, Any]:
        from oandapyV20.endpoints.positions import PositionClose

        return self._request(
            PositionClose(
                accountID=self._account_id,
                instrument=instrument,
                data={"longUnits": "ALL", "shortUnits": "ALL"},
            )
        )


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------

class MockOandaClient:
    """In-memory mock used in tests and dry runs."""

    def __init__(self, balance: float = 10_000.0, price: float = 1.10) -> None:
        self.balance = balance
        self.price = price
        self.positions: dict[str, PositionSnapshot] = {}
        self.orders: list[OrderTicket] = []
        self.closed = False

    def account_summary(self) -> dict[str, Any]:
        return {"account": {"balance": str(self.balance), "currency": "USD"}}

    def current_price(self, instrument: str | None = None) -> Quote:
        instrument = instrument or get_settings().instrument.pair
        bid = self.price - 0.00004
        ask = self.price + 0.00004
        pip = get_settings().instrument.pip
        return Quote(bid=bid, ask=ask, spread_pips=(ask - bid) / pip, timestamp="mock")

    def open_positions(self) -> list[PositionSnapshot]:
        return list(self.positions.values())

    def place_market_order(self, instrument: str, units: int,
                           sl_price: float | None = None,
                           tp_price: float | None = None,
                           client_tag: str | None = None) -> OrderTicket:
        ticket = OrderTicket(
            order_id=f"mock-{len(self.orders) + 1}",
            instrument=instrument,
            units=units,
            price=self.price,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        self.orders.append(ticket)
        prev = self.positions.get(instrument)
        prev_units = prev.units if prev else 0
        prev_avg = prev.average_price if prev else self.price
        new_units = prev_units + units
        if new_units == 0:
            self.positions.pop(instrument, None)
        else:
            new_avg = (prev_avg * prev_units + self.price * units) / new_units
            self.positions[instrument] = PositionSnapshot(
                instrument=instrument, units=new_units,
                average_price=new_avg, unrealized_pnl=0.0,
            )
        return ticket

    def close_position(self, instrument: str) -> dict[str, Any]:
        self.positions.pop(instrument, None)
        self.closed = True
        return {"status": "closed", "instrument": instrument}
