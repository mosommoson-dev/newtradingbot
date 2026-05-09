"""Async trading loop.

Consumes a strategy + risk manager + broker (real or mock) and runs an event
loop:

    while running:
        1. fetch the latest H1 bar (closed)
        2. update features / cached predictions if needed
        3. ask the strategy for a signal
        4. apply the schedule + risk filters
        5. submit / close orders
        6. emit Telegram + dashboard updates
        7. sleep until the next bar
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from ..config import get_settings
from ..data.fetcher import OandaHistoricalFetcher, fetch_yfinance
from ..execution.oanda_client import MockOandaClient
from ..execution.order_manager import OrderManager
from ..execution.risk_manager import Account, RiskManager, stats_from_trades
from ..ml import _indicators as ind
from ..strategy import get_strategy
from .health_check import HealthState
from .scheduler import evaluate_schedule, is_forex_open
from .telegram_notify import TelegramNotifier


@dataclass
class TraderState:
    running: bool = True
    paused: bool = False
    health: HealthState = field(default_factory=HealthState)


@dataclass
class TraderConfig:
    strategy_name: str = "mean_reversion"
    granularity: str = "H1"
    poll_seconds: int = 60
    use_mock: bool = True
    closed_trades: pd.DataFrame | None = None


class LiveTrader:
    """Self-contained async trader that runs against any broker client."""

    def __init__(self, config: TraderConfig | None = None,
                 broker: Any | None = None,
                 notifier: TelegramNotifier | None = None) -> None:
        self.cfg = config or TraderConfig()
        self.settings = get_settings()
        self.broker = broker or MockOandaClient(balance=self.settings.backtest.initial_balance)
        self.notifier = notifier or TelegramNotifier()
        self.risk = RiskManager()
        self.orders = OrderManager(self.broker)
        self.strategy = get_strategy(self.cfg.strategy_name)
        self.state = TraderState()
        self.account = Account(
            starting_balance=self.settings.backtest.initial_balance,
            balance=self.settings.backtest.initial_balance,
            peak_equity=self.settings.backtest.initial_balance,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("LiveTrader start: strategy={}", self.cfg.strategy_name)
        try:
            while self.state.running:
                self.state.health.heartbeat()
                try:
                    await self._tick()
                except Exception as exc:
                    self.state.health.record_error(str(exc))
                    logger.exception("Trader tick failed: {}", exc)
                await asyncio.sleep(self.cfg.poll_seconds)
        finally:
            logger.info("LiveTrader stopped")

    def pause(self) -> None:
        self.state.paused = True
        self.notifier.send("⏸️ Trader paused")

    def resume(self) -> None:
        self.state.paused = False
        self.notifier.send("▶️ Trader resumed")

    def stop(self) -> None:
        self.state.running = False

    def emergency_close(self) -> None:
        self.orders.emergency_close_all()
        self.notifier.send("🚨 Emergency close requested")

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        if self.state.paused:
            return
        now = datetime.now(timezone.utc)
        if not is_forex_open(now):
            return

        df = await asyncio.to_thread(self._fetch_recent)
        if df.empty:
            logger.warning("No fresh data this tick")
            return

        signals = self.strategy.generate_signals(df)
        signal = int(signals.iloc[-1])
        quote = self.broker.current_price()
        sched = evaluate_schedule(now, quote.spread_pips)
        if not sched.allowed:
            logger.debug("Schedule blocked: {}", sched.reason)
            return

        positions = self.broker.open_positions()
        held_units = sum(p.units for p in positions
                         if p.instrument == self.settings.instrument.pair)
        held = 1 if held_units > 0 else (-1 if held_units < 0 else 0)

        if signal == 0 and held != 0:
            self.broker.close_position(self.settings.instrument.pair)
            self.notifier.trade_close(direction=held, pnl_usd=0.0, pnl_pips=0.0)
            return
        if signal == held or signal == 0:
            return

        # Need to flip / open
        if held != 0:
            self.broker.close_position(self.settings.instrument.pair)

        atr = ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1]
        stop_pips = max(5.0, atr / self.settings.instrument.pip * 2)
        decision = self.risk.evaluate_entry(
            account=self.account,
            stop_loss_pips=stop_pips,
            spread_pips=quote.spread_pips,
            now=now,
        )
        if not decision.allowed:
            logger.info("Risk blocked: {}", decision.reason)
            return

        result = self.orders.submit(
            direction=signal, decision=decision,
            entry_price=quote.ask if signal > 0 else quote.bid,
            atr_in_price=atr,
            client_tag=self.cfg.strategy_name,
        )
        if result.submitted and result.ticket:
            t = result.ticket
            self.notifier.trade_open(
                direction=signal, lots=decision.lot_size,
                entry=t.price, sl=t.sl_price or 0.0, tp=t.tp_price or 0.0,
            )
            self.account.open_positions += 1
            self.account.open_risk += decision.risk_per_trade

        # Re-tune Kelly using historical trades, if we have them.
        if self.cfg.closed_trades is not None and len(self.cfg.closed_trades):
            stats = stats_from_trades(self.cfg.closed_trades, lookback=50)
            logger.debug("Trade stats: WR={:.2%} avg_win={:.4f} avg_loss={:.4f}",
                         stats.win_rate, stats.avg_win_pct, stats.avg_loss_pct)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_recent(self) -> pd.DataFrame:
        try:
            fetcher = OandaHistoricalFetcher(granularity=self.cfg.granularity)
            df = fetcher.fetch(start="2024-01-01")
            if df.empty:
                df = fetch_yfinance(granularity=self.cfg.granularity, start="2024-01-01")
            return df.tail(500)
        except Exception as exc:
            logger.warning("Live data fetch fell back to yfinance: {}", exc)
            return fetch_yfinance(granularity=self.cfg.granularity, start="2024-01-01").tail(500)
