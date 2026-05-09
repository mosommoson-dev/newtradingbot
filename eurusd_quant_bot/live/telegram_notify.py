"""Telegram notification helper.

The notifier is fire-and-forget: any failure to deliver is logged but never
raises (we never want a missed alert to take down the trading loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from loguru import logger

from ..config import get_settings

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class TelegramNotifier:
    bot_token: str | None = None
    chat_id: str | None = None

    def __post_init__(self) -> None:
        cfg = get_settings().telegram
        self.bot_token = self.bot_token or cfg.bot_token
        self.chat_id = self.chat_id or cfg.chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            logger.debug("Telegram disabled; would have sent: {}", message[:60])
            return False
        try:
            url = TELEGRAM_API.format(token=self.bot_token)
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Convenience formatters
    # ------------------------------------------------------------------

    def trade_open(self, *, direction: int, lots: float, entry: float,
                   sl: float, tp: float) -> bool:
        emoji = "🟢" if direction > 0 else "🔴"
        side = "BUY" if direction > 0 else "SELL"
        return self.send(
            f"{emoji} {side} EUR/USD | {lots:.2f} lots | Entry: {entry:.5f} | "
            f"SL: {sl:.5f} | TP: {tp:.5f}"
        )

    def trade_close(self, *, direction: int, pnl_usd: float, pnl_pips: float) -> bool:
        emoji = "💰" if pnl_usd >= 0 else "💔"
        side = "BUY" if direction > 0 else "SELL"
        return self.send(
            f"{emoji} CLOSE {side} EUR/USD | PnL: ${pnl_usd:+,.2f} ({pnl_pips:+.1f} pips)"
        )

    def kill_switch(self, reason: str) -> bool:
        return self.send(f"🚨 KILL SWITCH: {reason}")

    def daily_summary(self, *, equity: float, daily_pnl: float, drawdown: float,
                      n_trades: int, n_wins: int, n_losses: int) -> bool:
        return self.send(
            f"📊 Daily summary | Equity: ${equity:,.2f} | "
            f"PnL: ${daily_pnl:+,.2f} | DD: {drawdown:+.2%} | "
            f"Trades: {n_trades} (W:{n_wins} / L:{n_losses})"
        )
