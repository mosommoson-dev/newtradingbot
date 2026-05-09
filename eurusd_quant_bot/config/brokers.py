"""Broker-specific credentials and endpoint resolution.

Credentials are *only* read from environment variables -- never hard-coded.
This module is the single point of truth for which OANDA endpoint we hit
("practice" vs "live") and for the account ID and personal access token.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

OANDA_PRACTICE_REST: Final[str] = "https://api-fxpractice.oanda.com"
OANDA_LIVE_REST: Final[str] = "https://api-fxtrade.oanda.com"
OANDA_PRACTICE_STREAM: Final[str] = "https://stream-fxpractice.oanda.com"
OANDA_LIVE_STREAM: Final[str] = "https://stream-fxtrade.oanda.com"


@dataclass(frozen=True)
class OandaCredentials:
    environment: str   # "practice" | "live"
    token: str | None
    account_id: str | None

    @property
    def rest_url(self) -> str:
        return OANDA_LIVE_REST if self.environment == "live" else OANDA_PRACTICE_REST

    @property
    def stream_url(self) -> str:
        return OANDA_LIVE_STREAM if self.environment == "live" else OANDA_PRACTICE_STREAM

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.account_id)


def load_oanda_credentials() -> OandaCredentials:
    """Read OANDA creds from environment variables.

    Returns a credentials object; callers must check ``is_configured`` before
    making API calls.  The bot must never crash at import-time when credentials
    are missing -- only when the user explicitly tries to trade.
    """
    return OandaCredentials(
        environment=os.environ.get("OANDA_ENV", "practice"),
        token=os.environ.get("OANDA_TOKEN") or None,
        account_id=os.environ.get("OANDA_ACCOUNT_ID") or None,
    )
