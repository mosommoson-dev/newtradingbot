"""Slippage model.

A realistic slippage estimate for EUR/USD on OANDA, parameterised on:
    * Base slippage (~0.3 pips on quiet markets).
    * 10% of ATR -- accounts for volatility shocks.
    * A small notional-volume premium.
    * A 1-pip penalty during news windows.

The function is intentionally pure (no IO) so it is trivially unit-tested.
"""

from __future__ import annotations

from ..config import get_settings


def calculate_slippage(
    atr_in_price: float,
    notional_usd: float,
    is_news: bool = False,
) -> float:
    """Return slippage in pips for a single fill.

    Parameters
    ----------
    atr_in_price : float
        ATR(14) in *price* (not pips).  Pass ``ATR / pip`` if you have it
        in pips already.
    notional_usd : float
        Trade notional in USD (used to add a small premium for large fills).
    is_news : bool
        If True, add a 1-pip penalty for trades placed near scheduled news.
    """
    settings = get_settings().instrument
    base = settings.base_slippage_pips
    pip = settings.pip
    atr_pips = max(0.0, atr_in_price / pip)
    atr_component = 0.1 * atr_pips
    if notional_usd > 10_000_000:
        volume_component = 0.05
    else:
        volume_component = 0.02
    news_penalty = 1.0 if is_news else 0.0
    total = base + atr_component + volume_component + news_penalty
    # Cap and round so we don't return absurd values for crazy ATR spikes.
    total = min(total, 5.0)
    return round(float(total), 2)
