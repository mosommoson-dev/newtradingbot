"""Trade-schedule helpers (forex week + news + spread filters)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..config import get_settings


@dataclass
class ScheduleDecision:
    allowed: bool
    reason: str


def is_forex_open(now: datetime) -> bool:
    """Forex opens Sunday 22:00 UTC and closes Friday 21:00 UTC."""
    cfg = get_settings().schedule
    dow = now.weekday()
    hour = now.hour
    # Saturday closed
    if dow == 5:
        return False
    # Sunday before open
    if dow == cfg.week_open_dow and hour < cfg.week_open_hour:
        return False
    # Friday after close
    if dow == cfg.week_close_dow and hour >= cfg.week_close_hour:
        return False
    return True


def in_low_vol_window(now: datetime) -> bool:
    cfg = get_settings().schedule
    h = now.hour
    return any(start <= h < end for start, end in cfg.avoid_low_vol_windows)


def in_high_value_window(now: datetime) -> bool:
    cfg = get_settings().schedule
    h = now.hour
    return any(start <= h < end for start, end in cfg.high_value_windows)


def evaluate_schedule(
    now: datetime,
    spread_pips: float,
    *,
    in_news_blackout: bool = False,
    minutes_into_session: int | None = None,
) -> ScheduleDecision:
    cfg = get_settings().schedule
    risk_cfg = get_settings().risk
    if not is_forex_open(now):
        return ScheduleDecision(False, "forex_closed")
    if in_news_blackout:
        return ScheduleDecision(False, "news_blackout")
    if spread_pips > risk_cfg.max_spread_pips:
        return ScheduleDecision(False, f"spread_{spread_pips:.2f}_too_wide")
    if minutes_into_session is not None and minutes_into_session < cfg.skip_first_minutes_of_session:
        return ScheduleDecision(False, "session_open_blackout")
    return ScheduleDecision(True, "ok")


def today_news_times(events: pd.DataFrame, when: datetime | None = None) -> list[datetime]:
    """Convert a Forex Factory / FRED-style calendar frame into a list of UTC
    datetimes for today's events.

    The frame must contain a ``time`` column with timezone-aware timestamps.
    """
    when = when or datetime.now(timezone.utc)
    if events.empty or "time" not in events.columns:
        return []
    today = when.date()
    mask = pd.to_datetime(events["time"], utc=True).dt.date == today
    return list(pd.to_datetime(events.loc[mask, "time"], utc=True).dt.to_pydatetime())
