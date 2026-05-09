"""Health-check + auto-restart helper.

Designed to be hit by:
    - the Docker ``HEALTHCHECK`` instruction, and
    - the Flask dashboard ``/health`` endpoint.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class HealthState:
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: str | None = None
    restart_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def heartbeat(self) -> None:
        with self.lock:
            self.last_heartbeat = datetime.now(timezone.utc)

    def record_error(self, message: str) -> None:
        with self.lock:
            self.last_error = message

    def record_restart(self) -> None:
        with self.lock:
            self.restart_count += 1

    def to_dict(self) -> dict[str, str | int]:
        with self.lock:
            return {
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "last_error": self.last_error or "",
                "restart_count": self.restart_count,
                "uptime_seconds": int(time.time() - self.started_at.timestamp()),
            }


def is_healthy(state: HealthState, *, max_silence_seconds: int = 120) -> bool:
    age = (datetime.now(timezone.utc) - state.last_heartbeat).total_seconds()
    return age < max_silence_seconds
