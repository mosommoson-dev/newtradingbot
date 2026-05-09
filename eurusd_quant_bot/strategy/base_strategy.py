"""Abstract base strategy.

Concrete strategies must implement :meth:`generate_signals`, which returns a
``Series`` of ``int``s aligned with the input index using the convention:

    * ``+1`` -> long entry (or maintain long)
    * ``-1`` -> short entry (or maintain short)
    * ``0``  -> flat / exit

The :class:`Signal` dataclass carries optional metadata (target SL/TP, model
confidence) used by the live engine when sizing and routing orders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Signal:
    """A single trading signal at a specific bar."""

    ts: pd.Timestamp
    direction: int                       # -1, 0, +1
    confidence: float = 1.0
    sl: float | None = None              # absolute SL price (optional)
    tp: float | None = None              # absolute TP price (optional)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or 1, got {self.direction!r}")


class BaseStrategy(ABC):
    """Abstract base class.

    Subclasses are pure: ``generate_signals`` should be deterministic given
    the same data and parameters.  Stateful behaviour (open positions, sizing)
    lives in the execution layer, not in the strategy itself.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a ``-1/0/+1`` integer Series aligned with ``data.index``."""

    def required_columns(self) -> set[str]:
        """Columns the strategy must see; checked before calling.

        Subclasses override; default expects raw OHLCV.
        """
        return {"open", "high", "low", "close"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_columns(self, data: pd.DataFrame) -> None:
        missing = self.required_columns() - set(data.columns)
        if missing:
            raise ValueError(f"{self.name}: data missing columns: {sorted(missing)}")

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{self.__class__.__name__}({self.params})"
