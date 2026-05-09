"""Trading strategies."""

from __future__ import annotations

from .base_strategy import BaseStrategy, Signal
from .mean_reversion import MeanReversionStrategy
from .ml_strategy import MLEnsembleStrategy
from .pairs_trading import (
    KalmanResult,
    PairsSignal,
    PairsTradingStrategy,
    kalman_pairs,
    rolling_half_life,
)
from .portfolio import AllocatorOutput, combine
from .trend_following import TrendFollowingStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "trend_following": TrendFollowingStrategy,
    "ml_ensemble": MLEnsembleStrategy,
}


def get_strategy(name: str, **params: object) -> BaseStrategy:
    """Construct a strategy by registry name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**params)


__all__ = [
    "STRATEGY_REGISTRY",
    "AllocatorOutput",
    "BaseStrategy",
    "KalmanResult",
    "MLEnsembleStrategy",
    "MeanReversionStrategy",
    "PairsSignal",
    "PairsTradingStrategy",
    "Signal",
    "TrendFollowingStrategy",
    "combine",
    "get_strategy",
    "kalman_pairs",
    "rolling_half_life",
]
