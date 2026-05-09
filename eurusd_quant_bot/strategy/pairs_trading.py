"""Pairs trading strategy with a Kalman filter hedge ratio.

The classic cointegration-based pairs trade:

    spread_t = a_t - (alpha_t + beta_t * b_t)

We treat ``[alpha_t, beta_t]`` as a hidden state evolving as a random walk
and recover it on-line with a 2-state Kalman filter (no future information
in the estimate at any time ``t``).  This adapts to slow regime drift
without the lag of a fixed-window rolling OLS.

Trading rule:
    * Long spread (``+1``)  when z-score of spread crosses below ``-z_entry``
    * Short spread (``-1``) when z-score of spread crosses above ``+z_entry``
    * Exit (``0``)         when ``|z| < z_exit``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series
    spread_var: pd.Series  # innovation variance (one-step-ahead)


def kalman_pairs(
    price_a: pd.Series,
    price_b: pd.Series,
    obs_var: float = 1e-3,
    trans_var: float = 1e-5,
) -> KalmanResult:
    """Run a 2-state Kalman filter on a price pair.

    State: ``x = [alpha, beta]`` evolving as a random walk
        x_{t} = x_{t-1} + w_t,    w ~ N(0, Q)  with Q = trans_var * I
    Observation:
        a_t = [1, b_t] @ x_t + v_t,   v ~ N(0, R)  with R = obs_var

    All estimates at time ``t`` use only data up to and including time ``t``;
    the function is strictly causal.
    """
    if len(price_a) != len(price_b):
        raise ValueError("price_a and price_b must have identical length")
    a = np.asarray(price_a.astype(float).values)
    b = np.asarray(price_b.astype(float).values)

    n = len(a)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    spread_var = np.full(n, np.nan)

    # Initial state and covariance
    x = np.array([0.0, 1.0])  # alpha=0, beta=1
    P = np.eye(2) * 10.0      # large prior variance
    Q = np.eye(2) * trans_var
    R = obs_var

    for t in range(n):
        # Predict
        # x_pred = x (random walk)
        P_pred = P + Q
        # Observe
        H = np.array([1.0, b[t]])  # (2,)
        y_hat = float(H @ x)
        innov = a[t] - y_hat
        S = float(H @ P_pred @ H.T + R)  # innovation variance, scalar
        K = (P_pred @ H) / S            # Kalman gain (2,)
        # Update
        x = x + K * innov
        P = P_pred - np.outer(K, H) @ P_pred

        alpha[t] = x[0]
        beta[t] = x[1]
        spread[t] = innov          # one-step-ahead residual = a - (alpha+beta*b)
        spread_var[t] = S

    idx = price_a.index
    return KalmanResult(
        alpha=pd.Series(alpha, index=idx, name="alpha"),
        beta=pd.Series(beta, index=idx, name="beta"),
        spread=pd.Series(spread, index=idx, name="spread"),
        spread_var=pd.Series(spread_var, index=idx, name="spread_var"),
    )


@dataclass
class PairsSignal:
    position: pd.Series   # -1 / 0 / +1 on the spread (long spread = long A, short beta*B)
    beta: pd.Series       # hedge ratio for sizing leg B at each bar
    z: pd.Series          # z-score of the spread (rolling)
    spread: pd.Series


def rolling_ols_pairs(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 252,
) -> KalmanResult:
    """Compute a strictly-causal rolling-OLS hedge ratio and spread.

    Returns the same shape ``KalmanResult`` so callers can switch hedge
    modes without changing downstream code.  ``spread_var`` is the rolling
    variance of the residual.
    """
    if len(price_a) != len(price_b):
        raise ValueError("price_a and price_b must have identical length")
    a = price_a.astype(float)
    b = price_b.astype(float)
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = (cov / var).rename("beta")
    alpha = (a.rolling(window).mean() - beta * b.rolling(window).mean()).rename("alpha")
    spread = (a - alpha - beta * b).rename("spread")
    spread_var = spread.rolling(window).var().rename("spread_var")
    return KalmanResult(alpha=alpha, beta=beta, spread=spread, spread_var=spread_var)


class PairsTradingStrategy:
    """Cointegration-based pairs trading with selectable hedge ratio mode.

    Parameters
    ----------
    z_entry : float
        |z| threshold required to open a position.
    z_exit : float
        |z| threshold below which an open position is closed (must be < z_entry).
    z_lookback : int
        Window for computing the rolling mean / std of the spread used to form
        the trading z-score.
    hedge_mode : {"kalman", "rolling_ols"}
        ``kalman``: 2-state Kalman filter on (alpha, beta), adaptive but
        sensitive to the obs/trans variance ratio.
        ``rolling_ols``: classical fixed-window OLS, much simpler and tends
        to preserve the cointegration mean-reversion signal better.
    hedge_window : int
        Window length for ``rolling_ols`` mode.
    kalman_obs_var, kalman_trans_var : float
        Kalman filter measurement and transition variances.
    """

    name: str = "pairs_trading"

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_lookback: int = 60,
        hedge_mode: str = "rolling_ols",
        hedge_window: int = 252,
        kalman_obs_var: float = 1e-3,
        kalman_trans_var: float = 1e-5,
    ) -> None:
        if z_exit >= z_entry:
            raise ValueError(f"z_exit ({z_exit}) must be < z_entry ({z_entry})")
        if z_lookback < 5:
            raise ValueError(f"z_lookback must be >= 5, got {z_lookback}")
        if hedge_mode not in ("kalman", "rolling_ols"):
            raise ValueError(f"hedge_mode must be kalman|rolling_ols, got {hedge_mode!r}")
        if hedge_window < 20:
            raise ValueError(f"hedge_window must be >= 20, got {hedge_window}")
        if kalman_obs_var <= 0 or kalman_trans_var <= 0:
            raise ValueError("Kalman variances must be positive")
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)
        self.z_lookback = int(z_lookback)
        self.hedge_mode = hedge_mode
        self.hedge_window = int(hedge_window)
        self.kalman_obs_var = float(kalman_obs_var)
        self.kalman_trans_var = float(kalman_trans_var)

    # ------------------------------------------------------------------

    def fit_predict(self, price_a: pd.Series, price_b: pd.Series) -> PairsSignal:
        """Run the strategy and return the spread position over time."""
        if self.hedge_mode == "kalman":
            kf = kalman_pairs(
                price_a, price_b,
                obs_var=self.kalman_obs_var,
                trans_var=self.kalman_trans_var,
            )
        else:
            kf = rolling_ols_pairs(price_a, price_b, window=self.hedge_window)
        spread = kf.spread
        # z-score using rolling stats (causal)
        roll_mean = spread.rolling(self.z_lookback).mean()
        roll_std = spread.rolling(self.z_lookback).std()
        z = (spread - roll_mean) / roll_std

        pos = pd.Series(np.nan, index=spread.index, dtype=float)
        # Vectorised entry rule:
        pos.loc[z <= -self.z_entry] = 1.0   # spread is too low -> long it
        pos.loc[z >= self.z_entry] = -1.0   # spread is too high -> short it
        # Hold until z reverts inside [-z_exit, +z_exit], then exit.
        pos = pos.ffill().fillna(0.0)
        flat_mask = z.abs() < self.z_exit
        pos.loc[flat_mask] = 0.0
        # Re-forward-fill the remaining NaNs (warmup) with 0
        pos = pos.fillna(0.0).astype(int)

        return PairsSignal(position=pos, beta=kf.beta, z=z, spread=spread)


__all__ = [
    "KalmanResult",
    "PairsSignal",
    "PairsTradingStrategy",
    "kalman_pairs",
    "rolling_ols_pairs",
]
