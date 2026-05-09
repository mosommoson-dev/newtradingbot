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
    half_life: pd.Series | None = None  # rolling half-life of mean reversion (days)


def rolling_half_life(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Ornstein-Uhlenbeck half-life of mean reversion.

    Fits ``Δs_t = κ · s_{t-1} + ε_t`` over a rolling window; the half-life is
    ``-ln(2) / κ``.  Returns ``NaN`` whenever the spread is trending (κ ≥ 0),
    which is exactly the regime we want the gate to block.
    """
    delta = spread.diff()
    lagged = spread.shift(1)
    cov = delta.rolling(window).cov(lagged)
    var = lagged.rolling(window).var()
    kappa = cov / var.replace(0, np.nan)
    # Only mean-reverting (kappa < 0) gets a positive, finite half-life
    hl = -np.log(2.0) / kappa.where(kappa < 0)
    return hl.rename("half_life")


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
    stop_loss_z : float
        Hard exit when the absolute spread z-score breaches this level.
        Defends against cointegration breaks where the spread keeps
        diverging instead of reverting.  Set to ``0`` (or any non-positive
        value) to disable the stop entirely.  Recommended: ~2x ``z_entry``.
    max_holding_days : int
        Force-exit after this many bars in a single position.  Cointegration
        relationships should mean-revert quickly; if a trade hasn't reverted
        within this horizon the relationship has likely broken.  Set to
        ``0`` to disable.
    max_half_life : float
        Only allow new entries when the rolling Ornstein-Uhlenbeck
        half-life of the spread is finite and ``<= max_half_life`` bars.
        This gates entries on whether the spread is currently mean-
        reverting (rather than trending).  Set to ``0`` to disable.
    half_life_window : int
        Rolling window used to estimate the OU half-life.
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
        stop_loss_z: float = 0.0,
        max_holding_days: int = 0,
        max_half_life: float = 0.0,
        half_life_window: int = 60,
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
        if stop_loss_z != 0 and stop_loss_z <= z_entry:
            raise ValueError(
                f"stop_loss_z ({stop_loss_z}) must be > z_entry ({z_entry}) "
                f"or 0 to disable")
        if max_holding_days < 0:
            raise ValueError(
                f"max_holding_days must be >= 0, got {max_holding_days}")
        if max_half_life < 0:
            raise ValueError(
                f"max_half_life must be >= 0, got {max_half_life}")
        if half_life_window < 10:
            raise ValueError(
                f"half_life_window must be >= 10, got {half_life_window}")
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)
        self.z_lookback = int(z_lookback)
        self.hedge_mode = hedge_mode
        self.hedge_window = int(hedge_window)
        self.kalman_obs_var = float(kalman_obs_var)
        self.kalman_trans_var = float(kalman_trans_var)
        self.stop_loss_z = float(stop_loss_z)
        self.max_holding_days = int(max_holding_days)
        self.max_half_life = float(max_half_life)
        self.half_life_window = int(half_life_window)

    # ------------------------------------------------------------------

    def fit_predict(self, price_a: pd.Series, price_b: pd.Series) -> PairsSignal:
        """Run the strategy and return the spread position over time.

        With all three Tier-1 guards (`stop_loss_z`, `max_holding_days`,
        `max_half_life`) disabled the position vector is identical to the
        classic vectorised z-score rule (entries on |z|>=z_entry,
        forward-filled until |z|<z_exit).  When any guard is on, the
        loop is iterative so we can track per-trade state.
        """
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

        use_stop = self.stop_loss_z > 0
        use_max_hold = self.max_holding_days > 0
        use_hl = self.max_half_life > 0

        # Half-life only computed when the gate is enabled, otherwise NaN.
        if use_hl:
            half_life = rolling_half_life(spread, window=self.half_life_window)
        else:
            half_life = pd.Series(np.nan, index=spread.index, name="half_life")

        if not (use_stop or use_max_hold or use_hl):
            # Vectorised classic rule (preserves the original baseline).
            pos = pd.Series(np.nan, index=spread.index, dtype=float)
            pos.loc[z <= -self.z_entry] = 1.0
            pos.loc[z >= self.z_entry] = -1.0
            pos = pos.ffill().fillna(0.0)
            pos.loc[z.abs() < self.z_exit] = 0.0
            pos = pos.fillna(0.0).astype(int)
            return PairsSignal(position=pos, beta=kf.beta, z=z, spread=spread,
                                half_life=half_life)

        # Iterative path with per-trade state for the guards.
        n = len(z)
        z_arr = z.to_numpy()
        hl_arr = half_life.to_numpy()
        pos_arr = np.zeros(n, dtype=np.int64)
        cur = 0           # current open position
        bars_in_pos = 0   # bars in the current open trade
        # When a guard fires, we want to stay flat for the rest of THIS bar
        # (no same-bar re-entry) — the next bar can re-enter normally.
        # So `stopped_this_bar` is consulted only inside the entry block.
        for i in range(n):
            zi = z_arr[i]
            if not np.isfinite(zi):
                pos_arr[i] = 0
                continue

            stopped_this_bar = False
            if cur != 0:
                bars_in_pos += 1
                if abs(zi) < self.z_exit:
                    cur = 0
                    bars_in_pos = 0
                elif use_stop and abs(zi) >= self.stop_loss_z:
                    cur = 0
                    bars_in_pos = 0
                    stopped_this_bar = True
                elif use_max_hold and bars_in_pos > self.max_holding_days:
                    cur = 0
                    bars_in_pos = 0
                    stopped_this_bar = True

            if cur == 0 and not stopped_this_bar:
                # Half-life gate filters entry; pure entry rule otherwise.
                hl_ok = True
                if use_hl:
                    hli = hl_arr[i]
                    hl_ok = bool(np.isfinite(hli)) and (hli <= self.max_half_life)
                if hl_ok:
                    if zi <= -self.z_entry:
                        cur = 1
                        bars_in_pos = 1
                    elif zi >= self.z_entry:
                        cur = -1
                        bars_in_pos = 1

            pos_arr[i] = cur

        pos = pd.Series(pos_arr, index=spread.index, dtype=int, name="position")
        return PairsSignal(position=pos, beta=kf.beta, z=z, spread=spread,
                            half_life=half_life)


__all__ = [
    "KalmanResult",
    "PairsSignal",
    "PairsTradingStrategy",
    "kalman_pairs",
    "rolling_half_life",
    "rolling_ols_pairs",
]
