"""Walk-forward optimization with Optuna.

The walk-forward loop slices the dataset into rolling (train, test) windows.
On each window we use Optuna to maximise out-of-sample Sharpe (with drawdown
and profit-factor floors), then record the OOS performance.  The final
"deployment" parameters are the median of the per-fold best parameters,
which prevents picking a single overfit configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from ..config import get_settings
from ..strategy.base_strategy import BaseStrategy
from .engine import BacktestConfig, run_backtest
from .metrics import PerformanceMetrics


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, float | int]
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    median_params: dict[str, float | int]
    summary: pd.DataFrame


def _generate_folds(idx: pd.DatetimeIndex,
                    train_months: int,
                    test_months: int,
                    step_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp,
                                                   pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = idx[0].normalize()
    end = idx[-1].normalize()
    cur = start
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append((train_start, train_end, test_start, test_end))
        cur = cur + pd.DateOffset(months=step_months)
    return folds


def _suggest_param(trial, name: str, lo: float | int, hi: float | int, kind: str):
    if kind == "int":
        return trial.suggest_int(name, int(lo), int(hi))
    return trial.suggest_float(name, float(lo), float(hi))


def walk_forward(
    data: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    *,
    n_trials: int | None = None,
    train_months: int | None = None,
    test_months: int | None = None,
    step_months: int | None = None,
    score_fn: Callable[[PerformanceMetrics], float] | None = None,
) -> WalkForwardResult:
    """Run walk-forward optimization.

    Parameters
    ----------
    data : DataFrame
        OHLCV frame.
    strategy_cls : type
        Strategy class with a ``search_space`` static method.
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("optuna is required for walk-forward optimization") from e

    settings = get_settings()
    n_trials = n_trials or settings.backtest.optuna_trials
    train_months = train_months or settings.backtest.walk_forward_train_months
    test_months = test_months or settings.backtest.walk_forward_test_months
    step_months = step_months or settings.backtest.walk_forward_step_months
    score_fn = score_fn or _default_score

    space = strategy_cls.search_space()
    folds_idx = _generate_folds(
        pd.DatetimeIndex(data.index), train_months, test_months, step_months
    )
    folds: list[WalkForwardFold] = []

    for ts0, ts1, ts2, ts3 in folds_idx:
        train = data[(data.index >= ts0) & (data.index < ts1)]
        test = data[(data.index >= ts2) & (data.index < ts3)]
        if len(train) < 200 or len(test) < 50:
            continue

        def objective(trial: optuna.Trial,
                      _train: pd.DataFrame = train) -> float:
            params = {n: _suggest_param(trial, n, lo, hi, kind)
                      for n, (lo, hi, kind) in space.items()}
            strat = strategy_cls(**params)
            sigs = strat.generate_signals(_train)
            res = run_backtest(_train, sigs, config=BacktestConfig())
            return score_fn(res.stats)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params

        # Re-evaluate on both train and test with best params.
        strat = strategy_cls(**best)
        sigs_train = strat.generate_signals(train)
        sigs_test = strat.generate_signals(test)
        in_sample = run_backtest(train, sigs_train).stats
        out_of_sample = run_backtest(test, sigs_test).stats

        folds.append(
            WalkForwardFold(
                train_start=ts0, train_end=ts1, test_start=ts2, test_end=ts3,
                best_params=best, in_sample=in_sample, out_of_sample=out_of_sample,
            )
        )
        logger.info(
            "Fold {} -> {}: IS Sharpe={:.2f}, OOS Sharpe={:.2f}, OOS DD={:.2%}",
            ts2.date(), ts3.date(),
            in_sample.sharpe_ratio, out_of_sample.sharpe_ratio,
            out_of_sample.max_drawdown,
        )

    if not folds:
        raise RuntimeError("Walk-forward produced 0 valid folds")

    df = pd.DataFrame(
        [
            {
                "test_start": f.test_start,
                "test_end": f.test_end,
                "is_sharpe": f.in_sample.sharpe_ratio,
                "oos_sharpe": f.out_of_sample.sharpe_ratio,
                "oos_max_dd": f.out_of_sample.max_drawdown,
                "oos_pf": f.out_of_sample.profit_factor,
                **{f"p_{k}": v for k, v in f.best_params.items()},
            }
            for f in folds
        ]
    )
    median_params: dict[str, float | int] = {}
    for k in space:
        col = f"p_{k}"
        if col in df.columns:
            val = float(np.nanmedian(df[col]))
            median_params[k] = round(val) if space[k][2] == "int" else val
    return WalkForwardResult(folds=folds, median_params=median_params, summary=df)


def _default_score(m: PerformanceMetrics) -> float:
    settings = get_settings().backtest
    if (
        m.max_drawdown < -settings.accept_max_drawdown
        or m.profit_factor < settings.accept_min_profit_factor
        or m.sharpe_ratio < settings.accept_min_sharpe
    ):
        return -10.0 + m.sharpe_ratio
    return m.sharpe_ratio
