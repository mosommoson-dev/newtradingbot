"""Top-level entry point.

Usage::

    python -m eurusd_quant_bot.main --mode backtest --strategy mean_reversion \
        --start 2020-01-01 --end 2024-12-31

    python -m eurusd_quant_bot.main --mode walkforward --strategy trend_following \
        --trials 50

    python -m eurusd_quant_bot.main --mode paper   --strategy portfolio
    python -m eurusd_quant_bot.main --mode live    --strategy portfolio

The single ``--mode`` flag dispatches between research, paper, and live
modes; the live OANDA endpoint is selected via the ``OANDA_ENV`` env var
(``practice`` vs ``live``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from .backtest import run_backtest, run_monte_carlo, walk_forward
from .config import get_settings
from .config.settings import REPORTS_DIR
from .data import clean_ohlcv, fetch_yfinance
from .live import LiveTrader, TraderConfig
from .strategy import STRATEGY_REGISTRY, get_strategy


def _setup_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=settings.runtime.log_level)
    logger.add(
        settings.runtime.log_level
        and (Path(__file__).resolve().parents[1] / "logs" / "bot.log"),
        rotation="50 MB",
        retention=10,
        level=settings.runtime.log_level,
    )


def _load_data(start: str, end: str | None, granularity: str) -> pd.DataFrame:
    df = fetch_yfinance(granularity=granularity, start=start, end=end)
    return clean_ohlcv(df)


def _backtest(args: argparse.Namespace) -> int:
    df = _load_data(args.start, args.end, args.granularity)
    if df.empty:
        logger.error("Loaded 0 bars; aborting backtest")
        return 1
    strat = get_strategy(args.strategy)
    signals = strat.generate_signals(df)
    result = run_backtest(df, signals)
    out = REPORTS_DIR / f"backtest_{args.strategy}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({
        "strategy": args.strategy,
        "stats": result.stats.to_dict(),
        "n_bars": len(df),
        "n_trades": len(result.trades),
        "start": args.start,
        "end": args.end,
        "granularity": args.granularity,
    }, default=str, indent=2))
    logger.info("Backtest done. Stats: {}", result.stats.to_dict())
    print(json.dumps(result.stats.to_dict(), indent=2, default=str))
    return 0


def _walkforward(args: argparse.Namespace) -> int:
    df = _load_data(args.start, args.end, args.granularity)
    if df.empty:
        logger.error("Loaded 0 bars; aborting walk-forward")
        return 1
    strat_cls = STRATEGY_REGISTRY[args.strategy]
    if not hasattr(strat_cls, "search_space"):
        logger.error("Strategy '{}' has no search_space()", args.strategy)
        return 1
    res = walk_forward(df, strat_cls, n_trials=args.trials)
    out = REPORTS_DIR / f"walkforward_{args.strategy}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.csv"
    res.summary.to_csv(out, index=False)
    logger.info("Walk-forward median params: {}", res.median_params)
    print(json.dumps(res.median_params, indent=2, default=str))
    return 0


def _montecarlo(args: argparse.Namespace) -> int:
    df = _load_data(args.start, args.end, args.granularity)
    strat = get_strategy(args.strategy)
    signals = strat.generate_signals(df)
    bt = run_backtest(df, signals)
    if bt.trades.empty:
        logger.error("No trades produced; aborting Monte Carlo")
        return 1
    mc = run_monte_carlo(bt.trades, n_runs=args.runs)
    print(json.dumps(mc.summary(), indent=2, default=str))
    return 0


def _paper(args: argparse.Namespace) -> int:
    cfg = TraderConfig(strategy_name=args.strategy, use_mock=True)
    trader = LiveTrader(cfg)
    asyncio.run(trader.run())
    return 0


def _live(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.runtime.mode != "live":
        logger.warning(
            "Bot mode is '{}' but you are invoking --mode live. Set BOT_MODE=live in .env "
            "and confirm OANDA_ENV=live to actually trade with real money.",
            settings.runtime.mode,
        )
    from .execution.oanda_client import OandaClient
    cfg = TraderConfig(strategy_name=args.strategy, use_mock=False)
    trader = LiveTrader(cfg, broker=OandaClient())
    asyncio.run(trader.run())
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eurusd_quant_bot")
    p.add_argument("--mode", default="backtest",
                   choices=["backtest", "walkforward", "montecarlo", "paper", "live"])
    p.add_argument("--strategy", default="mean_reversion",
                   choices=sorted(STRATEGY_REGISTRY))
    p.add_argument("--start", default=get_settings().backtest.in_sample_start)
    p.add_argument("--end", default=None)
    p.add_argument("--granularity", default=get_settings().backtest.granularity)
    p.add_argument("--trials", type=int, default=get_settings().backtest.optuna_trials)
    p.add_argument("--runs", type=int, default=get_settings().backtest.monte_carlo_runs)
    return p


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _parser().parse_args(argv)
    if args.mode == "backtest":
        return _backtest(args)
    if args.mode == "walkforward":
        return _walkforward(args)
    if args.mode == "montecarlo":
        return _montecarlo(args)
    if args.mode == "paper":
        return _paper(args)
    if args.mode == "live":
        return _live(args)
    logger.error("Unknown mode: {}", args.mode)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
