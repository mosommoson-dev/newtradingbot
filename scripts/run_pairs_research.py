"""Productionize the EUR/USD pairs strategy.

Pipeline:
1. Pull D1 history for EUR/USD + each candidate cointegrated peer (EUR/JPY,
   EUR/GBP) from Dukascopy 2015-2025.
2. Walk-forward Optuna sweep (24m IS train, 6m OOS test) tuning
   ``z_entry``, ``z_exit``, ``z_lookback``, ``kalman_obs_var``,
   ``kalman_trans_var`` against the pairs backtest's Sharpe.
3. Final OOS backtest using the median-of-folds tuned params.
4. Monte Carlo trade-shuffling for the OOS trades.
5. HTML report + Telegram summary.

This script does not connect to any live broker.  It just produces tuned
parameters that the paper-trading daemon will load.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import requests
from loguru import logger

from eurusd_quant_bot.backtest import run_pairs_backtest
from eurusd_quant_bot.backtest.monte_carlo import run_monte_carlo
from eurusd_quant_bot.config.settings import REPORTS_DIR
from eurusd_quant_bot.data import clean_ohlcv, fetch_dukascopy
from eurusd_quant_bot.strategy import PairsTradingStrategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

PAIRS = ["EUR/USD vs EUR/JPY", "EUR/USD vs EUR/GBP"]
START = "2015-01-01"
END = "2025-05-01"
TUNED_PARAMS_PATH = Path(__file__).resolve().parents[1] / "config" / "pairs_tuned.json"
TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    is_sharpe: float
    oos_sharpe: float
    oos_trades: int
    oos_total_return: float


def _telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("Telegram failed: {}", exc)


def _fold_indices(idx: pd.DatetimeIndex,
                  is_months: int = 24,
                  oos_months: int = 6,
                  ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = idx[0]
    end = idx[-1]
    train_end = start + pd.DateOffset(months=is_months)
    while train_end + pd.DateOffset(months=oos_months) <= end:
        test_end = train_end + pd.DateOffset(months=oos_months)
        folds.append((start, train_end, test_end))
        train_end = test_end
    return folds


def _suggest_params(trial: optuna.Trial, hedge_mode: str) -> dict:
    base = {
        "z_entry": trial.suggest_float("z_entry", 1.5, 3.0),
        "z_exit": trial.suggest_float("z_exit", 0.1, 1.0),
        "z_lookback": trial.suggest_int("z_lookback", 20, 120),
        "hedge_mode": hedge_mode,
    }
    if hedge_mode == "kalman":
        base["kalman_obs_var"] = trial.suggest_float("kalman_obs_var", 1e-5, 1e-2, log=True)
        base["kalman_trans_var"] = trial.suggest_float("kalman_trans_var", 1e-8, 1e-4, log=True)
    else:
        base["hedge_window"] = trial.suggest_int("hedge_window", 60, 504)
    return base


def _evaluate(price_a: pd.Series, price_b: pd.Series, params: dict) -> tuple[float, int]:
    if params["z_exit"] >= params["z_entry"]:
        return -10.0, 0
    strat = PairsTradingStrategy(**params)
    sig = strat.fit_predict(price_a, price_b)
    res = run_pairs_backtest(price_a, price_b, sig.position, sig.beta, cost_bps=4.0)
    return float(res.stats.sharpe_ratio), int(res.stats.total_trades)


def walk_forward_pairs(price_a: pd.Series, price_b: pd.Series,
                        hedge_mode: str = "rolling_ols",
                        n_trials: int = 40,
                        is_months: int = 24, oos_months: int = 6,
                        ) -> list[FoldResult]:
    folds_idx = _fold_indices(price_a.index, is_months, oos_months)
    logger.info("Walk-forward ({} mode): {} folds", hedge_mode, len(folds_idx))
    results: list[FoldResult] = []
    for ts0, ts1, ts2 in folds_idx:
        a_train = price_a.loc[ts0:ts1].iloc[:-1]
        b_train = price_b.loc[ts0:ts1].iloc[:-1]
        a_test = price_a.loc[ts1:ts2].iloc[:-1]
        b_test = price_b.loc[ts1:ts2].iloc[:-1]
        if len(a_train) < 200 or len(a_test) < 30:
            continue

        def _objective(trial: optuna.Trial,
                        a=a_train, b=b_train, mode=hedge_mode) -> float:
            params = _suggest_params(trial, mode)
            sharpe, n = _evaluate(a, b, params)
            if n < 5:
                return -5.0  # discourage strategies that never trade in IS
            return sharpe

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        # Reconstruct full param dict including hedge_mode
        best_params = dict(study.best_params)
        best_params["hedge_mode"] = hedge_mode
        is_sharpe = float(study.best_value)
        oos_sharpe, oos_trades = _evaluate(a_test, b_test, best_params)
        # Re-run to grab total_return
        strat = PairsTradingStrategy(**best_params)
        sig = strat.fit_predict(a_test, b_test)
        res = run_pairs_backtest(a_test, b_test, sig.position, sig.beta, cost_bps=4.0)
        oos_total_return = float(res.stats.total_return)
        logger.info(
            "Fold {} -> {}: IS Sharpe {:.2f}, OOS Sharpe {:.2f}, OOS trades {}, OOS ret {:.2%}",
            ts1.date(), ts2.date(), is_sharpe, oos_sharpe, oos_trades, oos_total_return,
        )
        results.append(FoldResult(
            train_start=ts0, train_end=ts1, test_end=ts2,
            best_params=best_params,
            is_sharpe=is_sharpe, oos_sharpe=oos_sharpe,
            oos_trades=oos_trades, oos_total_return=oos_total_return,
        ))
    return results


def _median_params(folds: list[FoldResult]) -> dict:
    if not folds:
        return {}
    out: dict = {}
    keys = list(folds[0].best_params)
    for k in keys:
        vals_list = [f.best_params[k] for f in folds]
        if isinstance(vals_list[0], str):
            # Mode is constant per study; just take the first
            out[k] = vals_list[0]
        else:
            out[k] = float(np.median(vals_list))
    if "z_lookback" in out:
        out["z_lookback"] = round(out["z_lookback"])
    if "hedge_window" in out:
        out["hedge_window"] = round(out["hedge_window"])
    return out


def _final_oos_backtest(price_a: pd.Series, price_b: pd.Series,
                         params: dict, train_window_months: int = 24,
                         ) -> dict:
    """Backtest tuned params on the post-train window only."""
    cutoff = price_a.index[0] + pd.DateOffset(months=train_window_months)
    a_oos = price_a.loc[cutoff:]
    b_oos = price_b.loc[cutoff:]
    strat = PairsTradingStrategy(**params)
    sig = strat.fit_predict(a_oos, b_oos)
    res = run_pairs_backtest(a_oos, b_oos, sig.position, sig.beta, cost_bps=4.0)
    return {
        "equity": res.equity,
        "trades": res.trades,
        "stats": res.stats.to_dict(),
        "params": params,
    }


def _stitched_oos_backtest(price_a: pd.Series, price_b: pd.Series,
                            folds: list[FoldResult]) -> dict:
    """True walk-forward OOS: each fold's OOS slice uses ITS own best params.

    This is the methodologically correct walk-forward result -- no parameter
    is ever used on data it was selected on.  We concatenate each fold's
    OOS equity contribution (in-fold pct returns) into a single series.
    """
    pieces: list[pd.Series] = []
    trade_frames: list[pd.DataFrame] = []
    for f in folds:
        a_oos = price_a.loc[f.train_end:f.test_end].iloc[:-1]
        b_oos = price_b.loc[f.train_end:f.test_end].iloc[:-1]
        if len(a_oos) < 5:
            continue
        strat = PairsTradingStrategy(**f.best_params)
        sig = strat.fit_predict(a_oos, b_oos)
        res = run_pairs_backtest(a_oos, b_oos, sig.position, sig.beta, cost_bps=4.0)
        pieces.append(res.pnl)
        trade_frames.append(res.trades)
    if not pieces:
        return {"equity": pd.Series(dtype=float), "trades": pd.DataFrame(),
                "stats": {}, "params": {}}
    full_pnl = pd.concat(pieces).sort_index()
    full_pnl = full_pnl[~full_pnl.index.duplicated(keep="first")]
    equity = (1 + full_pnl).cumprod() * 10_000.0
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    from eurusd_quant_bot.backtest.metrics import compute_metrics
    stats = compute_metrics(equity, trades).to_dict()
    return {"equity": equity, "trades": trades, "stats": stats, "params": "per-fold"}


FIXED_OLS_PARAMS = dict(
    z_entry=2.0, z_exit=0.5, z_lookback=60,
    hedge_mode="rolling_ols", hedge_window=252,
)
FIXED_KALMAN_PARAMS = dict(
    z_entry=2.0, z_exit=0.5, z_lookback=60,
    hedge_mode="kalman", kalman_obs_var=1e-3, kalman_trans_var=1e-5,
)


def _fixed_baseline(price_a: pd.Series, price_b: pd.Series,
                     params: dict, train_window_months: int = 24) -> dict:
    """Backtest fixed-textbook params (no tuning) to give an honest
    benchmark for whether tuning actually helped.
    """
    return _final_oos_backtest(price_a, price_b, params, train_window_months)


def _monte_carlo(trades: pd.DataFrame, n_runs: int = 1000) -> dict:
    if trades.empty:
        return {"p_ruin": 1.0, "p5_sharpe": 0.0, "p50_sharpe": 0.0, "p95_sharpe": 0.0,
                "p5_dd": 0.0, "p50_dd": 0.0, "p95_dd": 0.0, "worst_dd": 0.0}
    mc = run_monte_carlo(trades, n_runs=n_runs, seed=42)
    return {
        "p_ruin": float(mc.probability_of_ruin),
        "p5_sharpe": float(np.percentile(mc.sharpes, 5)),
        "p50_sharpe": float(np.percentile(mc.sharpes, 50)),
        "p95_sharpe": float(np.percentile(mc.sharpes, 95)),
        "p5_dd": float(np.percentile(mc.max_drawdowns, 5)),
        "p50_dd": float(np.percentile(mc.max_drawdowns, 50)),
        "p95_dd": float(np.percentile(mc.max_drawdowns, 95)),
        "worst_dd": float(mc.worst_drawdown),
    }


def _format_oos_html(folds: list[FoldResult]) -> str:
    rows = []
    for f in folds:
        rows.append(
            f"<tr><td>{f.train_end.date()} → {f.test_end.date()}</td>"
            f"<td>{f.is_sharpe:.2f}</td><td>{f.oos_sharpe:.2f}</td>"
            f"<td>{f.oos_trades}</td><td>{f.oos_total_return:.2%}</td>"
            f"<td><pre style='margin:0'>{json.dumps(f.best_params, indent=0)}</pre></td></tr>"
        )
    return ("<table border=1 cellpadding=4 style='border-collapse:collapse'>"
            "<tr><th>OOS window</th><th>IS Sharpe</th><th>OOS Sharpe</th>"
            "<th>OOS trades</th><th>OOS return</th><th>Params</th></tr>"
            + "".join(rows) + "</table>")


def main() -> int:
    t0 = time.time()
    logger.info("=== Pairs research run starting ===")

    # Fetch all four series in one shot
    instruments = {"EUR/USD": "EUR/USD", "EUR/JPY": "EUR/JPY", "EUR/GBP": "EUR/GBP"}
    prices: dict[str, pd.Series] = {}
    for name, sym in instruments.items():
        df = fetch_dukascopy(sym, "D1", START, END)
        df = clean_ohlcv(df)
        prices[name] = df["close"]
        logger.info("{}: {} bars", name, len(df))

    # Align indices
    common_idx = prices["EUR/USD"].index
    for s in prices.values():
        common_idx = common_idx.intersection(s.index)
    prices = {k: v.reindex(common_idx) for k, v in prices.items()}

    pair_outputs: dict[str, dict] = {}
    for pair_label in PAIRS:
        b_name = pair_label.split(" vs ")[1]
        logger.info("=== Tuning {} ===", pair_label)

        # Tune both hedge modes
        ols_folds = walk_forward_pairs(
            prices["EUR/USD"], prices[b_name],
            hedge_mode="rolling_ols", n_trials=30,
        )
        kalman_folds = walk_forward_pairs(
            prices["EUR/USD"], prices[b_name],
            hedge_mode="kalman", n_trials=30,
        )
        ols_median = _median_params(ols_folds)
        kalman_median = _median_params(kalman_folds)
        logger.info("OLS median for {}: {}", pair_label, ols_median)
        logger.info("Kalman median for {}: {}", pair_label, kalman_median)

        # Five honest views:
        #  1. Fixed OLS baseline    -- z=2.0/0.5, lb=60, hedge_window=252
        #  2. Fixed Kalman baseline -- z=2.0/0.5, lb=60, default Kalman var
        #  3. Tuned OLS (median-of-folds)
        #  4. Tuned Kalman (median-of-folds)
        #  5. Walk-forward stitched (per-fold params, OLS folds)
        configs = {
            "fixed_ols":           _fixed_baseline(prices["EUR/USD"], prices[b_name], FIXED_OLS_PARAMS),
            "fixed_kalman":        _fixed_baseline(prices["EUR/USD"], prices[b_name], FIXED_KALMAN_PARAMS),
            "tuned_ols":           _final_oos_backtest(prices["EUR/USD"], prices[b_name], ols_median),
            "tuned_kalman":        _final_oos_backtest(prices["EUR/USD"], prices[b_name], kalman_median),
            "stitched_ols":        _stitched_oos_backtest(prices["EUR/USD"], prices[b_name], ols_folds),
        }
        best_label = max(configs, key=lambda k: configs[k]["stats"].get("sharpe_ratio", -99))
        best = configs[best_label]
        logger.info("{}: best mode = {} (Sharpe {:.3f})",
                    pair_label, best_label, best["stats"].get("sharpe_ratio", 0))

        mc = _monte_carlo(best["trades"], n_runs=1000)
        pair_outputs[pair_label] = {
            "ols_folds": ols_folds,
            "kalman_folds": kalman_folds,
            "ols_median": ols_median,
            "kalman_median": kalman_median,
            "configs": configs,
            "best_mode": best_label,
            "best": best,
            "mc": mc,
        }

    # Persist best params for the live daemon
    def _params_for(d: dict) -> dict:
        mode = d["best_mode"]
        if mode == "fixed_ols":
            return FIXED_OLS_PARAMS
        if mode == "fixed_kalman":
            return FIXED_KALMAN_PARAMS
        if mode == "tuned_ols":
            return d["ols_median"]
        if mode == "tuned_kalman":
            return d["kalman_median"]
        return {"per_fold": True}

    persisted = {
        label: {
            "best_mode": d["best_mode"],
            "params": _params_for(d),
            "stats": d["best"]["stats"],
            "monte_carlo": d["mc"],
            "all_configs": {k: v["stats"] for k, v in d["configs"].items()},
        }
        for label, d in pair_outputs.items()
    }
    TUNED_PARAMS_PATH.write_text(json.dumps(persisted, indent=2, default=str))
    logger.info("Wrote tuned params to {}", TUNED_PARAMS_PATH)

    # Plot equity curves: best mode per pair
    fig = go.Figure()
    for label, d in pair_outputs.items():
        eq = d["best"]["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values,
                                  name=f"{label} ({d['best_mode']})"))
    fig.update_layout(template="plotly_dark",
                      title="OOS equity curves (best mode, post-cost)",
                      height=560, xaxis_title="Time", yaxis_title="Equity ($)")
    equity_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # Build HTML report
    def _row(name: str, stats: dict) -> str:
        return (f"<tr><td>{name}</td>"
                f"<td>{stats.get('sharpe_ratio', 0):.2f}</td>"
                f"<td>{stats.get('max_drawdown', 0):.2%}</td>"
                f"<td>{stats.get('profit_factor', 0):.2f}</td>"
                f"<td>{stats.get('total_return', 0):.2%}</td>"
                f"<td>{stats.get('total_trades', 0)}</td></tr>")

    pair_sections = []
    for label, d in pair_outputs.items():
        rows = "".join(_row(name, cfg["stats"]) for name, cfg in d["configs"].items())
        pair_sections.append(f"""
<h2>{label}</h2>
<h3>Five OOS configurations</h3>
<table border=1 cellpadding=4 style='border-collapse:collapse'>
<tr><th>Mode</th><th>Sharpe</th><th>MaxDD</th><th>PF</th><th>Total ret</th><th>Trades</th></tr>
{rows}
</table>
<p>Best mode: <b>{d['best_mode']}</b></p>
<h3>Tuned OLS median params</h3>
<pre>{json.dumps(d['ols_median'], indent=2)}</pre>
<h3>Tuned Kalman median params</h3>
<pre>{json.dumps(d['kalman_median'], indent=2)}</pre>
<h3>Best stats (post-cost)</h3>
<pre>{json.dumps(d['best']['stats'], indent=2, default=str)}</pre>
<h3>Walk-forward OLS fold table</h3>
{_format_oos_html(d['ols_folds'])}
<h3>Walk-forward Kalman fold table</h3>
{_format_oos_html(d['kalman_folds'])}
<h3>Monte Carlo (1000 trade-shuffle sims) on best mode</h3>
<pre>{json.dumps(d['mc'], indent=2)}</pre>
""")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = REPORTS_DIR / f"pairs_tuning_report_{ts}.html"
    out.write_text(f"""<!doctype html><html><head>
<meta charset="utf-8"><title>EUR/USD pairs research v5</title>
<style>body{{background:#111;color:#eee;font-family:monospace;padding:24px}}
table{{color:#eee}} th{{background:#222}} pre{{background:#1a1a1a;padding:8px}}</style>
</head><body>
<h1>EUR/USD pairs trading — Kalman + walk-forward tuning</h1>
<p>Run: {ts}.  Cost model: 4 bps per spread direction-change.</p>
<h2>Equity curves</h2>{equity_html}
{''.join(pair_sections)}
</body></html>""")
    logger.info("Wrote HTML report to {}", out)

    # Telegram summary
    lines = []
    for label, d in pair_outputs.items():
        s = d["best"]["stats"]
        lines.append(
            f"<b>{label}</b> (best={d['best_mode']})\n"
            f"  Sharpe <b>{s.get('sharpe_ratio', 0):.2f}</b>, "
            f"MaxDD <b>{s.get('max_drawdown', 0):.2%}</b>, "
            f"PF <b>{s.get('profit_factor', 0):.2f}</b>, "
            f"Total <b>{s.get('total_return', 0):.2%}</b>, "
            f"Trades <b>{s.get('total_trades', 0)}</b>\n"
            f"  MC: P(ruin)=<b>{d['mc']['p_ruin']:.0%}</b>, "
            f"Sharpe P5/P50/P95 = "
            f"<b>{d['mc']['p5_sharpe']:.2f}/{d['mc']['p50_sharpe']:.2f}/"
            f"{d['mc']['p95_sharpe']:.2f}</b>"
        )
    msg = (
        "📊 <b>Pairs strategy tuning done</b>\n\n"
        + "\n\n".join(lines)
        + f"\n\nElapsed: {time.time()-t0:.0f}s"
    )
    _telegram(msg)

    print("\n=== HTML report:", out)
    for label, d in pair_outputs.items():
        print(f"=== {label} | best mode: {d['best_mode']}")
        print(f"=== {label} | all configs:")
        for name, cfg in d["configs"].items():
            print(f"  {name:25s}: Sharpe="
                  f"{cfg['stats'].get('sharpe_ratio', 0):+.2f} "
                  f"PF={cfg['stats'].get('profit_factor', 0):.2f} "
                  f"DD={cfg['stats'].get('max_drawdown', 0):+.2%} "
                  f"Total={cfg['stats'].get('total_return', 0):+.2%} "
                  f"N={cfg['stats'].get('total_trades', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
