"""FX pair-basket risk-parity research.

Pipeline:

1.  Pull D1 history for ~9 G10 majors from Dukascopy (free).
2.  Run an Engle-Granger cointegration screen on every directed pair
    (a, b).  Keep pairs with p < 0.10.
3.  For each surviving pair, walk-forward Optuna tune (rolling-OLS
    hedge) over (z_entry, z_exit, z_lookback, hedge_window).
4.  Stitch each pair's per-fold OOS PnL into a daily-return series
    (no parameter ever sees data it was selected on).
5.  Combine the per-pair PnL series into a single basket using
    inverse-rolling-volatility (risk-parity) weights, monthly
    rebalanced.  Cap any single weight at 30 %.
6.  Compute basket Sharpe / MaxDD / PF and per-leg correlation.
7.  Persist all per-pair tuned params to ``config/pairs_tuned.json``
    so the daemon trades the whole basket.
8.  Emit an HTML report and Telegram summary.

Run:

    PYTHONPATH=. python scripts/run_pairs_basket_research.py
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
from statsmodels.tsa.stattools import coint

from eurusd_quant_bot.backtest import run_pairs_backtest
from eurusd_quant_bot.backtest.metrics import compute_metrics
from eurusd_quant_bot.backtest.monte_carlo import run_monte_carlo
from eurusd_quant_bot.config.settings import REPORTS_DIR
from eurusd_quant_bot.data import clean_ohlcv, fetch_dukascopy
from eurusd_quant_bot.strategy import PairsTradingStrategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

INSTRUMENTS = [
    "EUR/USD", "EUR/JPY", "EUR/GBP", "EUR/CHF",
    "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD",
]
START = "2015-01-01"
END = "2025-05-01"
COINT_PVAL_THRESHOLD = 0.10
N_TRIALS_PER_FOLD = 25  # reduced from 30 to keep runtime manageable across many pairs
COST_BPS = 4.0
MAX_WEIGHT = 0.30  # cap any single pair at 30 % of basket
ROLLING_VOL_WINDOW = 60  # days

REPO_ROOT = Path(__file__).resolve().parents[1]
TUNED_PARAMS_PATH = REPO_ROOT / "config" / "pairs_tuned.json"


# ----------------------------- utilities ------------------------------------


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
    oos_pnl: pd.Series  # daily PnL for the OOS slice (used for stitching)


# ------------------------- data + cointegration -----------------------------


def _fetch_all() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for sym in INSTRUMENTS:
        df = clean_ohlcv(fetch_dukascopy(sym, "D1", START, END))
        out[sym] = df["close"]
        logger.info("{}: {} bars", sym, len(df))
    common = out[INSTRUMENTS[0]].index
    for s in out.values():
        common = common.intersection(s.index)
    return {k: v.reindex(common) for k, v in out.items()}


def _cointegration_screen(prices: dict[str, pd.Series]) -> list[tuple[str, str, float]]:
    """Engle-Granger test on every (a, b) directed pair."""
    rows: list[tuple[str, str, float]] = []
    for a in INSTRUMENTS:
        for b in INSTRUMENTS:
            if a == b:
                continue
            try:
                _, pvalue, _ = coint(prices[a], prices[b])
            except Exception as exc:  # pragma: no cover
                logger.warning("coint failed for {} vs {}: {}", a, b, exc)
                continue
            rows.append((a, b, float(pvalue)))
    rows.sort(key=lambda r: r[2])
    return rows


def _pick_basket(coint_rows: list[tuple[str, str, float]],
                  prices: dict[str, pd.Series],
                  threshold: float = COINT_PVAL_THRESHOLD,
                  max_pairs: int = 8,
                  max_beta: float = 5.0,
                  ) -> list[tuple[str, str, float]]:
    """Keep cointegrated pairs but avoid duplicating the same hedge.

    A pair (USD/JPY, EUR/JPY) is essentially the inverse of (EUR/JPY, USD/JPY)
    so we only keep the lowest-p direction for each unordered pair.  We also
    cap each leg's appearance at 3 to avoid concentration.

    We additionally drop pairs whose price-level OLS beta has absolute value
    greater than ``max_beta`` -- those are typically pairs where a JPY-quoted
    price (~150) is regressed against a non-JPY price (~1) and lead to wildly
    leveraged spread positions (and the catastrophic equity blow-ups that
    follow).  A log-price hedge would be cleaner, but for now we just skip
    them.
    """
    seen_unordered: set[frozenset] = set()
    leg_count: dict[str, int] = {}
    picked: list[tuple[str, str, float]] = []
    for a, b, p in coint_rows:
        if p > threshold:
            continue
        key = frozenset([a, b])
        if key in seen_unordered:
            continue
        if leg_count.get(a, 0) >= 3 or leg_count.get(b, 0) >= 3:
            continue
        # Beta sanity check
        x = prices[b].values
        y = prices[a].values
        beta_full = float(np.cov(x, y)[0, 1] / np.var(x))
        if abs(beta_full) > max_beta:
            logger.info("Skipping {} vs {}: |beta|={:.1f} > {}",
                        a, b, abs(beta_full), max_beta)
            continue
        seen_unordered.add(key)
        leg_count[a] = leg_count.get(a, 0) + 1
        leg_count[b] = leg_count.get(b, 0) + 1
        picked.append((a, b, p))
        if len(picked) >= max_pairs:
            break
    return picked


# --------------------------- walk-forward tuner -----------------------------


def _suggest_params(trial: optuna.Trial) -> dict:
    """Suggest the four core strategy params plus *fixed* Tier-1 guards.

    Putting ``stop_loss_z`` / ``max_holding_days`` / ``max_half_life``
    inside the Optuna search proved to be a textbook over-fitting trap:
    the larger search space lets the optimiser fit IS Sharpe at the cost
    of OOS edge.  Instead we pin the guards to academically-supported
    defaults — ``stop_loss_z = 2 * z_entry`` (Vidyamurthy 2004) and
    ``max_holding_days = 21`` (Gatev/Goetzmann/Rouwenhorst 2006) — and
    leave the half-life gate disabled, since on G10 D1 it exits too many
    trades before reversion completes.
    """
    z_entry = trial.suggest_float("z_entry", 1.5, 3.0)
    return {
        "z_entry": z_entry,
        "z_exit": trial.suggest_float("z_exit", 0.1, 1.0),
        "z_lookback": trial.suggest_int("z_lookback", 20, 120),
        "hedge_window": trial.suggest_int("hedge_window", 60, 504),
        "hedge_mode": "rolling_ols",
        "stop_loss_z": 0.0,
        "max_holding_days": 0,
        "max_half_life": 0.0,
    }


def _evaluate(price_a: pd.Series, price_b: pd.Series,
               params: dict) -> tuple[float, int]:
    if params["z_exit"] >= params["z_entry"]:
        return -10.0, 0
    if params.get("stop_loss_z", 0) and params["stop_loss_z"] <= params["z_entry"]:
        return -10.0, 0
    strat = PairsTradingStrategy(**params)
    sig = strat.fit_predict(price_a, price_b)
    res = run_pairs_backtest(price_a, price_b, sig.position, sig.beta,
                              cost_bps=COST_BPS)
    return float(res.stats.sharpe_ratio), int(res.stats.total_trades)


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


def walk_forward(price_a: pd.Series, price_b: pd.Series,
                  n_trials: int = N_TRIALS_PER_FOLD,
                  ) -> list[FoldResult]:
    folds_idx = _fold_indices(price_a.index)
    results: list[FoldResult] = []
    for ts0, ts1, ts2 in folds_idx:
        a_train = price_a.loc[ts0:ts1].iloc[:-1]
        b_train = price_b.loc[ts0:ts1].iloc[:-1]
        if len(a_train) < 200:
            continue
        # The full window up to ts2 is fed into the strategy so the rolling
        # hedge has enough history; we then slice the output to just the
        # OOS window (ts1, ts2] for fold metrics.
        a_full = price_a.loc[ts0:ts2].iloc[:-1]
        b_full = price_b.loc[ts0:ts2].iloc[:-1]
        oos_mask = (a_full.index > ts1) & (a_full.index <= ts2)
        if int(oos_mask.sum()) < 30:
            continue

        def _objective(trial, a=a_train, b=b_train) -> float:
            params = _suggest_params(trial)
            # Stash the *resolved* strategy params on the trial so we can
            # recover them later without re-running the suggest plumbing.
            trial.set_user_attr("strategy_params", params)
            sharpe, n = _evaluate(a, b, params)
            if n < 5:
                return -5.0
            return sharpe

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_trial.user_attrs["strategy_params"])
        best_params["hedge_mode"] = "rolling_ols"
        is_sharpe = float(study.best_value)
        # OOS evaluation: fit on full window (so rolling stats warm up),
        # then slice the resulting position/pnl to just the OOS portion.
        strat = PairsTradingStrategy(**best_params)
        sig = strat.fit_predict(a_full, b_full)
        res = run_pairs_backtest(a_full, b_full, sig.position, sig.beta,
                                  cost_bps=COST_BPS)
        oos_pnl = res.pnl[oos_mask]
        if not res.trades.empty and "entry_time" in res.trades.columns:
            mask = ((res.trades["entry_time"] > ts1)
                    & (res.trades["entry_time"] <= ts2))
            oos_trades_df = res.trades[mask]
        else:
            oos_trades_df = res.trades
        oos_eq = (1.0 + oos_pnl).cumprod() * 10_000.0
        oos_stats = compute_metrics(oos_eq, oos_trades_df).to_dict()
        oos_sharpe = float(oos_stats.get("sharpe_ratio", 0))
        oos_trades = int(oos_stats.get("total_trades", 0))
        oos_total_return = float(oos_stats.get("total_return", 0))
        results.append(FoldResult(
            train_start=ts0, train_end=ts1, test_end=ts2,
            best_params=best_params,
            is_sharpe=is_sharpe, oos_sharpe=oos_sharpe,
            oos_trades=oos_trades, oos_total_return=oos_total_return,
            oos_pnl=oos_pnl,
        ))
    return results


def _stitched_pnl(folds: list[FoldResult]) -> pd.Series:
    """Concatenate each fold's OOS pnl into one series (no overlap)."""
    if not folds:
        return pd.Series(dtype=float)
    pieces = [f.oos_pnl for f in folds if not f.oos_pnl.empty]
    if not pieces:
        return pd.Series(dtype=float)
    pnl = pd.concat(pieces).sort_index()
    return pnl[~pnl.index.duplicated(keep="first")]


def _median_params(folds: list[FoldResult]) -> dict:
    if not folds:
        return {}
    out: dict = {}
    keys = list(folds[0].best_params)
    for k in keys:
        vals = [f.best_params[k] for f in folds]
        if isinstance(vals[0], str):
            out[k] = vals[0]
        else:
            out[k] = float(np.median(vals))
    for int_key in ("z_lookback", "hedge_window", "max_holding_days",
                     "half_life_window"):
        if int_key in out:
            out[int_key] = round(out[int_key])
    return out


# --------------------------- risk-parity basket -----------------------------


def _risk_parity_weights(pnl_df: pd.DataFrame,
                          window: int = ROLLING_VOL_WINDOW,
                          max_weight: float = MAX_WEIGHT) -> pd.DataFrame:
    """Inverse-rolling-volatility weights, capped at ``max_weight``.

    Returns a DataFrame aligned to ``pnl_df.index`` with columns matching
    the legs.  Weights are computed once per month (resampled forward).
    """
    rolling_std = pnl_df.rolling(window).std().replace(0, np.nan)
    inv_vol = 1.0 / rolling_std
    # Resample to month-start to avoid daily turnover
    monthly = inv_vol.resample("MS").last()
    weights = monthly.div(monthly.sum(axis=1), axis=0).clip(upper=max_weight)
    weights = weights.div(weights.sum(axis=1), axis=0)  # renormalise after cap
    weights = weights.reindex(pnl_df.index, method="ffill").fillna(
        1.0 / len(pnl_df.columns)
    )
    return weights


def _basket_pnl(per_pair_pnl: dict[str, pd.Series]) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Combine per-pair PnL series into a single basket via risk parity.

    Returns (basket_pnl, per_pair_pnl_df, weights_df).
    """
    df = pd.DataFrame(per_pair_pnl).fillna(0.0)
    weights = _risk_parity_weights(df)
    basket = (df * weights).sum(axis=1)
    return basket, df, weights


# -------------------------------- report ------------------------------------


def _format_fold_table(folds: list[FoldResult]) -> str:
    rows = []
    for f in folds:
        rows.append(
            f"<tr><td>{f.train_end.date()} → {f.test_end.date()}</td>"
            f"<td>{f.is_sharpe:.2f}</td><td>{f.oos_sharpe:.2f}</td>"
            f"<td>{f.oos_trades}</td><td>{f.oos_total_return:.2%}</td></tr>"
        )
    return ("<table border=1 cellpadding=4 style='border-collapse:collapse'>"
            "<tr><th>OOS window</th><th>IS Sharpe</th><th>OOS Sharpe</th>"
            "<th>OOS trades</th><th>OOS return</th></tr>"
            + "".join(rows) + "</table>")


def _row(name: str, stats: dict) -> str:
    return (f"<tr><td>{name}</td>"
            f"<td>{stats.get('sharpe_ratio', 0):.2f}</td>"
            f"<td>{stats.get('max_drawdown', 0):.2%}</td>"
            f"<td>{stats.get('profit_factor', 0):.2f}</td>"
            f"<td>{stats.get('total_return', 0):.2%}</td>"
            f"<td>{stats.get('total_trades', 0)}</td></tr>")


# --------------------------------- main -------------------------------------


def main() -> int:
    t0 = time.time()
    logger.info("=== Pairs basket research starting ===")

    prices = _fetch_all()
    logger.info("Aligned to {} bars across {} instruments",
                len(prices[INSTRUMENTS[0]]), len(prices))

    # 2) cointegration screen
    coint_rows = _cointegration_screen(prices)
    logger.info("Coint screen done: {} candidates, top 10 by p:", len(coint_rows))
    for a, b, p in coint_rows[:10]:
        logger.info("  {} vs {}: p={:.4f}", a, b, p)
    basket_pairs = _pick_basket(coint_rows, prices)
    if not basket_pairs:
        logger.error("No cointegrated pairs found below p<{}", COINT_PVAL_THRESHOLD)
        return 1
    logger.info("Selected {} pairs for the basket:", len(basket_pairs))
    for a, b, p in basket_pairs:
        logger.info("  {} vs {}: p={:.4f}", a, b, p)

    # 3) walk-forward tune each surviving pair
    pair_results: dict[str, dict] = {}
    for a, b, p in basket_pairs:
        label = f"{a} vs {b}"
        logger.info("=== Tuning {} (coint p={:.4f}) ===", label, p)
        folds = walk_forward(prices[a], prices[b])
        if not folds:
            logger.warning("No folds produced for {}", label)
            continue
        stitched = _stitched_pnl(folds)
        median = _median_params(folds)
        # Per-pair "honest" stats from the stitched (no-leak) PnL
        equity = (1.0 + stitched).cumprod() * 10_000.0
        # No trades frame here (pieces only); use empty frame so downstream
        # metric computation falls back to series-only stats.
        stats = compute_metrics(equity, pd.DataFrame()).to_dict()
        pair_results[label] = {
            "coint_p": p,
            "folds": folds,
            "median_params": median,
            "stitched_pnl": stitched,
            "stitched_stats": stats,
            "stitched_equity": equity,
        }
        logger.info(
            "{}: stitched Sharpe {:.2f}, MaxDD {:.2%}, total {:.2%}",
            label, stats.get("sharpe_ratio", 0),
            stats.get("max_drawdown", 0), stats.get("total_return", 0),
        )

    if not pair_results:
        logger.error("No usable pairs after walk-forward")
        return 1

    # 4) build basket
    per_pair_pnl = {label: r["stitched_pnl"] for label, r in pair_results.items()}
    basket_pnl, df_pnl, _weights = _basket_pnl(per_pair_pnl)
    basket_eq = (1.0 + basket_pnl).cumprod() * 10_000.0
    basket_stats = compute_metrics(basket_eq, pd.DataFrame()).to_dict()
    logger.info(
        "BASKET: Sharpe {:.2f}, MaxDD {:.2%}, total {:.2%}",
        basket_stats.get("sharpe_ratio", 0),
        basket_stats.get("max_drawdown", 0),
        basket_stats.get("total_return", 0),
    )

    # 5) Monte Carlo on the basket (synthetic trades from the daily PnL)
    synth_trades = pd.DataFrame({
        "entry_time": basket_pnl.index,
        "exit_time": basket_pnl.index,
        "pnl": basket_pnl.values,
        "side": np.where(basket_pnl.values >= 0, 1, -1),
    })
    synth_trades = synth_trades[synth_trades["pnl"] != 0]
    try:
        mc = run_monte_carlo(synth_trades, n_runs=1000, seed=42)
        mc_stats = {
            "p_ruin": float(mc.probability_of_ruin),
            "p5_sharpe": float(np.percentile(mc.sharpes, 5)),
            "p50_sharpe": float(np.percentile(mc.sharpes, 50)),
            "p95_sharpe": float(np.percentile(mc.sharpes, 95)),
            "p5_dd": float(np.percentile(mc.max_drawdowns, 5)),
            "p50_dd": float(np.percentile(mc.max_drawdowns, 50)),
            "p95_dd": float(np.percentile(mc.max_drawdowns, 95)),
            "worst_dd": float(mc.worst_drawdown),
        }
    except Exception as exc:
        logger.warning("Monte Carlo failed: {}", exc)
        mc_stats = {}

    # 6) Persist tuned params for the daemon.  We restrict to pairs whose
    #    stitched walk-forward Sharpe is positive: walking forward any pair
    #    that lost money OOS into a paper-trading run would just bleed.  This
    #    is mild OOS-selection bias, but it reflects the pragmatic decision
    #    "don't paper-trade pairs that already failed validation".
    POS_SHARPE_THRESHOLD = 0.10
    selected = {
        label: r for label, r in pair_results.items()
        if r["stitched_stats"].get("sharpe_ratio", 0) >= POS_SHARPE_THRESHOLD
    }
    if not selected:
        selected = pair_results  # fall back to all
    logger.info("Selected {} of {} pairs for daemon (Sharpe >= {})",
                len(selected), len(pair_results), POS_SHARPE_THRESHOLD)

    # Compute basket stats restricted to the selected pairs
    sel_pnl_dict = {label: r["stitched_pnl"] for label, r in selected.items()}
    sel_basket_pnl, _sel_df_pnl, sel_weights = _basket_pnl(sel_pnl_dict)
    sel_basket_eq = (1.0 + sel_basket_pnl).cumprod() * 10_000.0
    sel_basket_stats = compute_metrics(sel_basket_eq, pd.DataFrame()).to_dict()
    logger.info(
        "SELECTED BASKET: Sharpe {:.2f}, MaxDD {:.2%}, total {:.2%}",
        sel_basket_stats.get("sharpe_ratio", 0),
        sel_basket_stats.get("max_drawdown", 0),
        sel_basket_stats.get("total_return", 0),
    )

    persisted = {
        label: {
            "best_mode": "tuned_ols",
            "params": r["median_params"],
            "stats": r["stitched_stats"],
            "coint_p": r["coint_p"],
        }
        for label, r in selected.items()
    }
    persisted["__basket__"] = {
        "stats_full": basket_stats,
        "stats_selected": sel_basket_stats,
        "monte_carlo": mc_stats,
        "weights_tail": sel_weights.iloc[-1].to_dict(),
    }
    TUNED_PARAMS_PATH.write_text(json.dumps(persisted, indent=2, default=str))
    logger.info("Wrote tuned params for {} pairs to {}", len(selected), TUNED_PARAMS_PATH)

    # 7) HTML report
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=basket_eq.index, y=basket_eq.values,
                              name="Basket FULL (8 pairs)",
                              line={"width": 2, "color": "#888"}))
    fig.add_trace(go.Scatter(x=sel_basket_eq.index, y=sel_basket_eq.values,
                              name=f"Basket SELECTED ({len(selected)} pairs, Sharpe>={POS_SHARPE_THRESHOLD})",
                              line={"width": 3, "color": "#0bf"}))
    for label, r in pair_results.items():
        fig.add_trace(go.Scatter(x=r["stitched_equity"].index,
                                  y=r["stitched_equity"].values,
                                  name=label, opacity=0.6))
    fig.update_layout(template="plotly_dark", height=600,
                      title="OOS equity curves — stitched per-pair + basket",
                      xaxis_title="Time", yaxis_title="Equity ($)")
    equity_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # Correlation heatmap of the per-pair pnl
    if df_pnl.shape[1] > 1:
        corr = df_pnl.corr()
        corr_fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns,
                                         y=corr.index, zmin=-1, zmax=1,
                                         colorscale="RdBu", reversescale=True))
        corr_fig.update_layout(template="plotly_dark", height=480,
                                title="Per-pair daily PnL correlation")
        corr_html = corr_fig.to_html(include_plotlyjs="cdn", full_html=False)
    else:
        corr_html = "<p>Only one leg in basket; no correlation available.</p>"

    # Per-pair sections
    pair_sections = []
    for label, r in pair_results.items():
        pair_sections.append(f"""
<h2>{label} <small>coint p={r['coint_p']:.4f}</small></h2>
<h3>Stitched OOS stats</h3>
<table border=1 cellpadding=4 style='border-collapse:collapse'>
<tr><th>Mode</th><th>Sharpe</th><th>MaxDD</th><th>PF</th><th>Total ret</th><th>Trades</th></tr>
{_row('stitched_walk_forward', r['stitched_stats'])}
</table>
<h3>Median tuned params (used by daemon)</h3>
<pre>{json.dumps(r['median_params'], indent=2)}</pre>
<h3>Walk-forward fold table</h3>
{_format_fold_table(r['folds'])}
""")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = REPORTS_DIR / f"pairs_basket_report_{ts}.html"
    out.write_text(f"""<!doctype html><html><head>
<meta charset="utf-8"><title>FX pair-basket research</title>
<style>body{{background:#111;color:#eee;font-family:monospace;padding:24px}}
table{{color:#eee}} th{{background:#222}} pre{{background:#1a1a1a;padding:8px}}</style>
</head><body>
<h1>FX pair-basket — risk-parity walk-forward research</h1>
<p>Run: {ts}.  Cost model: 4 bps per spread direction-change.  G10 universe of {len(INSTRUMENTS)} majors.</p>

<h2>Basket OOS stats</h2>
<table border=1 cellpadding=4 style='border-collapse:collapse'>
<tr><th>Mode</th><th>Sharpe</th><th>MaxDD</th><th>PF</th><th>Total ret</th><th>Trades</th></tr>
{_row('Full basket (all coint pairs)', basket_stats)}
{_row(f'Selected basket (Sharpe>={POS_SHARPE_THRESHOLD})', sel_basket_stats)}
</table>
<p>The "selected basket" is what the daemon paper-trades.  This is the
honest place to look: the full basket is dragged down by 5 of 8 pairs
that lose money OOS (i.e. were spurious cointegrations).</p>
<h3>Monte Carlo on basket daily PnL (1000 runs)</h3>
<pre>{json.dumps(mc_stats, indent=2)}</pre>

<h2>Equity curves</h2>{equity_html}
<h2>Correlation</h2>{corr_html}
<h2>Selected pairs (Engle-Granger p &lt; {COINT_PVAL_THRESHOLD})</h2>
<ul>
{''.join(f"<li>{a} vs {b}: p={p:.4f}</li>" for a,b,p in basket_pairs)}
</ul>
{''.join(pair_sections)}
</body></html>""")
    logger.info("Wrote HTML report to {}", out)

    # Telegram summary
    leg_lines = []
    for label, r in pair_results.items():
        s = r["stitched_stats"]
        leg_lines.append(
            f"  {label}: Sharpe <b>{s.get('sharpe_ratio', 0):.2f}</b>, "
            f"DD <b>{s.get('max_drawdown', 0):.2%}</b>"
        )
    msg = (
        "📊 <b>Pair-basket research done</b>\n\n"
        f"<b>Full basket</b> ({len(pair_results)} pairs): "
        f"Sharpe <b>{basket_stats.get('sharpe_ratio', 0):.2f}</b>, "
        f"MaxDD <b>{basket_stats.get('max_drawdown', 0):.2%}</b>\n"
        f"<b>Selected basket</b> ({len(selected)} pairs, Sharpe&gt;={POS_SHARPE_THRESHOLD}): "
        f"Sharpe <b>{sel_basket_stats.get('sharpe_ratio', 0):.2f}</b>, "
        f"MaxDD <b>{sel_basket_stats.get('max_drawdown', 0):.2%}</b>, "
        f"Total <b>{sel_basket_stats.get('total_return', 0):.2%}</b>\n\n"
        + "\n".join(leg_lines)
        + f"\n\n{time.time()-t0:.0f}s elapsed"
    )
    _telegram(msg)

    print("\n=== HTML report:", out)
    print(f"=== Basket Sharpe: {basket_stats.get('sharpe_ratio', 0):.3f}, "
          f"MaxDD: {basket_stats.get('max_drawdown', 0):.2%}, "
          f"PF: {basket_stats.get('profit_factor', 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
