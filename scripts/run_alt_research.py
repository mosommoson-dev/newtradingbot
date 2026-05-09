"""Alternative research: try prediction problems where edge might actually exist.

Three independent experiments in one run:

1. **Multi-week directional model** -- XGBoost predicting whether the EUR/USD
   D1 close 20 trading days ahead is higher than today's close, using the
   full macro+price feature matrix.  If mean OOS accuracy on walk-forward
   folds is materially > 50%, we have an edge.  Backtest the resulting
   long/short signal end-to-end.

2. **Cross-pair cointegration screen** -- Engle-Granger ADF test on the
   residuals of OLS regressions of EUR/USD on each of EUR/GBP, EUR/JPY,
   GBP/USD, USD/JPY.  Rank by p-value.

3. **Stat-arb backtest on the lowest-p-value pair** -- standard z-score
   mean-reversion on the cointegration spread (long-short at +/-2 sigma,
   close at 0).  Compute Sharpe, MaxDD, PF on a 2-year rolling-OLS spread.

Outputs:
  - HTML report under reports/.
  - Telegram headline summary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from loguru import logger

from eurusd_quant_bot.backtest import run_backtest
from eurusd_quant_bot.config.settings import REPORTS_DIR
from eurusd_quant_bot.data import (
    align_macro,
    clean_ohlcv,
    fetch_dukascopy,
    fetch_fred_bundle,
)
from eurusd_quant_bot.ml.features import build_features, feature_names
from eurusd_quant_bot.ml.models import EnsembleModel
from eurusd_quant_bot.ml.training import drop_correlated, make_target

REPORT_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
HTML_PATH = REPORTS_DIR / f"alt_research_report_{REPORT_TS}.html"

PAIRS = ["EUR/USD", "EUR/GBP", "EUR/JPY", "GBP/USD", "USD/JPY"]
START = "2015-01-01"
END = "2025-05-01"


def _telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram send failed: {}", exc)


# ---------------------------------------------------------------------------
# Experiment 1: 20-day directional model
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    accuracy: float


def run_directional(df: pd.DataFrame, macro_aligned: pd.DataFrame, horizon: int = 20) -> dict:
    logger.info("=== Experiment 1: {}-day directional model ===", horizon)
    feats = build_features(df, macro=macro_aligned)
    target = make_target(feats["close"], horizon)
    full = feats.join(target).dropna(subset=["target"])

    cols = feature_names(feats)  # exclude OHLCV + target
    X = full[cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    full = full.loc[X.index]
    y = full["target"].astype(int)

    X = drop_correlated(X, threshold=0.9)
    logger.info("Feature matrix after cleanup: {} rows x {} cols", *X.shape)

    folds_idx = []
    start = X.index[0]
    end = X.index[-1]
    train_end = start + pd.DateOffset(months=24)
    while train_end + pd.DateOffset(months=6) <= end:
        test_end = train_end + pd.DateOffset(months=6)
        folds_idx.append((start, train_end, test_end))
        train_end = test_end

    results: list[Fold] = []
    oos_proba = pd.Series(np.nan, index=X.index, name="p_up")

    for ts0, ts1, ts2 in folds_idx:
        train_mask = (X.index >= ts0) & (X.index < ts1)
        test_mask = (X.index >= ts1) & (X.index < ts2)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if len(X_train) < 200 or len(X_test) < 50:
            continue
        model = EnsembleModel(weights={"xgb": 0.6, "logreg": 0.4})
        model.fit(X_train, y_train, members=["xgb", "logreg"])
        proba = model.predict_proba(X_test)
        p_up = proba[:, 1]
        preds = (p_up > 0.5).astype(int)
        acc = float(np.mean(preds == y_test.values))
        results.append(Fold(ts1, ts2, len(X_train), len(X_test), acc))
        oos_proba.loc[X_test.index] = p_up
        logger.info("Fold {} -> {}: train={}, test={}, acc={:.4f}", ts1.date(), ts2.date(),
                    len(X_train), len(X_test), acc)

    fold_df = pd.DataFrame([{
        "train_end": f.train_end, "test_end": f.test_end,
        "n_train": f.n_train, "n_test": f.n_test, "accuracy": f.accuracy,
    } for f in results])
    mean_acc = fold_df["accuracy"].mean() if not fold_df.empty else 0.0
    logger.info("Mean OOS accuracy: {:.4f} across {} folds", mean_acc, len(fold_df))

    # Backtest the OOS signal
    sigs = pd.Series(0, index=df.index, dtype=int, name="signal")
    valid = oos_proba.dropna()
    long_mask = valid > 0.55
    short_mask = valid < 0.45
    sigs.loc[valid.index[long_mask]] = 1
    sigs.loc[valid.index[short_mask]] = -1
    bt = run_backtest(df, sigs)
    stats = bt.stats.to_dict()
    n_signals = int((sigs != 0).sum())
    logger.info("Directional OOS backtest: {} signals, stats: {}", n_signals, stats)

    return {
        "fold_df": fold_df,
        "mean_oos_accuracy": float(mean_acc),
        "horizon": horizon,
        "stats": stats,
        "n_signals": n_signals,
        "equity": bt.equity,
    }


# ---------------------------------------------------------------------------
# Experiment 2: cointegration screen
# ---------------------------------------------------------------------------

def run_cointegration(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from statsmodels.tsa.stattools import coint

    logger.info("=== Experiment 2: cointegration screen ===")
    base = prices["EUR/USD"]["close"].rename("EUR/USD")
    rows = []
    for pair, df_p in prices.items():
        if pair == "EUR/USD":
            continue
        other = df_p["close"].rename(pair)
        merged = pd.concat([base, other], axis=1).dropna()
        if len(merged) < 1000:
            continue
        # Engle-Granger test (statsmodels.coint)
        score, pvalue, _ = coint(merged["EUR/USD"], merged[pair])
        # Rolling 2-year OLS spread Z-score range as a regime hint
        rows.append({"pair": pair, "n_obs": len(merged), "coint_score": float(score), "p_value": float(pvalue)})
    coint_df = pd.DataFrame(rows).sort_values("p_value")
    logger.info("\n{}", coint_df.to_string(index=False))
    return coint_df


# ---------------------------------------------------------------------------
# Experiment 3: stat-arb on best cointegrated pair
# ---------------------------------------------------------------------------

def run_pairs_trade(eurusd: pd.DataFrame, other: pd.DataFrame, other_name: str) -> dict:
    logger.info("=== Experiment 3: pairs trade EUR/USD vs {} ===", other_name)
    merged = pd.concat([eurusd["close"].rename("a"), other["close"].rename("b")], axis=1).dropna()
    # Rolling OLS hedge ratio + spread
    win = 252  # 1 year for hedge ratio estimation
    a, b = merged["a"], merged["b"]
    # Compute rolling beta
    cov = a.rolling(win).cov(b)
    var_b = b.rolling(win).var()
    beta = (cov / var_b).ffill()
    spread = a - beta * b
    spread_mean = spread.rolling(60).mean()
    spread_std = spread.rolling(60).std()
    z = (spread - spread_mean) / spread_std

    # Position: long EUR/USD, short other when z < -2; opposite when z > +2; flat at |z| < 0.5
    pos_a = pd.Series(0, index=z.index, dtype=int)
    pos_a.loc[z < -2.0] = 1
    pos_a.loc[z > 2.0] = -1
    # Hold until z crosses 0
    pos_a = pos_a.replace(0, np.nan).ffill().fillna(0)
    flat_mask = z.abs() < 0.5
    pos_a.loc[flat_mask] = 0

    daily_ret = a.pct_change().fillna(0.0)
    daily_ret_b = b.pct_change().fillna(0.0)
    # Pair PnL: long A, short beta units of B
    pnl = pos_a.shift(1) * (daily_ret - beta.shift(1).fillna(0) * daily_ret_b)
    pnl = pnl.fillna(0.0)
    # Transaction costs: 0.8 pips spread + ~$0.07/lot commission on each leg.
    # As fraction of price: ~ 0.0001 per leg => 0.0002 per round-trip leg switch
    # on this paired position (we hold both A and B legs, so 2x).  Apply when
    # position changes.
    pos_change = pos_a.diff().abs().fillna(0.0)
    cost_per_change = 0.0004  # 4 bps per direction-change on the spread
    pnl = pnl - pos_change * cost_per_change
    # Equity: only kicks in after warmup window (NaN beta region) settles
    valid_pnl = pnl.loc[pnl.first_valid_index():] if pnl.first_valid_index() else pnl
    equity = (1 + valid_pnl).cumprod() * 10_000.0

    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() else 0.0
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
    n_trades = int((pos_a.diff().abs() > 0).sum())

    # Profit factor
    wins_sum = pnl[pnl > 0].sum()
    losses_sum = -pnl[pnl < 0].sum()
    pf = float(wins_sum / losses_sum) if losses_sum > 0 else float("nan")

    stats = {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "profit_factor": pf,
        "n_trades": n_trades,
        "pair": f"EUR/USD vs {other_name}",
    }
    logger.info("Pairs stats: {}", stats)
    return {"stats": stats, "equity": equity, "z": z, "spread": spread}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    logger.info("=== Alt research run starting ===")
    _telegram("🟢 EUR/USD bot: alt research started (20d directional + cointegration + stat-arb).")

    # --- Data --------------------------------------------------------------
    prices = {}
    for pair in PAIRS:
        df = clean_ohlcv(fetch_dukascopy(pair=pair, granularity="D1", start=START, end=END))
        prices[pair] = df
        logger.info("{}: {} bars", pair, len(df))
    macro = fetch_fred_bundle()
    macro_aligned = align_macro(prices["EUR/USD"], macro) if not macro.empty else None

    # --- Experiment 1 ------------------------------------------------------
    dir_results = run_directional(prices["EUR/USD"], macro_aligned, horizon=20)

    # --- Experiment 2 ------------------------------------------------------
    coint_df = run_cointegration(prices)

    # --- Experiment 3 ------------------------------------------------------
    # Run pairs trade on every pair where cointegration p < 0.10
    pairs_runs: list[dict] = []
    for _, row in coint_df.iterrows():
        if row["p_value"] >= 0.10:
            continue
        pair_name = row["pair"]
        result = run_pairs_trade(prices["EUR/USD"], prices[pair_name], pair_name)
        pairs_runs.append(result)

    # --- HTML report -------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dir_results["equity"].index, y=dir_results["equity"].values,
                              name="20-day directional (OOS)"))
    for run in pairs_runs:
        fig.add_trace(go.Scatter(x=run["equity"].index, y=run["equity"].values,
                                  name=f"Pairs: {run['stats']['pair']}"))
    fig.update_layout(template="plotly_dark", title="Equity curves", height=540,
                      xaxis_title="Time", yaxis_title="Equity ($)")
    equity_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    fold_table = (
        dir_results["fold_df"].to_html(index=False, classes="metrics", border=0)
        if not dir_results["fold_df"].empty else "(no folds)"
    )
    coint_table = coint_df.to_html(index=False, classes="metrics", border=0) if not coint_df.empty else "(no pairs)"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Alt research report</title>
<style>
body {{ background:#0e1117; color:#e6e6e6; font-family:system-ui,sans-serif; margin:32px; }}
h1, h2 {{ color:#fafafa; }}
table.metrics {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.metrics th, table.metrics td {{ padding:6px 10px; border-bottom:1px solid #2a2f3a; text-align:right; }}
table.metrics th:first-child, table.metrics td:first-child {{ text-align:left; }}
pre {{ background:#1e222a; padding:14px; border-radius:8px; overflow:auto; font-size:12px; }}
</style></head><body>
<h1>EUR/USD alt research report</h1>
<p>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} | Elapsed {time.time()-t0:.0f}s</p>

<h2>Experiment 1: {dir_results['horizon']}-day directional model (XGBoost + LogReg)</h2>
<p>Mean OOS accuracy: <b>{dir_results['mean_oos_accuracy']:.4f}</b> ({len(dir_results['fold_df'])} folds)</p>
{fold_table}
<p>Backtest (OOS, threshold 0.55/0.45):</p>
<pre>{json.dumps(dir_results['stats'], indent=2, default=str)}</pre>
<p>Total OOS signals: {dir_results['n_signals']}</p>

<h2>Experiment 2: Cointegration screen (EUR/USD vs other majors)</h2>
{coint_table}

<h2>Experiment 3: Pairs trading on every cointegrated spread (p &lt; 0.10)</h2>
<pre>{json.dumps([r['stats'] for r in pairs_runs] or 'no cointegrated pairs', indent=2, default=str)}</pre>

<h2>Equity curves</h2>
{equity_html}
</body></html>
"""
    HTML_PATH.write_text(html)
    logger.info("Wrote HTML report to {}", HTML_PATH)

    # --- Telegram summary --------------------------------------------------
    pair_lines = "\n".join(
        f"  {r['stats']['pair']}: Sharpe <b>{r['stats']['sharpe']:.2f}</b> "
        f"MaxDD <b>{r['stats']['max_drawdown']:.2%}</b> "
        f"PF <b>{r['stats']['profit_factor']:.2f}</b> "
        f"Tot <b>{r['stats']['total_return']:.2%}</b>"
        for r in pairs_runs
    ) or "  (none cointegrated)"
    headline = (
        "📊 <b>EUR/USD alt research done</b>\n\n"
        f"<b>20d directional</b>:\n"
        f"  Mean OOS acc: <b>{dir_results['mean_oos_accuracy']:.4f}</b>\n"
        f"  OOS Sharpe: <b>{dir_results['stats'].get('sharpe_ratio', 0):.2f}</b>\n"
        f"  OOS PF: <b>{dir_results['stats'].get('profit_factor', 0):.2f}</b>\n\n"
        f"<b>Pairs stat-arb</b>:\n{pair_lines}\n\n"
        f"Elapsed: {time.time()-t0:.0f}s"
    )
    _telegram(headline)

    print("\n=== HTML report:", HTML_PATH)
    print("=== 20d directional:", json.dumps(dir_results['stats'], indent=2, default=str))
    print("=== Cointegration:\n", coint_df.to_string(index=False))
    for run in pairs_runs:
        print(f"=== Pairs {run['stats']['pair']}:",
              json.dumps(run['stats'], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
