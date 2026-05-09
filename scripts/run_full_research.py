"""End-to-end validation runner (Dukascopy + FRED + walk-forward).

Steps:
  1. Pull EUR/USD H1 and D1 from Dukascopy (2015 -> present).
  2. Pull the configured FRED macro bundle (DFF / ECBDFR / CPIAUCSL / PAYEMS /
     DTWEXBGS / VIXCLS) and align it to the price index.
  3. Run Mean Reversion + Trend Following backtests on H1 and D1.
  4. Run walk-forward Optuna (50 trials) on the best strategy + granularity.
  5. Run Monte Carlo on the walk-forward best.
  6. Run a 30s paper-trading dry run against the in-memory mock broker.
  7. Write an HTML report with equity curves and stats, push a Telegram
     summary, and print the report path.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
from loguru import logger

from eurusd_quant_bot.backtest import run_backtest, run_monte_carlo, walk_forward
from eurusd_quant_bot.config.settings import REPORTS_DIR
from eurusd_quant_bot.data import (
    align_macro,
    clean_ohlcv,
    fetch_dukascopy,
    fetch_fred_bundle,
)
from eurusd_quant_bot.live import LiveTrader, TraderConfig
from eurusd_quant_bot.ml.features import build_features
from eurusd_quant_bot.strategy import (
    STRATEGY_REGISTRY,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)

REPORT_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
HTML_PATH = REPORTS_DIR / f"research_report_{REPORT_TS}.html"


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


def _backtest(label: str, df: pd.DataFrame, strategy) -> dict:
    signals = strategy.generate_signals(df)
    bt = run_backtest(df, signals)
    stats = bt.stats.to_dict()
    logger.info("[{}] {}", label, stats)
    return {
        "label": label,
        "stats": stats,
        "n_bars": len(df),
        "n_trades": len(bt.trades),
        "equity": bt.equity,
        "trades": bt.trades,
    }


def _equity_html(results: list[dict]) -> str:
    fig = go.Figure()
    for r in results:
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=r["label"], mode="lines"))
    fig.update_layout(
        template="plotly_dark",
        title="Equity curves (Dukascopy backtests)",
        height=520,
        xaxis_title="Time",
        yaxis_title="Equity ($)",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _stats_table(results: list[dict]) -> str:
    rows = []
    for r in results:
        s = r["stats"]
        rows.append({
            "strategy": r["label"],
            "n_bars": r["n_bars"],
            "n_trades": r["n_trades"],
            **{k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in s.items()},
        })
    df = pd.DataFrame(rows)
    return df.to_html(index=False, classes="metrics", border=0)


def _write_html(results: list[dict], extras: dict[str, str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>EUR/USD research report (v2)</title>
<style>
body {{ background:#0e1117; color:#e6e6e6; font-family:system-ui,sans-serif; margin:32px; }}
h1, h2 {{ color:#fafafa; }}
table.metrics {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.metrics th, table.metrics td {{ padding:6px 10px; border-bottom:1px solid #2a2f3a; text-align:right; }}
table.metrics th:first-child, table.metrics td:first-child {{ text-align:left; }}
pre {{ background:#1e222a; padding:14px; border-radius:8px; overflow:auto; font-size:12px; }}
</style></head><body>
<h1>EUR/USD Quant Trading Bot — research report (v2: Dukascopy + FRED)</h1>
<p>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>

<h2>Data sources</h2>
<pre>{extras.get('data_summary', '')}</pre>

<h2>Backtest summary</h2>
{_stats_table(results)}

<h2>Equity curves</h2>
{_equity_html(results)}

<h2>Walk-forward optimization (Optuna, 50 trials)</h2>
<pre>{extras.get('walkforward', 'skipped')}</pre>

<h2>Tuned-strategy backtest</h2>
<pre>{extras.get('tuned_backtest', 'skipped')}</pre>

<h2>Monte Carlo (best strategy)</h2>
<pre>{extras.get('montecarlo', 'skipped')}</pre>

<h2>Paper trading dry run</h2>
<pre>{extras.get('paper', 'skipped')}</pre>
</body></html>
"""
    HTML_PATH.write_text(html)
    logger.info("Wrote HTML report to {}", HTML_PATH)


async def _paper_dry_run(seconds: int = 30) -> str:
    cfg = TraderConfig(strategy_name="mean_reversion", use_mock=True, poll_seconds=2)
    trader = LiveTrader(cfg)
    task = asyncio.create_task(trader.run())
    await asyncio.sleep(seconds)
    trader.stop()
    try:
        await asyncio.wait_for(task, timeout=10)
    except asyncio.TimeoutError:
        logger.warning("Paper task did not exit within 10s")
    return json.dumps(trader.state.health.to_dict(), indent=2, default=str)


def main() -> int:
    t0 = time.time()
    logger.info("=== Research run v2 starting (Dukascopy + FRED) ===")
    _telegram("🟢 EUR/USD bot: v2 research run started (Dukascopy + FRED).")

    # --- Data ----------------------------------------------------------------
    df_h1 = clean_ohlcv(fetch_dukascopy(granularity="H1", start="2015-01-01", end="2025-05-01"))
    df_d1 = clean_ohlcv(fetch_dukascopy(granularity="D1", start="2015-01-01", end="2025-05-01"))
    macro = fetch_fred_bundle()
    macro_aligned_h1 = align_macro(df_h1, macro) if not macro.empty else None
    macro_aligned_d1 = align_macro(df_d1, macro) if not macro.empty else None

    data_summary = (
        f"H1: {len(df_h1)} bars from {df_h1.index[0]} to {df_h1.index[-1]}\n"
        f"D1: {len(df_d1)} bars from {df_d1.index[0]} to {df_d1.index[-1]}\n"
        f"FRED: {macro.shape[0]} rows x {macro.shape[1]} cols\n"
        f"FRED columns: {list(macro.columns)}\n"
    )
    logger.info("\n{}", data_summary)

    # Feature build (sanity-check counts; not required for the rules-based
    # backtest but exercises the macro alignment path).
    feats_h1 = build_features(df_h1, macro=macro_aligned_h1)
    feats_d1 = build_features(df_d1, macro=macro_aligned_d1)
    data_summary += (
        f"H1 features: {feats_h1.shape}\n"
        f"D1 features: {feats_d1.shape}\n"
    )

    # --- Backtests on default params ----------------------------------------
    results: list[dict] = [
        _backtest("MR D1 2015-2025", df_d1, MeanReversionStrategy()),
        _backtest("TF D1 2015-2025", df_d1, TrendFollowingStrategy()),
        _backtest("MR H1 2015-2025", df_h1, MeanReversionStrategy()),
        _backtest("TF H1 2015-2025", df_h1, TrendFollowingStrategy()),
    ]

    # --- Walk-forward Optuna -------------------------------------------------
    # 50 trials on D1 mean reversion (D1 has 10y of history, suitable for
    # 24m/6m walk-forward).
    wf_summary = "skipped"
    tuned_summary = "skipped"
    try:
        logger.info("Running walk-forward optimization (D1 MR, 50 trials)...")
        wf = walk_forward(df_d1, STRATEGY_REGISTRY["mean_reversion"], n_trials=50)
        wf_summary = (
            wf.summary.to_string(index=False)
            + "\n\nMedian params:\n"
            + json.dumps(wf.median_params, indent=2, default=str)
        )
        # Re-run backtest with tuned params for the report
        tuned_strategy = STRATEGY_REGISTRY["mean_reversion"](**wf.median_params)
        tuned_signals = tuned_strategy.generate_signals(df_d1)
        tuned_bt = run_backtest(df_d1, tuned_signals)
        tuned_summary = json.dumps(tuned_bt.stats.to_dict(), indent=2, default=str)
        results.append({
            "label": "MR D1 TUNED",
            "stats": tuned_bt.stats.to_dict(),
            "n_bars": len(df_d1),
            "n_trades": len(tuned_bt.trades),
            "equity": tuned_bt.equity,
            "trades": tuned_bt.trades,
        })
    except Exception as exc:
        wf_summary = f"walk-forward failed: {exc}"
        logger.exception("walk-forward error")

    # --- Monte Carlo on the best by trade count -----------------------------
    best = max(results, key=lambda r: r["n_trades"])
    mc_summary = "skipped"
    if best["n_trades"] > 5:
        mc = run_monte_carlo(best["trades"], n_runs=1000)
        mc_summary = json.dumps(mc.summary(), indent=2, default=str)
        logger.info("Monte Carlo summary: {}", mc_summary)

    # --- Paper trading dry run ----------------------------------------------
    paper_summary = "skipped"
    try:
        paper_summary = asyncio.run(_paper_dry_run(seconds=30))
        logger.info("Paper dry-run health: {}", paper_summary)
    except Exception as exc:
        paper_summary = f"paper dry-run failed: {exc}"
        logger.exception("paper error")

    _write_html(results, extras={
        "data_summary": data_summary,
        "walkforward": wf_summary,
        "tuned_backtest": tuned_summary,
        "montecarlo": mc_summary,
        "paper": paper_summary,
    })

    headline = (
        "📊 <b>EUR/USD v2 research run done</b>\n"
        f"Best (by #trades): <b>{best['label']}</b>\n"
        f"  trades: {best['n_trades']}\n"
        f"  Sharpe: {best['stats'].get('sharpe_ratio', 0):.2f}\n"
        f"  MaxDD: {best['stats'].get('max_drawdown', 0):.2%}\n"
        f"  Total return: {best['stats'].get('total_return', 0):.2%}\n"
        f"  Profit factor: {best['stats'].get('profit_factor', 0):.2f}\n"
        f"\nElapsed: {time.time()-t0:.1f}s"
        f"\nReport: <code>{HTML_PATH.name}</code>"
    )
    _telegram(headline)

    print("\n=== HTML report:", HTML_PATH)
    print("=== Best strategy:", best["label"])
    print(json.dumps(best["stats"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
