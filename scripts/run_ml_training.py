"""ML ensemble walk-forward training + OOS backtest.

Pipeline:
  1. Pull EUR/USD H1 from Dukascopy (2015-present) and FRED macro bundle.
  2. Build the full feature matrix (~108 features).
  3. Walk-forward train XGBoost + LogReg ensemble on 24m IS / 6m OOS folds.
  4. Stitch OOS predictions into a single series and run the
     :class:`MLEnsembleStrategy` on top of it.
  5. Backtest the resulting OOS signal end-to-end.
  6. Combine the ML signal with the tuned MR-D1 signal in the
     :class:`PortfolioAllocator` and backtest the combined equity curve.
  7. Telegram + HTML report.

Notes:
  * LSTM is skipped automatically if TensorFlow isn't installed (it is not in
    this venv).  XGBoost + LogReg already give us a robust ensemble.
  * OOS-only evaluation is critical -- we never look at IS results to gauge
    edge.
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

from eurusd_quant_bot.backtest import run_backtest, run_monte_carlo
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
from eurusd_quant_bot.strategy import MeanReversionStrategy, MLEnsembleStrategy

REPORT_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
HTML_PATH = REPORTS_DIR / f"ml_training_report_{REPORT_TS}.html"


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


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    accuracy: float


def fold_indices(idx: pd.DatetimeIndex, train_months: int, test_months: int):
    folds = []
    start = idx[0].normalize()
    end = idx[-1].normalize()
    train_end = start + pd.DateOffset(months=train_months)
    while train_end + pd.DateOffset(months=test_months) <= end:
        test_end = train_end + pd.DateOffset(months=test_months)
        folds.append((start, train_end, train_end, test_end))
        train_end = test_end
    return folds


def main() -> int:
    t0 = time.time()
    logger.info("=== ML walk-forward training starting ===")
    _telegram("🟢 EUR/USD bot: ML walk-forward training started.")

    # --- Data ----------------------------------------------------------------
    df = clean_ohlcv(fetch_dukascopy(granularity="H1", start="2015-01-01", end="2025-05-01"))
    macro = fetch_fred_bundle()
    macro_aligned = align_macro(df, macro) if not macro.empty else None
    logger.info("H1 bars: {}, macro shape: {}", len(df), macro.shape if macro is not None else None)

    # --- Features ------------------------------------------------------------
    features = build_features(df, macro=macro_aligned)
    target = make_target(features["close"], horizon=4)
    full = features.join(target).dropna(subset=["target"])
    logger.info("Feature matrix: {} rows x {} cols", *full.shape)

    # CRITICAL: build feature list from the FEATURES frame (no target) so we
    # never include the label as a predictor.
    cols = feature_names(features)
    X = full[cols].astype(float)
    # Drop NaN/inf rows
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    full = full.loc[X.index]
    y = full["target"].astype(int)
    assert "target" not in X.columns, "Target leaked into features"

    X = drop_correlated(X, threshold=0.9)
    logger.info("Post-correlation pruning: {} cols", X.shape[1])

    # --- Walk-forward training ----------------------------------------------
    folds = fold_indices(X.index, train_months=24, test_months=6)
    logger.info("Total folds: {}", len(folds))

    fold_results: list[FoldResult] = []
    oos_proba = pd.Series(np.nan, index=X.index, name="p_up")

    for i, (ts0, ts1, ts2, ts3) in enumerate(folds):
        train_mask = (X.index >= ts0) & (X.index < ts1)
        test_mask = (X.index >= ts2) & (X.index < ts3)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if len(X_train) < 1000 or len(X_test) < 100:
            logger.info("Skipping fold {} (insufficient data)", i)
            continue

        # Ensemble with only XGB+LogReg (LSTM skipped if TF missing)
        model = EnsembleModel(weights={"xgb": 0.6, "logreg": 0.4})
        model.fit(X_train, y_train, members=["xgb", "logreg"])

        proba = model.predict_proba(X_test)
        p_up = proba[:, 1]
        preds = (p_up > 0.5).astype(int)
        acc = float(np.mean(preds == y_test.values))
        fold_results.append(FoldResult(
            train_start=ts0, train_end=ts1, test_start=ts2, test_end=ts3,
            n_train=len(X_train), n_test=len(X_test), accuracy=acc,
        ))
        oos_proba.loc[X_test.index] = p_up
        logger.info("Fold {}: train={}, test={}, acc={:.4f}",
                    i, len(X_train), len(X_test), acc)

    fold_df = pd.DataFrame([{
        "train_start": f.train_start, "train_end": f.train_end,
        "test_start": f.test_start, "test_end": f.test_end,
        "n_train": f.n_train, "n_test": f.n_test, "accuracy": f.accuracy,
    } for f in fold_results])
    logger.info("\n{}", fold_df.to_string(index=False))

    mean_acc = fold_df["accuracy"].mean() if not fold_df.empty else 0.0
    logger.info("Mean OOS accuracy across folds: {:.4f}", mean_acc)

    # --- Save predictions ----------------------------------------------------
    from eurusd_quant_bot.config.settings import ARTIFACT_DIR
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"p_up": oos_proba}).dropna()
    pred_path = ARTIFACT_DIR / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("Wrote {} OOS predictions to {}", len(pred_df), pred_path)

    # --- Backtest ML strategy on OOS predictions -----------------------------
    ml_strategy = MLEnsembleStrategy(cached_predictions_path=str(pred_path))
    # Limit to the OOS window (skip warmup)
    df_oos = df.loc[pred_df.index.min():pred_df.index.max()]
    ml_signals = ml_strategy.generate_signals(df_oos)
    logger.info("ML signals: {} non-zero bars (long={}, short={})",
                int((ml_signals != 0).sum()),
                int((ml_signals > 0).sum()),
                int((ml_signals < 0).sum()))

    ml_bt = run_backtest(df_oos, ml_signals)
    ml_stats = ml_bt.stats.to_dict()
    logger.info("ML OOS backtest stats: {}", ml_stats)

    # Monte Carlo on ML trades
    ml_mc_summary: dict | str = "skipped"
    if len(ml_bt.trades) > 5:
        mc = run_monte_carlo(ml_bt.trades, n_runs=1000)
        ml_mc_summary = mc.summary()

    # --- Combined portfolio (MR-D1 tuned + ML ensemble) ---------------------
    df_d1 = clean_ohlcv(fetch_dukascopy(granularity="D1", start="2015-01-01", end="2025-05-01"))
    mr_strategy = MeanReversionStrategy(
        z_score_period=48,
        z_score_entry=1.78,
        rsi_period=16,
        bb_period=18,
        bb_std=1.94,
    )
    mr_signals = mr_strategy.generate_signals(df_d1)
    mr_bt = run_backtest(df_d1, mr_signals)
    mr_stats = mr_bt.stats.to_dict()
    logger.info("MR-D1 TUNED stats: {}", mr_stats)

    # Use the portfolio allocator on the equity curves of each strategy.
    # Resample equity to a common frequency (D1) for comparison.
    ml_equity_d1 = ml_bt.equity.resample("D").last().ffill()
    mr_equity_d1 = mr_bt.equity.reindex(ml_equity_d1.index).ffill()

    common_equity = pd.concat({"ml_ensemble": ml_equity_d1, "mr_d1_tuned": mr_equity_d1}, axis=1).dropna()
    daily_returns = common_equity.pct_change(fill_method=None).fillna(0.0)
    # Risk parity: weight ~ 1 / std
    stds = daily_returns.std()
    inv_vol = 1.0 / stds.replace(0, np.nan)
    weights = (inv_vol / inv_vol.sum()).fillna(0.5)
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    portfolio_equity = (1 + portfolio_returns).cumprod() * 10_000.0
    portfolio_stats = {
        "weights": weights.to_dict(),
        "annual_return": (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) ** (252 / max(len(portfolio_equity), 1)) - 1,
        "sharpe": float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() else 0.0,
        "max_drawdown": float((portfolio_equity / portfolio_equity.cummax() - 1).min()),
        "total_return": float(portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1),
    }
    logger.info("Portfolio stats: {}", portfolio_stats)

    # The formal allocator expects all 3 default strategies (mean_reversion,
    # trend_following, ml_ensemble); we only have 2 here.  Use the simple
    # risk-parity weights computed above as the headline.
    allocator_summary = json.dumps({
        "note": "Formal PortfolioAllocator skipped (default config expects 3 strategies). Headline = risk-parity weights above.",
    }, indent=2)

    # --- HTML report ---------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml_bt.equity.index, y=ml_bt.equity.values, name="ML ensemble (OOS)"))
    fig.add_trace(go.Scatter(x=mr_bt.equity.index, y=mr_bt.equity.values, name="MR-D1 tuned"))
    fig.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity.values, name="Portfolio (risk parity)"))
    fig.update_layout(template="plotly_dark", title="Equity curves", height=540,
                      xaxis_title="Time", yaxis_title="Equity ($)")
    equity_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    fold_table = fold_df.to_html(index=False, classes="metrics", border=0) if not fold_df.empty else "(no folds)"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ML training report</title>
<style>
body {{ background:#0e1117; color:#e6e6e6; font-family:system-ui,sans-serif; margin:32px; }}
h1, h2 {{ color:#fafafa; }}
table.metrics {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.metrics th, table.metrics td {{ padding:6px 10px; border-bottom:1px solid #2a2f3a; text-align:right; }}
table.metrics th:first-child, table.metrics td:first-child {{ text-align:left; }}
pre {{ background:#1e222a; padding:14px; border-radius:8px; overflow:auto; font-size:12px; }}
</style></head><body>
<h1>EUR/USD ML training report</h1>
<p>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} | Elapsed {time.time()-t0:.0f}s</p>

<h2>Walk-forward folds</h2>
{fold_table}
<p>Mean OOS accuracy: <b>{mean_acc:.4f}</b> ({len(fold_df)} folds)</p>

<h2>ML ensemble OOS backtest</h2>
<pre>{json.dumps(ml_stats, indent=2, default=str)}</pre>

<h2>ML Monte Carlo</h2>
<pre>{json.dumps(ml_mc_summary, indent=2, default=str)}</pre>

<h2>MR D1 (tuned) reference</h2>
<pre>{json.dumps(mr_stats, indent=2, default=str)}</pre>

<h2>Combined portfolio (risk parity)</h2>
<pre>{json.dumps(portfolio_stats, indent=2, default=str)}</pre>

<h2>Portfolio allocator (formal)</h2>
<pre>{allocator_summary}</pre>

<h2>Equity curves</h2>
{equity_html}
</body></html>
"""
    HTML_PATH.write_text(html)
    logger.info("Wrote HTML report to {}", HTML_PATH)

    headline = (
        "📊 <b>EUR/USD ML training done</b>\n"
        f"Mean OOS accuracy: <b>{mean_acc:.4f}</b> across {len(fold_df)} folds\n"
        f"ML OOS Sharpe: <b>{ml_stats.get('sharpe_ratio', 0):.2f}</b>\n"
        f"ML OOS MaxDD: <b>{ml_stats.get('max_drawdown', 0):.2%}</b>\n"
        f"ML OOS PF: <b>{ml_stats.get('profit_factor', 0):.2f}</b>\n"
        f"ML OOS trades: <b>{ml_stats.get('total_trades', 0)}</b>\n\n"
        f"Portfolio (MR-D1 + ML, risk parity):\n"
        f"  Sharpe {portfolio_stats['sharpe']:.2f}, "
        f"MaxDD {portfolio_stats['max_drawdown']:.2%}, "
        f"Total {portfolio_stats['total_return']:.2%}\n\n"
        f"Elapsed: {time.time()-t0:.0f}s"
    )
    _telegram(headline)

    print("\n=== HTML report:", HTML_PATH)
    print("=== ML stats:", json.dumps(ml_stats, indent=2, default=str))
    print("=== Portfolio:", json.dumps(portfolio_stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
