"""Flask + Plotly dashboard.

Pages:
    - ``/``           : equity curve, daily PnL, open positions
    - ``/trades``     : trade table with filters
    - ``/strategy``   : per-strategy metrics
    - ``/risk``       : current exposure, drawdown, position sizes
    - ``/settings``   : view current settings (read-only)
    - ``/health``     : JSON health-check endpoint (used by Docker)
    - ``/api/equity`` : JSON time-series for the equity chart

The dashboard reads from the same Postgres tables that the trader writes to.
When the database is unavailable (e.g. in tests) the views fall back to
demo data generated from yfinance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template
from loguru import logger
from plotly.utils import PlotlyJSONEncoder

from ..config import get_settings
from ..live.health_check import HealthState

app = Flask(__name__, template_folder="templates", static_folder="static")
HEALTH = HealthState()


# ---------------------------------------------------------------------------
# Data accessors -- isolated so they can be swapped for tests / fixtures
# ---------------------------------------------------------------------------

def _load_equity() -> pd.DataFrame:
    try:
        from sqlalchemy import text

        from ..data.database import Database

        db = Database()
        with db.connect() as conn:
            df = pd.read_sql(text("SELECT ts, equity, drawdown FROM equity ORDER BY ts"), conn)
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                return df
    except Exception as exc:
        logger.debug("Equity from DB unavailable: {}", exc)

    # Fallback synthetic curve so the dashboard renders without a DB.
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    eq = 10_000 * (1 + pd.Series(range(len(idx)), index=idx) * 0.0001)
    dd = (eq / eq.cummax() - 1)
    return pd.DataFrame({"ts": idx, "equity": eq.values, "drawdown": dd.values})


def _load_trades() -> pd.DataFrame:
    try:
        from sqlalchemy import text

        from ..data.database import Database

        db = Database()
        with db.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 500"), conn)
            return df
    except Exception as exc:
        logger.debug("Trades from DB unavailable: {}", exc)
    return pd.DataFrame(columns=["entry_time", "exit_time", "strategy",
                                 "direction", "pnl", "pnl_pips", "lot_size"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health() -> dict:
    HEALTH.heartbeat()
    return jsonify(HEALTH.to_dict())


@app.route("/api/equity")
def api_equity() -> dict:
    df = _load_equity()
    return jsonify({
        "ts": df["ts"].astype(str).tolist(),
        "equity": df["equity"].tolist(),
        "drawdown": df["drawdown"].tolist(),
    })


def _equity_figure(df: pd.DataFrame) -> dict:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["equity"], mode="lines",
                             name="Equity", line={"color": "#1f77b4"}))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["drawdown"], mode="lines",
                             name="Drawdown", yaxis="y2",
                             line={"color": "#d62728", "dash": "dot"}))
    fig.update_layout(
        title="Equity curve and drawdown",
        xaxis={"title": "Time"},
        yaxis={"title": "Equity (USD)"},
        yaxis2={"title": "Drawdown", "overlaying": "y", "side": "right",
                "tickformat": ",.1%"},
        legend={"x": 0.0, "y": 1.1},
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


@app.route("/")
def index() -> str:
    df = _load_equity()
    fig_json = _equity_figure(df)
    settings = get_settings()
    context = {
        "fig_json": fig_json,
        "now": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "mode": settings.runtime.mode,
            "max_daily_loss": f"{settings.risk.max_daily_loss:.0%}",
            "max_total_drawdown": f"{settings.risk.max_total_drawdown:.0%}",
            "risk_per_trade": f"{settings.risk.risk_per_trade:.0%}",
        },
    }
    return render_template("index.html", **context)


@app.route("/trades")
def trades() -> str:
    df = _load_trades()
    rows = df.head(200).to_dict(orient="records")
    return render_template("trades.html", rows=rows)


@app.route("/risk")
def risk() -> str:
    settings = asdict(get_settings().risk)
    return render_template("risk.html", risk=settings)


@app.route("/settings")
def settings_view() -> str:
    s = get_settings()
    flat = {
        "runtime": asdict(s.runtime),
        "instrument": asdict(s.instrument),
        "risk": asdict(s.risk),
        "backtest": asdict(s.backtest),
        "schedule": asdict(s.schedule),
    }
    return render_template("settings.html", settings=flat)


def main() -> None:  # pragma: no cover -- entry point for `python -m`
    settings = get_settings().runtime
    app.run(host=settings.dashboard_host, port=settings.dashboard_port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()
