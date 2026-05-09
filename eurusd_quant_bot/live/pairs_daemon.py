"""Paper-trading daemon for the cointegration pairs strategy.

The daemon:
  1. Loads tuned parameters from ``config/pairs_tuned.json``.
  2. Wakes up once a day, pulls the latest D1 OHLCV history for the leg pair
     from Dukascopy (free, no auth).
  3. Runs ``PairsTradingStrategy`` on the full window and reads the *current*
     position (-1 / 0 / +1) from the last bar.
  4. Updates the in-memory paper book: opens a new spread trade when the
     position flips from 0 to +/-1, closes it when it flips back to 0.
  5. Persists state to JSON on disk so restarts are safe.
  6. Sends a Telegram daily summary with current equity, today's PnL,
     open trade details, and the latest z-score.

The daemon does NOT connect to a real broker.  All fills are at the closing
price of the most recent D1 bar.  Transaction costs (4 bps per direction
change) are subtracted from PnL.

Run as a long-lived process (Python ``main`` loop) or via the docker-compose
``pairs-daemon`` service.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..backtest.pairs_engine import run_pairs_backtest
from ..data import clean_ohlcv, fetch_dukascopy
from ..strategy.pairs_trading import PairsTradingStrategy
from .telegram_notify import TelegramNotifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TUNED_PATH = REPO_ROOT / "config" / "pairs_tuned.json"
DEFAULT_STATE_PATH = REPO_ROOT / "data" / "paper_state.json"


@dataclass
class PaperTrade:
    pair: str
    direction: int  # +1 long spread, -1 short
    entry_a: float
    entry_b: float
    entry_beta: float
    entry_time: str
    pnl_pct: float = 0.0


@dataclass
class PaperState:
    equity: float = 10_000.0
    starting_equity: float = 10_000.0
    # one open trade per pair, keyed by the pair label
    open_trades: dict[str, PaperTrade] = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)
    last_run_utc: str | None = None
    daily_pnl_history: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "starting_equity": self.starting_equity,
            "open_trades": {k: asdict(v) for k, v in self.open_trades.items()},
            "closed_trades": self.closed_trades,
            "last_run_utc": self.last_run_utc,
            "daily_pnl_history": self.daily_pnl_history,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PaperState:
        # Back-compat: support old single open_trade format
        open_trades: dict[str, PaperTrade] = {}
        if "open_trades" in raw:
            for k, v in (raw.get("open_trades") or {}).items():
                if v:
                    open_trades[k] = PaperTrade(**v)
        elif raw.get("open_trade"):
            ot = PaperTrade(**raw["open_trade"])
            open_trades[ot.pair] = ot
        return cls(
            equity=raw.get("equity", 10_000.0),
            starting_equity=raw.get("starting_equity", 10_000.0),
            open_trades=open_trades,
            closed_trades=raw.get("closed_trades", []),
            last_run_utc=raw.get("last_run_utc"),
            daily_pnl_history=raw.get("daily_pnl_history", {}),
        )


def _split_pair_label(label: str) -> tuple[str, str]:
    """``EUR/USD vs EUR/JPY`` -> (``EUR/USD``, ``EUR/JPY``)."""
    parts = [p.strip() for p in label.split("vs")]
    if len(parts) != 2:
        raise ValueError(f"Cannot parse pair label {label!r}")
    return parts[0], parts[1]


def _load_state(path: Path) -> PaperState:
    if not path.exists():
        return PaperState()
    try:
        return PaperState.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Could not parse {} ({}); resetting state", path, exc)
        return PaperState()


def _save_state(state: PaperState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, default=str))


def _load_tuned_params(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tuned-params file not found: {path}.\n"
                                f"Run scripts/run_pairs_research.py first.")
    return json.loads(path.read_text())


def _fetch_pair(pair_a: str, pair_b: str, lookback_days: int = 800,
                granularity: str = "D1") -> tuple[pd.Series, pd.Series]:
    """Pull recent OHLCV for both legs of the spread."""
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    start = start_dt.isoformat()
    end = end_dt.isoformat()
    df_a = clean_ohlcv(fetch_dukascopy(pair_a, granularity, start, end))
    df_b = clean_ohlcv(fetch_dukascopy(pair_b, granularity, start, end))
    common = df_a.index.intersection(df_b.index)
    return df_a.loc[common, "close"], df_b.loc[common, "close"]


def _step_one_pair(label: str, params: dict, state: PaperState,
                    notifier: TelegramNotifier, *,
                    lookback_days: int = 800,
                    cost_bps: float = 4.0,
                    ) -> dict[str, Any]:
    """Run the strategy once for a single pair and update the state."""
    pair_a, pair_b = _split_pair_label(label)
    a_close, b_close = _fetch_pair(pair_a, pair_b, lookback_days=lookback_days)
    if len(a_close) < 200:
        raise RuntimeError(f"Not enough history for {label}")

    # Strip non-strategy fields (best_mode, hedge_mode is OK)
    cleaned = {k: v for k, v in params.items()
               if k in ("z_entry", "z_exit", "z_lookback", "hedge_mode",
                        "hedge_window", "kalman_obs_var", "kalman_trans_var")}
    strat = PairsTradingStrategy(**cleaned)
    sig = strat.fit_predict(a_close, b_close)
    res = run_pairs_backtest(a_close, b_close, sig.position, sig.beta,
                              cost_bps=cost_bps,
                              starting_equity=state.starting_equity)

    # Position decision uses the *latest* bar
    cur_pos = int(sig.position.iloc[-1])
    cur_z = float(sig.z.iloc[-1])
    cur_beta = float(sig.beta.iloc[-1])
    last_a = float(a_close.iloc[-1])
    last_b = float(b_close.iloc[-1])
    last_eq = float(res.equity.iloc[-1])
    today = a_close.index[-1].strftime("%Y-%m-%d")

    # Reconcile open trade with current desired position (per pair)
    open_t = state.open_trades.get(label)
    if open_t is None and cur_pos != 0:
        # Open a new spread trade
        state.open_trades[label] = PaperTrade(
            pair=label, direction=cur_pos,
            entry_a=last_a, entry_b=last_b, entry_beta=cur_beta,
            entry_time=today,
        )
        notifier.send(
            f"🟢 OPEN spread {label} | dir={'LONG' if cur_pos>0 else 'SHORT'} "
            f"| z={cur_z:+.2f} | a={last_a:.5f} b={last_b:.5f} beta={cur_beta:.4f}"
        )
    elif open_t is not None and (cur_pos == 0 or cur_pos != open_t.direction):
        # Close existing trade (and if cur_pos != 0, open the new one)
        d = open_t.direction
        pnl_a = (last_a - open_t.entry_a) / open_t.entry_a
        pnl_b = (last_b - open_t.entry_b) / open_t.entry_b
        gross = d * (pnl_a - open_t.entry_beta * pnl_b)
        # Approximate round-trip cost
        net = gross - 2 * (cost_bps / 1e4)
        state.closed_trades.append({
            "pair": label, "entry": open_t.entry_time, "exit": today,
            "direction": d, "pnl_pct": float(net),
        })
        emoji = "💰" if net > 0 else "💔"
        notifier.send(f"{emoji} CLOSE {label} dir={d:+d} | PnL {net:+.2%}")
        state.open_trades.pop(label, None)
        if cur_pos != 0:
            state.open_trades[label] = PaperTrade(
                pair=label, direction=cur_pos,
                entry_a=last_a, entry_b=last_b, entry_beta=cur_beta,
                entry_time=today,
            )
            notifier.send(
                f"🔁 FLIP {label} | new dir={'LONG' if cur_pos>0 else 'SHORT'} "
                f"| z={cur_z:+.2f}"
            )

    # Update equity from backtest re-run (deterministic on history).  We do
    # NOT compound equity across pairs in this simple paper book; each pair
    # contributes a *snapshot* of its own backtest equity at this instant.
    # For a multi-pair portfolio book the user would need a portfolio engine.
    state.equity = last_eq

    return {
        "pair": label,
        "z": cur_z,
        "beta": cur_beta,
        "position": cur_pos,
        "open_trade": asdict(state.open_trades.get(label)) if state.open_trades.get(label) else None,
        "equity": last_eq,
        "today": today,
    }


def run_once(*, tuned_path: Path = DEFAULT_TUNED_PATH,
              state_path: Path = DEFAULT_STATE_PATH,
              cost_bps: float = 4.0) -> dict[str, Any]:
    """One full strategy update + summary message.  Returns a status dict."""
    notifier = TelegramNotifier()
    state = _load_state(state_path)
    tuned = _load_tuned_params(tuned_path)

    pair_status: list[dict[str, Any]] = []
    for label, conf in tuned.items():
        try:
            status = _step_one_pair(label, conf["params"], state, notifier,
                                     cost_bps=cost_bps)
            pair_status.append(status)
        except Exception as exc:
            logger.exception("Pair {} step failed: {}", label, exc)
            notifier.send(f"⚠️ Pair {label} step failed: {exc}")

    # Daily summary
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = state.equity - state.starting_equity
    state.daily_pnl_history[today] = daily_pnl
    state.last_run_utc = datetime.now(timezone.utc).isoformat()

    n_open = sum(1 for s in pair_status if s.get("open_trade"))
    n_closed = len(state.closed_trades)
    summary_lines = [
        "📊 <b>Pairs paper run</b>",
        f"Equity: <b>${state.equity:,.2f}</b> (start ${state.starting_equity:,.2f}, "
        f"PnL ${daily_pnl:+,.2f})",
        f"Open trades: <b>{n_open}</b> | Closed all-time: <b>{n_closed}</b>",
    ]
    for s in pair_status:
        ot = s.get("open_trade")
        otline = (f"open dir={ot['direction']:+d}, entry={ot['entry_time']}"
                  if ot else "flat")
        summary_lines.append(
            f"  {s['pair']}: z={s['z']:+.2f} pos={s['position']:+d} ({otline})"
        )
    notifier.send("\n".join(summary_lines))

    _save_state(state, state_path)
    logger.info("Run done: equity=${:.2f}, open={}", state.equity, n_open)
    return {"equity": state.equity, "pairs": pair_status,
            "daily_pnl": daily_pnl}


def run_forever(*, interval_seconds: int = 24 * 3600,
                 tuned_path: Path = DEFAULT_TUNED_PATH,
                 state_path: Path = DEFAULT_STATE_PATH,
                 cost_bps: float = 4.0) -> None:
    """Long-lived loop.  Exits on SIGTERM/SIGINT cleanly."""
    stop = {"flag": False}

    def _sig_handler(*_a):
        logger.info("Signal received, shutting down")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    logger.info("Starting paper daemon (interval={} s)", interval_seconds)
    notifier = TelegramNotifier()
    notifier.send("🤖 <b>Pairs paper daemon started</b>")
    while not stop["flag"]:
        try:
            run_once(tuned_path=tuned_path, state_path=state_path, cost_bps=cost_bps)
        except Exception as exc:
            logger.exception("Daemon iteration failed: {}", exc)
            notifier.send(f"⚠️ Daemon iteration failed: {exc}")
        # Sleep in 1-second slices so SIGTERM is responsive
        for _ in range(interval_seconds):
            if stop["flag"]:
                break
            time.sleep(1)
    notifier.send("🛑 <b>Pairs paper daemon stopped</b>")


def _main() -> int:
    interval = int(os.getenv("PAIRS_DAEMON_INTERVAL_SECONDS", str(24 * 3600)))
    once = os.getenv("PAIRS_DAEMON_ONCE", "0") == "1"
    if once:
        run_once()
        return 0
    run_forever(interval_seconds=interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
