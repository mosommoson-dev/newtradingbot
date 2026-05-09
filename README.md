# EUR/USD Quant Trading Bot

Production-oriented quantitative trading system for the EUR/USD forex pair built around three
independent strategies (mean reversion, trend following, ML ensemble) combined into a
risk-parity portfolio with strict risk management, OANDA v20 execution, full backtesting and
walk-forward optimization, monitoring via Telegram + Flask dashboard, and Docker-based
deployment.

> **Status:** Paper-trading scaffolding only. Do **not** point this at a live OANDA account
> until you have completed at least 30 days of profitable paper trading and verified that the
> kill switches behave as expected.

## Features

- **Strategies**
  - Mean reversion (Z-score + RSI + Bollinger Band %B, ranging-market filter via ADX)
  - Trend following (EMA alignment + MACD + ADX, ATR-based trailing stop)
  - ML ensemble (XGBoost + LSTM + logistic regression baseline, weighted on validation)
  - Risk-parity portfolio allocator with rolling 30-day Sharpe-based weighting
- **Backtesting**
  - Vectorized engine with realistic spread, commission, slippage, and next-bar fills
  - No look-ahead bias (every feature uses only past data)
  - Walk-forward optimization (Optuna, 24m train → 6m OOS, rolling)
  - Monte Carlo robustness (1,000 simulations with shuffled trades, perturbed slippage)
- **Live trading**
  - OANDA v20 client (demo and live endpoints toggled by a single env flag)
  - Order manager with market/limit/stop, SL/TP, ATR trailing stops
  - Risk manager: half-Kelly position sizing, vol-targeting, per-trade / daily / total
    drawdown limits, kill switch
  - Slippage model parameterized on ATR and notional volume
  - News + spread + session filter (avoids 30 min around NFP/FOMC/ECB/CPI)
- **Monitoring**
  - PostgreSQL + TimescaleDB schema for trades, equity, signals, features
  - Telegram notifications on every trade, daily/weekly summaries, kill-switch alerts
  - Flask + Plotly dashboard (equity curve, drawdown, monthly heatmap, rolling Sharpe)
- **Deployment**
  - Dockerfile (non-root user, healthcheck) + docker-compose (bot + Timescale + Redis +
    Grafana)
  - One-shot `run.sh` for build / paper / live / deploy

## Project layout

```
eurusd_quant_bot/
├── config/        # All tunables and broker credentials
├── data/          # Fetchers, preprocessing, Postgres/Timescale schema
├── strategy/      # Base + 3 strategies + portfolio allocator
├── backtest/      # Vectorized engine, metrics, walk-forward, Monte Carlo
├── execution/     # OANDA client, order/risk manager, slippage model
├── live/          # Async trader loop, scheduler, Telegram, health check
├── ml/            # Feature engineering, models, training, prediction
├── dashboard/     # Flask app + Plotly templates
├── tests/         # pytest suite (incl. no-lookahead test)
├── docker/        # Dockerfile + docker-compose.yml
├── notebooks/     # Research notebooks
├── main.py        # Paper ↔ live entry point
├── run.sh         # One-command deploy
├── requirements.txt
└── .env.example
```

## Quickstart

### 1. Local install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit with your credentials
```

> `ta-lib` requires the underlying C library (`apt-get install libta-lib0 libta-lib-dev` on
> Debian/Ubuntu). The bot falls back to the pure-Python `ta` package automatically when
> `ta-lib` is unavailable, so most code paths run without it.

### 2. Run the test suite

```bash
pytest -q
```

### 3. Run a backtest on yfinance EUR/USD data

```bash
python -m eurusd_quant_bot.main --mode backtest --strategy mean_reversion \
    --start 2020-01-01 --end 2024-12-31
```

The HTML report is written to `reports/backtest_<strategy>_<timestamp>.html`.

### 3b. Use Dukascopy for clean H1/D1 history (recommended over yfinance)

`yfinance` H1 is capped at the last 730 days. Dukascopy (free, no auth) returns
10+ years of EUR/USD H1 in seconds:

```python
from eurusd_quant_bot.data import fetch_dukascopy, clean_ohlcv
df = clean_ohlcv(fetch_dukascopy(granularity="H1", start="2015-01-01"))
```

The end-to-end research harness at `scripts/run_full_research.py` pulls
Dukascopy + FRED, runs both strategies, executes a 50-trial Optuna walk-forward,
runs Monte Carlo, and writes a v2 HTML report:

```bash
PYTHONPATH=. python scripts/run_full_research.py
```

### 4. Walk-forward optimization

```bash
python -m eurusd_quant_bot.main --mode walkforward --strategy trend_following \
    --trials 50
```

### 5. Paper trade on OANDA demo

```bash
# .env: OANDA_ENV=practice, OANDA_TOKEN=..., OANDA_ACCOUNT_ID=...
python -m eurusd_quant_bot.main --mode paper --strategy portfolio
```

### 6. Live trade (only after 30+ days of profitable paper trading)

```bash
# .env: OANDA_ENV=live, OANDA_TOKEN=..., OANDA_ACCOUNT_ID=...
python -m eurusd_quant_bot.main --mode live --strategy portfolio
```

### 7. Dashboard

```bash
python -m eurusd_quant_bot.dashboard.app  # http://localhost:5000
```

### 8. Docker

```bash
cd docker && docker compose up -d
```

This brings up the bot, TimescaleDB (Postgres 14), Redis, Grafana, and the
**pairs paper-trading daemon** described below.

### 9. Pairs paper-trading daemon

A long-lived service that runs the cointegration pairs strategy on Dukascopy
D1 data once per day and posts a Telegram daily summary.

**One-time tuning (run one of these before starting the daemon):**

```bash
# (a) Original 2-pair EUR/JPY + EUR/GBP study (rolling-OLS vs Kalman)
PYTHONPATH=. python scripts/run_pairs_research.py

# (b) G10 pair-basket study -- screens 9 majors via Engle-Granger,
# walk-forward tunes each survivor, builds a risk-parity basket and
# persists the *positive-Sharpe-only* legs for the daemon.
PYTHONPATH=. python scripts/run_pairs_basket_research.py
```

Either script writes:
- `config/pairs_tuned.json` — the params the daemon will load
- `reports/pairs_*_report_*.html` — walk-forward + Monte Carlo report

The basket script is what we currently recommend: more legs, better
diversification, higher OOS Sharpe (~0.45 vs ~0.2 for the single pair).

**Start the daemon (runs forever, wakes once per day):**

```bash
cd docker && docker compose up -d pairs-daemon
```

Or directly:

```bash
PYTHONPATH=. python -m eurusd_quant_bot.live.pairs_daemon
```

For a one-shot iteration (no looping):

```bash
PYTHONPATH=. PAIRS_DAEMON_ONCE=1 python -m eurusd_quant_bot.live.pairs_daemon
```

**State persistence:** the daemon writes `data/paper_state.json` (equity,
open trades, closed trades, daily PnL history) so restarts are safe.

**Telegram messages:** on each run the daemon emits a single summary, plus
real-time `OPEN/CLOSE/FLIP` messages whenever a trade fires.  Set
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to enable.

**Honest expectation:** even with the basket diversification trick, OOS
edge is modest (basket Sharpe ~0.4–0.5 with the positive-Sharpe filter,
~0.2 for the single EUR/JPY pair).  Most of the 36 candidate G10
cointegrations turn out to be spurious under walk-forward; the basket
result depends on the AUD/USD vs EUR/CHF "risk-on factor" pair.  Treat
the 30-day paper run as a sanity check on infrastructure and risk model,
not as a guarantee of profitability.

## Risk model (summary)

| Limit                  | Value     | Action                                         |
| ---------------------- | --------- | ---------------------------------------------- |
| Per-trade risk         | 1%        | Reject orders larger than this                 |
| Correlated risk        | 3%        | Skip new EUR pair if exposure already 3%       |
| Daily loss             | 1%        | Reduce risk per trade to 0.5%                  |
| Daily loss             | 2%        | Stop trading for the day (kill switch)         |
| Total drawdown         | 5%        | Stop trading, send alert                       |
| Total drawdown         | 10%       | Emergency close all positions                  |
| Spread filter          | > 1.5 pip | Skip new entries                               |
| News blackout          | ±30 min   | Skip new entries around NFP/FOMC/ECB/CPI       |

## Telegram commands

The bot listens for the following commands from the configured `TELEGRAM_CHAT_ID`:

| Command       | Effect                                                 |
| ------------- | ------------------------------------------------------ |
| `/status`     | Send current equity, open positions, daily PnL         |
| `/pause`      | Stop opening new trades (keeps existing ones)          |
| `/resume`     | Resume new trades                                      |
| `/closeall`   | Close every open position immediately (manual override)|
| `/kill`       | Trip the kill switch (no new trades until restart)     |

## Critical rules

1. **NEVER** trade with real money until 30+ days of profitable paper trading.
2. **ALWAYS** use stop losses — no exceptions.
3. **NEVER** risk more than 1% per trade, 2% per day, 5% total.
4. **ALWAYS** log everything — you need data to improve.
5. **AVOID** overfitting: if in-sample Sharpe > 3.0 it is almost certainly overfitted.
6. **START SMALL:** 0.01 lots on live until 30 days profitable.
7. **TEST THE KILL SWITCH** manually before going live.
8. **NO** Martingale, **NO** grid, **NO** averaging down. Quant strategies only.
9. Monitor spread; if it exceeds 2 pips consistently, pause trading.
10. Always have a manual override (Telegram `/closeall`).

## License

MIT — see `LICENSE` if present, otherwise this repository is unlicensed and provided as-is.
