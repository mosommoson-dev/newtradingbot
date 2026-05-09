#!/usr/bin/env bash
# One-command convenience wrapper for the EUR/USD quant bot.
#
# Usage:
#   ./run.sh test                 # run pytest
#   ./run.sh backtest             # quick smoke backtest on yfinance
#   ./run.sh walkforward          # run walk-forward optimization
#   ./run.sh paper                # paper trading on the mock broker
#   ./run.sh live                 # live trading (BOT_MODE=live, real OANDA)
#   ./run.sh dashboard            # start the Flask dashboard on :5000
#   ./run.sh docker-up            # docker compose up -d
#   ./run.sh docker-down          # docker compose down
#   ./run.sh deploy --cloud aws --region us-east-1   # build + ECR + EC2

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CMD="${1:-help}"
shift || true

case "$CMD" in
  test)
    pytest -q "$@"
    ;;
  backtest)
    python -m eurusd_quant_bot.main --mode backtest "$@"
    ;;
  walkforward)
    python -m eurusd_quant_bot.main --mode walkforward "$@"
    ;;
  montecarlo)
    python -m eurusd_quant_bot.main --mode montecarlo "$@"
    ;;
  paper)
    python -m eurusd_quant_bot.main --mode paper "$@"
    ;;
  live)
    BOT_MODE=live python -m eurusd_quant_bot.main --mode live "$@"
    ;;
  dashboard)
    python -m eurusd_quant_bot.dashboard.app
    ;;
  docker-up)
    cd docker && docker compose up -d --build
    ;;
  docker-down)
    cd docker && docker compose down
    ;;
  deploy)
    echo "AWS deploy is a placeholder. Provision an EC2 box, install Docker, set up an ECR"
    echo "repo, then on the box run: cd docker && docker compose pull && docker compose up -d."
    ;;
  help|*)
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
    ;;
esac
