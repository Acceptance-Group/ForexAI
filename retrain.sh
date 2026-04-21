#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

echo "============================================"
echo " ForexAI — Retrain & Restart"
echo "============================================"

echo ""
echo "[1/3] Training direction ensemble (XGB+LGBM+CB)..."
USE_ALL_DATA=1 python3 -u trainer_xgb.py

echo ""
echo "[2/3] Training sub-models..."
USE_ALL_DATA=1 python3 -u trainer_multi.py

echo ""
echo "[3/3] Running backtest..."
python3 -u backtest_chart.py

echo ""
echo "Restarting dashboard..."
sudo systemctl restart forexai

echo ""
echo "Done. Models:"
ls -la models/
echo ""
echo "Dashboard: sudo systemctl status forexai"