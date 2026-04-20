#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

echo "=== ForexAI Start ==="

# Pull latest code
echo "[1/4] Updating code..."
git pull 2>/dev/null || echo "  (no git or no changes)"

# Kill old process on port 5000
echo "[2/4] Killing old process..."
fuser -k 5000/tcp 2>/dev/null || true
sleep 1
fuser -k 5000/tcp 2>/dev/null || true
sleep 1

# Activate venv
echo "[3/4] Activating environment..."
source "${PROJECT_DIR}/.venv/bin/activate"

# Start
echo "[4/4] Starting Bot + Dashboard on port 5000..."
echo ""
exec python3 dashboard.py 5000