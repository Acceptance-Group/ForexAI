#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_USER="$(whoami)"
DOMAIN="forex.screwltd.com"
APP_PORT=5000

echo "============================================"
echo " ForexAI — Full Install (Ubuntu Server)"
echo " Domain: ${DOMAIN}"
echo " Project: ${PROJECT_DIR}"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────────────
echo ""
echo "[1/7] Installing system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    git build-essential \
    libssl-dev libffi-dev

# ── 2. Remove old packages & create fresh venv ─────────────────────
echo ""
echo "[2/7] Creating fresh Python venv..."
rm -rf "${PROJECT_DIR}/.venv"
python3 -m venv "${PROJECT_DIR}/.venv"
source "${PROJECT_DIR}/.venv/bin/activate"

# ── 3. Install Python dependencies (CPU-only) ──────────────────────
echo ""
echo "[3/7] Installing Python dependencies (CPU-only)..."
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.24.0,<2.0"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "${PROJECT_DIR}/requirements.txt"
pip install gunicorn
pip install --force-reinstall "numpy>=1.24.0,<2.0"

mkdir -p "${PROJECT_DIR}/data" "${PROJECT_DIR}/models"

python3 -c "import torch; print(f'PyTorch {torch.__version__} CPU={not torch.cuda.is_available()}')"

# ── 4. Check cTrader API connectivity ───────────────────────────────
echo ""
echo "[4/7] Checking cTrader API connectivity (demo.ctraderapi.com:5035)..."
if timeout 10 bash -c 'echo > /dev/tcp/demo.ctraderapi.com/5035' 2>/dev/null; then
    echo "  OK: cTrader API порт 5035 доступен"
    CTRADER_OK=1
elif timeout 10 bash -c 'echo > /dev/tcp/demo.ctraderapi.com/8443' 2>/dev/null; then
    echo "  OK: cTrader API порт 8443 доступен"
    CTRADER_OK=1
else
    echo "  WARNING: cTrader API недоступен."
    echo "  Открытие портов..."
    sudo apt-get install -y ncat 2>/dev/null || true
    echo ""
    echo "  Попытка подключения с деталями:"
    echo "  --- Порт 5035 (TCP Protobuf) ---"
    timeout 5 nc -zv demo.ctraderapi.com 5035 2>&1 || echo "  ЗАКРЫТ" || true
    echo "  --- Порт 8443 (SSL) ---"
    timeout 5 nc -zv demo.ctraderapi.com 8443 2>&1 || echo "  ЗАКРЫТ" || true
    echo "  --- Порт 443 (HTTPS) ---"
    timeout 5 nc -zv demo.ctraderapi.com 443 2>&1 || echo "  ЗАКРЫТ" || true
    echo ""
    CTRADER_OK=0
fi

# ── 5. Train models ────────────────────────────────────────────────
echo ""
echo "[5/7] Training models (USE_ALL_DATA=1)..."
cd "${PROJECT_DIR}"
TRAIN_OK=0
if [ "${CTRADER_OK:-0}" = "1" ]; then
    echo "  Запуск обучения..."
    echo "  Подробности в /tmp/forexai_train.log"
    if USE_ALL_DATA=1 python3 -u trainer_xgb.py 2>&1 | tee /tmp/forexai_train.log; then
        if USE_ALL_DATA=1 python3 -u trainer_multi.py 2>&1 | tee -a /tmp/forexai_train.log; then
            if python3 -u backtest_chart.py 2>&1 | tee -a /tmp/forexai_train.log; then
                TRAIN_OK=1
            fi
        fi
    fi
else
    echo "  ПРОПУСК: cTrader API недоступен."
    echo "  Запустите обучение вручную когда API будет доступен:"
    echo ""
    echo "    source ${PROJECT_DIR}/.venv/bin/activate"
    echo "    USE_ALL_DATA=1 python3 trainer_xgb.py"
    echo "    USE_ALL_DATA=1 python3 trainer_multi.py"
    echo "    python3 backtest_chart.py"
    echo ""
fi

# ── 6. Nginx + SSL ─────────────────────────────────────────────────
echo ""
echo "[6/7] Configuring nginx + SSL..."
cat <<EOF | sudo tee /etc/nginx/sites-available/forexai
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/forexai /etc/nginx/sites-enabled/forexai
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

sudo certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos --register-unsafely-without-email || {
    echo ""
    echo "WARNING: certbot failed. HTTP only for now."
    echo "Fix DNS or rate-limit and run: sudo certbot --nginx -d ${DOMAIN}"
}

# ── 7. Systemd service ─────────────────────────────────────────────
echo ""
echo "[7/7] Creating systemd service..."
cat <<EOF | sudo tee /etc/systemd/system/forexai.service
[Unit]
Description=ForexAI Dashboard
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/.venv/bin/python3 ${PROJECT_DIR}/dashboard.py 5000
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable forexai
sudo systemctl start forexai

echo ""
echo "============================================"
echo " DONE"
echo "============================================"
echo ""
echo "  Dashboard:  https://${DOMAIN}"
echo "  Status:     sudo systemctl status forexai"
echo "  Logs:        sudo journalctl -u forexai -f"
echo "  Restart:    sudo systemctl restart forexai"
echo ""
if [ "${TRAIN_OK}" = "1" ]; then
    echo "  Models:     OK (trained)"
else
    echo "  Models:     NOT TRAINED (cTrader API was unreachable)"
    echo ""
    echo "  To train later when API is available:"
    echo "    source ${PROJECT_DIR}/.venv/bin/activate"
    echo "    USE_ALL_DATA=1 python3 trainer_xgb.py"
    echo "    USE_ALL_DATA=1 python3 trainer_multi.py"
    echo "    python3 backtest_chart.py"
    echo "    sudo systemctl restart forexai"
fi
echo ""
echo "  Models directory:"
ls -la "${PROJECT_DIR}/models/" 2>/dev/null || echo "  (empty)"
echo ""
echo "============================================"