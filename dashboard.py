import json
import os
import sys
import time
import threading
import datetime as _dt

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys, 'stderr') and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template_string, jsonify, request

from config import (
    DATA_CONFIG, RISK_CONFIG, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, BACKTEST_CONFIG, RETRAIN_CONFIG,
)
from data_loader import build_dataset, load_raw_prices, compute_atr, compute_adx
from broker import init_broker, get_broker, shutdown_broker
from ensemble import predict_direction_proba, predict_direction_proba_all, reload_models

VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"
MEANREV_MODEL_PATH = "models/meanrev_model.json"
MEANREV_SCALER_PATH = "models/meanrev_scaler.pkl"
SYMBOL = "EURUSD"
TRADE_LOG = "trade_log.json"
SIGNAL_LOG = "signal_log.json"

MODEL_FILES = [
    os.path.join("models", "dir_xgb.json"),
    os.path.join("models", "dir_lgbm.txt"),
    os.path.join("models", "dir_cb.cbm"),
    VOL_MODEL_PATH,
    MEANREV_MODEL_PATH,
]


def get_model_age_days():
    newest = 0
    for f in MODEL_FILES:
        if os.path.exists(f):
            newest = max(newest, os.path.getmtime(f))
    if newest == 0:
        return 999
    return (time.time() - newest) / 86400


def auto_retrain_if_needed():
    if not RETRAIN_CONFIG.get("auto_retrain", True):
        return False
    max_age = RETRAIN_CONFIG.get("max_model_age_days", 7)
    age = get_model_age_days()
    if age > max_age:
        print(f"\n{'='*60}", flush=True)
        print(f"  MODEL AGE: {age:.1f} days (max: {max_age})", flush=True)
        print(f"  Auto-retraining...", flush=True)
        print(f"{'='*60}", flush=True)
        import subprocess
        print("  Running trainer_xgb.py...", flush=True)
        result = subprocess.run(
            [".venv/bin/python", "trainer_xgb.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            capture_output=True, text=True, env={**os.environ, "USE_ALL_DATA": "1"},
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-20:]:
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  Retrain FAILED: {result.stderr[-500:]}")
            return False
        print("  Running trainer_multi.py...", flush=True)
        result2 = subprocess.run(
            [".venv/bin/python", "trainer_multi.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            capture_output=True, text=True, env={**os.environ, "USE_ALL_DATA": "1"},
        )
        if result2.stdout:
            for line in result2.stdout.strip().split("\n")[-20:]:
                print(f"  {line}")
        if result2.returncode != 0:
            print(f"  Multi-model retrain FAILED: {result2.stderr[-500:]}")
            return False
        print(f"  Retrain complete. New model age: {get_model_age_days():.1f} days", flush=True)
        reload_models()
        print("  Models reloaded in memory.", flush=True)
        return True
    return False


def _startup_cleanup():
    print("=" * 60, flush=True)
    print("  STARTUP CLEANUP", flush=True)
    print("=" * 60, flush=True)

    cache_files = [
        DATA_CONFIG.get("parquet_path", "data/eurusd_d1_features.parquet"),
        "data/eurusd_d1_features_labels.parquet",
        DATA_CONFIG.get("raw_parquet_path", "data/eurusd_d1_raw.parquet"),
    ]
    deleted = 0
    for f in cache_files:
        if not f or not os.path.exists(f):
            continue
        try:
            import pandas as _pd
            _pd.read_parquet(f)
        except Exception as e:
            print(f"  CORRUPTED: {f} — removing", flush=True)
            try:
                os.remove(f)
                deleted += 1
            except Exception:
                pass

    if deleted > 0:
        print(f"  Cleaned {deleted} corrupted cache files", flush=True)
    else:
        print("  Cache OK", flush=True)
    print("=" * 60, flush=True)

    model_ok = True
    for f in MODEL_FILES:
        if not os.path.exists(f):
            print(f"  MISSING MODEL: {f}", flush=True)
            model_ok = False

    if not model_ok:
        print("  Models missing! Will retrain...", flush=True)
        return True
    return False


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def _utciso():
    return _utcnow().isoformat()

_threshold_cache = {}

app = Flask(__name__)

last_signal = {}
trade_history = []
signal_history = []
bot_status = {"running": False, "last_check": None, "last_signal_time": None, "errors": []}
cached_candles = None
cached_candles_ts = 0
_cached_analysis = None
_cached_analysis_ts = 0
_broker_lock = threading.Lock()
_broker_connected = False


def _broker_keepalive():
    global _broker_connected, _cached_account, _cached_candles, cached_candles, cached_candles_ts
    import logging
    log = logging.getLogger(__name__) if logging.getLogger(__name__).handlers else None

    while True:
        time.sleep(5)
        try:
            if not _broker_connected:
                msg = "Broker keepalive: connecting..."
                if log: log.info(msg)
                else: print(msg)
                try:
                    shutdown_broker()
                except Exception:
                    pass
                try:
                    if init_broker():
                        _broker_connected = True
                        msg = "✓ Broker keepalive: connected!"
                        if log: log.info(msg)
                        else: print(msg)
                    else:
                        msg = "✗ Broker keepalive: failed (check config.py tokens)"
                        if log: log.error(msg)
                        else: print(msg)
                        time.sleep(10)
                        continue
                except Exception as e:
                    msg = f"✗ Broker connection error: {e}"
                    if log: log.error(msg)
                    else: print(msg)
                    time.sleep(10)
                    continue

            try:
                broker = get_broker()
                info = broker.account_info()
                if info is not None:
                    _broker_connected = True
                    positions = broker.positions_get(symbol=SYMBOL)
                    pos_list = []
                    if positions:
                        for p in positions:
                            pos_list.append({
                                "ticket": p.ticket,
                                "symbol": p.symbol,
                                "type": "BUY" if p.type == 0 else "SELL",
                                "volume": p.volume,
                                "price_open": p.price_open,
                                "price_current": p.price_current,
                                "sl": p.sl,
                                "tp": p.tp,
                                "profit": p.profit,
                                "time": p.time.isoformat() if p.time and isinstance(p.time, _dt.datetime) else "",
                            })
                    tick = broker.symbol_info_tick(SYMBOL)
                    prev_bid = _cached_account.get("bid", 0) if _cached_account else 0
                    prev_ask = _cached_account.get("ask", 0) if _cached_account else 0
                    new_bid = round(tick.bid, 5) if tick and tick.bid > 0 else prev_bid
                    new_ask = round(tick.ask, 5) if tick and tick.ask > 0 else prev_ask
                    result = {
                        "balance": info.balance,
                        "equity": info.equity,
                        "profit": info.profit,
                        "margin": info.margin,
                        "free_margin": info.margin_free,
                        "leverage": info.leverage,
                        "server": info.server,
                        "positions": pos_list,
                        "bid": new_bid,
                        "ask": new_ask,
                        "spread": round((new_ask - new_bid) * 100000, 1) if new_ask > 0 and new_bid > 0 and new_ask > new_bid else 0,
                        "connected": True,
                    }
                    _cached_account = result
                    _cached_account_ts = time.time()
                else:
                    _broker_connected = False
            except Exception as e:
                print(f"Broker keepalive error: {e}")
                _broker_connected = False
                try:
                    shutdown_broker()
                except Exception:
                    pass

            try:
                if _broker_connected:
                    broker = get_broker()
                    tf_d1 = broker.TIMEFRAME_D1
                    rates = broker.copy_rates_from_pos(SYMBOL, tf_d1, 0, 200)
                    if rates is not None and len(rates) > 0:
                        candles = []
                        for r in rates:
                            dt = _dt.datetime.fromtimestamp(r["time"])
                            candles.append({
                                "time": int(r["time"]),
                                "date": dt.strftime("%Y-%m-%d"),
                                "open": round(float(r["open"]), 5),
                                "high": round(float(r["high"]), 5),
                                "low": round(float(r["low"]), 5),
                                "close": round(float(r["close"]), 5),
                                "volume": int(r["tick_volume"]),
                            })
                        cached_candles = candles
                        cached_candles_ts = time.time()
            except Exception as e:
                print(f"Candle update error: {e}")

        except Exception as e:
            print(f"Broker keepalive outer error: {e}")
            time.sleep(10)


def ensure_broker():
    global _broker_connected
    import logging
    log = logging.getLogger(__name__) if logging.getLogger(__name__).handlers else None
    
    with _broker_lock:
        try:
            broker = get_broker()
            info = broker.account_info()
            if info is not None:
                _broker_connected = True
                return True
        except Exception:
            pass
        try:
            if init_broker():
                broker = get_broker()
                info = broker.account_info()
                if info is not None:
                    _broker_connected = True
                    msg = f"✓ Broker connected: {info.server} Balance=${info.balance:,.2f}"
                    if log: log.info(msg)
                    else: print(msg)
                    return True
        except Exception as e:
            msg = f"✗ Broker init error: {e}"
            if log: log.error(msg)
            else: print(msg)
        _broker_connected = False


def _bg_compute_signal():
    try:
        compute_signal(force_refresh=False)
    except Exception as e:
        print(f"BG signal error: {e}")
        return False


def load_logs():
    global trade_history, last_signal, signal_history
    try:
        if os.path.exists(TRADE_LOG):
            with open(TRADE_LOG, "r") as f:
                trade_history = json.load(f)
    except Exception:
        trade_history = []
    try:
        if os.path.exists(SIGNAL_LOG):
            with open(SIGNAL_LOG, "r") as f:
                signals = json.load(f)
            if signals:
                signal_history = signals
                last_signal = signals[-1]
    except Exception:
        signal_history = []
        last_signal = {}


def save_trade(trade_data):
    global trade_history
    trade_history.append(trade_data)
    with open(TRADE_LOG, "w") as f:
        json.dump(trade_history[-500:], f, indent=2)


def save_signal(sig):
    global last_signal, signal_history
    last_signal = sig
    signal_history.append(sig)
    with open(SIGNAL_LOG, "w") as f:
        json.dump(signal_history[-500:], f, indent=2)


_cached_signal = None
_cached_signal_ts = 0
SIGNAL_CACHE_SEC = 10


def compute_signal(force_refresh=False):
    global last_signal, _cached_signal, _cached_signal_ts
    if not force_refresh and _cached_signal is not None and (time.time() - _cached_signal_ts) < SIGNAL_CACHE_SEC:
        return _cached_signal
    try:
        prices_df = load_raw_prices()
        feats_df = build_dataset(force_download=force_refresh)
        common_idx = feats_df.index.intersection(prices_df.index)
        feats_df = feats_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]

        dir_prob, dir_probs = predict_direction_proba(feats_df.values[-1:].reshape(1, -1))
        dir_prob = float(dir_prob[0])

        vol_model = xgb.XGBRegressor()
        vol_model.load_model(VOL_MODEL_PATH)
        with open(VOL_SCALER_PATH, "rb") as f:
            vol_scaler = pickle.load(f)
        vol_scaled = vol_scaler.transform(feats_df.values)
        vol_pred = vol_model.predict(vol_scaled)
        vol_pct = pd.Series(vol_pred).rolling(252, min_periods=30).quantile(0.20).values
        vol_threshold = float(vol_pct[-1]) if not pd.isna(vol_pct[-1]) else float(np.median(vol_pred))
        current_vol = vol_pred[-1]
        vol_high = current_vol > vol_threshold

        mr_model = xgb.XGBClassifier()
        mr_model.load_model(MEANREV_MODEL_PATH)
        with open(MEANREV_SCALER_PATH, "rb") as f:
            mr_scaler = pickle.load(f)
        mr_scaled = mr_scaler.transform(feats_df.values)
        mr_prob = mr_model.predict_proba(mr_scaled[-1:].reshape(1, -1))[0, 1]

        close = prices_df["close"].values
        high = prices_df["high"].values
        low = prices_df["low"].values
        atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
        atr_mean = atr_series.rolling(60, min_periods=1).mean()
        adx_series = compute_adx(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])

        current_atr = atr_series.iloc[-1]
        current_atr_mean = atr_mean.iloc[-1]
        current_adx = adx_series.iloc[-1]
        atr_pips = current_atr * 10000

        no_trade_buy_above = DATA_CONFIG["no_trade_buy_above"]
        no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]
        tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
        min_sl_pips = DATA_CONFIG.get("min_sl_pips", 40.0)
        min_adx = DATA_CONFIG.get("min_adx", 0.30)

        sl_pips = max(atr_pips, min_sl_pips)
        vol_ratio = current_atr / current_atr_mean if not np.isnan(current_atr_mean) and current_atr_mean > 0 else 1.0
        dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
        dyn_tp_sl = min(dyn_tp_sl, 4.0)
        tp_pips = sl_pips * dyn_tp_sl

        adx_ok = current_adx >= min_adx * 100
        breakeven_pips = DATA_CONFIG.get("breakeven_pips", 15)
        trail_pips = DATA_CONFIG.get("trail_pips", 10)

        if dir_prob > no_trade_buy_above:
            signal = "BUY"
        elif dir_prob < no_trade_sell_below:
            signal = "SELL"
        else:
            signal = "HOLD"

        reasons = []
        if not adx_ok:
            signal = "HOLD"
        if signal != "HOLD" and not vol_high:
            signal = "HOLD"
        if signal == "HOLD":
            reasons.append(f"P(UP)={dir_prob:.3f} in [{no_trade_sell_below}, {no_trade_buy_above}]")

        sig = {
            "timestamp": _utciso(),
            "signal": signal,
            "prob_up": float(dir_prob),
            "prob_xgb": float(dir_probs["xgb"][0]),
            "prob_lgbm": float(dir_probs["lgbm"][0]),
            "prob_cb": float(dir_probs["cb"][0]),
            "mr_prob": float(mr_prob),
            "vol_pred": float(current_vol),
            "vol_pct20": float(vol_threshold),
            "vol_high": bool(vol_high),
            "adx": float(current_adx),
            "atr_pips": float(atr_pips),
            "sl_pips": float(sl_pips),
            "tp_pips": float(tp_pips),
            "dyn_tp_sl": float(dyn_tp_sl),
            "reasons": reasons,
            "breakeven": breakeven_pips,
            "trail": trail_pips,
        }
        save_signal(sig)
        _cached_signal = sig
        _cached_signal_ts = time.time()
        return sig
    except Exception as e:
        err = {"timestamp": _utciso(), "signal": "ERROR", "error": str(e)}
        _cached_signal = err
        _cached_signal_ts = time.time()
        return err


def get_chart_analysis(n_bars=200):
    global _cached_analysis, _cached_analysis_ts
    now = time.time()
    if _cached_analysis is not None and (now - _cached_analysis_ts) < 300:
        return _cached_analysis
    try:
        prices_df = load_raw_prices()
        feats_df = build_dataset(force_download=False)
        common_idx = feats_df.index.intersection(prices_df.index)
        feats_df = feats_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]

        prob_up_all, _ = predict_direction_proba_all(feats_df.values)

        vol_model = xgb.XGBRegressor()
        vol_model.load_model(VOL_MODEL_PATH)
        with open(VOL_SCALER_PATH, "rb") as f:
            vol_scaler = pickle.load(f)
        vol_scaled = vol_scaler.transform(feats_df.values)
        vol_pred_all = vol_model.predict(vol_scaled)
        vol_pct = pd.Series(vol_pred_all).rolling(252, min_periods=30).quantile(0.20).values

        close = prices_df["close"].values
        high = prices_df["high"].values
        low = prices_df["low"].values
        atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
        adx_series = compute_adx(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])

        no_trade_buy = DATA_CONFIG["no_trade_buy_above"]
        no_trade_sell = DATA_CONFIG["no_trade_sell_below"]
        min_adx = DATA_CONFIG.get("min_adx", 0.30)

        analysis = []
        start = max(0, len(feats_df) - n_bars)
        for i in range(start, len(feats_df)):
            p = prob_up_all[i]
            vh = bool(vol_pred_all[i] > (vol_pct[i] if i < len(vol_pct) and not pd.isna(vol_pct[i]) else np.median(vol_pred_all)))
            ax = float(adx_series.iloc[i])
            atr = float(atr_series.iloc[i])

            if p > no_trade_buy and ax >= min_adx * 100 and vh:
                sig = "BUY"
            elif p < no_trade_sell and ax >= min_adx * 100 and vh:
                sig = "SELL"
            elif ax < min_adx * 100:
                sig = "HOLD_ADX"
            elif not vh:
                sig = "HOLD_VOL"
            else:
                sig = "HOLD"

            dt = prices_df.index[i]
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)[:10]

            analysis.append({
                "date": date_str,
                "prob_up": round(float(p), 4),
                "vol_high": vh,
                "adx": round(ax, 2),
                "atr": round(atr, 5),
                "atr_pips": round(atr * 10000, 1),
                "signal": sig,
            })

        _cached_analysis = analysis
        _cached_analysis_ts = now
        return analysis
    except Exception as e:
        print(f"Chart analysis error: {e}")
        return _cached_analysis or []


_cached_account = {}
_cached_account_ts = 0
_ACCOUNT_CACHE_TTL = 10


def get_account_info():
    return _cached_account if _cached_account else {"connected": False}


def get_candles(count=200):
    return cached_candles or []


def get_signal_markers():
    markers = []
    for s in signal_history:
        if s.get("signal") in ("BUY", "SELL"):
            markers.append({
                "date": s.get("timestamp", "")[:10],
                "signal": s["signal"],
                "prob_up": s.get("prob_up", 0),
                "reasons": s.get("reasons", []),
            })
    return markers[-50:]


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForexAI Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
body{font-family:'Inter',system-ui,sans-serif;background:#0a0a0a;color:#e5e5e5}
.glass{background:rgba(23,23,23,0.8);backdrop-filter:blur(12px);border:1px solid rgba(64,64,64,0.2)}
* { scrollbar-width: none }
</style>
</head>
<body class="min-h-screen p-4">
<div class="max-w-[1600px] mx-auto space-y-4">
  <!-- Header -->
  <div class="glass rounded-2xl p-4">
    <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
      <div class="flex flex-col sm:flex-row sm:items-center gap-4">
        <div>
          <h1 class="text-xl font-bold tracking-tight">ForexAI</h1>
          <p class="text-xs text-neutral-400">EUR/USD AI Trader</p>
        </div>
        <div class="flex items-baseline gap-3">
          <span class="text-2xl lg:text-3xl font-bold font-mono" id="live-price">--</span>
          <span class="text-sm font-mono" id="price-chg">--</span>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-2 text-xs text-neutral-400 font-mono">
        <span class="px-2 py-1 bg-neutral-800/50 rounded">O <span id="oh_o">--</span></span>
        <span class="px-2 py-1 bg-neutral-800/50 rounded">H <span id="oh_h">--</span></span>
        <span class="px-2 py-1 bg-neutral-800/50 rounded">L <span id="oh_l">--</span></span>
        <span class="px-2 py-1 bg-neutral-800/50 rounded">C <span id="oh_c" class="text-neutral-200">--</span></span>
      </div>
    </div>
    <div class="mt-4 pt-3 border-t border-neutral-800 flex flex-wrap items-center gap-3 text-sm">
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full" id="conn-dot"></span>
        <span class="text-neutral-400" id="conn-txt">Connecting...</span>
      </div>
      <span class="px-3 py-1 rounded-full text-xs font-medium bg-neutral-800" id="bot-badge">BOT OFF</span>
      <span class="text-xs text-neutral-500 ml-auto" id="server-name">--</span>
    </div>
  </div>

  <!-- Main Content: Charts + Signal -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    <!-- Charts Column -->
    <div class="lg:col-span-2 space-y-4">
      <div class="glass rounded-2xl p-1">
        <div id="priceChart" class="min-h-[350px]"></div>
        <div id="probChart" class="h-28 border-t border-neutral-800"></div>
      </div>
    </div>

    <!-- Signal & Account Column -->
    <div class="space-y-4">
      <div class="glass rounded-2xl p-5 text-center" id="sig-card">
        <div class="text-xs uppercase tracking-wider text-neutral-500 mb-2">AI Signal</div>
        <div class="text-4xl font-bold mb-3" id="sig-val">--</div>
        <div class="h-2 bg-neutral-800 rounded-full overflow-hidden mb-2">
          <div class="h-full transition-all duration-500" id="prob-bar" style="width:50%"></div>
        </div>
        <div class="flex justify-between text-xs text-neutral-500 mb-3 font-mono">
          <span>0 ← SELL</span>
          <span id="prob-num" class="text-neutral-200 font-bold">0.50</span>
          <span>BUY → 1</span>
        </div>
        <div class="flex justify-center gap-3 mb-3 text-xs font-mono">
          <span class="text-cyan-400">XGB <span id="ens-xgb">--</span></span>
          <span class="text-emerald-400">LGBM <span id="ens-lgbm">--</span></span>
          <span class="text-amber-400">CB <span id="ens-cb">--</span></span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-left">
          <div class="bg-neutral-800/50 rounded-lg p-2">
            <div class="text-[10px] uppercase text-neutral-500">Vol Filter</div>
            <div class="text-base font-bold font-mono" id="f-vol">--</div>
          </div>
          <div class="bg-neutral-800/50 rounded-lg p-2">
            <div class="text-[10px] uppercase text-neutral-500">ADX</div>
            <div class="text-base font-bold font-mono" id="f-adx">--</div>
          </div>
          <div class="bg-neutral-800/50 rounded-lg p-2">
            <div class="text-[10px] uppercase text-neutral-500">ATR</div>
            <div class="text-base font-bold font-mono" id="f-atr">--</div>
          </div>
          <div class="bg-neutral-800/50 rounded-lg p-2">
            <div class="text-[10px] uppercase text-neutral-500">SL/TP</div>
            <div class="text-base font-bold font-mono" id="f-sltp">--</div>
          </div>
        </div>
        <div id="filter-rows" class="mt-2 text-left text-xs space-y-1"></div>
        <div id="ai-log" class="mt-2 text-xs text-neutral-400 font-mono text-left break-all"></div>
      </div>

      <div class="glass rounded-2xl p-4">
        <div class="text-xs uppercase tracking-wider text-neutral-500 mb-2">Account</div>
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div class="bg-neutral-800/30 rounded p-2">
            <div class="text-[10px] text-neutral-500 uppercase">Balance</div>
            <div class="font-bold font-mono" id="bal">--</div>
          </div>
          <div class="bg-neutral-800/30 rounded p-2">
            <div class="text-[10px] text-neutral-500 uppercase">Equity</div>
            <div class="font-bold font-mono" id="eq">--</div>
          </div>
          <div class="bg-neutral-800/30 rounded p-2">
            <div class="text-[10px] text-neutral-500 uppercase">Free Margin</div>
            <div class="font-bold font-mono" id="fm">--</div>
          </div>
          <div class="bg-neutral-800/30 rounded p-2">
            <div class="text-[10px] text-neutral-500 uppercase">Spread</div>
            <div class="font-bold font-mono" id="sp">--</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Equity Curve (отдельная строка, компактная) -->
  <div class="glass rounded-2xl p-4">
    <h3 class="text-sm font-semibold mb-2 text-neutral-300">Equity Curve</h3>
    <canvas id="equityChart" height="100"></canvas>
  </div>

  <!-- Positions (компактная таблица) -->
  <div class="glass rounded-2xl overflow-hidden">
    <div class="px-4 py-2 border-b border-neutral-800 flex items-center justify-between">
      <span class="text-sm font-semibold text-neutral-300">Open Positions</span>
      <span class="text-xs bg-neutral-800 px-2 py-1 rounded" id="pos-cnt">0</span>
    </div>
    <div class="overflow-x-auto" id="pos-table-wrap">
      <table class="w-full text-sm">
        <thead class="text-xs uppercase text-neutral-500 bg-neutral-800/50">
          <tr>
            <th class="text-left px-3 py-2 font-medium">Time</th>
            <th class="text-left px-3 py-2 font-medium">Type</th>
            <th class="text-left px-3 py-2 font-medium">Lots</th>
            <th class="text-left px-3 py-2 font-medium">Entry</th>
            <th class="text-left px-3 py-2 font-medium">Current</th>
            <th class="text-left px-3 py-2 font-medium">SL</th>
            <th class="text-left px-3 py-2 font-medium">TP</th>
            <th class="text-right px-3 py-2 font-medium">P&L</th>
          </tr>
        </thead>
        <tbody id="pos-body" class="font-mono text-xs"></tbody>
      </table>
    </div>
    <div class="text-center text-neutral-500 py-6 hidden" id="no-pos">No open positions</div>
  </div>

  <!-- Performance & Signals (1:1 grid) -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div class="glass rounded-2xl p-4">
      <div class="text-sm font-semibold text-neutral-300 mb-3">Performance</div>
      <div class="grid grid-cols-3 gap-2">
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Trades</div>
          <div class="text-lg font-bold" id="p-tr">0</div>
        </div>
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Win Rate</div>
          <div class="text-lg font-bold" id="p-wr">--</div>
        </div>
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Total P&L</div>
          <div class="text-lg font-bold" id="p-pnl">$0</div>
        </div>
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Avg P&L</div>
          <div class="text-lg font-bold" id="p-avg">$0</div>
        </div>
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Long</div>
          <div class="text-lg font-bold" id="p-long">0/0</div>
        </div>
        <div class="bg-neutral-800/50 rounded-lg p-2">
          <div class="text-[10px] text-neutral-500 uppercase">Short</div>
          <div class="text-lg font-bold" id="p-short">0/0</div>
        </div>
      </div>
    </div>

    <div class="glass rounded-2xl p-4">
      <div class="text-sm font-semibold text-neutral-300 mb-3">Recent Signals</div>
      <div class="overflow-y-auto max-h-[160px]">
        <table class="w-full text-sm">
          <thead class="text-xs uppercase text-neutral-500 sticky top-0 bg-neutral-900">
            <tr>
              <th class="text-left py-1 font-medium">Time</th>
              <th class="text-left py-1 font-medium">Sig</th>
              <th class="text-left py-1 font-medium">P(UP)</th>
              <th class="text-left py-1 font-medium">Vol</th>
              <th class="text-left py-1 font-medium">ADX</th>
              <th class="text-right py-1 font-medium">P&L</th>
            </tr>
          </thead>
          <tbody id="sig-body" class="font-mono text-xs"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="text-center text-xs text-neutral-600 pt-2">
    Auto-refresh 10s · Last: <span id="last-upd">--</span> · XGBoost + LightGBM + CatBoost Ensemble
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
let equityChart=null,priceChart=null,probChart=null;
let candleSeries=null,volSeries=null,probLine=null,probThresh60=null,probThresh45=null;

function initCharts(){
  const pc=document.getElementById('priceChart');
  if(!pc)return;
  priceChart=LightweightCharts.createChart(pc,{
    width:pc.clientWidth,height:400,
    layout:{background:{type:'solid',color:'transparent'},textcolor:'#a3a3a3',fontSize:11,fontFamily:'JetBrains Mono'},
    grid:{vertLines:{color:'#262626'},horzLines:{color:'#262626'}},
    crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:'#0ea5e955',style:2},horzLine:{color:'#0ea5e955',style:2}},
    rightPriceScale:{bordercolor:'#404040',scaleMargins:{top:0.05,bottom:0.15}},
    timeScale:{bordercolor:'#404040',timeVisible:false},
  });
  candleSeries=priceChart.addCandlestickSeries({
    upColor:'#2dd4bf',downColor:'#f472b6',borderUpColor:'#2dd4bf',borderDownColor:'#f472b6',
    wickUpColor:'rgba(45,212,191,0.5)',wickDownColor:'rgba(244,114,182,0.5)',
  });
  volSeries=priceChart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:''});
  volSeries.priceScale().applyOptions({scaleMargins:{top:0.88,bottom:0}});

  const prc=document.getElementById('probChart');
  if(!prc)return;
  probChart=LightweightCharts.createChart(prc,{
    width:prc.clientWidth,height:120,
    layout:{background:{type:'solid',color:'transparent'},textcolor:'#a3a3a3',fontSize:10,fontFamily:'JetBrains Mono'},
    grid:{vertLines:{color:'#262626'},horzLines:{color:'#262626'}},
    rightPriceScale:{bordercolor:'#404040',scaleMargins:{top:0.05,bottom:0.05}},
    timeScale:{bordercolor:'#404040',timeVisible:false},
  });
  probLine=probChart.addLineSeries({color:'#a855f7',lineWidth:2,lastValueVisible:true,priceLineVisible:true,priceFormat:{type:'price',precision:2,minMove:0.01}});
  probThresh60=probChart.addLineSeries({color:'rgba(45,212,191,0.5)',lineWidth:1,lineStyle:2,lineVisible:true});
  probThresh45=probChart.addLineSeries({color:'rgba(244,114,182,0.5)',lineWidth:1,lineStyle:2,lineVisible:true});
}

let lastCandleTime=null;
function renderCandles(candles){
  if(!candleSeries||!candles||!candles.length)return;
  // Validate candle structure
  if(!candles[0].date || candles[0].open === undefined)return;
  const newCandles=candles.map(c=>({time:c.date,open:c.open,high:c.high,low:c.low,close:c.close}));
  const newVol=candles.map(c=>({time:c.date,value:c.volume||0,color:c.close>=c.open?'rgba(45,212,191,0.3)':'rgba(244,114,182,0.3)'}));
  
  // Если время последней свечи изменилось — обновляем
  const lastTime=candles[candles.length-1].date;
  if(lastTime!==lastCandleTime){
    candleSeries.setData(newCandles);
    volSeries.setData(newVol);
    lastCandleTime=lastTime;
    priceChart.timeScale().scrollToRealTime();
  }
  
  const l=candles[candles.length-1];
  document.getElementById('oh_o').textContent=l.open.toFixed(5);
  document.getElementById('oh_h').textContent=l.high.toFixed(5);
  document.getElementById('oh_l').textContent=l.low.toFixed(5);
  document.getElementById('oh_c').textContent=l.close.toFixed(5);
}

function renderAnalysis(analysis){
  if(!analysis||!analysis.length||!probLine)return;
  probLine.setData(analysis.map(a=>({time:a.date,value:a.prob_up})));
  probThresh60.setData(analysis.map(a=>({time:a.date,value:0.60})));
  probThresh45.setData(analysis.map(a=>({time:a.date,value:0.45})));
  const markers=[];
  for(const a of analysis){
    if(a.signal==='BUY')markers.push({time:a.date,position:'belowBar',color:'#2dd4bf',shape:'arrowUp',text:'B'});
    else if(a.signal==='SELL')markers.push({time:a.date,position:'aboveBar',color:'#f472b6',shape:'arrowDown',text:'S'});
  }
  candleSeries.setMarkers(markers.sort((a,b)=>a.time.localeCompare(b.time)));
  probChart.timeScale().fitContent();
}

function updateUI(data){
  const brk=data.broker,sig=data.signal,perf=data.performance||{},hist=data.history||[];

  const conn=brk&&brk.connected;
  const dot=document.getElementById('conn-dot');
  if(dot)dot.className='w-2 h-2 rounded-full '+(conn?'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]':'bg-rose-500');
  document.getElementById('conn-txt').textContent=conn?'cTrader Live':'Offline';
  document.getElementById('server-name').textContent=conn?brk.server:'--';

  const bb=document.getElementById('bot-badge');
  if(data.bot_status&&data.bot_status.running){
    bb.className='px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
    bb.textContent='BOT ACTIVE';
  }else{
    bb.className='px-3 py-1 rounded-full text-xs font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30';
    bb.textContent='BOT OFF';
  }

  const lp=document.getElementById('live-price');
  const pc=document.getElementById('price-chg');
  if(conn&&brk.bid>0){
    lp.textContent=brk.bid.toFixed(5);
    if(data.candles&&data.candles.length>1){
      const prev=data.candles[data.candles.length-2].close;
      const d=brk.bid-prev;
      const pct=d/prev*100;
      pc.innerHTML='<span class="'+(d>=0?'text-emerald-400':'text-rose-400')+'">'+(d>=0?'+':'')+(d*10000).toFixed(1)+'p · '+(pct>=0?'+':'')+pct.toFixed(3)+'%</span>';
    }
  }else{lp.textContent='--';pc.textContent='';}

  if(conn){
    document.getElementById('bal').textContent='$'+brk.balance.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});
    document.getElementById('eq').textContent='$'+brk.equity.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});
    document.getElementById('fm').textContent='$'+brk.free_margin.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});
    document.getElementById('sp').textContent=brk.spread.toFixed(1);
  }

  if(sig){
    const sc=document.getElementById('sig-card');
    const sv=document.getElementById('sig-val');
    const s=sig.signal||'HOLD';
    sv.textContent=s;
    const sColor=s==='BUY'?'text-cyan-400':s==='SELL'?'text-pink-400':'text-amber-400';
    const sGlow=s==='BUY'?'shadow-[0_0_60px_rgba(34,211,238,0.2)]':s==='SELL'?'shadow-[0_0_60px_rgba(244,114,182,0.2)]':'shadow-[0_0_60px_rgba(251,191,36,0.1)]';
    sv.className='text-5xl font-bold mb-4 '+sColor;
    sc.className='glass rounded-2xl p-6 text-center '+sGlow;

    const p=sig.prob_up||0;
    document.getElementById('prob-num').textContent=p.toFixed(4);
    const pb=document.getElementById('prob-bar');
    pb.style.width=(p*100)+'%';
    pb.className='h-full transition-all duration-500 '+(p>0.6?'bg-cyan-400':p<0.45?'bg-pink-400':'bg-amber-400');

    if(sig.prob_xgb!==undefined){
      document.getElementById('ens-xgb').textContent=sig.prob_xgb.toFixed(3);
      document.getElementById('ens-lgbm').textContent=sig.prob_lgbm.toFixed(3);
      document.getElementById('ens-cb').textContent=sig.prob_cb.toFixed(3);
    }

    const fv=document.getElementById('f-vol');
    fv.textContent=sig.vol_high?'PASS':'BLOCK';
    fv.className='text-lg font-bold font-mono '+(sig.vol_high?'text-emerald-400':'text-rose-400');

    const fa=document.getElementById('f-adx');
    fa.textContent=(sig.adx||0).toFixed(2);
    fa.className='text-lg font-bold font-mono '+((sig.adx||0)>=30?'text-emerald-400':'text-amber-400');

    document.getElementById('f-atr').textContent=(sig.atr_pips||0).toFixed(1)+'p';
    document.getElementById('f-sltp').textContent=(sig.sl_pips||0).toFixed(0)+'/'+(sig.tp_pips||0).toFixed(0)+'p';

    let frHTML='';
const adxOk=(sig.adx||0)>=30;
            frHTML+='<div class="flex items-center gap-2"><span class="w-4 h-4 rounded flex items-center justify-center text-[10px] '+(adxOk?'bg-emerald-500/20 text-emerald-400':'bg-rose-500/20 text-rose-400')+'">'+(adxOk?'✓':'✗')+'</span><span>ADX '+(sig.adx||0).toFixed(2)+(adxOk?' ≥ 30':' < 30')+'</span></div>';
    var vThresh=sig.vol_pct20?(sig.vol_pct20*10000).toFixed(1)+'p':'--';
    frHTML+='<div class="flex items-center gap-2"><span class="w-4 h-4 rounded flex items-center justify-center text-[10px] '+(sig.vol_high?'bg-emerald-500/20 text-emerald-400':'bg-rose-500/20 text-rose-400')+'">'+(sig.vol_high?'✓':'✗')+'</span><span>Vol '+(sig.vol_high?'>':'≤')+' pct20 '+vThresh+'</span></div>';
    if(sig.reasons&&sig.reasons.length){
      sig.reasons.forEach(r=>{
        frHTML+='<div class="flex items-center gap-2"><span class="w-4 h-4 rounded flex items-center justify-center text-[10px] bg-slate-700 text-neutral-400">i</span><span class="text-neutral-400">'+r+'</span></div>';
      });
    }
    document.getElementById('filter-rows').innerHTML=frHTML;

    const pClass=p>0.6?'text-cyan-400':p<0.45?'text-pink-400':'text-amber-400';
    document.getElementById('ai-log').innerHTML='P(UP)=<span class="'+pClass+' font-bold">'+p.toFixed(4)+'</span> · Vol=<span class="'+(sig.vol_high?'text-emerald-400':'text-rose-400')+'">'+(sig.vol_high?'HIGH':'LOW')+'</span> · ADX=<span class="'+(adxOk?'text-emerald-400':'text-amber-400')+'">'+(sig.adx||0).toFixed(2)+'</span> → <span class="'+pClass+' font-bold">'+s+'</span>';
  }

  const posB=document.getElementById('pos-body');
  const noP=document.getElementById('no-pos');
  const posWrap=document.getElementById('pos-table-wrap');
  if(brk&&brk.positions&&brk.positions.length>0){
    noP.classList.add('hidden');
    if(posWrap)posWrap.classList.remove('hidden');
    document.getElementById('pos-cnt').textContent=brk.positions.length;
    posB.innerHTML=brk.positions.map(p=>{
      const pc=p.profit>=0?'text-emerald-400':'text-rose-400';
      return '<tr class="border-b border-neutral-800 hover:bg-neutral-800/30"><td class="px-4 py-2">'+p.time.substring(11,19)+'</td><td class="px-4 py-2 '+(p.type==='BUY'?'text-emerald-400':'text-rose-400')+' font-bold">'+p.type+'</td><td class="px-4 py-2">'+p.volume.toFixed(2)+'</td><td class="px-4 py-2">'+p.price_open.toFixed(5)+'</td><td class="px-4 py-2">'+p.price_current.toFixed(5)+'</td><td class="px-4 py-2 text-neutral-400">'+p.sl.toFixed(5)+'</td><td class="px-4 py-2 text-neutral-400">'+p.tp.toFixed(5)+'</td><td class="px-4 py-2 '+pc+' font-bold text-right">$'+p.profit.toFixed(2)+'</td></tr>';
    }).join('');
  }else{posB.innerHTML='';noP.classList.remove('hidden');if(posWrap)posWrap.classList.add('hidden');document.getElementById('pos-cnt').textContent='0';}

  const wr=perf.win_rate||0;
  const pnl=perf.total_pnl||0;
  document.getElementById('p-tr').textContent=perf.total_trades||0;
  const wrEl=document.getElementById('p-wr');wrEl.textContent=wr?wr.toFixed(1)+'%':'--';
  wrEl.className='text-xl font-bold '+(wr>=60?'text-emerald-400':wr>=50?'text-neutral-200':'text-rose-400');
  const pnlEl=document.getElementById('p-pnl');pnlEl.textContent=(pnl>=0?'+':'')+'$'+Math.abs(pnl).toFixed(0);
  pnlEl.className='text-xl font-bold '+(pnl>=0?'text-emerald-400':'text-rose-400');
  document.getElementById('p-avg').textContent='$'+(perf.avg_pnl||0).toFixed(2);
  document.getElementById('p-long').textContent=(perf.long_wins||0)+'/'+(perf.long_total||0);
  document.getElementById('p-short').textContent=(perf.short_wins||0)+'/'+(perf.short_total||0);

  if(hist.length>0){
    document.getElementById('sig-body').innerHTML=hist.slice(-20).reverse().map(h=>{
      const sc=h.signal==='BUY'?'text-emerald-400':h.signal==='SELL'?'text-rose-400':'text-amber-400';
      const pn=h.pnl||0;
      const pnC=pn>=0?'text-emerald-400':'text-rose-400';
      return '<tr class="border-b border-neutral-800"><td class="py-1">'+(h.timestamp||'').substring(5,16)+'</td><td class="'+sc+' font-bold">'+h.signal+'</td><td>'+(h.prob_up||0).toFixed(3)+'</td><td>'+(h.vol_pred?(h.vol_pred*10000).toFixed(1):'--')+'</td><td>'+(h.adx||0).toFixed(2)+'</td><td class="'+pnC+' text-right">'+(pn!==0?'$'+pn.toFixed(2):'--')+'</td></tr>';
    }).join('');
  }

  const eqD=data.equity_history||[];
  if(eqD.length>1)updateEquity(eqD);
  document.getElementById('last-upd').textContent=new Date().toLocaleTimeString();
}

function updateEquity(data){
  const ctx=document.getElementById('equityChart');
  if(!ctx)return;
  if(equityChart)equityChart.destroy();
  const labels=data.map(e=>{const d=new Date(e.time);return(d.getMonth()+1)+'/'+d.getDate()});
  equityChart=new Chart(ctx.getContext('2d'),{
    type:'line',
    data:{labels:labels,datasets:[{label:'Equity',data:data.map(e=>e.equity),borderColor:'#06b6d4',backgroundColor:'rgba(6,182,212,0.1)',fill:true,tension:0.4,pointRadius:0,borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{
      x:{grid:{color:'#262626',drawBorder:false},ticks:{color:'#737373',maxTicksLimit:6,font:{size:9,family:'JetBrains Mono'},padding:5}},
      y:{grid:{color:'#262626',drawBorder:false},ticks:{color:'#737373',callback:v=>'$'+v.toLocaleString(),font:{size:9,family:'JetBrains Mono'},padding:5}}
    }}
  });
}

async function fetchStatus(){
  try{
    const r=await fetch('/api/status');
    if(!r.ok) throw new Error('HTTP '+r.status);
    const data=await r.json();
    // Only render candles if we have valid data with proper structure
    if(data.candles && data.candles.length > 0 && data.candles[0].date){
      if(!candleSeries) initCharts();
      if(candleSeries) renderCandles(data.candles);
    }
    updateUI(data);
  }catch(e){
    document.getElementById('conn-dot').className='w-2 h-2 rounded-full bg-rose-500';
    document.getElementById('conn-txt').textContent='Reconnecting...';
    console.error('fetchStatus error:',e);
  }
}

async function fetchAnalysis(){
  try{
    const r=await fetch('/api/analysis');
    if(!r.ok) return;
    const analysis=await r.json();
    if(analysis&&analysis.length>0){
      if(!candleSeries)initCharts();
      renderAnalysis(analysis);
    }
  }catch(e){
    console.error('fetchAnalysis error:',e);
  }
}

window.addEventListener('resize',()=>{
  if(priceChart)priceChart.applyOptions({width:document.getElementById('priceChart').clientWidth});
  if(probChart)probChart.applyOptions({width:document.getElementById('probChart').clientWidth});
});

setInterval(()=>{fetchStatus();fetchAnalysis();},10000);
fetchStatus();
fetchAnalysis();
</script>
</body>
</html>
'''

@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE)

_cached_candles = None
_cached_candles_ts = 0
_CANDLES_CACHE_TTL = 300

def _bg_update_candles():
    global _cached_candles, _cached_candles_ts
    try:
        feats = build_dataset(force_download=False)
        prices = load_raw_prices()
        common_idx = feats.index.intersection(prices.index)
        prices = prices.loc[common_idx]
        
        candles = []
        for i in range(max(0, len(prices)-200), len(prices)):
            dt = prices.index[i]
            candles.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": float(prices.iloc[i]["open"]),
                "high": float(prices.iloc[i]["high"]),
                "low": float(prices.iloc[i]["low"]),
                "close": float(prices.iloc[i]["close"]),
                "volume": float(prices.iloc[i].get("tick_volume", 0)),
            })
        _cached_candles = candles
        _cached_candles_ts = time.time()
    except Exception as e:
        print(f"BG candles error: {e}")

@app.route("/api/status")
def api_status():
    try:
        broker_info = get_account_info()
    except Exception:
        broker_info = {"connected": False}
    sig = _cached_signal if _cached_signal else {"signal": "PENDING", "timestamp": _utciso()}
    if not _cached_signal:
        threading.Thread(target=_bg_compute_signal, daemon=True).start()

    # Update candles in background if stale
    if _cached_candles is None or (time.time() - _cached_candles_ts) > _CANDLES_CACHE_TTL:
        threading.Thread(target=_bg_update_candles, daemon=True).start()

    long_wins = sum(1 for t in trade_history if t.get("side") == "BUY" and (t.get("pnl", 0) > 0))
    long_total = sum(1 for t in trade_history if t.get("side") == "BUY")
    short_wins = sum(1 for t in trade_history if t.get("side") == "SELL" and (t.get("pnl", 0) > 0))
    short_total = sum(1 for t in trade_history if t.get("side") == "SELL")
    wins = long_wins + short_wins
    total = long_total + short_total
    pnl_list = [t.get("pnl", 0) for t in trade_history if t.get("pnl") is not None]

    equity_history = []
    balance = broker_info.get("balance", 10000) if broker_info and broker_info.get("connected") else 10000
    equity = broker_info.get("equity", balance) if broker_info and broker_info.get("connected") else balance
    equity_history.append({"time": _utciso(), "equity": equity})
    for t in trade_history[-100:]:
        equity_history.append({"time": t.get("timestamp", ""), "equity": t.get("equity_after", 10000)})

    perf = {
        "total_trades": total,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "total_pnl": sum(pnl_list) if pnl_list else 0,
        "avg_pnl": float(np.mean(pnl_list)) if pnl_list else 0,
        "best_trade": max(pnl_list) if pnl_list else 0,
        "worst_trade": min(pnl_list) if pnl_list else 0,
        "long_wins": long_wins, "long_total": long_total,
        "short_wins": short_wins, "short_total": short_total,
    }

    return jsonify({
        "broker": broker_info or {},
        "signal": sig,
        "history": trade_history[-50:],
        "performance": perf,
        "equity_history": equity_history[-200:],
        "bot_status": bot_status,
        "candles": _cached_candles or [],
        "markers": get_signal_markers(),
        "analysis": [],  # Client should fetch /api/analysis separately
        "model_age_days": get_model_age_days(),
    })

@app.route("/api/analysis")
def api_analysis():
    return jsonify(get_chart_analysis(200))

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    global _cached_analysis, _cached_analysis_ts
    _cached_analysis = None
    _cached_analysis_ts = 0
    sig = compute_signal(force_refresh=True)
    return jsonify({"signal": sig})

@app.route("/api/trade", methods=["POST"])
def api_trade():
    from trader_d1 import init_broker, run_trading_cycle, shutdown_broker
    if init_broker():
        try:
            result = run_trading_cycle()
            return jsonify({"success": True, "result": str(result)})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
        finally:
            shutdown_broker()
    return jsonify({"success": False, "error": "Broker not connected"})

@app.route("/api/close", methods=["POST"])
def api_close():
    from trader_d1 import init_broker, get_open_position, close_position, shutdown_broker
    if init_broker():
        try:
            pos = get_open_position()
            if pos:
                close_position(pos)
                return jsonify({"success": True})
            return jsonify({"success": False, "error": "No position"})
        finally:
            shutdown_broker()
    return jsonify({"success": False, "error": "Broker not connected"})

@app.route("/api/signal")
def api_signal():
    sig = compute_signal()
    return jsonify(sig)

@app.route("/api/diagnose")
def api_diagnose():
    import logging
    log = logging.getLogger(__name__) if logging.getLogger(__name__).handlers else None
    
    issues = []
    from config import CTRADER_CONFIG
    if not CTRADER_CONFIG.get('access_token'):
        issues.append("Missing access_token in config.py")
    if not CTRADER_CONFIG.get('account_id'):
        issues.append("Missing account_id in config.py")
    if not CTRADER_CONFIG.get('client_id'):
        issues.append("Missing client_id in config.py")
    
    try:
        from broker import init_broker, get_broker
        if init_broker():
            broker = get_broker()
            info = broker.account_info()
            if info:
                return jsonify({
                    "status": "ok",
                    "connected": True,
                    "server": info.server,
                    "balance": info.balance,
                    "account_id": CTRADER_CONFIG.get('account_id'),
                    "issues": issues if issues else None
                })
            else:
                issues.append("Connected but account_info() returned None - wrong account_id?")
        else:
            issues.append("init_broker() returned False - check tokens and network")
    except Exception as e:
        issues.append(f"Connection error: {str(e)}")
    
    return jsonify({
        "status": "error",
        "connected": False,
        "issues": issues,
        "config_account_id": CTRADER_CONFIG.get('account_id'),
        "help": "Get new tokens at https://ctraderapi.com/"
    })

_active_order_info = None

def run_bot_with_dashboard(host="0.0.0.0", port=5000):
    from trader_d1 import build_signal, get_open_position, close_position, place_order, modify_sl, get_current_price, SYMBOL, BREAKEVEN_PIPS, TRAIL_PIPS, MAX_HOLD_BARS

    print(f"\n{'='*60}")
    print(" /$$$$$$$$                                       /$$$$$$  /$$$$$$")
    print("| $$_____/                                      /$$__  $$|_  $$_/")
    print("| $$     /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$| $$  \\ $$  | $$  ")
    print("| $$$$$ /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$$$$$$$  | $$  ")
    print("| $$__/| $$  \\ $$| $$  \\__/| $$$$$$$$ \\  $$$$/ | $$__  $$  | $$  ")
    print("| $$   | $$  | $$| $$      | $$_____/  >$$  $$ | $$  | $$  | $$  ")
    print("| $$   |  $$$$$$/| $$      |  $$$$$$$ /$$/\\  $$| $$  | $$ /$$$$$$")
    print("|__/    \\______/ |__/       \\_______/|__/  \\__/|__/  |__/|______/")
    print(f"  Dashboard: http://localhost:{port} | powered by moonway")
    print(f"{'='*60}")

    if not init_broker():
        print("Cannot connect to broker! Starting dashboard anyway...")
    else:
        global _broker_connected
        _broker_connected = True

    bot_status["running"] = True
    last_trade_date = None
    _last_retrain_date = None

    def bot_loop():
        nonlocal last_trade_date, _last_retrain_date
        global _active_order_info, _cached_signal, _cached_signal_ts, _cached_analysis, _cached_analysis_ts
        while True:
            try:
                now = _utcnow()
                today = now.strftime("%Y-%m-%d")

                if now.hour == 0 and now.minute < 10 and today != last_trade_date:
                    print(f"\n--- D1 bar close {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
                    try:
                        sig = compute_signal()
                        signal = sig.get("signal", "HOLD")
                        bot_status["last_check"] = now.isoformat()
                        bot_status["last_signal_time"] = sig.get("timestamp", "")
                        bot_status["errors"] = bot_status.get("errors", [])[-5:]

                        if signal != "HOLD":
                            sig_info = build_signal()
                            existing = get_open_position()
                            if existing is not None:
                                pos_type = "BUY" if existing.type == 0 else "SELL"
                                same_dir = (pos_type == "BUY" and signal == "BUY") or (pos_type == "SELL" and signal == "SELL")
                                if same_dir:
                                    print(f"  Already in {pos_type} matching {signal}. Holding.")
                                else:
                                    print(f"  Closing {pos_type} (new signal: {signal})...")
                                    close_position(existing)
                                    time.sleep(1)
                                    existing = None

                            if existing is None:
                                order_info = place_order(sig_info)
                                if order_info:
                                    _active_order_info = order_info
                                    try:
                                        broker = get_broker()
                                        info = broker.account_info()
                                        equity_before = info.equity if info else 0
                                    except Exception:
                                        equity_before = 0
                                    trade_data = {
                                        "timestamp": now.isoformat(),
                                        "signal": signal,
                                        "side": signal,
                                        "prob_up": sig.get("prob_up", 0),
                                        "vol_pred": sig.get("vol_pred", 0),
                                        "adx": sig.get("adx", 0),
                                        "equity_before": equity_before,
                                    }
                                    save_trade(trade_data)
                        else:
                            print(f"  Signal: HOLD, no trade.")
                        last_trade_date = today
                    except Exception as e:
                        bot_status["errors"] = bot_status.get("errors", []) + [str(e)]
                        print(f"  Bot trade error: {e}")

                retrain_hour = RETRAIN_CONFIG.get("retrain_hour_utc", 23)
                if now.hour == retrain_hour and _last_retrain_date != today:
                    print(f"\n--- Scheduled retrain at {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
                    try:
                        did_retrain = auto_retrain_if_needed()
                        if did_retrain:
                            _cached_signal = None
                            _cached_signal_ts = 0
                            _cached_analysis = None
                            _cached_analysis_ts = 0
                            print("  Signal/analysis caches invalidated.")
                        _last_retrain_date = today
                    except Exception as e:
                        print(f"  Scheduled retrain error: {e}")

                time.sleep(60)
            except Exception as e:
                bot_status["errors"] = bot_status.get("errors", []) + [str(e)]
                time.sleep(60)

    def trailing_loop():
        global _active_order_info
        while True:
            time.sleep(60)
            try:
                if not _broker_connected:
                    continue
                position = get_open_position()
                if position is None:
                    if _active_order_info is not None:
                        print("  Position closed (TP/SL hit)")
                        _active_order_info = None
                    continue

                is_buy = position.type == 0
                entry = position.price_open
                current_sl = position.sl
                ask, bid = get_current_price()
                if not ask or not bid:
                    continue

                be_price = entry + (1 / 10000 if is_buy else -1 / 10000)
                be_triggered = (is_buy and current_sl >= be_price) or (not is_buy and current_sl <= be_price)

                if is_buy:
                    if be_triggered:
                        trail_sl = bid - TRAIL_PIPS / 10000
                        if trail_sl > current_sl and trail_sl > entry:
                            modify_sl(position, trail_sl)
                    else:
                        unrealized = (bid - entry) * 10000
                        if unrealized >= BREAKEVEN_PIPS:
                            new_sl = entry + 1 / 10000
                            modify_sl(position, new_sl)
                else:
                    if be_triggered:
                        trail_sl = ask + TRAIL_PIPS / 10000
                        if trail_sl < current_sl and trail_sl < entry:
                            modify_sl(position, trail_sl)
                    else:
                        unrealized = (entry - ask) * 10000
                        if unrealized >= BREAKEVEN_PIPS:
                            new_sl = entry - 1 / 10000
                            modify_sl(position, new_sl)
            except Exception as e:
                print(f"  Trailing loop error: {e}")

    def startup_init():
        try:
            need_retrain = _startup_cleanup()
            if need_retrain:
                print("  Starting auto-retrain...", flush=True)
                auto_retrain_if_needed()
            print("  Startup init complete.", flush=True)
        except Exception as e:
            import traceback
            print(f"  Startup init error: {e}", flush=True)
            traceback.print_exc()

    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    keepalive_thread = threading.Thread(target=_broker_keepalive, daemon=True)
    keepalive_thread.start()

    trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
    trailing_thread.start()

    init_thread = threading.Thread(target=startup_init, daemon=True)
    init_thread.start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dashboard.py           - Run bot + dashboard (port 5000)")
        print("  python dashboard.py 8080      - Custom port")
        print("  python dashboard.py dashboard - Dashboard only (no trading)")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "dashboard":
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dashboard.log'),
                logging.StreamHandler()
            ]
        )
        log = logging.getLogger(__name__)
        
        log.info("=" * 60)
        log.info("ForexAI Dashboard Starting...")
        log.info("powered by moonway")
        log.info("=" * 60)
        log.info("Initializing broker connection...")
        
        if init_broker():
            _broker_connected = True
            log.info("✓ Broker connected!")
        else:
            log.error("✗ Broker connection failed. Check tokens in config.py")
            log.error("Get new token: https://ctraderapi.com/")
        
        threading.Thread(target=_broker_keepalive, daemon=True).start()
        load_logs()

        def _dash_init():
            try:
                _startup_cleanup()
            except Exception as e:
                log.error(f"Startup cleanup error: {e}")

        threading.Thread(target=_dash_init, daemon=True).start()
        bot_status["running"] = False
        log.info(f"Dashboard running on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        port = int(cmd) if cmd.isdigit() else 5000
        load_logs()
        run_bot_with_dashboard(port=port)


