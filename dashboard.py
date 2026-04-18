import json
import os
import time
import threading
import datetime
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template_string, jsonify, request

from config import (
    DATA_CONFIG, RISK_CONFIG, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, BACKTEST_CONFIG,
)
from data_loader import build_dataset, load_raw_prices, compute_atr, compute_adx
from broker import init_broker, get_broker, shutdown_broker

VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"
MEANREV_MODEL_PATH = "models/meanrev_model.json"
MEANREV_SCALER_PATH = "models/meanrev_scaler.pkl"
SYMBOL = "EURUSD"
TRADE_LOG = "trade_log.json"
SIGNAL_LOG = "signal_log.json"

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


def ensure_broker():
    global _broker_connected
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
                    print(f"Broker connected: {info.server} Balance=${info.balance:,.2f}")
                    return True
        except Exception as e:
            print(f"Broker init error: {e}")
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
SIGNAL_CACHE_SEC = 60


def compute_signal(force_refresh=False):
    global last_signal, _cached_signal, _cached_signal_ts
    if not force_refresh and _cached_signal is not None and (time.time() - _cached_signal_ts) < SIGNAL_CACHE_SEC:
        return _cached_signal
    try:
        feats_df = build_dataset(force_download=force_refresh)
        prices_df = load_raw_prices()
        common_idx = feats_df.index.intersection(prices_df.index)
        feats_df = feats_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]

        with open(SCALER_SAVE_PATH, "rb") as f:
            scaler = pickle.load(f)
        dir_model = xgb.XGBClassifier()
        dir_model.load_model(MODEL_SAVE_PATH.replace(".pth", ".json"))
        scaled = scaler.transform(feats_df.values)
        dir_prob = dir_model.predict_proba(scaled[-1:].reshape(1, -1))[0, 1]

        vol_model = xgb.XGBRegressor()
        vol_model.load_model(VOL_MODEL_PATH)
        with open(VOL_SCALER_PATH, "rb") as f:
            vol_scaler = pickle.load(f)
        vol_scaled = vol_scaler.transform(feats_df.values)
        vol_pred = vol_model.predict(vol_scaled)
        vol_median = np.median(vol_pred[:int(len(vol_pred) * 0.7)])
        current_vol = vol_pred[-1]
        vol_high = current_vol > vol_median

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

        adx_ok = current_adx >= min_adx
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
            reasons.append(f"ADX {current_adx:.2f} < {min_adx}")
        if signal != "HOLD" and not vol_high:
            signal = "HOLD"
            reasons.append(f"Vol {current_vol:.5f} <= median {vol_median:.5f}")
        if signal == "HOLD" and not reasons:
            reasons.append(f"P(UP)={dir_prob:.3f} in [{no_trade_sell_below}, {no_trade_buy_above}]")

        sig = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "signal": signal,
            "prob_up": float(dir_prob),
            "mr_prob": float(mr_prob),
            "vol_pred": float(current_vol),
            "vol_median": float(vol_median),
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
        err = {"timestamp": datetime.datetime.utcnow().isoformat(), "signal": "ERROR", "error": str(e)}
        _cached_signal = err
        _cached_signal_ts = time.time()
        return err


def get_chart_analysis(n_bars=200):
    global _cached_analysis, _cached_analysis_ts
    now = time.time()
    if _cached_analysis is not None and (now - _cached_analysis_ts) < 300:
        return _cached_analysis

    try:
        feats_df = build_dataset(force_download=False)
        prices_df = load_raw_prices()
        common_idx = feats_df.index.intersection(prices_df.index)
        feats_df = feats_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]

        with open(SCALER_SAVE_PATH, "rb") as f:
            scaler = pickle.load(f)
        dir_model = xgb.XGBClassifier()
        dir_model.load_model(MODEL_SAVE_PATH.replace(".pth", ".json"))
        scaled = scaler.transform(feats_df.values)
        prob_up_all = dir_model.predict_proba(scaled)[:, 1]

        vol_model = xgb.XGBRegressor()
        vol_model.load_model(VOL_MODEL_PATH)
        with open(VOL_SCALER_PATH, "rb") as f:
            vol_scaler = pickle.load(f)
        vol_scaled = vol_scaler.transform(feats_df.values)
        vol_pred_all = vol_model.predict(vol_scaled)
        vol_median = np.median(vol_pred_all[:int(len(vol_pred_all) * 0.7)])

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
            vh = bool(vol_pred_all[i] > vol_median)
            ax = float(adx_series.iloc[i])
            atr = float(atr_series.iloc[i])

            if p > no_trade_buy and ax >= min_adx and vh:
                sig = "BUY"
            elif p < no_trade_sell and ax >= min_adx and vh:
                sig = "SELL"
            elif ax < min_adx:
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
    global _cached_account, _cached_account_ts
    now = time.time()
    if _cached_account and (now - _cached_account_ts) < _ACCOUNT_CACHE_TTL:
        return _cached_account

    if not _broker_connected:
        try:
            with _broker_lock:
                broker = get_broker()
                info = broker.account_info()
        except Exception:
            return _cached_account if _cached_account else {"connected": False}

    try:
        broker = get_broker()
        info = broker.account_info()
        if info is None:
            return _cached_account if _cached_account else {"connected": False}
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
                    "time": p.time.isoformat() if p.time and isinstance(p.time, datetime.datetime) else "",
                })
        tick = broker.symbol_info_tick(SYMBOL)
        result = {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "server": info.server,
            "positions": pos_list,
            "bid": tick.bid if tick else 0,
            "ask": tick.ask if tick else 0,
            "spread": (tick.ask - tick.bid) * 100000 if tick else 0,
            "connected": True,
        }
        _cached_account = result
        _cached_account_ts = now
        return result
    except Exception:
        return _cached_account if _cached_account else {"connected": False}


def get_candles(count=200):
    global cached_candles, cached_candles_ts
    now = time.time()
    if cached_candles is not None and (now - cached_candles_ts) < 30:
        return cached_candles

    try:
        with _broker_lock:
            if not _broker_connected and not ensure_broker():
                return cached_candles or []
            broker = get_broker()
            tf_d1 = broker.TIMEFRAME_D1
            rates = broker.copy_rates_from_pos(SYMBOL, tf_d1, 0, count)
        if rates is None or len(rates) == 0:
            return cached_candles or []

        candles = []
        for r in rates:
            dt = datetime.datetime.fromtimestamp(r["time"])
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
        cached_candles_ts = now
        return candles
    except Exception as e:
        print(f"Error fetching candles: {e}")
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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EUR/USD AI Trader</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{--bg0:#0a0e17;--bg1:#0f1923;--bg2:#141e30;--bg3:#1a2744;--txt:#e0e6ed;--txt2:#8892a4;--txt3:#4a5568;--blu:#448aff;--grn:#00e676;--red:#ff1744;--yel:#ffc400;--pur:#7c4dff;--bdr:#1e3050}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg0);color:var(--txt);overflow-x:hidden}
.mono{font-family:'JetBrains Mono',monospace}
.wrap{max-width:1800px;margin:0 auto;padding:8px 12px}

.topbar{display:flex;justify-content:space-between;align-items:center;padding:10px 20px;background:linear-gradient(135deg,var(--bg2),var(--bg3));border-radius:14px;margin-bottom:8px;border:1px solid var(--bdr)}
.brand{display:flex;align-items:center;gap:12px}
.brand-name{font-size:15px;font-weight:800;letter-spacing:1px;background:linear-gradient(135deg,var(--blu),var(--pur));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.pair-info{display:flex;align-items:baseline;gap:10px;margin-left:16px}
.live-price{font-size:32px;font-weight:800;font-family:'JetBrains Mono',monospace;letter-spacing:-1px}
.live-price.up{color:var(--grn)}.live-price.dn{color:var(--red)}
.price-chg{font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace}
.price-chg.up{color:var(--grn)}.price-chg.dn{color:var(--red)}
.topbar-r{display:flex;align-items:center;gap:12px}
.pill{display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;background:var(--bg3);font-size:11px;font-weight:600;color:var(--txt2)}
.dot{width:7px;height:7px;border-radius:50%}
.dot-on{background:var(--grn);box-shadow:0 0 8px var(--grn);animation:blink 2s infinite}
.dot-off{background:var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

.layout{display:grid;grid-template-columns:1fr 320px;gap:8px;margin-bottom:8px}

.chart-box{background:var(--bg1);border-radius:14px;border:1px solid var(--bdr);overflow:hidden;display:flex;flex-direction:column}
.chart-hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid var(--bdr);background:var(--bg2)}
.chart-hdr .sym{font-size:13px;font-weight:800;letter-spacing:0.5px}
.chart-hdr .ohlc{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--txt2);display:flex;gap:8px}
.chart-hdr .ohlc span{display:inline-flex;align-items:center;gap:2px}
.ohlc .lbl{color:var(--txt3);font-size:9px}



.side{display:flex;flex-direction:column;gap:8px}

.sig-card{background:var(--bg1);border-radius:14px;padding:16px;position:relative;overflow:hidden;border:1px solid var(--bdr);transition:border-color .3s}
.sig-card.buy{border-color:rgba(0,230,118,.4);box-shadow:0 0 30px rgba(0,230,118,.06)}
.sig-card.sell{border-color:rgba(255,23,68,.4);box-shadow:0 0 30px rgba(255,23,68,.06)}
.sig-card.hold{border-color:rgba(255,196,0,.3);box-shadow:0 0 30px rgba(255,196,0,.04)}
.sig-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.sig-card.buy::before{background:linear-gradient(90deg,#00e676,#00c853)}
.sig-card.sell::before{background:linear-gradient(90deg,#ff1744,#d50000)}
.sig-card.hold::before{background:linear-gradient(90deg,#ffc400,#ff9100)}

.sig-main{text-align:center;padding:6px 0 4px}
.sig-label{font-size:9px;text-transform:uppercase;letter-spacing:2px;color:var(--txt3);margin-bottom:2px}
.sig-val{font-size:44px;font-weight:900;letter-spacing:3px}
.sig-card.buy .sig-val{color:var(--grn);text-shadow:0 0 40px rgba(0,230,118,.3)}
.sig-card.sell .sig-val{color:var(--red);text-shadow:0 0 40px rgba(255,23,68,.3)}
.sig-card.hold .sig-val{color:var(--yel);text-shadow:0 0 40px rgba(255,196,0,.3)}

.prob-track{height:5px;border-radius:3px;background:var(--bg3);margin:6px 0 2px;position:relative}
.prob-fill{height:100%;border-radius:3px;transition:width .6s ease}
.prob-labels{display:flex;justify-content:space-between;font-size:9px;color:var(--txt3);margin-bottom:8px;font-family:'JetBrains Mono',monospace}

.filters{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.f-item{background:var(--bg3);border-radius:8px;padding:7px 9px}
.f-item .f-lbl{font-size:8px;text-transform:uppercase;letter-spacing:1px;color:var(--txt3);margin-bottom:1px}
.f-item .f-val{font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace}
.f-item .f-val.g{color:var(--grn)}.f-item .f-val.r{color:var(--red)}.f-item .f-val.y{color:var(--yel)}.f-item .f-val.b{color:var(--blu)}
.filter-row{display:flex;align-items:center;gap:4px;margin-top:6px;padding:4px 8px;border-radius:6px;background:var(--bg3);font-size:11px}
.filter-row.pass{color:var(--grn)}.filter-row.block{color:var(--red)}
.filter-dot{width:6px;height:6px;border-radius:50%}
.filter-dot.on{background:var(--grn)}.filter-dot.off{background:var(--red)}
.ai-thinking{margin-top:6px;padding:6px 8px;border-radius:6px;background:var(--bg3);font-size:10px;color:var(--txt2);line-height:1.5;font-family:'JetBrains Mono',monospace}
.ai-thinking b.g{color:var(--grn)}.ai-thinking b.r{color:var(--red)}.ai-thinking b.y{color:var(--yel)}

.acct-box{background:var(--bg1);border-radius:14px;border:1px solid var(--bdr);padding:12px}
.acct-title{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--txt3);margin-bottom:8px}
.acct-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.a-item{background:var(--bg3);border-radius:8px;padding:7px 9px}
.a-item .a-lbl{font-size:8px;text-transform:uppercase;letter-spacing:1px;color:var(--txt3);margin-bottom:1px}
.a-item .a-val{font-size:16px;font-weight:800;font-family:'JetBrains Mono',monospace}

.pos-box{background:var(--bg1);border-radius:14px;border:1px solid var(--bdr);padding:0;overflow:hidden;margin-bottom:8px}
.sec-hdr{padding:8px 14px;border-bottom:1px solid var(--bdr);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--txt3);display:flex;justify-content:space-between;align-items:center}
.sec-hdr .cnt{color:var(--txt2);font-weight:500}
table{width:100%;border-collapse:collapse;font-size:11px}
th{background:var(--bg3);padding:6px 10px;text-align:left;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--txt3);position:sticky;top:0}
td{padding:5px 10px;border-bottom:1px solid var(--bdr);font-family:'JetBrains Mono',monospace;font-size:11px}
tr:hover td{background:var(--bg3)}
.g{color:var(--grn)!important}.r{color:var(--red)!important}.y{color:var(--yel)!important}.b{color:var(--blu)!important}
.empty{text-align:center;padding:16px;color:var(--txt3);font-size:12px}

.bottom{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;align-items:start}
.perf-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.perf-box{background:var(--bg1);border-radius:14px;border:1px solid var(--bdr);padding:12px}
.perf-box .sec-hdr{padding:0 4px 8px;margin-bottom:0;border:none}
.p-item{background:var(--bg3);border-radius:10px;border:1px solid var(--bdr);padding:8px 10px}
.p-item .p-lbl{font-size:8px;text-transform:uppercase;letter-spacing:1px;color:var(--txt3);margin-bottom:1px}
.p-item .p-val{font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace}

.chart-wrap{background:var(--bg1);border-radius:14px;border:1px solid var(--bdr);overflow:hidden;margin-bottom:8px}
.chart-wrap .sec-hdr{padding:8px 14px;border-bottom:1px solid var(--bdr)}


.foot{text-align:center;padding:6px;color:var(--txt3);font-size:10px}
@media(max-width:1200px){.layout{grid-template-columns:1fr}}
@media(max-width:768px){.bottom{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-name">EUR/USD AI TRADER</div>
      <div class="pair-info">
        <div class="live-price mono" id="live-price">--</div>
        <div class="price-chg mono" id="price-chg">--</div>
      </div>
    </div>
    <div class="topbar-r">
      <div class="pill"><div class="dot" id="conn-dot"></div><span id="conn-txt">Connecting...</span></div>
      <div class="pill" id="server-name">--</div>
    </div>
  </div>

  <div class="layout">
    <div class="chart-box">
      <div class="chart-hdr">
        <div class="sym">EUR/USD &middot; D1</div>
        <div class="ohlc" id="ohlc-bar">
          <span><span class="lbl">O</span> <span id="oh_o">--</span></span>
          <span><span class="lbl">H</span> <span id="oh_h">--</span></span>
          <span><span class="lbl">L</span> <span id="oh_l">--</span></span>
          <span><span class="lbl">C</span> <span id="oh_c">--</span></span>
        </div>
      </div>
      <div id="priceChart"></div>
      <div id="probChart"></div>
    </div>

    <div class="side">
      <div class="sig-card hold" id="sig-card">
        <div class="sig-main">
          <div class="sig-label">AI Signal</div>
          <div class="sig-val" id="sig-val">--</div>
        </div>
        <div class="prob-track"><div class="prob-fill" id="prob-fill" style="width:50%"></div></div>
        <div class="prob-labels"><span>0 &larr; SELL</span><span id="prob-num">0.50</span><span>BUY &rarr; 1</span></div>
        <div class="filters" id="filters">
          <div class="f-item"><div class="f-lbl">Vol Filter</div><div class="f-val" id="f-vol">--</div></div>
          <div class="f-item"><div class="f-lbl">ADX</div><div class="f-val" id="f-adx">--</div></div>
          <div class="f-item"><div class="f-lbl">ATR</div><div class="f-val" id="f-atr">--</div></div>
          <div class="f-item"><div class="f-lbl">SL / TP</div><div class="f-val" id="f-sltp">--</div></div>
          <div class="f-item"><div class="f-lbl">Ratio</div><div class="f-val" id="f-ratio">--</div></div>
          <div class="f-item"><div class="f-lbl">Trailing</div><div class="f-val" id="f-trail">--</div></div>
        </div>
        <div id="filter-rows"></div>
        <div class="ai-thinking" id="ai-log">Waiting...</div>
      </div>

      <div class="acct-box">
        <div class="acct-title">Account</div>
        <div class="acct-grid">
          <div class="a-item"><div class="a-lbl">Balance</div><div class="a-val" id="bal">--</div></div>
          <div class="a-item"><div class="a-lbl">Equity</div><div class="a-val" id="eq">--</div></div>
          <div class="a-item"><div class="a-lbl">Free Margin</div><div class="a-val" id="fm">--</div></div>
          <div class="a-item"><div class="a-lbl">Spread</div><div class="a-val" id="sp">--</div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="pos-box">
    <div class="sec-hdr">Open Positions <span class="cnt" id="pos-cnt">0</span></div>
    <table><thead><tr><th>Time</th><th>Type</th><th>Lots</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>P&L</th></tr></thead>
    <tbody id="pos-body"></tbody></table>
    <div class="empty" id="no-pos">No open positions</div>
  </div>

  <div class="bottom">
    <div class="perf-box">
      <div class="sec-hdr" style="padding:0 4px;margin-bottom:6px">Performance</div>
      <div class="perf-grid">
        <div class="p-item"><div class="p-lbl">Trades</div><div class="p-val" id="p-tr">0</div></div>
        <div class="p-item"><div class="p-lbl">Win Rate</div><div class="p-val" id="p-wr">--</div></div>
        <div class="p-item"><div class="p-lbl">Total P&L</div><div class="p-val" id="p-pnl">$0</div></div>
        <div class="p-item"><div class="p-lbl">Avg P&L</div><div class="p-val" id="p-avg">$0</div></div>
        <div class="p-item"><div class="p-lbl">Long</div><div class="p-val" id="p-long">0/0</div></div>
        <div class="p-item"><div class="p-lbl">Short</div><div class="p-val" id="p-short">0/0</div></div>
      </div>
    </div>
    <div class="pos-box">
      <div class="sec-hdr">Recent Signals</div>
      <div style="max-height:200px;overflow-y:auto">
        <table><thead><tr><th>Time</th><th>Signal</th><th>P(UP)</th><th>Vol</th><th>ADX</th><th>P&L</th></tr></thead>
        <tbody id="sig-body"></tbody></table>
      </div>
    </div>
  </div>

  <div class="chart-wrap">
    <div class="sec-hdr">Equity Curve</div>
    <canvas id="equityChart"></canvas>
  </div>

  <div class="foot">Auto-refresh 10s &middot; Last: <span id="last-upd">--</span> &middot; XGBoost + Vol Filter + Trailing Stop</div>
</div>

<script>
let equityChart=null;
let priceChart=null,probChart=null;
let candleSeries=null,volSeries=null,probLine=null,probThresh60=null,probThresh45=null;

function initCharts(){
    const pc=document.getElementById('priceChart');
    priceChart=LightweightCharts.createChart(pc,{
        width:pc.clientWidth,height:350,
        layout:{background:{type:'solid',color:'#0c1017'},textColor:'#8899b4',fontSize:11,fontFamily:'JetBrains Mono'},
        grid:{vertLines:{color:'#1a223540'},horzLines:{color:'#1a223540'}},
        crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:'#448aff55',style:2},horzLine:{color:'#448aff55',style:2}},
        rightPriceScale:{borderColor:'#1e2d4a',scaleMargins:{top:0.05,bottom:0.15}},
        timeScale:{borderColor:'#1e2d4a',timeVisible:false},
    });
    candleSeries=priceChart.addCandlestickSeries({
        upColor:'#00e676',downColor:'#ff1744',borderUpColor:'#00e676',borderDownColor:'#ff1744',
        wickUpColor:'#00e67680',wickDownColor:'#ff174480',
    });
    volSeries=priceChart.addHistogramSeries({
        priceFormat:{type:'volume'},priceScaleId:'',
    });
    volSeries.priceScale().applyOptions({scaleMargins:{top:0.88,bottom:0}});

    const prc=document.getElementById('probChart');
    probChart=LightweightCharts.createChart(prc,{
        width:prc.clientWidth,height:120,
        layout:{background:{type:'solid',color:'#0a0d12'},textColor:'#8899b4',fontSize:10,fontFamily:'JetBrains Mono'},
        grid:{vertLines:{color:'#1a223520'},horzLines:{color:'#1a223520'}},
        rightPriceScale:{borderColor:'#1e2d4a',scaleMargins:{top:0.05,bottom:0.05}},
        timeScale:{borderColor:'#1e2d4a',timeVisible:false},
    });
    probLine=probChart.addLineSeries({
        color:'#b388ff',lineWidth:2,lastValueVisible:true,priceLineVisible:true,
        priceFormat:{type:'price',precision:2,minMove:0.01},
    });
    // Threshold lines for BUY/SELL zones
    probThresh60=probChart.addLineSeries({color:'rgba(0,230,118,0.4)',lineWidth:1,lineStyle:2,lineVisible:true});
    probThresh45=probChart.addLineSeries({color:'rgba(255,23,68,0.4)',lineWidth:1,lineStyle:2,lineVisible:true});
}

function renderCandles(candles){
    if(!candleSeries||!candles||!candles.length)return;
    candleSeries.setData(candles.map(c=>({time:c.date,open:c.open,high:c.high,low:c.low,close:c.close})));
    volSeries.setData(candles.map(c=>({
        time:c.date,value:c.volume,
        color:c.close>=c.open?'rgba(0,230,118,0.15)':'rgba(255,23,68,0.15)',
    })));
    const l=candles[candles.length-1];
    document.getElementById('oh_o').textContent=l.open.toFixed(5);
    document.getElementById('oh_h').textContent=l.high.toFixed(5);
    document.getElementById('oh_l').textContent=l.low.toFixed(5);
    const cEl=document.getElementById('oh_c');
    cEl.textContent=l.close.toFixed(5);
    cEl.className=l.close>=l.open?'g':'r';
    priceChart.timeScale().fitContent();
}

function renderAnalysis(analysis){
    if(!analysis||!analysis.length||!probLine)return;
    const data=analysis.map(a=>({time:a.date,value:a.prob_up}));
    probLine.setData(data);

    probThresh60.setData(analysis.map(a=>({time:a.date,value:0.60})));
    probThresh45.setData(analysis.map(a=>({time:a.date,value:0.45})));

    const markers=[];
    for(const a of analysis){
        if(a.signal==='BUY')markers.push({time:a.date,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'B '+Math.round(a.prob_up*100)+'%'});
        else if(a.signal==='SELL')markers.push({time:a.date,position:'aboveBar',color:'#ff1744',shape:'arrowDown',text:'S '+Math.round(a.prob_up*100)+'%'});
    }
    candleSeries.setMarkers(markers.sort((a,b)=>a.time.localeCompare(b.time)));
    probChart.timeScale().fitContent();
}

function renderMarkers(markers){
    if(!candleSeries||!markers||!markers.length)return;
}

function updateUI(data){
    const brk=data.broker,sig=data.signal,perf=data.performance||{};hist=data.history||[];

    const conn=brk&&brk.connected;
    document.getElementById('conn-dot').className='dot '+(conn?'dot-on':'dot-off');
    document.getElementById('conn-txt').textContent=conn?'cTrader Live':'Offline';
    document.getElementById('server-name').textContent=conn?brk.server:'--';

    const lp=document.getElementById('live-price');
    const pc=document.getElementById('price-chg');
    if(conn&&brk.bid){
        lp.textContent=brk.bid.toFixed(5);
        if(data.candles&&data.candles.length>1){
            const prev=data.candles[data.candles.length-2].close;
            const d=brk.bid-prev;
            const pct=d/prev*100;
            pc.textContent=(d>=0?'+':'')+( d*10000).toFixed(1)+'p '+(pct>=0?'+':'')+pct.toFixed(3)+'%';
            pc.className='price-chg mono '+(d>=0?'up':'dn');
            lp.className='live-price mono '+(d>=0?'up':'dn');
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
        sc.className='sig-card '+(s==='BUY'?'buy':s==='SELL'?'sell':'hold');

        const p=sig.prob_up||0;
        document.getElementById('prob-num').textContent=p.toFixed(4);
        const pf=document.getElementById('prob-fill');
        pf.style.width=(p*100)+'%';
        pf.style.background=p>0.6?'linear-gradient(90deg,#00e676,#00c853)':p<0.45?'linear-gradient(90deg,#ff1744,#d50000)':'linear-gradient(90deg,#ffc400,#ff9100)';

        const fv=document.getElementById('f-vol');
        fv.textContent=sig.vol_high?'PASS':'BLOCK';
        fv.className='f-val '+(sig.vol_high?'g':'r');

        const fa=document.getElementById('f-adx');
        fa.textContent=(sig.adx||0).toFixed(2);
        fa.className='f-val '+((sig.adx||0)>=0.3?'g':'y');

        document.getElementById('f-atr').textContent=(sig.atr_pips||0).toFixed(1)+'p';
        document.getElementById('f-sltp').textContent=(sig.sl_pips||0).toFixed(0)+'/'+(sig.tp_pips||0).toFixed(0)+'p';
        document.getElementById('f-ratio').textContent=(sig.dyn_tp_sl||0).toFixed(1)+'x';
        document.getElementById('f-trail').textContent='BE'+(sig.breakeven||15)+' TR'+(sig.trail||10);

        let frHTML='';
        const adxOk=(sig.adx||0)>=0.3;
        frHTML+='<div class="filter-row '+(adxOk?'pass':'block')+'"><div class="filter-dot '+(adxOk?'on':'off')+'"></div> ADX '+(sig.adx||0).toFixed(2)+' '+(adxOk?'>= 0.30':'< 0.30')+'</div>';
        frHTML+='<div class="filter-row '+(sig.vol_high?'pass':'block')+'"><div class="filter-dot '+(sig.vol_high?'on':'off')+'"></div> Vol '+(sig.vol_high?'> median':'<= median')+'</div>';
        if(sig.reasons&&sig.reasons.length){
            sig.reasons.forEach(r=>{
                frHTML+='<div class="filter-row block"><div class="filter-dot off"></div> '+r+'</div>';
            });
        }
        document.getElementById('filter-rows').innerHTML=frHTML;

        const pClass=s>0.6?'g':p<0.45?'r':'y';
        document.getElementById('ai-log').innerHTML='P(UP)=<b class="'+pClass+'">'+p.toFixed(4)+'</b> | Vol=<b class="'+(sig.vol_high?'g':'r')+'">'+(sig.vol_high?'HIGH':'LOW')+'</b> | ADX=<b class="'+(adxOk?'g':'r')+'">'+(sig.adx||0).toFixed(2)+'</b> &rarr; <b class="'+pClass+'">'+s+'</b>';
    }

    const posB=document.getElementById('pos-body');
    const noP=document.getElementById('no-pos');
    if(brk&&brk.positions&&brk.positions.length>0){
        noP.style.display='none';
        document.getElementById('pos-cnt').textContent=brk.positions.length;
        posB.innerHTML=brk.positions.map(p=>{
            const pc=p.profit>=0?'g':'r';
            return '<tr><td>'+p.time.substring(11,19)+'</td><td class="'+(p.type==='BUY'?'g':'r')+'" style="font-weight:700">'+p.type+'</td><td>'+p.volume.toFixed(2)+'</td><td>'+p.price_open.toFixed(5)+'</td><td>'+p.price_current.toFixed(5)+'</td><td>'+p.sl.toFixed(5)+'</td><td>'+p.tp.toFixed(5)+'</td><td class="'+pc+'" style="font-weight:700">$'+p.profit.toFixed(2)+'</td></tr>';
        }).join('');
    }else{posB.innerHTML='';noP.style.display='block';document.getElementById('pos-cnt').textContent='0';}

    const wr=perf.win_rate||0;
    const pnl=perf.total_pnl||0;
    document.getElementById('p-tr').textContent=perf.total_trades||0;
    const wrEl=document.getElementById('p-wr');wrEl.textContent=wr?wr.toFixed(1)+'%':'--';
    wrEl.className='p-val '+(wr>=60?'g':wr>=50?'':'r');
    const pnlEl=document.getElementById('p-pnl');pnlEl.textContent=(pnl>=0?'+':'')+'$'+Math.abs(pnl).toFixed(0);
    pnlEl.className='p-val '+(pnl>=0?'g':'r');
    document.getElementById('p-avg').textContent='$'+(perf.avg_pnl||0).toFixed(2);
    document.getElementById('p-long').textContent=(perf.long_wins||0)+'/'+(perf.long_total||0);
    document.getElementById('p-short').textContent=(perf.short_wins||0)+'/'+(perf.short_total||0);

    if(hist.length>0){
        document.getElementById('sig-body').innerHTML=hist.slice(-20).reverse().map(h=>{
            const sc=h.signal==='BUY'?'g':h.signal==='SELL'?'r':'y';
            const pn=h.pnl||0;
            const pnC=pn>=0?'g':'r';
            return '<tr><td>'+(h.timestamp||'').substring(5,16)+'</td><td class="'+sc+'" style="font-weight:700">'+h.signal+'</td><td>'+(h.prob_up||0).toFixed(3)+'</td><td>'+(h.vol_pred?(h.vol_pred*10000).toFixed(1):'--')+'</td><td>'+(h.adx||0).toFixed(2)+'</td><td class="'+pnC+'">'+(pn!==0?'$'+pn.toFixed(2):'--')+'</td></tr>';
        }).join('');
    }

    const eqD=data.equity_history||[];
    if(eqD.length>1)updateEquity(eqD);
    document.getElementById('last-upd').textContent=new Date().toLocaleTimeString();
}

function updateEquity(data){
    const ctx=document.getElementById('equityChart').getContext('2d');
    if(equityChart)equityChart.destroy();
    const labels=data.map(e=>{const d=new Date(e.time);return(d.getMonth()+1)+'/'+d.getDate()});
    equityChart=new Chart(ctx,{
        type:'line',
        data:{labels:labels,datasets:[{label:'Equity',data:data.map(e=>e.equity),borderColor:'#448aff',backgroundColor:'rgba(68,138,255,0.06)',fill:true,tension:0.4,pointRadius:0,borderWidth:2}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{
            x:{grid:{color:'#1a223540'},ticks:{color:'#4e6082',maxTicksLimit:8,font:{size:9,family:'JetBrains Mono'}}},
            y:{grid:{color:'#1a223540'},ticks:{color:'#4e6082',callback:v=>'$'+v.toLocaleString(),font:{size:9,family:'JetBrains Mono'}}}
        }}
    });
}

async function fetchStatus(){
    try{
        const r=await fetch('/api/status');
        if(!r.ok) throw new Error('HTTP '+r.status);
        const data=await r.json();
        if(data.candles&&data.candles.length>0){
            if(!candleSeries)initCharts();
            renderCandles(data.candles);
            renderAnalysis(data.analysis||[]);
            renderMarkers(data.markers||[]);
        }
        updateUI(data);
    }catch(e){
        document.getElementById('conn-dot').className='dot dot-off';
        document.getElementById('conn-txt').textContent='Reconnecting...';
        console.error('fetchStatus error:',e);
    }
}

window.addEventListener('resize',()=>{
    if(priceChart)priceChart.applyOptions({width:document.getElementById('priceChart').clientWidth});
    if(probChart)probChart.applyOptions({width:document.getElementById('probChart').clientWidth});
});

setInterval(fetchStatus,10000);
fetchStatus();
if(typeof LightweightCharts!=='undefined'){initCharts();}
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/status")
def api_status():
    broker_info = get_account_info()
    sig = _cached_signal if _cached_signal else {"signal": "PENDING", "timestamp": datetime.datetime.utcnow().isoformat()}
    if not _cached_signal:
        threading.Thread(target=_bg_compute_signal, daemon=True).start()

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
    equity_history.append({"time": datetime.datetime.utcnow().isoformat(), "equity": equity})
    for t in trade_history[-100:]:
        equity_history.append({
            "time": t.get("timestamp", ""),
            "equity": t.get("equity_after", 10000),
        })

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

    candles = get_candles(200)
    markers = get_signal_markers()
    analysis = get_chart_analysis(200)

    return jsonify({
        "broker": broker_info or {},
        "signal": sig,
        "history": trade_history[-50:],
        "performance": perf,
        "equity_history": equity_history[-200:],
        "bot_status": bot_status,
        "candles": candles,
        "markers": markers,
        "analysis": analysis,
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


def run_bot_with_dashboard(host="0.0.0.0", port=5000):
    from trader_d1 import init_broker, build_signal, get_open_position, close_position, place_order, SYMBOL, BREAKEVEN_PIPS, TRAIL_PIPS

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

    bot_status["running"] = True
    last_trade_date = None

    def bot_loop():
        nonlocal last_trade_date
        while True:
            try:
                now = datetime.datetime.utcnow()
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
                            from trader_d1 import init_broker as tb_init2, run_trading_cycle, shutdown_broker as tb_shut2
                            if tb_init2():
                                try:
                                    result = run_trading_cycle()
                                    if result:
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
                                finally:
                                    tb_shut2()
                        last_trade_date = today
                    except Exception as e:
                        bot_status["errors"] = bot_status.get("errors", []) + [str(e)]

                time.sleep(60)
            except Exception as e:
                bot_status["errors"] = bot_status.get("errors", []) + [str(e)]
                time.sleep(60)

    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

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
        print("=" * 60)
        print(" /$$$$$$$$                                       /$$$$$$  /$$$$$$")
        print("| $$_____/                                      /$$__  $$|_  $$_/")
        print("| $$     /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$| $$  \\ $$  | $$  ")
        print("| $$$$$ /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$$$$$$$  | $$  ")
        print("| $$__/| $$  \\ $$| $$  \\__/| $$$$$$$$ \\  $$$$/ | $$__  $$  | $$  ")
        print("| $$   | $$  | $$| $$      | $$_____/  >$$  $$ | $$  | $$  | $$  ")
        print("| $$   |  $$$$$$/| $$      |  $$$$$$$ /$$/\\  $$| $$  | $$ /$$$$$$")
        print("|__/    \\______/ |__/       \\_______/|__/  \\__/|__/  |__/|______/")
        print("powered by moonway")
        print("=" * 60)
        threading.Thread(target=ensure_broker, daemon=True).start()
        load_logs()
        bot_status["running"] = False
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        port = int(cmd) if cmd.isdigit() else 5000
        load_logs()
        run_bot_with_dashboard(port=port)