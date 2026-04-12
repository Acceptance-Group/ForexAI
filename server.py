import os
import pickle
import threading
import time
import json
from datetime import datetime, timedelta

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string

from config import DATA_CONFIG, DEVICE, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLUMNS, FEATURE_WEIGHTS, RISK_CONFIG
from data_loader import load_dataset, load_raw_prices, build_dataset
from model import ForexClassifier

app = Flask(__name__)

latest_data = None
last_update = None
model = None
scaler = None
lock = threading.Lock()


def load_model_and_scalers():
    global model, scaler
    if model is None:
        model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        model.eval()

    if scaler is None and os.path.exists(SCALER_SAVE_PATH):
        with open(SCALER_SAVE_PATH, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler


def update_data():
    global latest_data, last_update
    while True:
        try:
            print(f"[{datetime.now()}] Updating data...")
            build_dataset()
            feats = load_dataset()
            prices = load_raw_prices()["EUR_USD"].values

            with lock:
                latest_data = {
                    "features": feats,
                    "prices": prices,
                    "timestamp": datetime.now(),
                }
                last_update = datetime.now()
            print(f"[{datetime.now()}] Data updated successfully")
        except Exception as e:
            print(f"[{datetime.now()}] Error updating data: {e}")
        time.sleep(600)


def get_prediction_data():
    model, scaler = load_model_and_scalers()
    with lock:
        if latest_data is None:
            return None
        feats = latest_data["features"]
        prices = latest_data["prices"]

    scaled = scaler.transform(feats.values)
    lookback = DATA_CONFIG["lookback"]
    current_price = prices[-1]
    no_trade_low = DATA_CONFIG["no_trade_low"]
    no_trade_high = DATA_CONFIG["no_trade_high"]
    risk_pct = RISK_CONFIG["risk_per_trade"]
    max_pos_frac = RISK_CONFIG["max_position_fraction"]

    with torch.no_grad():
        X_now = torch.tensor(scaled[-lookback:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prob_now = model.predict_proba(X_now).detach().cpu().item()

    direction = "UP" if prob_now > 0.5 else "DOWN"
    if prob_now > no_trade_high:
        signal_str = "BUY"
    elif prob_now < no_trade_low:
        signal_str = "SELL"
    else:
        signal_str = "HOLD"

    confidence = abs(prob_now - 0.5) * 2.0

    daily_ranges = np.diff(prices[-20:])
    atr = np.mean(np.abs(daily_ranges))
    atr_pct = atr / current_price * 100

    sl_distance = atr
    sl_pct = sl_distance / current_price if current_price > 0 else 0.01
    pos_fraction = min(risk_pct / sl_pct, max_pos_frac) if sl_pct > 0 else 0.0

    if signal_str == "BUY":
        sl_price = current_price - sl_distance
        tp_price = current_price * (1 + DATA_CONFIG["barrier_tp"])
    elif signal_str == "SELL":
        sl_price = current_price + sl_distance
        tp_price = current_price * (1 - DATA_CONFIG["barrier_tp"])
    else:
        sl_price = 0.0
        tp_price = 0.0

    n_days = min(30, len(feats) - lookback)
    recent_probs = []
    recent_dates = []
    recent_prices_list = []

    with torch.no_grad():
        for i in range(n_days):
            idx = len(feats) - n_days + i
            if idx < lookback:
                continue
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p = model.predict_proba(X).cpu().item()
            recent_probs.append(round(p * 100, 1))
            recent_dates.append(feats.index[idx].strftime("%Y-%m-%d"))
            recent_prices_list.append(round(prices[idx], 5))

    return {
        "timestamp": datetime.now().isoformat(),
        "current_price": round(current_price, 5),
        "direction": direction,
        "probability_up": round(prob_now * 100, 1),
        "probability_down": round((1 - prob_now) * 100, 1),
        "confidence": round(confidence * 100, 1),
        "signal": signal_str,
        "no_trade_low": no_trade_low,
        "no_trade_high": no_trade_high,
        "pos_fraction": round(pos_fraction, 3),
        "sl_price": round(sl_price, 5) if sl_price > 0 else None,
        "tp_price": round(tp_price, 5) if tp_price > 0 else None,
        "atr_value": round(atr, 5),
        "atr_pct": round(atr_pct, 4),
        "risk_per_trade": risk_pct,
        "temperature": model.temperature,
        "recent_probs": recent_probs,
        "recent_dates": recent_dates,
        "recent_prices": recent_prices_list,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EUR/USD Forex Classifier</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0d1117; color:#c9d1d9; font-family:'Segoe UI',system-ui,sans-serif; }
.header { background:#161b22; border-bottom:1px solid #30363d; padding:16px 24px; display:flex; align-items:center; justify-content:space-between; }
.header h1 { font-size:20px; color:#58a6ff; }
.status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; }
.status-dot.ok { background:#3fb950; }
.status-dot.err { background:#f85149; }
.signal-badge { padding:6px 16px; border-radius:6px; font-weight:700; font-size:18px; display:inline-block; }
.signal-buy { background:#1a6e2e; color:#3fb950; }
.signal-sell { background:#7d1a1a; color:#f85149; }
.signal-hold { background:#3d3d00; color:#d29922; }
.grid { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:16px; padding:20px; }
.card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }
.card h2 { font-size:13px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.card .value { font-size:28px; font-weight:700; }
.card .value.up { color:#3fb950; }
.card .value.down { color:#f85149; }
.card .value.neutral { color:#d29922; }
.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 20px 20px; }
.chart-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; }
.risk-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
.risk-item { text-align:center; padding:8px; background:#0d1117; border-radius:6px; }
.risk-item .label { font-size:11px; color:#8b949e; }
.risk-item .val { font-size:16px; font-weight:600; }
.refresh-info { font-size:12px; color:#8b949e; }
</style>
</head>
<body>
<div class="header">
  <h1>EUR/USD Forex Classifier</h1>
  <div>
    <span class="status-dot ok" id="statusDot"></span>
    <span class="refresh-info" id="refreshInfo">Loading...</span>
  </div>
</div>

<div class="grid">
  <div class="card" id="cardSignal">
    <h2>Signal</h2>
    <div id="signalBadge" class="signal-badge signal-hold">HOLD</div>
    <div style="margin-top:8px;font-size:14px;" id="directionText">--</div>
  </div>
  <div class="card">
    <h2>Probability</h2>
    <div class="value" id="probValue">--%</div>
    <div style="font-size:13px;color:#8b949e;margin-top:4px;">
      UP: <span id="probUp">--</span>% | DOWN: <span id="probDown">--</span>%
    </div>
  </div>
  <div class="card">
    <h2>Confidence</h2>
    <div class="value" id="confValue">--%</div>
    <div style="font-size:13px;color:#8b949e;margin-top:4px;">
      Thresholds: UP &gt; <span id="thrUp">--</span> | DN &lt; <span id="thrDn">--</span> | Dead Zone
    </div>
  </div>
  <div class="card">
    <h2>Price</h2>
    <div class="value" id="priceValue">--</div>
    <div class="risk-grid">
      <div class="risk-item"><div class="label">Pos Frac</div><div class="val" id="lotVal">--</div></div>
      <div class="risk-item"><div class="label">ATR %</div><div class="val" id="atrVal">--</div></div>
      <div class="risk-item"><div class="label">SL</div><div class="val" id="slVal">--</div></div>
      <div class="risk-item"><div class="label">TP</div><div class="val" id="tpVal">--</div></div>
    </div>
  </div>
</div>

<div class="charts-row">
  <div class="chart-card"><div id="chartPrice" style="height:380px;"></div></div>
  <div class="chart-card"><div id="chartProb" style="height:380px;"></div></div>
</div>
<div class="charts-row">
  <div class="chart-card"><div id="chartSignalStrength" style="height:380px;"></div></div>
  <div class="chart-card"><div id="chartDistribution" style="height:380px;"></div></div>
</div>

<script>
const PLOTLY_LAYOUT = {
  paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
  font:{color:'#8b949e',size:12},
  xaxis:{gridcolor:'#21262d'}, yaxis:{gridcolor:'#21262d'},
  margin:{l:50,r:20,t:40,b:50},
};

function updateDashboard() {
  fetch('/api/predict').then(r=>r.json()).then(d=>{
    document.getElementById('refreshInfo').textContent = 'Updated: ' + new Date().toLocaleTimeString();
    document.getElementById('statusDot').className = 'status-dot ok';

    const sig = d.signal;
    const badge = document.getElementById('signalBadge');
    badge.textContent = sig;
    badge.className = 'signal-badge ' + (sig==='BUY'?'signal-buy':sig==='SELL'?'signal-sell':'signal-hold');
    document.getElementById('directionText').textContent = 'Direction: ' + d.direction;

    document.getElementById('probUp').textContent = d.probability_up;
    document.getElementById('probDown').textContent = d.probability_down;
    const probClass = d.probability_up > 50 ? 'up' : 'down';
    document.getElementById('probValue').className = 'value ' + probClass;
    document.getElementById('probValue').textContent = d.probability_up + '% UP';

    document.getElementById('confValue').className = 'value ' + (d.confidence>50?'up':'neutral');
    document.getElementById('confValue').textContent = d.confidence + '%';
    document.getElementById('thrUp').textContent = d.no_trade_high;
    document.getElementById('thrDn').textContent = d.no_trade_low;

    document.getElementById('priceValue').className = 'value';
    document.getElementById('priceValue').textContent = d.current_price;
    document.getElementById('lotVal').textContent = d.pos_fraction;
    document.getElementById('atrVal').textContent = d.atr_pct + '%';
    document.getElementById('slVal').textContent = d.sl_price || '--';
    document.getElementById('tpVal').textContent = d.tp_price || '--';

    const dates = d.recent_dates;
    const probs = d.recent_probs;
    const prices = d.recent_prices;
    const colors = probs.map(p => p > d.no_trade_high*100 ? '#3fb950' : p < d.no_trade_low*100 ? '#f85149' : '#8b949e');
    const lowerThr = (d.no_trade_low*100).toFixed(1);
    const upperThr = (d.no_trade_high*100).toFixed(1);

    Plotly.newPlot('chartPrice', [{
      x:dates, y:prices, mode:'lines+markers',
      line:{color:'#58a6ff',width:1.5},
      marker:{size:6, color:colors}
    }], {
      ...PLOTLY_LAYOUT, title:{text:'Price + Signals',font:{color:'#c9d1d9',size:14}},
      yaxis:{...PLOTLY_LAYOUT.yaxis,title:'EUR/USD'}
    }, {responsive:true});

    Plotly.newPlot('chartProb', [{
      x:dates, y:probs, mode:'lines+markers',
      line:{color:'#d2a8ff',width:1.5},
      marker:{size:5}
    }], {
      ...PLOTLY_LAYOUT,
      title:{text:'P(UP) Over Time',font:{color:'#c9d1d9',size:14}},
      yaxis:{...PLOTLY_LAYOUT.yaxis,title:'P(UP) %',range:[30,70]},
      shapes:[
        {type:'line',x0:dates[0],x1:dates[dates.length-1],y0:'50',y1:'50',
         line:{color:'#484f58',width:1,dash:'dot'}},
        {type:'line',x0:dates[0],x1:dates[dates.length-1],y0:upperThr,y1:upperThr,
         line:{color:'#3fb950',width:1,dash:'dash'}},
        {type:'line',x0:dates[0],x1:dates[dates.length-1],y0:lowerThr,y1:lowerThr,
         line:{color:'#f85149',width:1,dash:'dash'}}
      ]
    }, {responsive:true});

    const strength = probs.map(p => (p - 50) / 50);
    const strColors = strength.map(s => s > 0 ? '#3fb950' : '#f85149');
    Plotly.newPlot('chartSignalStrength', [{
      x:dates, y:strength, type:'bar',
      marker:{color:strColors}
    }], {
      ...PLOTLY_LAYOUT,
      title:{text:'Signal Strength',font:{color:'#c9d1d9',size:14}},
      yaxis:{...PLOTLY_LAYOUT.yaxis,title:'Strength',range:[-0.5,0.5]},
      shapes:[{type:'line',x0:dates[0],x1:dates[dates.length-1],y0:0,y1:0,
               line:{color:'#484f58',width:1}}]
    }, {responsive:true});

    Plotly.newPlot('chartDistribution', [{
      x:probs, type:'histogram',
      marker:{color:'#388bfd'},
      nbinsx:15
    }], {
      ...PLOTLY_LAYOUT,
      title:{text:'P(UP) Distribution (Recent)',font:{color:'#c9d1d9',size:14}},
      xaxis:{...PLOTLY_LAYOUT.xaxis,title:'P(UP) %'},
      yaxis:{...PLOTLY_LAYOUT.yaxis,title:'Count'},
      shapes:[
        {type:'line',x0:'50',x1:'50',y0:0,y1:1,yref:'paper',
         line:{color:'#484f58',width:1,dash:'dot'}},
        {type:'line',x0:upperThr,x1:upperThr,y0:0,y1:1,yref:'paper',
         line:{color:'#3fb950',width:1,dash:'dash'}},
        {type:'line',x0:lowerThr,x1:lowerThr,y0:0,y1:1,yref:'paper',
         line:{color:'#f85149',width:1,dash:'dash'}}
      ]
    }, {responsive:true});

  }).catch(err=>{
    console.error(err);
    document.getElementById('statusDot').className='status-dot err';
  });
}

updateDashboard();
setInterval(updateDashboard, 60000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def status():
    with lock:
        return jsonify({
            "status": "running",
            "last_update": last_update.isoformat() if last_update else None,
            "data_available": latest_data is not None,
            "model_loaded": model is not None,
        })


@app.route("/api/predict")
def predict_api():
    data = get_prediction_data()
    if data is None:
        return jsonify({"error": "No data available"}), 503

    response = {
        "timestamp": data["timestamp"],
        "current_price": data["current_price"],
        "direction": data["direction"],
        "probability_up": data["probability_up"],
        "probability_down": data["probability_down"],
        "confidence": data["confidence"],
        "signal": data["signal"],
        "threshold_up": data["no_trade_high"],
        "threshold_down": data["no_trade_low"],
        "pos_fraction": data["pos_fraction"],
        "sl_price": data["sl_price"],
        "tp_price": data["tp_price"],
        "atr_value": data["atr_value"],
        "atr_pct": data["atr_pct"],
        "risk_per_trade": data["risk_per_trade"],
        "temperature": data["temperature"],
        "recent_probs": data["recent_probs"],
        "recent_dates": data["recent_dates"],
        "recent_prices": data["recent_prices"],
    }
    return jsonify(response)


if __name__ == "__main__":
    print("Loading initial data...")
    try:
        build_dataset()
        feats = load_dataset()
        prices = load_raw_prices()["EUR_USD"].values
        with lock:
            latest_data = {
                "features": feats,
                "prices": prices,
                "timestamp": datetime.now(),
            }
            last_update = datetime.now()
        print("Initial data loaded successfully")
    except Exception as e:
        print(f"Error loading initial data: {e}")

    load_model_and_scalers()

    thread = threading.Thread(target=update_data, daemon=True)
    thread.start()

    print("Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)