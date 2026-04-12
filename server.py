import os
import pickle
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from flask import Flask, jsonify, send_file
from sklearn.preprocessing import StandardScaler

from config import DATA_CONFIG, DEVICE, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLUMNS, FEATURE_WEIGHTS
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


@app.route("/")
def index():
    return """
    <h1>EUR/USD Classifier Server</h1>
    <p>Classification model with corrected direction.</p>
    <ul>
        <li><a href="/forecast">View Forecast Chart</a></li>
        <li><a href="/api/status">Status (JSON)</a></li>
        <li><a href="/api/predict">Prediction (JSON)</a></li>
    </ul>
    """


@app.route("/forecast")
def forecast():
    chart_path = generate_forecast_chart()
    if chart_path and os.path.exists(chart_path):
        return send_file(chart_path, mimetype="image/png")
    return "Error generating forecast", 500


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
    model, scaler = load_model_and_scalers()
    with lock:
        if latest_data is None:
            return jsonify({"error": "No data available"}), 503
        feats = latest_data["features"]
        prices = latest_data["prices"]

    scaled = scaler.transform(feats.values)
    lookback = DATA_CONFIG["lookback"]
    current_price = prices[-1]
    threshold = DATA_CONFIG["trade_threshold"]

    with torch.no_grad():
        seq = scaled[-lookback:]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prob = model.predict_proba(X).cpu().item()
        signal, confidence, _ = model.trading_signal(X, threshold=threshold)

    direction = "UP" if prob > 0.5 else "DOWN"
    signal_str = "BUY" if signal.item() > 0 else "SELL" if signal.item() < 0 else "HOLD"

    daily_ranges = np.diff(prices[-20:])
    atr = np.mean(np.abs(daily_ranges))
    atr_pct = atr / current_price * 100

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "current_price": round(current_price, 5),
        "direction": direction,
        "probability_up": round(prob * 100, 1),
        "probability_down": round((1 - prob) * 100, 1),
        "confidence": round(confidence.item() * 100, 1),
        "signal": signal_str,
        "threshold": threshold,
        "atr_pct": round(atr_pct, 4),
        "temperature": model.temperature,
    })


def generate_forecast_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    model, scaler = load_model_and_scalers()
    with lock:
        if latest_data is None:
            return None
        feats = latest_data["features"]
        prices = latest_data["prices"]

    scaled = scaler.transform(feats.values)
    lookback = DATA_CONFIG["lookback"]
    current_price = prices[-1]
    last_date = feats.index[-1]
    threshold = DATA_CONFIG["trade_threshold"]

    n_days = min(30, len(feats) - lookback)
    probs = []
    dates_recent = []

    with torch.no_grad():
        for i in range(n_days):
            idx = len(feats) - n_days + i
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p = model.predict_proba(X).cpu().item()
            probs.append(p)
            dates_recent.append(feats.index[idx])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    with torch.no_grad():
        X = torch.tensor(scaled[-lookback:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prob_now = model.predict_proba(X).cpu().item()
        _, conf_now, _ = model.trading_signal(X, threshold=threshold)

    direction = "UP" if prob_now > 0.5 else "DOWN"
    with torch.no_grad():
        sig, _, _ = model.trading_signal(X, threshold=threshold)
        signal_str = "BUY" if sig.item() > 0 else "SELL" if sig.item() < 0 else "HOLD"

    fig.suptitle(
        f"EUR/USD  |  {current_price:.5f}  |  {direction} ({prob_now*100:.1f}%)  |  "
        f"{signal_str}  |  T={model.temperature}",
        fontsize=13, fontweight="bold"
    )

    # 1. Signal strength
    ax = axes[0, 0]
    signal_strength = [(p - 0.5) * 2 for p in probs]
    colors = ["green" if s > 0 else "red" for s in signal_strength]
    ax.bar(dates_recent, signal_strength, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Signal Strength")
    ax.set_ylabel("(P(UP)-0.5) x 2")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2. Price + signals
    ax = axes[0, 1]
    n_price = min(60, len(prices))
    recent_prices = prices[-n_price:]
    recent_price_dates = feats.index[-n_price:]
    ax.plot(recent_price_dates, recent_prices, "b-", linewidth=1.0, label="EUR/USD")

    for i, d in enumerate(dates_recent):
        color = "green" if probs[i] > threshold else "red" if probs[i] < (1 - threshold) else "gray"
        idx_offset = n_price - n_days + i
        if 0 <= idx_offset < n_price:
            ax.plot(d, recent_prices[idx_offset], "o", color=color, markersize=5)

    ax.set_title("Price + Signals (green=BUY, red=SELL)")
    ax.set_ylabel("EUR/USD")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 3. Probability histogram
    ax = axes[1, 0]
    ax.hist(probs, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="0.5")
    ax.axvline(threshold, color="orange", linestyle=":", linewidth=1.0, label=f"Threshold={threshold}")
    ax.axvline(1 - threshold, color="orange", linestyle=":", linewidth=1.0)
    ax.set_title(f"Recent P(UP) Distribution (T={model.temperature})")
    ax.set_xlabel("P(UP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Direction:  {direction}\n\n"
        f"P(UP):      {prob_now*100:.1f}%\n"
        f"P(DOWN):    {(1-prob_now)*100:.1f}%\n\n"
        f"Confidence: {conf_now.item()*100:.1f}%\n"
        f"Signal:     {signal_str}\n\n"
        f"Price:       {current_price:.5f}\n"
        f"Temperature: {model.temperature}"
    )
    color = "green" if prob_now > 0.5 else "red"
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=13,
            verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            color=color, family="monospace")

    plt.tight_layout()
    chart_path = "forecast_4panel.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


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