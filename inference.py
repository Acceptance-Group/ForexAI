import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import DATA_CONFIG, RISK_CONFIG, DEVICE, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_WEIGHTS
from data_loader import build_dataset, load_raw_prices
from model import ForexClassifier


def run_inference():
    print("Fetching latest data...")
    df = build_dataset()
    prices_df = load_raw_prices()

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    lookback = DATA_CONFIG["lookback"]
    scaled = scaler.transform(df.values)

    last_seq = scaled[-lookback:]
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_up = model.predict_proba(X).detach().cpu().item()

    current_price = prices_df["EUR_USD"].iloc[-1]
    no_trade_low = DATA_CONFIG["no_trade_low"]
    no_trade_high = DATA_CONFIG["no_trade_high"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    risk_pct = RISK_CONFIG["risk_per_trade"]
    max_pos_frac = RISK_CONFIG["max_position_fraction"]

    direction = "UP" if prob_up > 0.5 else "DOWN"
    if prob_up > no_trade_high:
        signal = "BUY"
    elif prob_up < no_trade_low:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = abs(prob_up - 0.5) * 2.0

    eurusd = prices_df["EUR_USD"].values
    atr = np.mean(np.abs(np.diff(eurusd[-20:])))
    sl_pct = atr / current_price if current_price > 0 else 0.01
    tp_pct = sl_pct * tp_sl_ratio
    pos_fraction = min(risk_pct / sl_pct, max_pos_frac) if sl_pct > 0 else 0.0

    if signal == "BUY":
        sl_price = current_price - atr
        tp_price = current_price + atr * tp_sl_ratio
    elif signal == "SELL":
        sl_price = current_price + atr
        tp_price = current_price - atr * tp_sl_ratio
    else:
        sl_price = 0.0
        tp_price = 0.0

    print(f"\n=== Inference Result ===")
    print(f"Current Price: {current_price:.5f}")
    print(f"P(UP)       : {prob_up:.4f}")
    print(f"P(DOWN)     : {1 - prob_up:.4f}")
    print(f"Direction   : {direction}")
    print(f"Signal      : {signal}")
    print(f"Confidence  : {confidence:.2%}")
    print(f"No-Trade    : [{no_trade_low}, {no_trade_high}]")
    print(f"TP/SL Ratio: {tp_sl_ratio}")
    print(f"Position    : {pos_fraction:.3f}")
    if signal != "HOLD":
        print(f"SL Price    : {sl_price:.5f}")
        print(f"TP Price    : {tp_price:.5f}")
        print(f"ATR         : {atr:.5f}")


if __name__ == "__main__":
    run_inference()