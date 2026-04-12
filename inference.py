import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from config import DATA_CONFIG, DEVICE, MODEL_SAVE_PATH, SCALER_SAVE_PATH
from data_loader import build_dataset
from model import ForexPredictor


def run_inference():
    print("Fetching latest data...")
    df = build_dataset()

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexPredictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    lookback = DATA_CONFIG["lookback"]
    scaled = scaler.transform(df.values)

    last_seq = scaled[-lookback:]
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(X).cpu().item()

    dummy = np.zeros((1, scaled.shape[1]))
    dummy[0, 0] = pred_scaled
    pred_real = scaler.inverse_transform(dummy)[0, 0]

    last_close = df.values[-1, 0]
    direction = "UP" if pred_real > last_close else "DOWN"

    print(f"\n=== Inference Result ===")
    print(f"Last Close  : {last_close:.5f}")
    print(f"Prediction  : {pred_real:.5f}")
    print(f"Direction   : {direction}")
    print(f"Change      : {pred_real - last_close:.5f} ({(pred_real / last_close - 1) * 100:.3f}%)")


if __name__ == "__main__":
    run_inference()