import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, DEVICE, RISK_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels
from model import ForexClassifier


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max
    return dd.min()


def calibrate_and_backtest():
    feats_df = load_dataset()
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    eurusd = prices_df["EUR_USD"].values

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scaled = scaler.transform(feats_df.values)
    lookback = DATA_CONFIG["lookback"]

    all_probs = []
    all_labels = []
    for idx in range(lookback, len(feats_df)):
        seq = scaled[idx - lookback:idx]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = model.predict_proba(X).detach().cpu().item()
        all_probs.append(prob)
        label_row = labels_df.iloc[idx - lookback + lookback]
        label_val = (label_row["label"] + 1) / 2.0
        all_labels.append(label_val)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    bt_start = pd.Timestamp(BACKTEST_CONFIG["start_date"])
    bt_end = pd.Timestamp(BACKTEST_CONFIG["end_date"])

    cal_end = bt_start
    cal_start = pd.Timestamp("2020-01-01")

    cal_mask = (feats_df.index[lookback:] >= cal_start) & (feats_df.index[lookback:] < cal_end)
    bt_mask = (feats_df.index[lookback:] >= bt_start) & (feats_df.index[lookback:] < bt_end)

    cal_probs = all_probs[cal_mask]
    cal_labels = all_labels[cal_mask]

    print(f"Calibration set: {len(cal_probs)} samples")
    print(f"  Raw prob range: [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")
    print(f"  Raw prob mean: {cal_probs.mean():.3f}, std: {cal_probs.std():.4f}")

    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(cal_probs, cal_labels)
    cal_probs_iso = iso_cal.transform(cal_probs)
    print(f"  Calibrated prob range: [{cal_probs_iso.min():.3f}, {cal_probs_iso.max():.3f}]")
    print(f"  Calibrated prob mean: {cal_probs_iso.mean():.3f}, std: {cal_probs_iso.std():.4f}")

    no_trade_low = 0.42
    no_trade_high = 0.58
    tp_sl_ratio = 2.5
    risk_pct = 0.01

    ma_period = DATA_CONFIG.get("trend_filter_ma", 50) or 50
    ma_vals = pd.Series(eurusd).rolling(ma_period).mean().values

    atr_series = pd.Series(eurusd).diff().abs().rolling(DATA_CONFIG["atr_period"]).mean()
    atr_mean = atr_series.rolling(60).mean()
    atr_threshold = atr_series.rolling(60).quantile(0.15)

    bt_probs = all_probs[bt_mask]
    bt_probs_iso = iso_cal.transform(bt_probs)

    bt_indices = np.where(bt_mask)[0]
    bt_indices = bt_indices + lookback
    bt_indices = bt_indices[bt_indices < len(eurusd) - 1]

    strategies = {
        "raw_nt4258": {"cal": False, "ntl": 0.42, "nth": 0.58, "tsl": 2.5},
        "raw_nt4555": {"cal": False, "ntl": 0.45, "nth": 0.55, "tsl": 2.5},
        "raw_nt4852": {"cal": False, "ntl": 0.48, "nth": 0.52, "tsl": 2.5},
        "cal_nt4555": {"cal": True, "ntl": 0.45, "nth": 0.55, "tsl": 2.5},
        "cal_nt4258": {"cal": True, "ntl": 0.42, "nth": 0.58, "tsl": 2.5},
        "cal_nt3565": {"cal": True, "ntl": 0.35, "nth": 0.65, "tsl": 2.5},
    }

    for name, cfg in strategies.items():
        ntl = cfg["ntl"]
        nth = cfg["nth"]
        use_cal = cfg["cal"]
        tsl = cfg["tsl"]

        returns_list = []
        n_trades = 0
        n_up = 0
        n_dn = 0
        n_win = 0
        n_loss = 0

        for bi, idx in enumerate(bt_indices):
            if use_cal:
                prob = bt_probs_iso[bi] if bi < len(bt_probs_iso) else bt_probs_iso[-1]
            else:
                prob = bt_probs[bi] if bi < len(bt_probs) else bt_probs[-1]

            current_price = eurusd[idx - 1]
            next_price = eurusd[idx]
            actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0

            idx_date = feats_df.index[idx] if idx < len(feats_df) else feats_df.index[-1]
            atr_val = atr_series.iloc[idx] if idx < len(atr_series) else np.nan
            atr_thr = atr_threshold.iloc[idx] if idx < len(atr_threshold) else np.nan
            atr_mean_val = atr_mean.iloc[idx] if idx < len(atr_mean) else np.nan
            ma_val = ma_vals[idx] if idx < len(ma_vals) else np.nan

            atr_ok = not np.isnan(atr_val) and not np.isnan(atr_thr) and atr_val > atr_thr

            if atr_ok:
                if prob > nth:
                    signal = 1.0
                elif prob < ntl:
                    signal = -1.0
                else:
                    signal = 0.0

                if signal != 0.0:
                    sl_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.01
                    if not np.isnan(atr_mean_val) and atr_mean_val > 0:
                        vol_ratio = atr_val / atr_mean_val
                        dyn_tp_sl = tsl / max(vol_ratio, 0.5)
                        dyn_tp_sl = min(dyn_tp_sl, 3.0)
                    else:
                        dyn_tp_sl = tsl

                    pos_frac = min(risk_pct / sl_pct, 1.0) if sl_pct > 0 else 0.0

                    if signal > 0:
                        if actual_ret >= sl_pct * dyn_tp_sl:
                            capped = sl_pct * dyn_tp_sl
                        elif actual_ret <= -sl_pct:
                            capped = -sl_pct
                        else:
                            capped = actual_ret
                        n_up += 1
                    else:
                        if actual_ret <= -sl_pct * dyn_tp_sl:
                            capped = sl_pct * dyn_tp_sl
                        elif actual_ret >= sl_pct:
                            capped = -sl_pct
                        else:
                            capped = -actual_ret
                        n_dn += 1

                    ret = signal * pos_frac * capped
                    if ret > 0:
                        n_win += 1
                    else:
                        n_loss += 1
                    returns_list.append(ret)
                    n_trades += 1
                else:
                    returns_list.append(0.0)
            else:
                returns_list.append(0.0)

        returns = np.array(returns_list)
        cum_ret = np.cumprod(1 + returns) - 1
        pf = profit_factor(returns[returns != 0]) if np.any(returns != 0) else 0
        md = max_drawdown(returns) if len(returns) > 0 else 0

        da_raw = np.mean(((bt_probs[:len(bt_indices)] > 0.5) == ((np.array([np.log(eurusd[i+1]/eurusd[i]) if i < len(eurusd)-1 else 0 for i in bt_indices])) > 0)) * 100)

        print(f"\n  {name}: Net=${10000*(1+cum_ret[-1]):,.0f} PF={pf:.2f} MD={abs(md)*100:.1f}% "
              f"Trades={n_trades} UP={n_up} DN={n_dn} WR={n_win/max(n_win+n_loss,1)*100:.1f}%")


if __name__ == "__main__":
    calibrate_and_backtest()